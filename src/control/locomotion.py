from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from src.core.config_loader import project_root


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


def yaw_from_quat(quaternion: np.ndarray) -> float:
    qw, _, _, qz = quaternion
    return 2.0 * float(np.arctan2(qz, qw))


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


class G1LocomotionPolicy:
    """unitree_rl_gym 预训练 G1 行走策略（TorchScript + PD 力矩控制）。"""

    def __init__(
        self,
        cfg: dict,
        leg_actuator_ids: list[int],
        leg_qpos_indices: list[int],
        leg_dof_indices: list[int],
    ) -> None:
        self.cfg = cfg
        self.leg_actuator_ids = leg_actuator_ids
        self.leg_qpos_indices = leg_qpos_indices
        self.leg_dof_indices = leg_dof_indices
        self.num_actions = int(cfg["num_actions"])
        self.num_obs = int(cfg["num_obs"])
        self.decimation = int(cfg["control_decimation"])
        self.simulation_dt = float(cfg["simulation_dt"])

        self.kps = np.asarray(cfg["kps"], dtype=np.float32)
        self.kds = np.asarray(cfg["kds"], dtype=np.float32)
        self.default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.dof_pos_scale = float(cfg["dof_pos_scale"])
        self.dof_vel_scale = float(cfg["dof_vel_scale"])
        self.action_scale = float(cfg["action_scale"])
        self.cmd_scale = np.asarray(cfg["cmd_scale"], dtype=np.float32)
        self.phase_period = float(cfg.get("phase_period", 0.8))

        policy_path = project_root() / cfg["policy_path"]
        if not policy_path.exists():
            raise FileNotFoundError(
                f"未找到行走策略 {policy_path}，请运行: bash scripts/download_policy.sh"
            )
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.cmd = np.zeros(3, dtype=np.float32)
        self.counter = 0
        self.active = False

    def reset(self) -> None:
        self.action[:] = 0.0
        self.target_dof_pos = self.default_angles.copy()
        self.cmd[:] = 0.0
        self.counter = 0
        self.active = False

    def warm_start_from_env(self, env) -> None:
        """从当前腿关节角平滑接管，避免搬运后突然切换默认站姿。"""
        self.target_dof_pos = self._leg_qpos(env).copy()
        self.action[:] = 0.0
        self.cmd[:] = 0.0
        self.counter = 0
        self.active = True

    def deactivate(self) -> None:
        self.active = False
        self.cmd[:] = 0.0

    def compute_cmd(self, env, target_xy: np.ndarray, arrive_threshold: float) -> tuple[np.ndarray, bool]:
        pos = env.pelvis_xy()
        diff = target_xy - pos
        dist = float(np.linalg.norm(diff))
        if dist <= arrive_threshold:
            return np.zeros(3, dtype=np.float32), True

        quat = env.data.qpos[3:7]
        yaw = yaw_from_quat(quat)
        desired_yaw = float(np.arctan2(diff[1], diff[0]))
        yaw_err = _wrap_pi(desired_yaw - yaw)

        forward = float(self.cfg.get("cmd_max_forward", 0.45))
        lateral = float(self.cfg.get("cmd_max_lateral", 0.18))
        max_yaw = float(self.cfg.get("cmd_max_yaw", 0.40))
        turn_yaw = float(self.cfg.get("cmd_turn_yaw_threshold", 0.35))
        turn_vx = float(self.cfg.get("cmd_turn_forward", 0.12))
        min_vx = float(self.cfg.get("cmd_min_forward", 0.22))
        speed_gain = float(self.cfg.get("speed_distance_gain", 0.75))

        if abs(yaw_err) > turn_yaw:
            vx = turn_vx
            vy = 0.0
            omega = float(np.clip(yaw_err * 1.2, -max_yaw, max_yaw))
        else:
            vx = min(forward, max(min_vx, dist * speed_gain))
            vy = float(np.clip(np.dot(diff / max(dist, 1e-6),
                                       np.array([-np.sin(yaw), np.cos(yaw)])) * 0.55,
                               -lateral, lateral))
            omega = float(np.clip(yaw_err * 1.5, -max_yaw, max_yaw))

        return np.array([vx, vy, omega], dtype=np.float32), False

    def _leg_qpos(self, env) -> np.ndarray:
        return np.array([env.data.qpos[i] for i in self.leg_qpos_indices], dtype=np.float32)

    def _leg_qvel(self, env) -> np.ndarray:
        return np.array([env.data.qvel[i] for i in self.leg_dof_indices], dtype=np.float32)

    def _build_obs(self, env) -> np.ndarray:
        qj = self._leg_qpos(env)
        dqj = self._leg_qvel(env)
        quat = env.data.qpos[3:7]
        omega = env.data.qvel[3:6]

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale
        grav = gravity_orientation(quat)
        omega = omega * self.ang_vel_scale

        t = self.counter * self.simulation_dt
        phase = (t % self.phase_period) / self.phase_period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = omega
        obs[3:6] = grav
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9 : 9 + self.num_actions] = qj
        obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj
        obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action
        obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = [sin_phase, cos_phase]
        return obs

    def apply_leg_torques(self, env) -> None:
        """PD 力矩写入 qfrc_applied；position 执行器设为当前角以避免与 kp=500 冲突。"""
        q = self._leg_qpos(env)
        dq = self._leg_qvel(env)
        tau = pd_control(
            self.target_dof_pos,
            q,
            self.kps,
            np.zeros_like(self.kds),
            dq,
            self.kds,
        )
        for i, dof_i in enumerate(self.leg_dof_indices):
            env.data.qfrc_applied[dof_i] = float(tau[i])
        for qpos_i, aid in zip(self.leg_qpos_indices, self.leg_actuator_ids):
            env.data.ctrl[aid] = float(env.data.qpos[qpos_i])

    def maybe_update_policy(self, env) -> None:
        self.counter += 1
        if self.counter % self.decimation != 0:
            return
        obs = self._build_obs(env)
        with torch.no_grad():
            act = self.policy(torch.from_numpy(obs).unsqueeze(0))
        self.action = act.detach().numpy().squeeze().astype(np.float32)
        self.target_dof_pos = self.action * self.action_scale + self.default_angles

    @classmethod
    def from_yaml(
        cls,
        model: mujoco.MjModel,
        leg_actuator_ids: list[int],
        leg_joint_names: list[str],
        path: Path | None = None,
    ) -> G1LocomotionPolicy:
        if path is None:
            path = project_root() / "config" / "locomotion_g1.yaml"
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        leg_qpos_indices: list[int] = []
        leg_dof_indices: list[int] = []
        for name in leg_joint_names:
            jid = model.joint(name).id
            leg_qpos_indices.append(int(model.jnt_qposadr[jid]))
            leg_dof_indices.append(int(model.jnt_dofadr[jid]))

        return cls(cfg, leg_actuator_ids, leg_qpos_indices, leg_dof_indices)
