from __future__ import annotations

import mujoco
import numpy as np

from src.control.locomotion import G1LocomotionPolicy

LEG_ACTUATOR_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


class PolicyWalker:
    """基于 unitree_rl_gym 预训练策略的真实行走控制器（PD 力矩 + RL 策略）。"""

    def __init__(self, locomotion: G1LocomotionPolicy, walk_cfg: dict, leg_actuator_ids: list[int]) -> None:
        self.locomotion = locomotion
        self.walk_cfg = walk_cfg
        self.leg_ids = leg_actuator_ids
        self.active = False
        self.lock_planar = False
        self._lock_xy: np.ndarray | None = None
        self._lock_z: float | None = None
        self._lock_orientation = True
        self._lock_yaw: float | None = None
        self._assist_target: np.ndarray | None = None
        self._upper_body_ctrl: dict | None = None
        self._hand_ctrl: tuple[list[int], np.ndarray] | None = None

    def reset(self, env) -> None:
        self.locomotion.reset()
        self.active = False
        self.lock_planar = False
        self._lock_xy = None
        self._lock_z = None
        self._lock_orientation = True
        self._lock_yaw = None
        self._assist_target = None
        self._upper_body_ctrl = None
        self._hand_ctrl = None

    def deactivate(self) -> None:
        self.locomotion.deactivate()
        self.active = False
        self.lock_planar = False
        self._lock_xy = None
        self._lock_z = None
        self._lock_orientation = True
        self._lock_yaw = None
        self._assist_target = None
        self._upper_body_ctrl = None
        self._hand_ctrl = None

    def set_upper_body(
        self,
        upper_body_ctrl: dict[str, np.ndarray] | None,
        hand_ids: list[int] | None = None,
        hand_pose: np.ndarray | None = None,
    ) -> None:
        """保存自定义上半身姿态；RL 策略仅控腿，腰/臂/手每步强制跟踪。"""
        self._upper_body_ctrl = upper_body_ctrl
        if hand_ids is not None and hand_pose is not None:
            self._hand_ctrl = (hand_ids, np.asarray(hand_pose, dtype=float))

    def enable_planar_lock(
        self,
        env,
        lock_z: bool = False,
        height: float | None = None,
        refresh_xy: bool = True,
        lock_orientation: bool = True,
        lock_yaw: float | None = None,
    ) -> None:
        """操作阶段固定骨盆；lock_yaw 指定朝向（默认用 spawn yaw）。"""
        self.lock_planar = True
        self.locomotion.deactivate()
        self.active = False
        self._assist_target = None
        self._lock_orientation = lock_orientation
        self._lock_yaw = lock_yaw
        if refresh_xy or self._lock_xy is None:
            self._lock_xy = env.pelvis_xy().copy()
        if height is not None:
            self._lock_z = float(height)
        elif lock_z:
            self._lock_z = float(env.data.qpos[2])
        else:
            self._lock_z = None

    def set_lock_height(self, height: float | None) -> None:
        self._lock_z = height

    def apply(
        self,
        env,
        target_xy: np.ndarray,
        upper_body_ctrl: dict[str, np.ndarray] | None = None,
        assist: bool = False,
    ) -> bool:
        self.active = True
        self.lock_planar = False
        self.locomotion.active = True
        self._assist_target = np.asarray(target_xy, dtype=float) if assist else None

        threshold = float(self.walk_cfg.get("arrive_threshold", 0.12))
        saved: dict[str, float] = {}
        if env.grasped_box is not None:
            for key, default in [
                ("cmd_max_forward", 0.45),
                ("cmd_min_forward", 0.30),
                ("cmd_turn_forward", 0.18),
            ]:
                saved[key] = float(self.locomotion.cfg.get(key, default))
            self.locomotion.cfg["cmd_max_forward"] = float(
                self.walk_cfg.get("carry_cmd_max_forward", 0.25)
            )
            self.locomotion.cfg["cmd_min_forward"] = 0.12
            self.locomotion.cfg["cmd_turn_forward"] = 0.10

        cmd, arrived = self.locomotion.compute_cmd(env, target_xy, threshold)
        for key, val in saved.items():
            self.locomotion.cfg[key] = val
        self.locomotion.cmd = cmd

        self.set_upper_body(upper_body_ctrl)

        if arrived:
            self.locomotion.deactivate()
            self.active = False
            self._assist_target = None
        return arrived

    def hold_balance(self, env, upper_body_ctrl: dict[str, np.ndarray] | None = None) -> None:
        """原地站立平衡：策略 cmd=0，由 RL 控腿，上层只控腰/臂。"""
        self.active = True
        self.lock_planar = False
        self.locomotion.active = True
        self.locomotion.cmd[:] = 0.0
        self._assist_target = None
        self.set_upper_body(upper_body_ctrl)

    @staticmethod
    def _apply_upper_body(env, upper_body_ctrl: dict) -> None:
        for name, ids_key in [
            ("waist", "waist_ids"),
            ("arm", "arm_ids"),
            ("left_arm", "left_arm_ids"),
        ]:
            if ids_key in upper_body_ctrl and name in upper_body_ctrl:
                for i, aid in enumerate(upper_body_ctrl[ids_key]):
                    env.data.ctrl[aid] = float(upper_body_ctrl[name][i])

    def _apply_custom_upper(self, env) -> None:
        if self._upper_body_ctrl is not None:
            self._apply_upper_body(env, self._upper_body_ctrl)
        if self._hand_ctrl is not None:
            ids, pose = self._hand_ctrl
            for i, aid in enumerate(ids):
                env.data.ctrl[aid] = float(pose[i])

    def _kinematic_upper(self, env) -> None:
        """持箱时直接把腰/臂/手关节角写进 qpos，避免被 RL 摆腿甩飞。"""
        if self._upper_body_ctrl is None:
            return
        self._apply_custom_upper(env)
        model = env.model
        data = env.data
        pairs = [
            (self._upper_body_ctrl.get("waist_ids"), self._upper_body_ctrl.get("waist")),
            (self._upper_body_ctrl.get("arm_ids"), self._upper_body_ctrl.get("arm")),
            (self._upper_body_ctrl.get("left_arm_ids"), self._upper_body_ctrl.get("left_arm")),
        ]
        for ids, pose in pairs:
            if ids is None or pose is None:
                continue
            for i, aid in enumerate(ids):
                joint_id = int(model.actuator(aid).trnid[0])
                qa = int(model.jnt_qposadr[joint_id])
                da = int(model.jnt_dofadr[joint_id])
                data.qpos[qa] = float(pose[i])
                data.qvel[da] = 0.0
        if self._hand_ctrl is not None:
            ids, pose = self._hand_ctrl
            for i, aid in enumerate(ids):
                joint_id = int(model.actuator(aid).trnid[0])
                qa = int(model.jnt_qposadr[joint_id])
                da = int(model.jnt_dofadr[joint_id])
                data.qpos[qa] = float(pose[i])
                data.qvel[da] = 0.0
        mujoco.mj_forward(model, data)

    def _sync_upper(self, env) -> None:
        if self._upper_body_ctrl is not None and (
            env.grasped_box is not None or self.lock_planar
        ):
            self._kinematic_upper(env)
        elif self.active or self.lock_planar:
            self._apply_custom_upper(env)

    def hold_carry(
        self,
        env,
        leg_pose: np.ndarray,
        upper_body_ctrl: dict[str, np.ndarray] | None = None,
    ) -> None:
        """搬运稳定：锁骨盆 + 固定腿角，不用 RL（避免偏载失衡）。"""
        self.locomotion.deactivate()
        self.active = False
        self.lock_planar = True
        self._assist_target = None
        for i, aid in enumerate(self.leg_ids):
            env.data.ctrl[aid] = float(leg_pose[i])
        self.set_upper_body(upper_body_ctrl)

    def _apply_velocity_assist(self, env) -> None:
        """速度辅助：RL 摆腿 + 物理推力，不用 kinematic 平移。"""
        if self._assist_target is None:
            return
        pos = env.pelvis_xy()
        diff = self._assist_target - pos
        dist = float(np.linalg.norm(diff))
        threshold = float(self.walk_cfg.get("arrive_threshold", 0.22))
        if dist <= threshold:
            return

        pelvis_id = env.model.body("pelvis").id
        direction = diff / max(dist, 1e-6)
        speed = float(self.walk_cfg.get("walk_assist_speed", 0.30))
        if env.grasped_box is not None and dist < 0.35:
            speed = max(speed, 0.32)
        desired_speed = min(speed, max(0.12, dist * 0.8))
        desired_v = direction * desired_speed

        pelvis_vel = self.data_cvel_xy(env, pelvis_id)
        mass = max(float(env.model.body_mass[pelvis_id]), 1.0)
        kp = float(self.walk_cfg.get("walk_assist_kp", 700.0))
        kd = float(self.walk_cfg.get("walk_assist_kd", 80.0))
        force_xy = kp * mass * (desired_v - pelvis_vel) - kd * mass * pelvis_vel
        env.data.xfrc_applied[pelvis_id, 0] += float(force_xy[0])
        env.data.xfrc_applied[pelvis_id, 1] += float(force_xy[1])

    @staticmethod
    def data_cvel_xy(env, body_id: int) -> np.ndarray:
        return np.array(
            [env.data.cvel[body_id, 3], env.data.cvel[body_id, 4]], dtype=float
        )

    def pre_physics(self, env) -> None:
        """物理步进前：RL 只写腿部力矩；持箱时上半身运动学锁定。"""
        env.data.qfrc_applied[:] = 0.0
        self._sync_upper(env)
        if self.active:
            self.locomotion.apply_leg_torques(env)
            self._apply_velocity_assist(env)

    def post_physics(self, env) -> None:
        """物理步进后：更新策略；持箱时再次锁定上肢。"""
        if self.active:
            self.locomotion.maybe_update_policy(env)
        elif self.lock_planar:
            self._hold_planar(env)
        if env.grasped_box is not None and self._upper_body_ctrl is not None:
            self._kinematic_upper(env)
        elif self.lock_planar and self._upper_body_ctrl is not None:
            self._kinematic_upper(env)

    def on_physics_step(self, env) -> None:
        """兼容旧调用：拆分为 pre/post。"""
        self.pre_physics(env)

    def _hold_planar(self, env) -> None:
        if self._lock_xy is not None:
            env.data.qpos[0:2] = self._lock_xy
            env.data.qvel[0:2] = 0.0
        if self._lock_z is not None:
            env.data.qpos[2] = self._lock_z
            env.data.qvel[2] = 0.0
        if self._lock_orientation:
            yaw = self._lock_yaw
            if yaw is None:
                yaw = float(env.spawn_cfg.get("yaw", 0.0))
            qw = np.cos(yaw / 2)
            qz = np.sin(yaw / 2)
            env.data.qpos[3:7] = np.array([qw, 0.0, 0.0, qz])
            env.data.qvel[3:6] = 0.0
