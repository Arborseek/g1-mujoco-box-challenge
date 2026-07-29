from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.core.config_loader import load_config, project_root
from src.env.task import BoxPickPlaceTask
from src.control.locomotion import G1LocomotionPolicy, yaw_from_quat
from src.control.walker import PolicyWalker, LEG_ACTUATOR_NAMES


@dataclass
class BoxPickPlaceEnv:
    model: mujoco.MjModel
    data: mujoco.MjData
    task: BoxPickPlaceTask
    box_body_ids: list[int]
    hand_body_id: int
    grasp_eq_ids: list[int]
    stand_ctrl: np.ndarray
    sim_cfg: dict
    grasp_cfg: dict
    spawn_cfg: dict
    walk_cfg: dict
    table_cfg: dict
    initial_box_qpos: np.ndarray
    gait_walker: PolicyWalker
    pelvis_body_id: int
    leg_actuator_ids: list[int]
    grasped_box: int | None = None
    carry_attach: str = "wrist"
    placed_box_qpos: dict[int, np.ndarray] | None = None

    @classmethod
    def create(cls, config_path: Path | None = None) -> BoxPickPlaceEnv:
        cfg = load_config(config_path)
        scene = project_root() / "assets" / "robots" / "g1" / "box_pick_place.xml"
        if not scene.exists():
            raise FileNotFoundError(f"场景文件不存在: {scene}，请先运行 bash setup.sh")

        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        sim_cfg = cfg["simulation"]
        model.opt.timestep = float(sim_cfg["dt"])

        # 保存箱子初始关节状态（reset 时恢复）
        mujoco.mj_forward(model, data)
        box_qpos_slices = []
        for i in range(int(cfg["num_boxes"])):
            jid = model.body(f"box_{i}").jntadr[0]
            adr = model.jnt_qposadr[jid]
            box_qpos_slices.append(data.qpos[adr : adr + 7].copy())
        initial_box_qpos = np.concatenate(box_qpos_slices)

        num_boxes = int(cfg["num_boxes"])
        box_body_ids = [model.body(f"box_{i}").id for i in range(num_boxes)]
        hand_body_id = model.body(cfg["grasp"]["hand_body"]).id
        grasp_eq_ids = [model.eq(f"grasp_weld_{i}").id for i in range(num_boxes)]
        pelvis_body_id = model.body("pelvis").id
        leg_actuator_ids = [model.actuator(n).id for n in LEG_ACTUATOR_NAMES]
        locomotion = G1LocomotionPolicy.from_yaml(model, leg_actuator_ids, LEG_ACTUATOR_NAMES)
        gait_walker = PolicyWalker(locomotion, cfg["walk"], leg_actuator_ids)

        stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        stand_ctrl = np.array(model.key_ctrl[stand_id]).copy()

        task = BoxPickPlaceTask(
            num_boxes=num_boxes,
            drop_zone_center=np.asarray(cfg["drop_zone"]["center"], dtype=float),
            drop_zone_half=np.asarray(cfg["drop_zone"]["half_size"], dtype=float),
            placement_velocity_threshold=float(cfg["placement"]["velocity_threshold"]),
            placement_height_min=float(cfg["placement"]["height_min"]),
        )

        env = cls(
            model=model,
            data=data,
            task=task,
            box_body_ids=box_body_ids,
            hand_body_id=hand_body_id,
            grasp_eq_ids=grasp_eq_ids,
            stand_ctrl=stand_ctrl,
            sim_cfg=sim_cfg,
            grasp_cfg=cfg["grasp"],
            spawn_cfg=cfg["spawn"],
            walk_cfg=cfg["walk"],
            table_cfg=cfg.get("table", {}),
            initial_box_qpos=initial_box_qpos,
            gait_walker=gait_walker,
            pelvis_body_id=pelvis_body_id,
            leg_actuator_ids=leg_actuator_ids,
        )
        env.reset()
        return env

    def reset(self) -> None:
        stand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        robot_nq = len(self.model.key_qpos[stand_id])
        self.data.qpos[:robot_nq] = self.model.key_qpos[stand_id]
        self.data.qvel[:robot_nq] = 0.0
        self.data.ctrl[:] = self.stand_ctrl

        spawn = np.asarray(self.spawn_cfg["pos"], dtype=float)
        self.data.qpos[0:3] = spawn
        yaw = float(self.spawn_cfg.get("yaw", 0.0))
        self.data.qpos[3:7] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

        # 恢复所有箱子到地面初始位置
        offset = 0
        for i in range(self.task.num_boxes):
            jid = self.model.body(f"box_{i}").jntadr[0]
            adr = self.model.jnt_qposadr[jid]
            self.data.qpos[adr : adr + 7] = self.initial_box_qpos[offset : offset + 7]
            dof_adr = self.model.jnt_dofadr[jid]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0
            offset += 7

        self.grasped_box = None
        self.carry_attach = "wrist"
        self.placed_box_qpos = {}
        self.task.placed = [False] * self.task.num_boxes
        for eq_id in self.grasp_eq_ids:
            self.data.eq_active[eq_id] = 0
        for bid in self.box_body_ids:
            self.model.body_gravcomp[bid] = 0.0
        self.gait_walker.reset(self)
        mujoco.mj_forward(self.model, self.data)

    def pelvis_xy(self) -> np.ndarray:
        return self.data.qpos[0:2].copy()

    def walk_toward(
        self,
        target_xy: np.ndarray,
        upper_body: dict | None = None,
        assist: bool = False,
    ) -> bool:
        """RL 行走；assist=True 时加速度辅助（非 kinematic 平移）。"""
        return self.gait_walker.apply(self, target_xy, upper_body, assist=assist)

    def body_pos(self, body_id: int) -> np.ndarray:
        return self.data.xpos[body_id].copy()

    def body_vel(self, body_id: int) -> np.ndarray:
        return self.data.cvel[body_id, 3:6].copy()

    def hand_pos(self) -> np.ndarray:
        return self.body_pos(self.hand_body_id)

    def box_positions(self) -> list[np.ndarray]:
        return [self.body_pos(bid) for bid in self.box_body_ids]

    def box_velocities(self) -> list[np.ndarray]:
        return [self.body_vel(bid) for bid in self.box_body_ids]

    def nearest_box(self) -> tuple[int, float]:
        hand = self.hand_pos()
        best_i, best_d = -1, float("inf")
        for i, pos in enumerate(self.box_positions()):
            if self.task.placed[i]:
                continue
            d = float(np.linalg.norm(hand - pos))
            if d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

    def try_grasp(self, box_id: int | None = None) -> bool:
        if self.grasped_box is not None:
            return False

        if box_id is None:
            box_id, dist = self.nearest_box()
            if box_id < 0:
                return False
        else:
            dist = float(np.linalg.norm(
                self.hand_pos() - self.body_pos(self.box_body_ids[box_id])
            ))

        threshold = float(self.grasp_cfg["distance_threshold"])
        if dist > threshold:
            return False

        for eq_id in self.grasp_eq_ids:
            self.data.eq_active[eq_id] = 0
        self.grasped_box = box_id
        self.model.body_gravcomp[self.box_body_ids[box_id]] = 1.0
        self.sync_carried_box()
        mujoco.mj_forward(self.model, self.data)
        return True

    def stabilize_table_boxes(self) -> None:
        """未抓取/未放置、仍在拾取台的箱子保持原位，避免被机器人碰飞。"""
        offset = 0
        for i in range(self.task.num_boxes):
            if self.grasped_box == i:
                offset += 7
                continue
            jid = self.model.body(f"box_{i}").jntadr[0]
            adr = self.model.jnt_qposadr[jid]
            dof = self.model.jnt_dofadr[jid]
            if self.task.placed[i]:
                if self.placed_box_qpos and i in self.placed_box_qpos:
                    self.data.qpos[adr : adr + 7] = self.placed_box_qpos[i]
                self.data.qvel[dof : dof + 6] = 0.0
                offset += 7
                continue
            init = self.initial_box_qpos[offset : offset + 3]
            pos = self.data.qpos[adr : adr + 3]
            # 已搬离拾取台的箱子不再钉回初始位（否则放货后会被传送回来）
            if float(np.linalg.norm(pos[:2] - init[:2])) > 0.35:
                offset += 7
                continue
            self.data.qpos[adr : adr + 7] = self.initial_box_qpos[offset : offset + 7]
            self.data.qvel[dof : dof + 6] = 0.0
            offset += 7

    def _palm_pos(self, side: str = "right") -> np.ndarray:
        name = "right_wrist_yaw_link" if side == "right" else "left_wrist_yaw_link"
        body_id = self.model.body(name).id
        mat = self.data.xmat[body_id].reshape(3, 3)
        y = -0.003 if side == "right" else 0.003
        local = np.array([0.0415, y, 0.0], dtype=float)
        return self.data.xpos[body_id] + mat @ local

    def _carry_cradle_pos(self) -> np.ndarray:
        """箱心 = 双掌中点 + 身前上方（与抱箱姿态配套）。"""
        pc = 0.5 * (self._palm_pos("right") + self._palm_pos("left"))
        pelvis_mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        offset = np.asarray(
            self.grasp_cfg.get("carry_palm_offset", [0.02, 0.0, 0.04]),
            dtype=float,
        )
        return pc + pelvis_mat @ offset

    def sync_carried_box(self) -> None:
        """持箱：抓取时跟右手腕，搬运时双手抱箱。"""
        if self.grasped_box is None:
            return
        i = self.grasped_box
        jid = self.model.body(f"box_{i}").jntadr[0]
        adr = self.model.jnt_qposadr[jid]
        dof = self.model.jnt_dofadr[jid]
        mode = self.carry_attach or self.grasp_cfg.get("carry_attach", "wrist")
        if mode == "cradle":
            pos = self._carry_cradle_pos()
            pelvis_mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
            yaw = float(np.arctan2(pelvis_mat[1, 0], pelvis_mat[0, 0]))
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float)
            self.data.qpos[adr : adr + 3] = pos
            self.data.qpos[adr + 3 : adr + 7] = quat
        else:
            mat = self.data.xmat[self.hand_body_id].reshape(3, 3)
            palm = np.asarray(self.grasp_cfg.get("palm_in_wrist", [0.042, -0.003, 0.0]), float)
            box_off = np.asarray(self.grasp_cfg.get("box_in_palm", [0.06, 0.0, 0.05]), float)
            offset = mat @ (palm + box_off)
            self.data.qpos[adr : adr + 3] = self.data.xpos[self.hand_body_id] + offset
            self.data.qpos[adr + 3 : adr + 7] = self.data.xquat[self.hand_body_id].copy()
        self.data.qvel[dof : dof + 6] = 0.0

    def snap_pelvis(self, xy: np.ndarray, face_xy: np.ndarray | None = None) -> None:
        """放货/操作前把骨盆钉到安全站位，避免顶桌弹飞箱子。"""
        self.data.qpos[0:2] = np.asarray(xy, dtype=float)[:2]
        self.data.qvel[0:6] = 0.0
        if face_xy is not None:
            diff = np.asarray(face_xy, dtype=float)[:2] - self.data.qpos[0:2]
            if float(np.linalg.norm(diff)) > 1e-4:
                yaw = float(np.arctan2(diff[1], diff[0]))
                self.data.qpos[3:7] = np.array(
                    [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float
                )
        mujoco.mj_forward(self.model, self.data)
        self.sync_carried_box()

    def is_fallen(self) -> bool:
        """骨盆高度明显低于正常站立，视为摔倒。"""
        spawn_z = float(self.spawn_cfg["pos"][2])
        return float(self.data.qpos[2]) < spawn_z - 0.22

    def recover_stand(self) -> None:
        """摔倒后扶起：恢复站立关节姿态，保留当前 xy/yaw 与持箱状态。"""
        stand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        robot_nq = len(self.model.key_qpos[stand_id])
        xy = self.pelvis_xy()
        yaw = float(yaw_from_quat(self.data.qpos[3:7]))
        spawn_z = float(self.spawn_cfg["pos"][2])
        grasped = self.grasped_box
        carry_mode = self.carry_attach

        self.data.qpos[:robot_nq] = self.model.key_qpos[stand_id]
        self.data.qvel[:robot_nq] = 0.0
        self.data.ctrl[:] = self.stand_ctrl
        self.data.qpos[0:2] = xy
        self.data.qpos[2] = spawn_z
        self.data.qpos[3:7] = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float
        )

        self.grasped_box = grasped
        self.carry_attach = carry_mode
        if grasped is not None:
            self.model.body_gravcomp[self.box_body_ids[grasped]] = 1.0

        self.gait_walker.reset(self)
        mujoco.mj_forward(self.model, self.data)
        self.sync_carried_box()

    def release(self) -> None:
        if self.grasped_box is None:
            return
        self.data.eq_active[self.grasp_eq_ids[self.grasped_box]] = 0
        self.model.body_gravcomp[self.box_body_ids[self.grasped_box]] = 0.0
        self.grasped_box = None

    def place_carried_box_on_table(self) -> None:
        """释放前把箱心放到放货区台面上（与拾取台同高）。"""
        if self.grasped_box is None:
            return
        i = self.grasped_box
        hs = float(self.grasp_cfg.get("box_half_size", 0.12))
        top_z = float(self.table_cfg.get("top_height", 0.73))
        center = self.task.drop_zone_center.copy()
        center[2] = top_z + hs
        jid = self.model.body(f"box_{i}").jntadr[0]
        adr = self.model.jnt_qposadr[jid]
        dof = self.model.jnt_dofadr[jid]
        pelvis_mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        yaw = float(np.arctan2(pelvis_mat[1, 0], pelvis_mat[0, 0]))
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float)
        self.data.qpos[adr : adr + 3] = center
        self.data.qpos[adr + 3 : adr + 7] = quat
        self.data.qvel[dof : dof + 6] = 0.0
        if self.placed_box_qpos is not None:
            self.placed_box_qpos[i] = self.data.qpos[adr : adr + 7].copy()
        self.task.placed[i] = True
        mujoco.mj_forward(self.model, self.data)

    def apply_elastic_band(self) -> None:
        carry_support = (
            self.grasped_box is not None
            and self.sim_cfg.get("carry_elastic_band", True)
            and not self.gait_walker.lock_planar
        )
        if not self.sim_cfg.get("enable_elastic_band", False) and not carry_support:
            return
        pelvis_id = self.model.body("pelvis").id
        if carry_support:
            anchor_z = float(self.spawn_cfg["pos"][2])
            pelvis_pos = self.data.xpos[pelvis_id]
            pelvis_vel = self.data.cvel[pelvis_id, 3:6]
            dz = anchor_z - pelvis_pos[2]
            kp = float(self.sim_cfg.get("carry_band_stiffness", 1400.0))
            kd = float(self.sim_cfg.get("carry_band_damping", 140.0))
            self.data.xfrc_applied[pelvis_id, 2] = kp * dz - kd * pelvis_vel[2]
            return
        anchor = np.array([0.0, 0.0, float(self.sim_cfg["elastic_band_height"])])
        pelvis_pos = self.data.xpos[pelvis_id]
        delta = anchor - pelvis_pos
        kp = float(self.sim_cfg.get("elastic_band_stiffness", 800.0))
        kd = float(self.sim_cfg.get("elastic_band_damping", 80.0))
        pelvis_vel = self.data.cvel[pelvis_id, 3:6]
        force = kp * delta - kd * pelvis_vel
        self.data.xfrc_applied[pelvis_id, :3] = force

    def step(self) -> list[int]:
        self.stabilize_table_boxes()
        self.apply_elastic_band()
        self.gait_walker.pre_physics(self)
        mujoco.mj_step(self.model, self.data)
        self.gait_walker.post_physics(self)
        self.sync_carried_box()
        self.data.xfrc_applied[:] = 0.0
        return self.task.update(self.box_positions(), self.box_velocities(), self.grasped_box)

    def run_viewer(self, controller=None, demo: bool = False) -> None:
        if controller is not None:
            controller.reset(self)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("控制: G=抓取  R=释放  Backspace=重置  Esc=退出")
            while viewer.is_running():
                if controller is not None:
                    controller.step(self)

                newly = self.step()
                for idx in newly:
                    print(f"[任务] 箱子 {idx} 已放置 -> {self.task.status_line()}")
                if self.task.is_complete:
                    print("[任务] 全部 10 个箱子已搬运完成!")
                    if demo:
                        break

                viewer.sync()
