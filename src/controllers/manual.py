from __future__ import annotations

import numpy as np

from src.controllers.demo import (
    CARRY_ARM_L,
    CARRY_ARM_R,
    CARRY_WAIST,
    CLOSE_HAND,
    DEFAULT_LEG,
    LEFT_ARM,
    REACH_ARM_L,
    REACH_ARM_R,
    REACH_WAIST,
    RIGHT_ARM,
    STAND_ARM_L,
    STAND_ARM_R,
    STAND_WAIST,
    WAIST,
)
from src.control.locomotion import yaw_from_quat


class ManualController:
    """手动模式：WASD 行走；G 够取+抱箱（与 demo 同一套上肢/搬运逻辑）。"""

    def __init__(self) -> None:
        self.target: np.ndarray | None = None
        self._upper: dict | None = None
        self.hand_r_ids: list[int] = []
        self.hand_l_ids: list[int] = []
        self.step_dist = 0.35
        self.strafe_dist = 0.28
        self.arrive_threshold = 0.14
        self._grasp_state: str | None = None
        self._grasp_timer = 0
        self._blend_steps = 90
        self._blend_from: dict[str, np.ndarray] | None = None

    def bind_env(self, env) -> None:
        waist_ids = [env.model.actuator(n).id for n in WAIST]
        arm_r_ids = [env.model.actuator(n).id for n in RIGHT_ARM]
        arm_l_ids = [env.model.actuator(n).id for n in LEFT_ARM]
        self.hand_r_ids = [
            env.model.actuator(n).id
            for n in [
                "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                "right_hand_index_0_joint", "right_hand_index_1_joint",
                "right_hand_middle_0_joint", "right_hand_middle_1_joint",
            ]
        ]
        self.hand_l_ids = [
            env.model.actuator(n).id
            for n in [
                "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                "left_hand_index_0_joint", "left_hand_index_1_joint",
                "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            ]
        ]
        self._upper = {
            "waist": STAND_WAIST,
            "arm": STAND_ARM_R,
            "left_arm": STAND_ARM_L,
            "waist_ids": waist_ids,
            "arm_ids": arm_r_ids,
            "left_arm_ids": arm_l_ids,
        }
        self.target = None
        self._reset_grasp_state()
        env.gait_walker.reset(env)
        env.gait_walker.set_upper_body(self._upper)

    def _upper_dict(self, waist, arm_r, arm_l) -> dict:
        assert self._upper is not None
        return {
            "waist": waist,
            "arm": arm_r,
            "left_arm": arm_l,
            "waist_ids": self._upper["waist_ids"],
            "arm_ids": self._upper["arm_ids"],
            "left_arm_ids": self._upper["left_arm_ids"],
        }

    def _reset_grasp_state(self) -> None:
        self._grasp_state = None
        self._grasp_timer = 0
        self._blend_from = None

    def _set_hands(self, env, v: np.ndarray) -> None:
        pose = np.asarray(v, dtype=float)[:7]
        for ids in (self.hand_r_ids, self.hand_l_ids):
            for i, aid in enumerate(ids):
                env.data.ctrl[aid] = float(pose[i])

    def _bind_carry(self, env, upper: dict) -> None:
        all_hand_ids = self.hand_r_ids + self.hand_l_ids
        hand_pose = np.concatenate([CLOSE_HAND[:7], CLOSE_HAND[:7]])
        env.gait_walker.set_upper_body(upper, all_hand_ids, hand_pose)

    def _apply_upper(self, env, waist, arm_r, arm_l) -> None:
        u = self._upper_dict(waist, arm_r, arm_l)
        for i, aid in enumerate(u["waist_ids"]):
            env.data.ctrl[aid] = float(waist[i])
        for i, aid in enumerate(u["arm_ids"]):
            env.data.ctrl[aid] = float(arm_r[i])
        for i, aid in enumerate(u["left_arm_ids"]):
            env.data.ctrl[aid] = float(arm_l[i])

    def _hold_manipulate(self, env, waist, arm_r, arm_l) -> None:
        upper = self._upper_dict(waist, arm_r, arm_l)
        env.gait_walker.hold_carry(env, DEFAULT_LEG, upper)
        env.gait_walker._lock_xy = env.pelvis_xy().copy()
        env.gait_walker._lock_z = float(env.spawn_cfg["pos"][2])
        env.gait_walker._lock_orientation = True
        env.gait_walker._lock_yaw = float(yaw_from_quat(env.data.qpos[3:7]))

    def _unlock_manipulate(self, env) -> None:
        env.gait_walker.deactivate()
        env.gait_walker.lock_planar = False
        env.gait_walker._lock_xy = None
        env.gait_walker._lock_z = None
        env.gait_walker._lock_yaw = None

    def stop(self, env) -> None:
        self.target = None
        if env.grasped_box is None and self._grasp_state is None:
            env.gait_walker.deactivate()

    def request_grasp(self, env) -> None:
        if env.grasped_box is not None or self._grasp_state is not None:
            return
        self.stop(env)
        self._grasp_state = "reach"
        self._grasp_timer = 0

    def release_box(self, env) -> None:
        env.release()
        env.carry_attach = "wrist"
        self._reset_grasp_state()
        self._unlock_manipulate(env)
        assert self._upper is not None
        env.gait_walker.set_upper_body(self._upper)
        self._set_hands(env, np.zeros(7))

    def recover(self, env) -> None:
        """扶起并恢复抱箱/行走控制。"""
        carrying = env.grasped_box is not None
        env.recover_stand()
        self._reset_grasp_state()
        self._unlock_manipulate(env)
        if carrying:
            env.carry_attach = "cradle"
            carry_upper = self._upper_dict(CARRY_WAIST, CARRY_ARM_R, CARRY_ARM_L)
            self._bind_carry(env, carry_upper)
            env.gait_walker.locomotion.warm_start_from_env(env)
            env.gait_walker.hold_balance(env, carry_upper)
            print("已起身（仍抱着箱），WASD 可继续搬运")
        else:
            assert self._upper is not None
            env.gait_walker.set_upper_body(self._upper)
            print("已起身，WASD 可继续行走")

    def on_key(self, keycode: int, env) -> bool:
        if keycode in (ord(" "),):
            self.stop(env)
            print("行走停止")
            return True

        if keycode in (ord("g"), ord("G")):
            self.request_grasp(env)
            return True
        if keycode in (ord("r"), ord("R")):
            self.release_box(env)
            print("已释放")
            return True
        if keycode in (ord("u"), ord("U")):
            self.recover(env)
            return True

        pos = env.pelvis_xy()
        base = self.target if self.target is not None else pos
        yaw = float(yaw_from_quat(env.data.qpos[3:7]))
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=float)

        moved = False
        if keycode in (ord("w"), ord("W"), 265):
            self.target = base + forward * self.step_dist
            moved = True
        elif keycode in (ord("s"), ord("S"), 264):
            self.target = base - forward * self.step_dist * 0.55
            moved = True
        elif keycode in (ord("a"), ord("A"), 263):
            self.target = base + left * self.strafe_dist
            moved = True
        elif keycode in (ord("d"), ord("D"), 262):
            self.target = base - left * self.strafe_dist
            moved = True

        if moved:
            if self._grasp_state is not None:
                return True
            env.gait_walker.locomotion.active = True
            return True
        return False

    def _step_grasp(self, env) -> bool:
        if self._grasp_state == "reach":
            self._hold_manipulate(env, REACH_WAIST, REACH_ARM_R, REACH_ARM_L)
            self._bind_carry(env, self._upper_dict(REACH_WAIST, REACH_ARM_R, REACH_ARM_L))
            self._set_hands(env, CLOSE_HAND)
            self._grasp_timer += 1
            if self._grasp_timer > 40:
                if env.try_grasp():
                    env.carry_attach = "wrist"
                    self._grasp_state = "blend"
                    self._grasp_timer = 0
                    self._blend_from = {
                        "waist": np.array([env.data.ctrl[i] for i in self._upper["waist_ids"]]),
                        "arm_r": np.array([env.data.ctrl[i] for i in self._upper["arm_ids"]]),
                        "arm_l": np.array([env.data.ctrl[i] for i in self._upper["left_arm_ids"]]),
                    }
                    print("抓取成功")
                else:
                    self._reset_grasp_state()
                    self._unlock_manipulate(env)
                    print("抓取失败（手未靠近箱子）")
            return True

        if self._grasp_state == "blend":
            a = min(1.0, self._grasp_timer / self._blend_steps)
            assert self._blend_from is not None
            w = (1 - a) * self._blend_from["waist"] + a * CARRY_WAIST
            r = (1 - a) * self._blend_from["arm_r"] + a * CARRY_ARM_R
            l = (1 - a) * self._blend_from["arm_l"] + a * CARRY_ARM_L
            self._hold_manipulate(env, w, r, l)
            self._bind_carry(env, self._upper_dict(w, r, l))
            self._grasp_timer += 1
            if self._grasp_timer >= self._blend_steps:
                env.carry_attach = "cradle"
                self._reset_grasp_state()
                self._unlock_manipulate(env)
                self._bind_carry(env, self._upper_dict(CARRY_WAIST, CARRY_ARM_R, CARRY_ARM_L))
            return True

        return False

    def step(self, env) -> None:
        if self._upper is None:
            self.bind_env(env)

        if self._step_grasp(env):
            return

        carry_upper = self._upper_dict(CARRY_WAIST, CARRY_ARM_R, CARRY_ARM_L)

        if env.grasped_box is not None:
            self._bind_carry(env, carry_upper)
            if self.target is None:
                env.gait_walker.hold_balance(env, carry_upper)
                return
            if not env.gait_walker.locomotion.active:
                env.gait_walker.locomotion.warm_start_from_env(env)
            old = float(env.walk_cfg.get("arrive_threshold", 0.22))
            env.walk_cfg["arrive_threshold"] = self.arrive_threshold
            arrived = env.walk_toward(self.target, carry_upper, assist=True)
            env.walk_cfg["arrive_threshold"] = old
            if arrived:
                self.target = None
                env.gait_walker.deactivate()
            return

        if self.target is None:
            env.gait_walker.deactivate()
            env.gait_walker.set_upper_body(self._upper)
            env.data.ctrl[:] = env.stand_ctrl
            return

        if not env.gait_walker.locomotion.active:
            env.gait_walker.locomotion.warm_start_from_env(env)
        old = float(env.walk_cfg.get("arrive_threshold", 0.22))
        env.walk_cfg["arrive_threshold"] = self.arrive_threshold
        arrived = env.walk_toward(self.target, self._upper, assist=True)
        env.walk_cfg["arrive_threshold"] = old
        if arrived:
            self.target = None
            env.gait_walker.deactivate()
