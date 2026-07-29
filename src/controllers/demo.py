from __future__ import annotations

from enum import Enum, auto

import numpy as np

from src.controllers.base import TeamController
from src.control.locomotion import yaw_from_quat


class DemoPhase(Enum):
    WALK_TO_PICK = auto()
    REACH = auto()
    GRASP = auto()
    STABILIZE = auto()
    WALK_ROUTE = auto()
    PLACE = auto()
    RELEASE = auto()
    RETRACT = auto()
    WALK_BACK = auto()


RIGHT_ARM = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEFT_ARM = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEGS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

DEFAULT_LEG = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, dtype=float)
STAND_WAIST = np.zeros(3)
STAND_ARM_R = np.array([0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0])
STAND_ARM_L = np.array([0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0])

# 站立够取（拾取台箱心 ~0.85m）
REACH_WAIST = np.zeros(3)
REACH_ARM_R = np.array([0.10, -0.45, 0.10, 0.30, 0.0, -0.70, 0.0])
REACH_ARM_L = np.array([0.10, 0.45, -0.10, 0.30, 0.0, 0.70, 0.0])

# 搬运：双臂内收前伸，手间距 ~36cm 托 30cm 箱
CARRY_WAIST = np.zeros(3)
CARRY_ARM_R = np.array([0.15, -0.15, 0.10, 0.15, 0.0, -0.65, 0.0])
CARRY_ARM_L = np.array([0.15, 0.15, -0.10, 0.15, 0.0, 0.65, 0.0])

PLACE_ARM_R = np.array([0.12, -0.40, 0.05, 0.35, 0.0, -0.55, 0.0])
PLACE_ARM_L = np.array([0.12, 0.40, -0.05, 0.35, 0.0, 0.55, 0.0])

CLOSE_HAND = np.array([-0.8, 0.6, -0.8, 0.0, 1.0, 0.0, 1.0, 0.0])
OPEN_HAND = np.zeros(7)


class DemoController(TeamController):
    """官方基线：RL 行走 + 站立够取/放桌（无深蹲）。"""

    def __init__(self) -> None:
        self.phase = DemoPhase.WALK_TO_PICK
        self.current_box = 0
        self.timer = 0
        self._move_steps = 200
        self.arm_r_ids: list[int] = []
        self.arm_l_ids: list[int] = []
        self.hand_r_ids: list[int] = []
        self.hand_l_ids: list[int] = []
        self.waist_ids: list[int] = []
        self.leg_ids: list[int] = []
        self.from_pose: dict[str, np.ndarray] = {}
        self.to_pose: dict[str, np.ndarray] = {}
        self.walk_cfg: dict = {}
        self._walk_target: np.ndarray | None = None
        self._walk_timer = 0
        self._carry_ready = False
        self._route: list[np.ndarray] = []

    def reset(self, env) -> None:
        self.phase = DemoPhase.WALK_TO_PICK
        self.current_box = 0
        self.timer = 0
        self.walk_cfg = env.walk_cfg
        self.arm_r_ids = [env.model.actuator(n).id for n in RIGHT_ARM]
        self.arm_l_ids = [env.model.actuator(n).id for n in LEFT_ARM]
        self.waist_ids = [env.model.actuator(n).id for n in WAIST]
        self.leg_ids = [env.model.actuator(n).id for n in LEGS]
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
        env.data.ctrl[:] = env.stand_ctrl
        self._set_hands(env, CLOSE_HAND[:7])
        env.gait_walker.reset(env)
        self._walk_target = None
        self._walk_timer = 0
        self._carry_ready = False
        self._route = []
        self._apply_upper(env, STAND_WAIST, STAND_ARM_R, STAND_ARM_L)

    def _set_hands(self, env, v: np.ndarray) -> None:
        pose = np.asarray(v, dtype=float)[:7]
        for ids in (self.hand_r_ids, self.hand_l_ids):
            for i, aid in enumerate(ids):
                env.data.ctrl[aid] = float(pose[i])

    def _bind_upper_carry(self, env, upper: dict) -> None:
        """行走时：下肢 RL，上半身固定为抱箱姿态。"""
        all_hand_ids = self.hand_r_ids + self.hand_l_ids
        hand_pose = np.concatenate([CLOSE_HAND[:7], CLOSE_HAND[:7]])
        env.gait_walker.set_upper_body(upper, all_hand_ids, hand_pose)

    def _bind_upper(self, env, upper: dict, hand: np.ndarray | None = None) -> None:
        env.gait_walker.set_upper_body(upper)
        if hand is not None:
            all_hand_ids = self.hand_r_ids + self.hand_l_ids
            hand_pose = np.concatenate([hand[:7], hand[:7]])
            env.gait_walker.set_upper_body(upper, all_hand_ids, hand_pose)

    def _set_hand(self, env, v: np.ndarray) -> None:
        self._set_hands(env, v)

    def _apply_upper(self, env, waist, arm_r, arm_l) -> None:
        for i, aid in enumerate(self.waist_ids):
            env.data.ctrl[aid] = float(waist[i])
        for i, aid in enumerate(self.arm_r_ids):
            env.data.ctrl[aid] = float(arm_r[i])
        for i, aid in enumerate(self.arm_l_ids):
            env.data.ctrl[aid] = float(arm_l[i])

    def _apply_legs(self, env, legs: np.ndarray) -> None:
        for i, aid in enumerate(self.leg_ids):
            env.data.ctrl[aid] = float(legs[i])

    def _upper(self, waist, arm_r, arm_l) -> dict:
        return {
            "waist": waist, "arm": arm_r, "left_arm": arm_l,
            "waist_ids": self.waist_ids, "arm_ids": self.arm_r_ids, "left_arm_ids": self.arm_l_ids,
        }

    def _begin_blend(self, env, waist, arm_r, arm_l, steps: int) -> None:
        self.from_pose = {
            "waist": np.array([env.data.ctrl[i] for i in self.waist_ids]),
            "arm_r": np.array([env.data.ctrl[i] for i in self.arm_r_ids]),
            "arm_l": np.array([env.data.ctrl[i] for i in self.arm_l_ids]),
        }
        self.to_pose = {"waist": waist.copy(), "arm_r": arm_r.copy(), "arm_l": arm_l.copy()}
        self.timer = 0
        self._move_steps = steps

    def _blend(self, env, alpha: float) -> None:
        a = min(1.0, alpha)
        w = (1 - a) * self.from_pose["waist"] + a * self.to_pose["waist"]
        r = (1 - a) * self.from_pose["arm_r"] + a * self.to_pose["arm_r"]
        l = (1 - a) * self.from_pose["arm_l"] + a * self.to_pose["arm_l"]
        self._apply_upper(env, w, r, l)

    def _blend_pose(self, env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = min(1.0, self.timer / self._move_steps)
        w = (1 - a) * self.from_pose["waist"] + a * self.to_pose["waist"]
        r = (1 - a) * self.from_pose["arm_r"] + a * self.to_pose["arm_r"]
        l = (1 - a) * self.from_pose["arm_l"] + a * self.to_pose["arm_l"]
        self._apply_upper(env, w, r, l)
        return w, r, l

    def _box_xy(self, env, bid: int) -> np.ndarray:
        o = bid * 7
        return np.array([float(env.initial_box_qpos[o]), float(env.initial_box_qpos[o + 1])])

    def _pick_stand(self, env, bid: int) -> np.ndarray:
        bx, by = self._box_xy(env, bid)
        off = float(self.walk_cfg.get("pick_stand_offset", -0.22))
        min_x = float(self.walk_cfg.get("pick_stand_min_x", -0.52))
        west_limit = float(self.walk_cfg.get("pick_stand_west_max_x", -0.42))
        ideal = bx + off
        if bx < 0.0:
            stand_x = max(min(ideal, west_limit), min_x)
            return np.array([stand_x, by])
        south_y = float(self.walk_cfg.get("pick_south_y", -0.52))
        if by > 0.1:
            south_y = float(self.walk_cfg.get("pick_south_y_row2", -0.32))
        stand_x = float(np.clip(bx, min_x, 0.62))
        return np.array([stand_x, south_y])

    def _pick_face_xy(self, env, bid: int) -> np.ndarray:
        bx, by = self._box_xy(env, bid)
        if bx < 0.0:
            stand = self._pick_stand(env, bid)
            return stand + np.array([1.0, 0.0])
        return np.array([bx, by])

    def _snap_pick_stand(self, env, bid: int) -> None:
        stand = self._pick_stand(env, bid)
        env.snap_pelvis(stand, self._pick_face_xy(env, bid))

    def _route_from_config(
        self,
        env,
        key: str,
        prepend_retreat: bool = False,
        bid: int | None = None,
    ) -> list[np.ndarray]:
        """从 config 读取固定路径点；可选在开头插入「从站位后退」。"""
        raw = self.walk_cfg.get(key)
        if not raw:
            return []
        pts = [np.asarray(p, dtype=float) for p in raw]
        if prepend_retreat and bid is not None:
            stand = self._pick_stand(env, bid)
            back = float(self.walk_cfg.get("retreat_back", 0.25))
            pts = [np.array([stand[0] - back, stand[1]])] + pts
        return pts

    def _carry_route(self, env, bid: int) -> list[np.ndarray]:
        """搬运路线：先撤到通道，再沿通道去放货区（不贴拾取台 y=0 行走）。"""
        stand = self._pick_stand(env, bid)
        back = float(self.walk_cfg.get("retreat_back", 0.35))
        lane_y = float(self.walk_cfg.get("path_lane_y", -0.90))
        west_x = float(self.walk_cfg.get("path_west_x", -0.90))
        head = [
            np.array([stand[0] - back, lane_y]),
            np.array([west_x, lane_y]),
        ]
        tail = self._route_from_config(env, "carry_route")
        if tail:
            return head + tail
        table = np.asarray(self.walk_cfg["table_stand"], float)
        return head + [np.array([table[0] - 0.10, lane_y]), table]

    def _back_route(self, env) -> list[np.ndarray]:
        """返回路线：先下到通道，再沿通道回 spawn。"""
        lane_y = float(self.walk_cfg.get("path_lane_y", -0.90))
        west_x = float(self.walk_cfg.get("path_west_x", -0.90))
        spawn = np.asarray(env.spawn_cfg["pos"][:2], float)
        pos = env.pelvis_xy()
        head: list[np.ndarray] = []
        if float(pos[1]) > lane_y + 0.12:
            head.append(np.array([float(pos[0]), lane_y]))
        head.append(np.array([west_x, lane_y]))
        tail = self._route_from_config(env, "return_route")
        if tail:
            out = head[:]
            for pt in tail:
                if out and float(np.linalg.norm(pt - out[-1])) < 0.12:
                    continue
                out.append(pt)
            return out
        return head + [spawn]

    @staticmethod
    def _face_target(env, target_xy: np.ndarray) -> None:
        """将骨盆 yaw 对准目标，避免沿通道时朝向错乱。"""
        pos = env.pelvis_xy()
        diff = np.asarray(target_xy, dtype=float) - pos
        if float(np.linalg.norm(diff)) < 1e-4:
            return
        yaw = float(np.arctan2(diff[1], diff[0]))
        env.data.qpos[3:7] = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float
        )
        env.data.qvel[3:6] = 0.0

    def _walk_route(
        self,
        env,
        upper: dict,
        assist: bool,
        threshold: float | None = None,
    ) -> bool:
        if not self._route:
            return True
        target = self._route[0]
        pos = env.pelvis_xy()
        dist = float(np.linalg.norm(target - pos))
        stuck_limit = int(self.walk_cfg.get("route_stuck_steps", 5000))
        stuck_dist = float(self.walk_cfg.get("route_stuck_dist", 0.30))
        if self._walk_timer > stuck_limit and dist < stuck_dist:
            self._route.pop(0)
            self._walk_timer = 0
            if self._route:
                self._face_target(env, self._route[0])
            return not self._route

        old = float(env.walk_cfg.get("arrive_threshold", 0.22))
        if threshold is not None:
            env.walk_cfg["arrive_threshold"] = threshold
        arrived = env.walk_toward(target, upper, assist=assist)
        env.walk_cfg["arrive_threshold"] = old
        if arrived:
            self._route.pop(0)
            self._walk_timer = 0
            if self._route:
                self._face_target(env, self._route[0])
        else:
            self._walk_timer += 1
        return not self._route

    def _drop_stand(self, env) -> np.ndarray:
        return np.asarray(self.walk_cfg["table_stand"], dtype=float)

    def _snap_drop_stand(self, env) -> None:
        stand = self._drop_stand(env)
        table_xy = env.task.drop_zone_center[:2]
        env.snap_pelvis(stand, table_xy)

    def _hold_manipulate(self, env, waist, arm_r, arm_l, lock_yaw: bool = True) -> None:
        """操作阶段：锁骨盆 + 固定腿，防止够取/持箱时摔倒。"""
        upper = self._upper(waist, arm_r, arm_l)
        env.gait_walker.hold_carry(env, DEFAULT_LEG, upper)
        env.gait_walker._lock_xy = env.pelvis_xy().copy()
        env.gait_walker._lock_z = float(env.spawn_cfg["pos"][2])
        env.gait_walker._lock_orientation = lock_yaw
        if lock_yaw:
            env.gait_walker._lock_yaw = float(yaw_from_quat(env.data.qpos[3:7]))
        else:
            env.gait_walker._lock_yaw = None

    def _ensure_carry_upper(self, env, upper: dict) -> None:
        """持箱阶段强制抱箱姿态（策略仅控腿）。"""
        self._bind_upper_carry(env, upper)
        self._set_hands(env, CLOSE_HAND[:7])

    def _prepare_next_pick(self, env) -> None:
        """每轮抓取前重置行走/持箱状态，并对准下一站位。"""
        env.carry_attach = "wrist"
        env.gait_walker.reset(env)
        self._walk_target = None
        self._walk_timer = 0
        self._carry_ready = False
        self._route = []

    def step(self, env) -> None:
        while self.current_box < env.task.num_boxes and env.task.placed[self.current_box]:
            self.current_box += 1
        if env.task.is_complete or self.current_box >= env.task.num_boxes:
            return

        table = np.asarray(self.walk_cfg["table_stand"], float)
        route_th = float(self.walk_cfg.get("route_arrive_threshold", 0.12))
        upper_stand = self._upper(STAND_WAIST, STAND_ARM_R, STAND_ARM_L)
        upper_carry = self._upper(CARRY_WAIST, CARRY_ARM_R, CARRY_ARM_L)

        if self.phase == DemoPhase.WALK_TO_PICK:
            if self._walk_target is None:
                self._walk_target = self._pick_stand(env, self.current_box)
                self._walk_timer = 0
                self._face_target(env, self._walk_target)
            pick_th = float(self.walk_cfg.get("pick_arrive_threshold", 0.08))
            old = float(env.walk_cfg.get("arrive_threshold", 0.22))
            env.walk_cfg["arrive_threshold"] = pick_th
            pos = env.pelvis_xy()
            dist = float(np.linalg.norm(pos - self._walk_target))
            stuck_limit = int(self.walk_cfg.get("pick_stuck_steps", 6000))
            stuck_dist = float(self.walk_cfg.get("pick_stuck_dist", 0.40))
            if self._walk_timer > stuck_limit and dist < stuck_dist:
                env.gait_walker.deactivate()
                self._snap_pick_stand(env, self.current_box)
                self._walk_target = None
                self._walk_timer = 0
                self.phase = DemoPhase.REACH
                self._begin_blend(
                    env, REACH_WAIST, REACH_ARM_R, REACH_ARM_L,
                    int(self.walk_cfg.get("reach_steps", 220)),
                )
                return
            arrived = env.walk_toward(
                self._walk_target, upper_stand, assist=True,
            )
            self._bind_upper(env, upper_stand, OPEN_HAND[:7])
            env.walk_cfg["arrive_threshold"] = old
            self._walk_timer += 1
            if arrived:
                env.gait_walker.deactivate()
                self._snap_pick_stand(env, self.current_box)
                self._walk_target = None
                self._walk_timer = 0
                self.phase = DemoPhase.REACH
                self._begin_blend(
                    env, REACH_WAIST, REACH_ARM_R, REACH_ARM_L,
                    int(self.walk_cfg.get("reach_steps", 220)),
                )
            return

        if self.phase == DemoPhase.REACH:
            self._hold_manipulate(env, *self._blend_pose(env))
            self.timer += 1
            if self.timer >= self._move_steps:
                self.phase = DemoPhase.GRASP
                self.timer = 0
            return

        if self.phase == DemoPhase.GRASP:
            self._hold_manipulate(env, REACH_WAIST, REACH_ARM_R, REACH_ARM_L)
            self._bind_upper_carry(
                env, self._upper(REACH_WAIST, REACH_ARM_R, REACH_ARM_L),
            )
            self._set_hands(env, CLOSE_HAND)
            self.timer += 1
            if self.timer > 40 and env.try_grasp(self.current_box):
                env.carry_attach = "wrist"
                self.phase = DemoPhase.STABILIZE
                self._begin_blend(
                    env, STAND_WAIST, CARRY_ARM_R, CARRY_ARM_L, 180,
                )
                self.timer = 0
            elif self.timer > 350:
                self.phase = DemoPhase.WALK_TO_PICK
                self._prepare_next_pick(env)
                self.timer = 0
            return

        if self.phase == DemoPhase.STABILIZE:
            waist, arm_r, arm_l = self._blend_pose(env)
            upper_blend = self._upper(waist, arm_r, arm_l)
            self._hold_manipulate(env, waist, arm_r, arm_l)
            self._ensure_carry_upper(env, upper_blend)
            self.timer += 1
            if self.timer >= self._move_steps:
                env.carry_attach = "cradle"
                env.gait_walker.deactivate()
                env.gait_walker.lock_planar = False
                env.gait_walker._lock_z = None
                env.gait_walker._lock_xy = None
                env.gait_walker._lock_orientation = True
                env.gait_walker._lock_yaw = None
                self._route = self._carry_route(env, self.current_box)
                if self._route:
                    self._face_target(env, self._route[0])
                self._carry_ready = False
                self._walk_timer = 0
                self.phase = DemoPhase.WALK_ROUTE
                self.timer = 0
            return

        if self.phase == DemoPhase.WALK_ROUTE:
            if not self._carry_ready:
                env.gait_walker.hold_balance(env, upper_carry)
                self._ensure_carry_upper(env, upper_carry)
                self.timer += 1
                if self.timer >= int(self.walk_cfg.get("carry_balance_steps", 300)):
                    env.gait_walker.locomotion.warm_start_from_env(env)
                    self._carry_ready = True
                    self.timer = 0
                return
            self._ensure_carry_upper(env, upper_carry)
            if self._walk_route(env, upper_carry, assist=True, threshold=route_th):
                env.gait_walker.deactivate()
                self._snap_drop_stand(env)
                self.phase = DemoPhase.PLACE
                self._begin_blend(
                    env, STAND_WAIST, PLACE_ARM_R, PLACE_ARM_L,
                    int(self.walk_cfg.get("place_steps", 180)),
                )
                self.timer = 0
            return

        if self.phase == DemoPhase.PLACE:
            waist, arm_r, arm_l = self._blend_pose(env)
            self._hold_manipulate(env, waist, arm_r, arm_l)
            self._ensure_carry_upper(env, self._upper(waist, arm_r, arm_l))
            self.timer += 1
            if self.timer >= self._move_steps:
                self.phase = DemoPhase.RELEASE
                self.timer = 0
            return

        if self.phase == DemoPhase.RELEASE:
            self._hold_manipulate(env, STAND_WAIST, PLACE_ARM_R, PLACE_ARM_L)
            if self.timer == 0:
                env.place_carried_box_on_table()
                env.release()
                self._set_hand(env, OPEN_HAND)
            self.timer += 1
            if self.timer > 80:
                self.phase = DemoPhase.RETRACT
                self._begin_blend(env, STAND_WAIST, STAND_ARM_R, STAND_ARM_L, 140)
                self.timer = 0
            return

        if self.phase == DemoPhase.RETRACT:
            self._hold_manipulate(env, *self._blend_pose(env))
            self.timer += 1
            if self.timer >= self._move_steps:
                self.phase = DemoPhase.WALK_BACK
                env.gait_walker.reset(env)
                self.timer = 0
            return

        if self.phase == DemoPhase.WALK_BACK:
            if not self._route:
                self._route = self._back_route(env)
                self._walk_timer = 0
                env.gait_walker.reset(env)
                if self._route:
                    self._face_target(env, self._route[0])
            if self._walk_route(env, upper_stand, assist=True, threshold=route_th):
                env.gait_walker.deactivate()
                if env.task.placed[self.current_box]:
                    self.current_box += 1
                if self.current_box < env.task.num_boxes and not env.task.is_complete:
                    self.phase = DemoPhase.WALK_TO_PICK
                    self._prepare_next_pick(env)
                self.timer = 0
            else:
                self._bind_upper(env, upper_stand, OPEN_HAND[:7])
            return
