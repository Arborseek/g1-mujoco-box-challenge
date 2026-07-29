from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoxPickPlaceTask:
    num_boxes: int
    drop_zone_center: np.ndarray
    drop_zone_half: np.ndarray
    placement_velocity_threshold: float
    placement_height_min: float
    placed: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.placed:
            self.placed = [False] * self.num_boxes

    @property
    def completed_count(self) -> int:
        return sum(self.placed)

    @property
    def is_complete(self) -> bool:
        return self.completed_count == self.num_boxes

    def update(
        self,
        box_positions: list[np.ndarray],
        box_velocities: list[np.ndarray],
        grasped_box: int | None = None,
    ) -> list[int]:
        """检查并更新放置状态，返回本轮新完成的箱子编号。"""
        newly_placed: list[int] = []
        for i, (pos, vel) in enumerate(zip(box_positions, box_velocities)):
            if self.placed[i] or i == grasped_box:
                continue
            in_zone = self._in_drop_zone(pos)
            slow = float(np.linalg.norm(vel)) < self.placement_velocity_threshold
            high_enough = float(pos[2]) >= self.placement_height_min
            if in_zone and slow and high_enough:
                self.placed[i] = True
                newly_placed.append(i)
        return newly_placed

    def _in_drop_zone(self, pos: np.ndarray) -> bool:
        delta = np.abs(pos - self.drop_zone_center)
        return bool(
            delta[0] <= self.drop_zone_half[0]
            and delta[1] <= self.drop_zone_half[1]
            and delta[2] <= self.drop_zone_half[2] + 0.05
        )

    def status_line(self) -> str:
        return f"已完成 {self.completed_count}/{self.num_boxes} 个箱子"
