from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.env.box_pick_place import BoxPickPlaceEnv


class TeamController(ABC):
    """参赛控制器统一接口。"""

    @abstractmethod
    def reset(self, env: BoxPickPlaceEnv) -> None:
        """环境 reset 后初始化队伍内部状态。"""

    @abstractmethod
    def step(self, env: BoxPickPlaceEnv) -> None:
        """每个仿真步写入控制量。"""
