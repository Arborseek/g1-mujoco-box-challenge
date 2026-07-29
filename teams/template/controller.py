"""参赛队提交模板。

复制本目录为 teams/<队名>/，在 controller.py 中实现 TeamController。
本地评测：

    python scripts/evaluate.py --controller teams/<队名>/controller.py
"""

from __future__ import annotations

from src.controllers.base import TeamController as ControllerBase
from src.controllers.demo import DemoController
from src.env import BoxPickPlaceEnv


class TeamController(ControllerBase):
    """在此类基础上扩展你的算法。"""

    def __init__(self) -> None:
        # 可先委托官方基线，再逐步替换各阶段逻辑
        self._baseline = DemoController()

    def reset(self, env: BoxPickPlaceEnv) -> None:
        self._baseline.reset(env)

    def step(self, env: BoxPickPlaceEnv) -> None:
        self._baseline.step(env)
