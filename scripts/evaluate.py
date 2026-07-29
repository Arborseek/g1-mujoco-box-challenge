#!/usr/bin/env python3
"""无头运行控制器并输出任务完成情况。"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controllers.demo import DemoController
from src.env import BoxPickPlaceEnv


def load_team_controller(path: Path):
    spec = importlib.util.spec_from_file_location("team_controller", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载控制器: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "TeamController"):
        raise ImportError(f"{path} 须导出 TeamController 类")
    return mod.TeamController()


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 搬箱仿真运行脚本")
    parser.add_argument(
        "--controller",
        type=Path,
        default=None,
        help="自定义控制器路径（默认官方基线 DemoController）",
    )
    parser.add_argument("--max-steps", type=int, default=200_000)
    args = parser.parse_args()

    env = BoxPickPlaceEnv.create()
    controller = load_team_controller(args.controller) if args.controller else DemoController()
    controller.reset(env)

    steps = 0
    while steps < args.max_steps and not env.task.is_complete:
        controller.step(env)
        env.step()
        steps += 1

    print(env.task.status_line())
    print(f"仿真步数: {steps}")
    if env.task.is_complete:
        print("任务完成：10 个箱子均已放置。")
    elif steps >= args.max_steps:
        print("已达步数上限，任务未全部完成。")


if __name__ == "__main__":
    main()
