#!/usr/bin/env python3
"""无显示器批量录制（CI / 生成 README 演示片）。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from src.controllers.demo import DemoController
from src.env import BoxPickPlaceEnv
from src.utils.video_recorder import VideoRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="无窗口批量录制 demo")
    parser.add_argument("--target-boxes", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "demo_2boxes.mp4")
    parser.add_argument("--max-steps", type=int, default=200_000)
    args = parser.parse_args()

    env = BoxPickPlaceEnv.create()
    controller = DemoController()
    controller.reset(env)

    steps = 0
    with VideoRecorder(env.model, args.output, sim_dt=float(env.sim_cfg["dt"])) as rec:
        print(f"无头录制 -> {args.output}，目标 {args.target_boxes} 箱")
        while steps < args.max_steps and env.task.completed_count < args.target_boxes:
            controller.step(env)
            newly = env.step()
            for idx in newly:
                print(f"[{steps}] 箱子 {idx} 已放置 -> {env.task.status_line()}")
            rec.capture(env.data)
            steps += 1

    print(f"完成: {env.task.status_line()} | 步数 {steps}")


if __name__ == "__main__":
    main()
