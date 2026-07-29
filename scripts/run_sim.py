#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer

from src.controllers.demo import DemoController
from src.controllers.manual import ManualController
from src.env import BoxPickPlaceEnv
from src.utils.video_recorder import VideoRecorder


def process_pending_keys(
    pending: deque[int],
    env: BoxPickPlaceEnv,
    manual: ManualController | None,
    controller: DemoController | None,
) -> None:
    while pending:
        keycode = pending.popleft()
        if manual is not None and manual.on_key(keycode, env):
            continue
        if controller is not None and keycode in (ord("R"), ord("r")):
            env.release()
            print("已释放")
        if keycode == 259:  # GLFW_KEY_BACKSPACE
            env.reset()
            if manual is not None:
                manual.bind_env(env)
            if controller is not None:
                controller.reset(env)
            print("场景已重置")


def main() -> None:
    parser = argparse.ArgumentParser(description="宇树 G1 搬箱仿真")
    parser.add_argument("--demo", action="store_true", help="自动演示搬运流程")
    parser.add_argument("--headless", action="store_true", help="无窗口运行（仅 demo 模式）")
    parser.add_argument("--record", action="store_true", help="启动窗口录制（可正常操作，Esc 结束并保存）")
    parser.add_argument(
        "--record-output",
        type=Path,
        default=ROOT / "assets" / "recording.mp4",
        help="录制输出路径",
    )
    parser.add_argument(
        "--record-stop-boxes",
        type=int,
        default=0,
        help="完成 N 箱后自动停止（0=不限制，Esc 手动结束）",
    )
    args = parser.parse_args()

    if args.record and args.headless:
        parser.error("--record 需配合可视化窗口，请勿与 --headless 同时使用")

    env = BoxPickPlaceEnv.create()
    controller = DemoController() if args.demo else None
    manual = None if args.demo else ManualController()

    if controller:
        controller.reset(env)
    elif manual:
        manual.bind_env(env)

    if args.headless and args.demo:
        steps = 0
        while steps < 200000 and not env.task.is_complete:
            controller.step(env)
            newly = env.step()
            for idx in newly:
                print(f"[任务] 箱子 {idx} 已放置 -> {env.task.status_line()}")
            steps += 1
        print(env.task.status_line())
        return

    pending_keys: deque[int] = deque()
    recorder: VideoRecorder | None = None

    try:
        if args.record:
            recorder = VideoRecorder(
                env.model,
                args.record_output,
                sim_dt=float(env.sim_cfg["dt"]),
            )
            print(f"● 录制中 -> {args.record_output}")
            print("  可照常操作；关闭窗口或按 Esc 结束并保存视频")

        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=pending_keys.append,
        ) as viewer:
            print("宇树 G1 搬箱仿真")
            if args.demo:
                print("演示模式：自动搬运 10 个箱子")
            else:
                print("手动模式：")
                print("  WASD / 方向键 = 行走（可长按）")
                print("  空格 = 停止")
                print("  G = 抓取  R = 释放  U = 起身  Backspace = 重置  Esc = 退出")

            while viewer.is_running():
                process_pending_keys(pending_keys, env, manual, controller)
                if controller:
                    controller.step(env)
                elif manual:
                    manual.step(env)
                newly = env.step()
                for idx in newly:
                    print(f"[任务] 箱子 {idx} 已放置 -> {env.task.status_line()}")

                if recorder is not None:
                    recorder.capture(env.data, viewer.cam)

                if env.task.is_complete:
                    print("[任务] 全部 10 个箱子已搬运完成!")
                    if args.demo:
                        break
                if (
                    args.record_stop_boxes > 0
                    and env.task.completed_count >= args.record_stop_boxes
                ):
                    print(f"[录制] 已完成 {args.record_stop_boxes} 箱，停止仿真")
                    break

                viewer.sync()
    finally:
        if recorder is not None:
            recorder.close()
            print(f"● 视频已保存: {args.record_output}（{recorder.frames} 帧）")


if __name__ == "__main__":
    main()
