from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np


class VideoRecorder:
    """MuJoCo 离屏渲染 + ffmpeg 编码。可与 viewer 并行：学生操作 viewer，后台按固定帧率采帧。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        output: Path,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        sim_dt: float = 0.002,
        stride: int = 0,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未找到 ffmpeg，请先安装: sudo apt install ffmpeg")

        self.stride = stride or max(1, int(round(1.0 / (fps * sim_dt))))
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self._default_cam = self._make_default_camera()
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(self.output),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.frames = 0
        self._step = 0

    @staticmethod
    def _make_default_camera() -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array([1.2, -0.2, 0.85])
        cam.distance = 5.5
        cam.elevation = -18.0
        cam.azimuth = 140.0
        return cam

    def capture(self, data: mujoco.MjData, camera: mujoco.MjvCamera | None = None) -> None:
        self._step += 1
        if (self._step - 1) % self.stride != 0:
            return
        cam = camera if camera is not None else self._default_cam
        self.renderer.update_scene(data, camera=cam)
        assert self._ffmpeg.stdin is not None
        self._ffmpeg.stdin.write(self.renderer.render().tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._ffmpeg.stdin:
            self._ffmpeg.stdin.close()
        self._ffmpeg.wait()

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *_) -> None:
        self.close()
