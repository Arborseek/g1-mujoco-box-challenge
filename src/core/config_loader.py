from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class Zone:
    center: np.ndarray
    half_size: np.ndarray

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zone:
        return cls(
            center=np.asarray(data["center"], dtype=float),
            half_size=np.asarray(data["half_size"], dtype=float),
        )

    def contains(self, point: np.ndarray) -> bool:
        delta = np.abs(point - self.center)
        return bool(np.all(delta[:2] <= self.half_size[:2]))


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = project_root() / "config" / "task.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
