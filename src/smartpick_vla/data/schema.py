"""Versioned trajectory records shared by simulation and real-log replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

QualityClass = Literal["accepted", "scratch", "unknown"]


@dataclass(frozen=True, slots=True)
class StepRecord:
    timestamp_s: float
    instruction: str
    robot_state: list[float]
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    image_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    schema_version: str
    episode_id: str
    source: Literal["sim", "real", "real_replay"]
    seed: int
    task_class: QualityClass
    instruction: str
    success: bool
    steps: list[StepRecord]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_action(action: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (5,):
        raise ValueError(f"action must have shape (5,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("action contains NaN or infinity")
    return array
