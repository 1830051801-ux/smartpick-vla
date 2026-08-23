"""ROS-independent pairing logic used by the safety bridge node."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite
from typing import Any

from smartpick_vla.real.types import ActionChunk

ChunkKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class ChunkPair:
    base: ActionChunk
    residual: ActionChunk


@dataclass(frozen=True, slots=True)
class PredictiveRisk:
    """ROS-independent world-model risk result for a dry-run preview."""

    collision_probability: float
    termination_probability: float
    wrong_bin_probability: float
    horizon: int
    blocked: bool
    model_id: str = "unavailable"
    wrong_pick_probability: float = 0.0
    max_state_std: float = 0.0
    risk_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        probabilities = (
            self.collision_probability,
            self.termination_probability,
            self.wrong_bin_probability,
            self.wrong_pick_probability,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("predictive probabilities must be finite and in [0,1]")
        if self.horizon < 1:
            raise ValueError("predictive horizon must be positive")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not isfinite(self.max_state_std) or self.max_state_std < 0.0:
            raise ValueError("max_state_std must be finite and non-negative")
        if any(not reason.strip() for reason in self.risk_reasons):
            raise ValueError("risk_reasons must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "collision_probability": self.collision_probability,
            "termination_probability": self.termination_probability,
            "wrong_bin_probability": self.wrong_bin_probability,
            "wrong_pick_probability": self.wrong_pick_probability,
            "max_state_std": self.max_state_std,
            "risk_reasons": list(self.risk_reasons),
            "horizon": self.horizon,
            "blocked": self.blocked,
            "model_id": self.model_id,
        }


class ChunkPairBuffer:
    """Pair base and residual chunks by task and sequence with bounded memory."""

    def __init__(self, max_pending: int = 32) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = max_pending
        self._base: dict[ChunkKey, ActionChunk] = {}
        self._residual: dict[ChunkKey, ActionChunk] = {}
        self._arrival_order: OrderedDict[ChunkKey, None] = OrderedDict()

    def add_base(self, chunk: ActionChunk) -> ChunkPair | None:
        key = (chunk.task_id, chunk.sequence_id)
        self._base[key] = chunk
        self._mark_latest(key)
        return self._pop_if_ready(key)

    def add_residual(self, chunk: ActionChunk) -> ChunkPair | None:
        key = (chunk.task_id, chunk.sequence_id)
        self._residual[key] = chunk
        self._mark_latest(key)
        return self._pop_if_ready(key)

    @property
    def pending_count(self) -> int:
        return len(set(self._base) | set(self._residual))

    def _pop_if_ready(self, key: ChunkKey) -> ChunkPair | None:
        if key in self._base and key in self._residual:
            self._arrival_order.pop(key, None)
            return ChunkPair(self._base.pop(key), self._residual.pop(key))
        self._trim()
        return None

    def _mark_latest(self, key: ChunkKey) -> None:
        self._arrival_order.pop(key, None)
        self._arrival_order[key] = None

    def _trim(self) -> None:
        while self.pending_count > self.max_pending:
            oldest, _ = self._arrival_order.popitem(last=False)
            self._base.pop(oldest, None)
            self._residual.pop(oldest, None)


def stamp_to_seconds(sec: int, nanosec: int) -> float:
    """Convert a ROS ``builtin_interfaces/Time`` pair without importing ROS 2."""

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("ROS time requires sec >= 0 and 0 <= nanosec < 1e9")
    return float(sec) + float(nanosec) * 1e-9


def seconds_to_stamp_parts(value_s: float) -> tuple[int, int]:
    """Convert non-negative finite seconds into normalized ROS time fields."""

    if not isfinite(value_s) or value_s < 0.0:
        raise ValueError("ROS timestamp must be finite and non-negative")
    sec = int(value_s)
    nanosec = round((value_s - sec) * 1_000_000_000)
    if nanosec == 1_000_000_000:
        sec += 1
        nanosec = 0
    return sec, nanosec
