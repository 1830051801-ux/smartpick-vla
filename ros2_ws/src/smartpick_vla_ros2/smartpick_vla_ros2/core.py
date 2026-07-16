"""ROS-independent pairing logic used by the safety bridge node."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite

from smartpick_vla.real.types import ActionChunk

ChunkKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class ChunkPair:
    base: ActionChunk
    residual: ActionChunk


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
