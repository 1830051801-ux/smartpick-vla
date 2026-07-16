"""Strongly typed contracts shared by real-log replay and ROS 2 adapters.

The canonical command is deliberately small and explicit: Cartesian translation
increments and yaw are expressed in ``base_link`` using metres and radians;
the gripper channel is normalized to ``[-1, 1]``.  Policies may operate on
NumPy arrays internally, but commands cross subsystem boundaries as these
dataclasses so that frame, timing, task, and sequence metadata cannot be lost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

BASE_FRAME: Final = "base_link"
ACTION_ORDER: Final = ("dx_m", "dy_m", "dz_m", "dyaw_rad", "gripper")
ACTION_DIM: Final = len(ACTION_ORDER)

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class CartesianDelta:
    """One policy action in ``base_link``.

    The first three values are translational increments in metres, ``dyaw_rad``
    is a yaw increment in radians, and ``gripper`` is dimensionless in
    ``[-1, 1]``.  Range and finiteness checks intentionally live in the safety
    supervisor so even malformed model output is represented and rejected with
    an auditable reason.
    """

    dx_m: float
    dy_m: float
    dz_m: float
    dyaw_rad: float
    gripper: float

    @classmethod
    def from_array(cls, values: Sequence[float] | npt.NDArray[np.floating[Any]]) -> CartesianDelta:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {array.shape}")
        return cls(*(float(value) for value in array))

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> CartesianDelta:
        try:
            return cls(*(float(value[name]) for name in ACTION_ORDER))
        except KeyError as error:
            raise ValueError(f"action is missing field {error.args[0]!r}") from error

    def as_array(self, *, dtype: npt.DTypeLike = np.float32) -> npt.NDArray[Any]:
        return np.asarray(
            (self.dx_m, self.dy_m, self.dz_m, self.dyaw_rad, self.gripper),
            dtype=dtype,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """A time-ordered block of Cartesian delta commands."""

    task_id: str
    sequence_id: int
    stamp_s: float
    dt_s: float
    actions: tuple[CartesianDelta, ...]
    frame_id: str = BASE_FRAME
    policy_id: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if not self.actions:
            raise ValueError("action chunk must contain at least one action")

    @property
    def horizon(self) -> int:
        return len(self.actions)

    def as_array(self) -> FloatArray:
        return np.stack([action.as_array(dtype=np.float32) for action in self.actions]).astype(
            np.float32,
            copy=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sequence_id": self.sequence_id,
            "stamp_s": self.stamp_s,
            "dt_s": self.dt_s,
            "frame_id": self.frame_id,
            "policy_id": self.policy_id,
            "action_order": list(ACTION_ORDER),
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_array(
        cls,
        values: Sequence[Sequence[float]] | npt.NDArray[np.floating[Any]],
        *,
        task_id: str,
        sequence_id: int,
        stamp_s: float,
        dt_s: float,
        frame_id: str = BASE_FRAME,
        policy_id: str = "unknown",
    ) -> ActionChunk:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != ACTION_DIM:
            raise ValueError(f"action chunk must have shape [H, {ACTION_DIM}], got {array.shape}")
        return cls(
            task_id=task_id,
            sequence_id=sequence_id,
            stamp_s=float(stamp_s),
            dt_s=float(dt_s),
            actions=tuple(CartesianDelta.from_array(row) for row in array),
            frame_id=frame_id,
            policy_id=policy_id,
        )


@dataclass(frozen=True, slots=True)
class RobotState:
    """Robot feedback required to validate an incremental action chunk."""

    stamp_s: float
    tcp_position_m: tuple[float, float, float]
    tcp_yaw_rad: float
    joint_positions_rad: tuple[float, ...]
    gripper: float
    frame_id: str = BASE_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "tcp_position_m", tuple(float(v) for v in self.tcp_position_m))
        object.__setattr__(
            self,
            "joint_positions_rad",
            tuple(float(v) for v in self.joint_positions_rad),
        )
        if len(self.tcp_position_m) != 3:
            raise ValueError("tcp_position_m must contain exactly three values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stamp_s": self.stamp_s,
            "frame_id": self.frame_id,
            "tcp_position_m": list(self.tcp_position_m),
            "tcp_yaw_rad": self.tcp_yaw_rad,
            "joint_positions_rad": list(self.joint_positions_rad),
            "gripper": self.gripper,
        }


@dataclass(frozen=True, slots=True)
class ComposedActionChunk:
    """Audit record for ``base + bounded residual = final``."""

    base: ActionChunk
    requested_residual: ActionChunk
    bounded_residual: ActionChunk
    final: ActionChunk


def zero_chunk_like(chunk: ActionChunk, *, policy_id: str = "zero_residual") -> ActionChunk:
    """Create an all-zero residual chunk with matching identity and timing."""

    return ActionChunk.from_array(
        np.zeros((chunk.horizon, ACTION_DIM), dtype=np.float32),
        task_id=chunk.task_id,
        sequence_id=chunk.sequence_id,
        stamp_s=chunk.stamp_s,
        dt_s=chunk.dt_s,
        frame_id=chunk.frame_id,
        policy_id=policy_id,
    )
