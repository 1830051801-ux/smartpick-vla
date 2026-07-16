"""Versioned real-robot log import with strict frame and unit validation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import RigidTransform
from .types import ACTION_ORDER, BASE_FRAME, CartesianDelta, RobotState

REAL_LOG_SCHEMA_VERSION = "1.0"
TRANSLATION_UNIT = "m"
ROTATION_UNIT = "rad"
TIME_UNIT = "s"
TASK_CLASSES = frozenset({"accepted", "scratch", "unknown"})


class RealLogValidationError(ValueError):
    """A real log is ambiguous, unsafe to replay, or violates the schema."""


@dataclass(frozen=True, slots=True)
class RealLogStep:
    schema_version: str
    episode_id: str
    step_index: int
    timestamp_s: float
    instruction: str
    task_class: str
    robot_state: RobotState
    action: CartesianDelta
    image_file: str | None = None
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "timestamp_s": self.timestamp_s,
            "instruction": self.instruction,
            "task_class": self.task_class,
            "frame_id": self.robot_state.frame_id,
            "translation_unit": TRANSLATION_UNIT,
            "rotation_unit": ROTATION_UNIT,
            "time_unit": TIME_UNIT,
            "robot_state": self.robot_state.to_dict(),
            "action": self.action.to_dict(),
            "action_order": list(ACTION_ORDER),
            "image_file": self.image_file,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class RealLogEpisode:
    schema_version: str
    episode_id: str
    steps: tuple[RealLogStep, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.episode_id.strip() or not self.steps:
            raise RealLogValidationError("episode_id must be non-empty and steps must not be empty")
        if any(step.episode_id != self.episode_id for step in self.steps):
            raise RealLogValidationError("every step must match the enclosing episode_id")
        if any(step.schema_version != self.schema_version for step in self.steps):
            raise RealLogValidationError("every step must match the enclosing schema_version")
        _validate_episode(self.episode_id, list(self.steps))

    @property
    def duration_s(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return self.steps[-1].timestamp_s - self.steps[0].timestamp_s

    @property
    def success(self) -> bool | None:
        values = [step.success for step in self.steps if step.success is not None]
        return values[-1] if values else None


def load_real_log(
    path: str | Path,
    *,
    calibration: RigidTransform | None = None,
    require_images: bool = False,
) -> tuple[RealLogEpisode, ...]:
    """Load JSONL or CSV logs and return base-frame episodes.

    Logs already in ``base_link`` need no calibration.  Any other frame is
    rejected unless an explicit transform from that frame to ``base_link`` is
    supplied.  This prevents an unlabelled camera-frame trajectory from being
    mistaken for a robot-base command.
    """

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(input_path)
    elif suffix == ".csv":
        rows = _read_csv(input_path)
    else:
        raise RealLogValidationError("real log must use .jsonl or .csv")
    if not rows:
        raise RealLogValidationError(f"real log is empty: {input_path}")

    grouped: dict[str, list[RealLogStep]] = defaultdict(list)
    for line_number, row in rows:
        try:
            _validate_contract(row)
            step = _parse_step(row)
            if step.robot_state.frame_id != BASE_FRAME:
                if calibration is None:
                    raise RealLogValidationError(
                        f"frame {step.robot_state.frame_id!r} requires explicit calibration to {BASE_FRAME!r}"
                    )
                step = transform_step_to_base(step, calibration)
            _validate_step(step, input_path, require_images=require_images)
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, RealLogValidationError):
                detail = str(error)
            else:
                detail = f"{type(error).__name__}: {error}"
            raise RealLogValidationError(f"{input_path}:{line_number}: {detail}") from error
        grouped[step.episode_id].append(step)

    episodes: list[RealLogEpisode] = []
    for episode_id, steps in grouped.items():
        _validate_episode(episode_id, steps)
        episodes.append(
            RealLogEpisode(
                schema_version=steps[0].schema_version,
                episode_id=episode_id,
                steps=tuple(steps),
            )
        )
    return tuple(episodes)


def save_real_log(path: str | Path, episodes: Iterable[RealLogEpisode]) -> None:
    """Write episodes as deterministic JSONL suitable for later replay."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            for step in episode.steps:
                handle.write(
                    json.dumps(step.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
                )


def transform_step_to_base(step: RealLogStep, transform: RigidTransform) -> RealLogStep:
    """Transform a yaw-only step into ``base_link`` using a calibrated transform."""

    source_frame = step.robot_state.frame_id
    if transform.source_frame != source_frame or transform.target_frame != BASE_FRAME:
        raise RealLogValidationError(
            f"calibration is {transform.source_frame}->{transform.target_frame}, expected {source_frame}->{BASE_FRAME}"
        )
    delta_xyz = transform.apply_vector((step.action.dx_m, step.action.dy_m, step.action.dz_m))
    yaw_offset = transform.yaw_offset_rad()
    state = RobotState(
        stamp_s=step.robot_state.stamp_s,
        tcp_position_m=transform.apply_point(step.robot_state.tcp_position_m),
        tcp_yaw_rad=step.robot_state.tcp_yaw_rad + yaw_offset,
        joint_positions_rad=step.robot_state.joint_positions_rad,
        gripper=step.robot_state.gripper,
        frame_id=BASE_FRAME,
    )
    action = CartesianDelta(
        dx_m=delta_xyz[0],
        dy_m=delta_xyz[1],
        dz_m=delta_xyz[2],
        dyaw_rad=step.action.dyaw_rad,
        gripper=step.action.gripper,
    )
    metadata = dict(step.metadata)
    metadata["coordinate_transform"] = {
        "source_frame": transform.source_frame,
        "target_frame": transform.target_frame,
    }
    return RealLogStep(
        schema_version=step.schema_version,
        episode_id=step.episode_id,
        step_index=step.step_index,
        timestamp_s=step.timestamp_s,
        instruction=step.instruction,
        task_class=step.task_class,
        robot_state=state,
        action=action,
        image_file=step.image_file,
        success=step.success,
        metadata=metadata,
    )


def _read_jsonl(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    rows: list[tuple[int, Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RealLogValidationError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise RealLogValidationError(
                    f"{path}:{line_number}: each JSONL row must be an object"
                )
            rows.append((line_number, value))
    return rows


def _read_csv(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [(line_number, row) for line_number, row in enumerate(reader, start=2)]


def _parse_step(row: Mapping[str, Any]) -> RealLogStep:
    if "robot_state" in row:
        state_payload = _mapping(row["robot_state"], "robot_state")
        action_payload = row["action"]
        timestamp_s = float(row["timestamp_s"])
        row_frame = _optional_string(row.get("frame_id"))
        state_frame = _optional_string(state_payload.get("frame_id"))
        if row_frame is not None and state_frame is not None and row_frame != state_frame:
            raise RealLogValidationError("top-level and robot-state frame_id values disagree")
        state = RobotState(
            stamp_s=float(state_payload.get("stamp_s", timestamp_s)),
            tcp_position_m=_float_tuple3(state_payload["tcp_position_m"], "tcp_position_m"),
            tcp_yaw_rad=float(state_payload["tcp_yaw_rad"]),
            joint_positions_rad=_float_tuple_any(state_payload.get("joint_positions_rad", ())),
            gripper=float(state_payload["gripper"]),
            frame_id=state_frame or row_frame or "",
        )
        action = _parse_action(action_payload)
        metadata_value = row.get("metadata", {})
        metadata = dict(_mapping(metadata_value, "metadata")) if metadata_value is not None else {}
    else:
        timestamp_s = float(row["timestamp_s"])
        state = RobotState(
            stamp_s=float(row.get("state_stamp_s") or timestamp_s),
            tcp_position_m=(float(row["tcp_x_m"]), float(row["tcp_y_m"]), float(row["tcp_z_m"])),
            tcp_yaw_rad=float(row["tcp_yaw_rad"]),
            joint_positions_rad=_parse_delimited_floats(row.get("joint_positions_rad", "")),
            gripper=float(row["state_gripper"]),
            frame_id=str(row["frame_id"]),
        )
        action = CartesianDelta(
            dx_m=float(row["dx_m"]),
            dy_m=float(row["dy_m"]),
            dz_m=float(row["dz_m"]),
            dyaw_rad=float(row["dyaw_rad"]),
            gripper=float(row["action_gripper"]),
        )
        metadata_text = str(row.get("metadata_json") or "").strip()
        metadata = (
            dict(_mapping(json.loads(metadata_text), "metadata_json")) if metadata_text else {}
        )

    return RealLogStep(
        schema_version=str(row.get("schema_version") or REAL_LOG_SCHEMA_VERSION),
        episode_id=str(row["episode_id"]),
        step_index=int(row["step_index"]),
        timestamp_s=timestamp_s,
        instruction=str(row["instruction"]),
        task_class=str(row["task_class"]),
        robot_state=state,
        action=action,
        image_file=_optional_string(row.get("image_file")),
        success=_optional_bool(row.get("success")),
        metadata=metadata,
    )


def _parse_action(value: Any) -> CartesianDelta:
    if isinstance(value, Mapping):
        return CartesianDelta.from_mapping(dict(value))
    if isinstance(value, (list, tuple)):
        return CartesianDelta.from_array(value)
    raise RealLogValidationError("action must be an object or a five-value array")


def _validate_contract(row: Mapping[str, Any]) -> None:
    schema_version = _optional_string(row.get("schema_version"))
    if schema_version is None:
        raise RealLogValidationError("missing required schema_version")
    if schema_version != REAL_LOG_SCHEMA_VERSION:
        raise RealLogValidationError(
            f"unsupported schema_version {schema_version!r}; expected {REAL_LOG_SCHEMA_VERSION!r}"
        )

    declarations = {
        "translation_unit": TRANSLATION_UNIT,
        "rotation_unit": ROTATION_UNIT,
        "time_unit": TIME_UNIT,
    }
    for field_name, expected in declarations.items():
        value = _optional_string(row.get(field_name))
        if value is None:
            raise RealLogValidationError(f"missing required {field_name}")
        if value != expected:
            raise RealLogValidationError(
                f"unsupported {field_name} {value!r}; expected {expected!r}"
            )

    row_frame = _optional_string(row.get("frame_id"))
    state_payload = row.get("robot_state")
    state_frame: str | None = None
    if isinstance(state_payload, Mapping):
        state_frame = _optional_string(state_payload.get("frame_id"))
    if row_frame is None and state_frame is None:
        raise RealLogValidationError("missing required frame_id")

    action_order = _parse_action_order(row.get("action_order"))
    if action_order != ACTION_ORDER:
        raise RealLogValidationError(
            f"action_order must be {list(ACTION_ORDER)!r}, got {list(action_order)!r}"
        )


def _parse_action_order(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise RealLogValidationError("action_order JSON must be a list")
            return tuple(str(item).strip() for item in parsed)
        return tuple(item.strip() for item in text.replace(",", ";").split(";") if item.strip())
    return ()


def _validate_step(step: RealLogStep, path: Path, *, require_images: bool) -> None:
    if step.schema_version != REAL_LOG_SCHEMA_VERSION:
        raise RealLogValidationError(
            f"unsupported schema_version {step.schema_version!r}; expected {REAL_LOG_SCHEMA_VERSION!r}"
        )
    if not step.episode_id.strip() or step.step_index < 0:
        raise RealLogValidationError("episode_id must be non-empty and step_index non-negative")
    if not step.instruction.strip():
        raise RealLogValidationError("instruction must not be empty")
    if step.task_class not in TASK_CLASSES:
        raise RealLogValidationError(
            f"task_class must be one of {sorted(TASK_CLASSES)!r}, got {step.task_class!r}"
        )
    if step.robot_state.frame_id != BASE_FRAME:
        raise RealLogValidationError(f"normalized log frame must be {BASE_FRAME!r}")
    values = np.asarray(
        (
            step.timestamp_s,
            step.robot_state.stamp_s,
            *step.robot_state.tcp_position_m,
            step.robot_state.tcp_yaw_rad,
            *step.robot_state.joint_positions_rad,
            step.robot_state.gripper,
            *step.action.as_array(dtype=np.float64),
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RealLogValidationError("log step contains NaN or infinity")
    if not -1.0 <= step.robot_state.gripper <= 1.0 or not -1.0 <= step.action.gripper <= 1.0:
        raise RealLogValidationError("gripper values must be normalized to [-1, 1]")
    if require_images:
        if step.image_file is None:
            raise RealLogValidationError("image_file is required")
        image_path = Path(step.image_file)
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise RealLogValidationError(f"image file does not exist: {image_path}")


def _validate_episode(episode_id: str, steps: list[RealLogStep]) -> None:
    if not steps:
        raise RealLogValidationError(f"episode {episode_id!r} is empty")
    for previous, current in pairwise(steps):
        if current.step_index <= previous.step_index:
            raise RealLogValidationError(f"episode {episode_id!r} has non-increasing step_index")
        if current.timestamp_s <= previous.timestamp_s:
            raise RealLogValidationError(f"episode {episode_id!r} has non-increasing timestamp_s")
        if current.robot_state.stamp_s <= previous.robot_state.stamp_s:
            raise RealLogValidationError(
                f"episode {episode_id!r} has non-increasing robot-state stamp_s"
            )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RealLogValidationError(f"{label} must be an object")
    return value


def _float_tuple3(value: Any, label: str) -> tuple[float, float, float]:
    output = _float_tuple_any(value)
    if len(output) != 3:
        raise RealLogValidationError(f"{label} must contain 3 values")
    return output[0], output[1], output[2]


def _float_tuple_any(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        return _parse_delimited_floats(value)
    if not isinstance(value, (list, tuple)):
        raise RealLogValidationError("expected a list of numeric values")
    return tuple(float(item) for item in value)


def _parse_delimited_floats(value: Any) -> tuple[float, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(float(item.strip()) for item in text.replace(",", ";").split(";") if item.strip())


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise RealLogValidationError(f"invalid boolean value {value!r}")
