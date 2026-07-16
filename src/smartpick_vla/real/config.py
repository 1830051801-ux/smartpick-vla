"""Validated runtime configuration for real-log and hardware preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from .calibration import RigidTransform
from .execution import GateConfig, gate_config_from_mapping
from .safety import SafetyLimits
from .system_parameters import SystemParameters
from .types import ACTION_DIM, ACTION_ORDER, BASE_FRAME


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    sample_period_s: float = 0.05
    speed: float = 1.0


@dataclass(frozen=True, slots=True)
class RealRuntimeConfig:
    schema_version: str
    safety: SafetyLimits
    execution: GateConfig
    replay: ReplayConfig
    camera_to_base: RigidTransform | None
    audit_jsonl: str | None
    camera: dict[str, Any]
    robot: dict[str, Any]
    system_parameters: SystemParameters


def load_real_config(path: str | Path) -> RealRuntimeConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("real runtime configuration must be a mapping")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "1.0":
        raise ValueError(f"unsupported real runtime schema_version {schema_version!r}")

    action = _section(payload, "action")
    if int(action.get("dim", ACTION_DIM)) != ACTION_DIM:
        raise ValueError(f"configured action dim must be {ACTION_DIM}")
    if tuple(action.get("order", ACTION_ORDER)) != ACTION_ORDER:
        raise ValueError(f"configured action order must be {ACTION_ORDER}")
    if str(action.get("frame_id", BASE_FRAME)) != BASE_FRAME:
        raise ValueError(f"canonical action frame must be {BASE_FRAME!r}")
    if str(action.get("translation_unit", "m")) != "m":
        raise ValueError("canonical translation unit must be 'm'")
    if str(action.get("rotation_unit", "rad")) != "rad":
        raise ValueError("canonical rotation unit must be 'rad'")
    gripper_range = tuple(float(value) for value in action.get("gripper_range", (-1.0, 1.0)))
    if gripper_range != (-1.0, 1.0):
        raise ValueError("canonical gripper range must be [-1.0, 1.0]")

    safety = SafetyLimits.from_mapping(_section(payload, "safety"))
    execution_payload = _section(payload, "execution")
    execution = gate_config_from_mapping(dict(execution_payload))
    audit_value = execution_payload.get("audit_jsonl")
    audit_jsonl = str(audit_value).strip() if audit_value is not None else None
    if audit_jsonl == "":
        audit_jsonl = None
    replay_payload = _section(payload, "replay")
    replay = ReplayConfig(
        sample_period_s=float(replay_payload.get("sample_period_s", 0.05)),
        speed=float(replay_payload.get("speed", 1.0)),
    )
    if replay.sample_period_s <= 0.0 or replay.speed <= 0.0:
        raise ValueError("replay sample period and speed must be positive")

    calibration = _section(payload, "calibration")
    camera_to_base: RigidTransform | None = None
    calibrated = calibration.get("calibrated", False)
    if not isinstance(calibrated, bool):
        raise ValueError("calibration.calibrated must be a YAML boolean")
    if calibrated:
        camera_to_base = RigidTransform(
            source_frame=str(calibration["source_frame"]),
            target_frame=str(calibration["target_frame"]),
            rotation=cast(
                tuple[tuple[float, float, float], ...],
                tuple(tuple(float(value) for value in row) for row in calibration["rotation"]),
            ),
            translation_m=cast(
                tuple[float, float, float],
                tuple(float(value) for value in calibration["translation_m"]),
            ),
        )
        if camera_to_base.target_frame != BASE_FRAME:
            raise ValueError(f"calibration target must be {BASE_FRAME!r}")

    return RealRuntimeConfig(
        schema_version=schema_version,
        safety=safety,
        execution=execution,
        replay=replay,
        camera_to_base=camera_to_base,
        audit_jsonl=audit_jsonl,
        camera=dict(_section(payload, "camera")),
        robot=dict(_section(payload, "robot")),
        system_parameters=SystemParameters.from_mapping(_section(payload, "system_parameters")),
    )


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return value
