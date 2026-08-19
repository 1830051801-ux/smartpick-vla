"""Planning-only bridge from XiaoU camera detections to six-axis target poses.

The module intentionally stops at a ROS 2-compatible pose preview. It can read
the project homography and per-object grasp profiles, but it never opens a
serial, CAN, or hardware executor. A real arm must still satisfy the separate
ROS 2 and hardware safety gates before any motion is considered.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, isfinite, sin
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from smartpick_vla.utils.io import atomic_write_json

XiaoUPhase = Literal["pregrasp", "grasp", "lift"]


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


@dataclass(frozen=True, slots=True)
class XiaoUHomography:
    """Validated pixel-to-robot-base planar calibration from the XiaoU workflow."""

    matrix: tuple[tuple[float, float, float], ...]
    mean_error_mm: float
    max_error_mm: float
    source_frame: str = "camera_pixels"
    target_frame: str = "base_link"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("homography matrix must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("homography matrix must be invertible")
        errors = np.asarray((self.mean_error_mm, self.max_error_mm), dtype=np.float64)
        if not np.isfinite(errors).all() or np.any(errors < 0.0):
            raise ValueError("homography errors must be finite and non-negative")
        if not self.source_frame.strip() or self.target_frame != "base_link":
            raise ValueError("homography must map a named camera frame into base_link")

    @property
    def matrix_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def project_pixel(self, u_px: float, v_px: float) -> tuple[float, float]:
        """Project pixel coordinates into base-link metres."""

        pixel = np.asarray((_finite_float(u_px, "u_px"), _finite_float(v_px, "v_px"), 1.0))
        homogeneous = self.matrix_array @ pixel
        if abs(float(homogeneous[2])) < 1e-12:
            raise ValueError("homography projection reached the line at infinity")
        point_mm = homogeneous[:2] / homogeneous[2]
        if not np.isfinite(point_mm).all():
            raise ValueError("homography projection is non-finite")
        return float(point_mm[0] / 1000.0), float(point_mm[1] / 1000.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "homography": [list(row) for row in self.matrix],
            "mean_error_mm": self.mean_error_mm,
            "max_error_mm": self.max_error_mm,
        }


@dataclass(frozen=True, slots=True)
class XiaoUDetection:
    """One stable camera detection supplied by the vision system."""

    label: str
    u_px: float
    v_px: float
    confidence: float
    stamp_s: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("detection label must not be empty")
        for name in ("u_px", "v_px", "confidence", "stamp_s"):
            _finite_float(getattr(self, name), name)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")
        if self.stamp_s < 0.0:
            raise ValueError("detection timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class XiaoUGraspProfile:
    """Measured or simulation-only vertical grasp contract for one object label."""

    label: str
    grasp_height_m: float
    approach_height_m: float
    lift_height_m: float
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("grasp profile label must not be empty")
        values = np.asarray(
            (self.grasp_height_m, self.approach_height_m, self.lift_height_m, self.yaw_rad),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("grasp profile values must be finite")
        if self.grasp_height_m < 0.0:
            raise ValueError("grasp height must be non-negative")
        if not self.grasp_height_m < self.approach_height_m <= self.lift_height_m:
            raise ValueError("grasp, approach, and lift heights must be strictly ordered")


@dataclass(frozen=True, slots=True)
class XiaoUPoseTarget:
    phase: XiaoUPhase
    position_m: tuple[float, float, float]
    yaw_rad: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("pose target position must be a finite 3-vector")
        if not isfinite(self.yaw_rad):
            raise ValueError("pose target yaw must be finite")

    def to_ros2_pose_stamped(self, *, stamp_s: float) -> dict[str, Any]:
        """Return a JSON-safe ``geometry_msgs/PoseStamped``-compatible preview."""

        half_yaw = self.yaw_rad * 0.5
        return {
            "header": {"frame_id": "base_link", "stamp_s": stamp_s},
            "pose": {
                "position": {
                    "x": self.position_m[0],
                    "y": self.position_m[1],
                    "z": self.position_m[2],
                },
                "orientation": {"x": 0.0, "y": 0.0, "z": sin(half_yaw), "w": cos(half_yaw)},
            },
        }


@dataclass(frozen=True, slots=True)
class XiaoUPlanPreview:
    """Auditable planning-only output for XiaoU's six-axis MoveIt target node."""

    task_id: str
    label: str
    detection_confidence: float
    detection_stamp_s: float
    calibration_max_error_mm: float
    targets: tuple[XiaoUPoseTarget, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.label.strip():
            raise ValueError("task_id and label must not be empty")
        if len(self.targets) != 3:
            raise ValueError("XiaoU plan preview must contain pregrasp, grasp, and lift targets")
        if tuple(target.phase for target in self.targets) != ("pregrasp", "grasp", "lift"):
            raise ValueError("XiaoU targets must use pregrasp, grasp, lift ordering")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "picksort-xiaou-plan-preview/v1",
            "planning_only": True,
            "real_motion_authorized": False,
            "task_id": self.task_id,
            "label": self.label,
            "detection_confidence": self.detection_confidence,
            "detection_stamp_s": self.detection_stamp_s,
            "calibration_max_error_mm": self.calibration_max_error_mm,
            "targets": [
                {
                    "phase": target.phase,
                    "position_m": list(target.position_m),
                    "yaw_rad": target.yaw_rad,
                    "ros2_pose_stamped_preview": target.to_ros2_pose_stamped(
                        stamp_s=self.detection_stamp_s
                    ),
                }
                for target in self.targets
            ],
        }


def load_xiaou_homography(path: str | Path) -> XiaoUHomography:
    """Load the established XiaoU ``workspace_homography.yaml`` contract."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("XiaoU homography file must contain a mapping")
    if str(payload.get("type", "")) != "pixel_to_robot_base_mm":
        raise ValueError("XiaoU homography type must be pixel_to_robot_base_mm")
    matrix = payload.get("homography")
    if not isinstance(matrix, list):
        raise ValueError("XiaoU homography must contain a 3x3 matrix")
    rows: list[tuple[float, ...]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)):
            raise ValueError("XiaoU homography rows must be sequences")
        values = tuple(_finite_float(value, "homography value") for value in row)
        if len(values) != 3:
            raise ValueError("XiaoU homography must contain exactly three columns")
        rows.append(values)
    if len(rows) != 3:
        raise ValueError("XiaoU homography must contain exactly three rows")
    return XiaoUHomography(
        matrix=cast(tuple[tuple[float, float, float], ...], tuple(rows)),
        mean_error_mm=_finite_float(payload.get("mean_error_mm"), "mean_error_mm"),
        max_error_mm=_finite_float(payload.get("max_error_mm"), "max_error_mm"),
    )


def load_xiaou_grasp_profiles(path: str | Path) -> dict[str, XiaoUGraspProfile]:
    """Load only complete grasp profiles and reject unknown measured values.

    The current XiaoU hardware profile deliberately stores unmeasured heights
    as ``null``. This function refuses those entries rather than assigning a
    generic grasp height, which keeps the planning bridge safe by default.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payload = (
        yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    )
    if not isinstance(payload, Mapping):
        raise ValueError("XiaoU grasp profile file must contain a mapping")
    profiles_value = payload.get("profiles", payload.get("classes"))
    if not isinstance(profiles_value, Mapping):
        raise ValueError("XiaoU grasp profile file must contain profiles/classes")
    profiles: dict[str, XiaoUGraspProfile] = {}
    incomplete: list[str] = []
    for label, raw_profile in profiles_value.items():
        if not isinstance(label, str) or not isinstance(raw_profile, Mapping):
            raise ValueError("XiaoU grasp profiles must map labels to objects")
        required = ("grasp_height_m", "approach_height_m")
        if any(raw_profile.get(name) is None for name in required):
            incomplete.append(label)
            continue
        lift_value = raw_profile.get("lift_height_m", raw_profile.get("approach_height_m"))
        if lift_value is None:
            incomplete.append(label)
            continue
        profiles[label] = XiaoUGraspProfile(
            label=label,
            grasp_height_m=_finite_float(raw_profile["grasp_height_m"], "grasp_height_m"),
            approach_height_m=_finite_float(raw_profile["approach_height_m"], "approach_height_m"),
            lift_height_m=_finite_float(lift_value, "lift_height_m"),
            yaw_rad=_finite_float(raw_profile.get("yaw_rad", 0.0), "yaw_rad"),
        )
    if incomplete:
        raise ValueError(
            "XiaoU grasp profiles are incomplete for: " + ", ".join(sorted(incomplete))
        )
    if not profiles:
        raise ValueError("XiaoU grasp profile file contains no complete profiles")
    return profiles


def build_xiaou_plan_preview(
    detection: XiaoUDetection,
    *,
    homography: XiaoUHomography,
    profiles: Mapping[str, XiaoUGraspProfile],
    task_id: str,
    minimum_confidence: float = 0.55,
    maximum_calibration_error_mm: float = 2.0,
) -> XiaoUPlanPreview:
    """Build a planning-only six-axis target sequence from a camera detection."""

    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if maximum_calibration_error_mm <= 0.0 or not isfinite(maximum_calibration_error_mm):
        raise ValueError("maximum_calibration_error_mm must be finite and positive")
    if detection.confidence < minimum_confidence:
        raise ValueError(
            f"detection confidence {detection.confidence:.3f} is below {minimum_confidence:.3f}"
        )
    if homography.max_error_mm > maximum_calibration_error_mm:
        raise ValueError(
            "homography max error "
            f"{homography.max_error_mm:.3f}mm exceeds {maximum_calibration_error_mm:.3f}mm"
        )
    profile = profiles.get(detection.label)
    if profile is None:
        raise ValueError(f"no complete XiaoU grasp profile exists for {detection.label!r}")
    x_m, y_m = homography.project_pixel(detection.u_px, detection.v_px)
    targets = (
        XiaoUPoseTarget("pregrasp", (x_m, y_m, profile.approach_height_m), profile.yaw_rad),
        XiaoUPoseTarget("grasp", (x_m, y_m, profile.grasp_height_m), profile.yaw_rad),
        XiaoUPoseTarget("lift", (x_m, y_m, profile.lift_height_m), profile.yaw_rad),
    )
    return XiaoUPlanPreview(
        task_id=task_id,
        label=detection.label,
        detection_confidence=detection.confidence,
        detection_stamp_s=detection.stamp_s,
        calibration_max_error_mm=homography.max_error_mm,
        targets=targets,
    )


def save_xiaou_plan_preview(preview: XiaoUPlanPreview, path: str | Path) -> Path:
    """Write a JSON planning artifact, never a hardware command."""

    destination = Path(path)
    atomic_write_json(destination, preview.to_dict())
    return destination
