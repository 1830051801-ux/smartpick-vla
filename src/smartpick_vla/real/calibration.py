"""Rigid camera-to-base calibration utilities for real-log import."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """A proper 3D rigid transform from ``source_frame`` to ``target_frame``."""

    source_frame: str
    target_frame: str
    rotation: tuple[tuple[float, float, float], ...]
    translation_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.source_frame.strip() or not self.target_frame.strip():
            raise ValueError("source and target frames must not be empty")
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rigid transform requires a 3x3 rotation and 3-vector translation")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("rigid transform contains NaN or infinity")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation matrix is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("rotation matrix must have determinant +1")

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.rotation, dtype=np.float64)

    @property
    def translation_vector(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.translation_m, dtype=np.float64)

    @classmethod
    def identity(cls, frame: str = "base_link") -> RigidTransform:
        return cls(
            source_frame=frame,
            target_frame=frame,
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation_m=(0.0, 0.0, 0.0),
        )

    def apply_point(self, point_m: Sequence[float]) -> tuple[float, float, float]:
        point = np.asarray(point_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("point must be a finite 3-vector")
        transformed = self.rotation_matrix @ point + self.translation_vector
        return tuple(float(value) for value in transformed)  # type: ignore[return-value]

    def apply_vector(self, vector: Sequence[float]) -> tuple[float, float, float]:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("vector must be a finite 3-vector")
        transformed = self.rotation_matrix @ value
        return tuple(float(component) for component in transformed)  # type: ignore[return-value]

    def yaw_offset_rad(self) -> float:
        """Return planar yaw offset when both frames have aligned z axes."""

        rotation = self.rotation_matrix
        if not np.allclose(rotation[2], (0.0, 0.0, 1.0), atol=1e-5) or not np.allclose(
            rotation[:, 2],
            (0.0, 0.0, 1.0),
            atol=1e-5,
        ):
            raise ValueError("5D yaw-only actions require source and target z axes to be aligned")
        return float(atan2(rotation[1, 0], rotation[0, 0]))

    def inverse(self) -> RigidTransform:
        rotation = self.rotation_matrix.T
        translation = -(rotation @ self.translation_vector)
        return RigidTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            rotation=_matrix_tuple(rotation),
            translation_m=_vector_tuple(translation),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "rotation": [list(row) for row in self.rotation],
            "translation_m": list(self.translation_m),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    transform: RigidTransform
    rmse_m: float
    max_error_m: float
    sample_count: int

    def __post_init__(self) -> None:
        errors = np.asarray((self.rmse_m, self.max_error_m), dtype=np.float64)
        if not np.isfinite(errors).all() or np.any(errors < 0.0):
            raise ValueError("calibration errors must be finite and non-negative")
        if self.sample_count < 0:
            raise ValueError("calibration sample_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            **self.transform.to_dict(),
            "rmse_m": self.rmse_m,
            "max_error_m": self.max_error_m,
            "sample_count": self.sample_count,
        }


def estimate_rigid_transform(
    source_points_m: Sequence[Sequence[float]] | npt.NDArray[np.floating[Any]],
    target_points_m: Sequence[Sequence[float]] | npt.NDArray[np.floating[Any]],
    *,
    source_frame: str = "camera_link",
    target_frame: str = "base_link",
) -> CalibrationResult:
    """Estimate a least-squares rigid transform using the Kabsch algorithm."""

    source = np.asarray(source_points_m, dtype=np.float64)
    target = np.asarray(target_points_m, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target points must have matching shape [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("at least three point correspondences are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("calibration points contain NaN or infinity")
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("calibration points must not be collinear")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    predicted = (rotation @ source.T).T + translation
    errors = np.linalg.norm(predicted - target, axis=1)
    transform = RigidTransform(
        source_frame=source_frame,
        target_frame=target_frame,
        rotation=_matrix_tuple(rotation),
        translation_m=_vector_tuple(translation),
    )
    return CalibrationResult(
        transform=transform,
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        max_error_m=float(np.max(errors)),
        sample_count=int(source.shape[0]),
    )


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".yaml", ".yml"}:
        output.write_text(yaml.safe_dump(result.to_dict(), sort_keys=False), encoding="utf-8")
    else:
        output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> CalibrationResult:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("calibration file must contain a mapping")
    if str(payload.get("schema_version", "")) != "1.0":
        raise ValueError("unsupported calibration schema_version")
    transform = RigidTransform(
        source_frame=str(payload["source_frame"]),
        target_frame=str(payload["target_frame"]),
        rotation=cast(
            tuple[tuple[float, float, float], ...],
            tuple(tuple(float(value) for value in row) for row in payload["rotation"]),
        ),
        translation_m=cast(
            tuple[float, float, float],
            tuple(float(value) for value in payload["translation_m"]),
        ),
    )
    return CalibrationResult(
        transform=transform,
        rmse_m=float(payload.get("rmse_m", 0.0)),
        max_error_m=float(payload.get("max_error_m", 0.0)),
        sample_count=int(payload.get("sample_count", 0)),
    )


def _matrix_tuple(matrix: npt.NDArray[np.float64]) -> tuple[tuple[float, float, float], ...]:
    return cast(
        tuple[tuple[float, float, float], ...],
        tuple(tuple(float(value) for value in row) for row in matrix),
    )


def _vector_tuple(vector: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    return cast(tuple[float, float, float], tuple(float(value) for value in vector))
