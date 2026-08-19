"""Explicit image-plane to robot-base calibration utilities.

The runtime controller only receives detected image pixels and a stored
calibration. MuJoCo world poses are used to create and evaluate the simulated
calibration artifact, never to choose an action at deployment time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _points(values: np.ndarray | list[list[float]], *, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 4:
        raise ValueError(f"{name} must have shape [N,2] with N >= 4")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


def _normalization_transform(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    mean_distance = float(np.linalg.norm(points - centroid, axis=1).mean())
    if mean_distance <= 1e-12:
        raise ValueError("calibration points must not be coincident")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.c_[points, np.ones(points.shape[0], dtype=np.float64)]
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def fit_homography(
    source_points: np.ndarray | list[list[float]],
    destination_points: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Fit a normalized DLT homography mapping source pixels to base-frame XY."""

    source = _points(source_points, name="source_points")
    destination = _points(destination_points, name="destination_points")
    if source.shape != destination.shape:
        raise ValueError("source_points and destination_points must have the same shape")

    source_normalized, source_transform = _normalization_transform(source)
    destination_normalized, destination_transform = _normalization_transform(destination)
    rows: list[list[float]] = []
    for (u, v), (x, y) in zip(source_normalized, destination_normalized, strict=True):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    _, singular_values, right_vectors = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    if singular_values[-1] <= 0.0 or singular_values[-2] <= 1e-14:
        raise ValueError("calibration correspondences are degenerate")
    normalized_homography = right_vectors[-1].reshape(3, 3)
    homography = np.linalg.inv(destination_transform) @ normalized_homography @ source_transform
    if abs(float(homography[2, 2])) <= 1e-12:
        raise ValueError("homography normalization failed")
    homography /= homography[2, 2]
    return homography


def transform_points(homography: np.ndarray, points: np.ndarray | list[list[float]]) -> np.ndarray:
    """Apply a 3x3 homography while preserving an optional single-point shape."""

    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    raw = np.asarray(points, dtype=np.float64)
    single = raw.ndim == 1
    source = raw[None, :] if single else raw
    source = _points(source, name="points") if source.shape[0] >= 4 else _single_or_many(source)
    homogeneous = np.c_[source, np.ones(source.shape[0], dtype=np.float64)]
    mapped = (matrix @ homogeneous.T).T
    denominator = mapped[:, 2]
    if np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("point maps to infinity")
    result = mapped[:, :2] / denominator[:, None]
    return result[0] if single else result


def _single_or_many(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError("points must have shape [N,2]")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    return points


@dataclass(frozen=True, slots=True)
class PlanarCalibration:
    """Versioned planar camera-to-base calibration record."""

    camera_name: str
    image_size: int
    plane_z_m: float
    pixel_to_base_h: np.ndarray
    reference_pixels: np.ndarray
    reference_base_xy_m: np.ndarray
    fit_rmse_mm: float

    def __post_init__(self) -> None:
        if not self.camera_name:
            raise ValueError("camera_name must be non-empty")
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if not np.isfinite(self.plane_z_m):
            raise ValueError("plane_z_m must be finite")
        matrix = np.asarray(self.pixel_to_base_h, dtype=np.float64)
        pixels = _points(self.reference_pixels, name="reference_pixels")
        world = _points(self.reference_base_xy_m, name="reference_base_xy_m")
        if pixels.shape != world.shape:
            raise ValueError("reference arrays must have matching shape")
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("pixel_to_base_h must be a finite 3x3 matrix")
        if self.fit_rmse_mm < 0 or not np.isfinite(self.fit_rmse_mm):
            raise ValueError("fit_rmse_mm must be finite and non-negative")
        object.__setattr__(self, "pixel_to_base_h", matrix.copy())
        object.__setattr__(self, "reference_pixels", pixels.copy())
        object.__setattr__(self, "reference_base_xy_m", world.copy())

    @classmethod
    def fit(
        cls,
        *,
        camera_name: str,
        image_size: int,
        plane_z_m: float,
        reference_pixels: np.ndarray | list[list[float]],
        reference_base_xy_m: np.ndarray | list[list[float]],
    ) -> PlanarCalibration:
        pixels = _points(reference_pixels, name="reference_pixels")
        world = _points(reference_base_xy_m, name="reference_base_xy_m")
        homography = fit_homography(pixels, world)
        projected = transform_points(homography, pixels)
        rmse_mm = float(np.sqrt(np.mean(np.sum((projected - world) ** 2, axis=1))) * 1000.0)
        return cls(
            camera_name=camera_name,
            image_size=image_size,
            plane_z_m=float(plane_z_m),
            pixel_to_base_h=homography,
            reference_pixels=pixels,
            reference_base_xy_m=world,
            fit_rmse_mm=rmse_mm,
        )

    def pixels_to_base_xy(self, pixels: np.ndarray | list[list[float]]) -> np.ndarray:
        return transform_points(self.pixel_to_base_h, pixels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "picksort-planar-calibration/v1",
            "camera_name": self.camera_name,
            "image_size": self.image_size,
            "plane_z_m": self.plane_z_m,
            "pixel_to_base_h": self.pixel_to_base_h.tolist(),
            "reference_pixels": self.reference_pixels.tolist(),
            "reference_base_xy_m": self.reference_base_xy_m.tolist(),
            "fit_rmse_mm": self.fit_rmse_mm,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanarCalibration:
        if payload.get("schema_version") != "picksort-planar-calibration/v1":
            raise ValueError("unsupported planar calibration schema")
        return cls(
            camera_name=str(payload["camera_name"]),
            image_size=int(payload["image_size"]),
            plane_z_m=float(payload["plane_z_m"]),
            pixel_to_base_h=np.asarray(payload["pixel_to_base_h"], dtype=np.float64),
            reference_pixels=np.asarray(payload["reference_pixels"], dtype=np.float64),
            reference_base_xy_m=np.asarray(payload["reference_base_xy_m"], dtype=np.float64),
            fit_rmse_mm=float(payload["fit_rmse_mm"]),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> PlanarCalibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("calibration JSON must contain an object")
        return cls.from_dict(payload)
