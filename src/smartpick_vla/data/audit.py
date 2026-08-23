"""Quality gates for synthetic embodied perception datasets.

The audit is deliberately independent from model training.  It validates the
archive contract, reports modality and label coverage, and produces an
episode-level split recommendation so frames from one rollout are not silently
scattered across train and validation sets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import runtime_snapshot, sha256_file

AUDIT_SCHEMA_VERSION = "picksort-perception-audit/v1"

_REQUIRED_ARRAYS = {
    "rgb",
    "bbox_xyxy",
    "episode_id",
    "camera_names",
    "quality_classes",
}


def _fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _finite_fraction(array: np.ndarray) -> float:
    return float(np.isfinite(array).sum() / max(1, array.size))


def _load_adjacent_manifest(dataset_path: Path) -> dict[str, Any] | None:
    manifest_path = dataset_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"path": str(manifest_path), "read_error": True}
    if not isinstance(payload, dict):
        return {"path": str(manifest_path), "invalid": True}
    return {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "payload": payload}


def _validate_base_shapes(archive: Any) -> dict[str, Any]:
    files = set(str(name) for name in archive.files)
    missing = sorted(_REQUIRED_ARRAYS.difference(files))
    if missing:
        raise ValueError(f"perception archive is missing required arrays: {missing}")

    rgb = np.asarray(archive["rgb"])
    if rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape [N,V,H,W,3]")
    samples, views, height, width, _ = rgb.shape
    if samples < 1 or views < 1 or height < 16 or width < 16:
        raise ValueError("rgb archive dimensions are too small")

    camera_names = tuple(str(value) for value in np.asarray(archive["camera_names"]).tolist())
    classes = tuple(str(value) for value in np.asarray(archive["quality_classes"]).tolist())
    if len(camera_names) != views or len(set(camera_names)) != views:
        raise ValueError("camera_names does not match the RGB view dimension")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("quality_classes must be non-empty and unique")

    bboxes = np.asarray(archive["bbox_xyxy"])
    expected_bbox_shape = (samples, views, len(classes), 4)
    if bboxes.shape != expected_bbox_shape:
        raise ValueError(f"bbox_xyxy must have shape {expected_bbox_shape}")

    episode_ids = np.asarray(archive["episode_id"])
    if episode_ids.shape != (samples,):
        raise ValueError("episode_id must have shape [N]")

    optional_shapes: dict[str, list[int] | None] = {}
    if "depth_m" in files:
        depth = np.asarray(archive["depth_m"])
        expected_depth_shape = (samples, views, height, width)
        if depth.shape != expected_depth_shape:
            raise ValueError(f"depth_m must have shape {expected_depth_shape}")
        optional_shapes["depth_m"] = list(depth.shape)
    for name, expected in (
        ("instance_mask", (samples, views, height, width)),
        ("visible_pixels", (samples, views, len(classes))),
        ("keypoint_xy", (samples, views, len(classes), 2)),
    ):
        if name in files:
            value = np.asarray(archive[name])
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            optional_shapes[name] = list(value.shape)
    for name in ("robot_state", "action", "instruction", "step_index", "target_object_index"):
        if name in files:
            value = np.asarray(archive[name])
            if value.shape[0] != samples:
                raise ValueError(f"{name} first dimension must equal sample count")
            optional_shapes[name] = list(value.shape)

    return {
        "samples": samples,
        "views": views,
        "image_size": [height, width],
        "camera_names": list(camera_names),
        "quality_classes": list(classes),
        "optional_shapes": optional_shapes,
        "array_names": sorted(files),
    }


def _episode_split(episode_ids: np.ndarray, *, seed: int = 20260822) -> dict[str, Any]:
    unique = np.unique(episode_ids)
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique)
    validation_count = max(1, round(unique.size * 0.2)) if unique.size > 1 else 0
    validation = shuffled[:validation_count]
    training = shuffled[validation_count:]
    return {
        "seed": seed,
        "strategy": "episode-disjoint shuffled 80/20 recommendation",
        "training_episode_count": int(training.size),
        "validation_episode_count": int(validation.size),
        "training_episode_ids": [int(value) for value in training.tolist()],
        "validation_episode_ids": [int(value) for value in validation.tolist()],
        "leakage_possible": bool(
            set(int(value) for value in training.tolist())
            & set(int(value) for value in validation.tolist())
        ),
    }


def audit_perception_dataset(
    dataset_path: str | Path,
    *,
    output_path: str | Path | None = None,
    require_depth: bool = False,
    max_duplicate_fraction: float = 0.20,
    split_seed: int = 20260822,
) -> dict[str, Any]:
    """Audit a generated RGB/RGB-D perception archive and write a JSON report.

    The function raises only for an unreadable or structurally malformed
    archive.  Content problems are returned as machine-readable errors and
    warnings so CI, a dataset curator, or a training pipeline can decide how
    strict the gate should be.
    """

    source = Path(dataset_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not 0.0 <= max_duplicate_fraction <= 1.0:
        raise ValueError("max_duplicate_fraction must be in [0,1]")
    with np.load(source, allow_pickle=False) as archive:
        shape_info = _validate_base_shapes(archive)
        rgb = np.asarray(archive["rgb"])
        bboxes = np.asarray(archive["bbox_xyxy"], dtype=np.float64)
        episode_ids = np.asarray(archive["episode_id"])
        classes = shape_info["quality_classes"]
        samples = int(shape_info["samples"])
        height, width = (int(value) for value in shape_info["image_size"])
        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        if rgb.dtype != np.uint8:
            warnings.append(
                {"code": "rgb_dtype", "message": f"RGB dtype is {rgb.dtype}, expected uint8"}
            )
        rgb_finite = _finite_fraction(rgb)
        if rgb_finite < 1.0:
            errors.append({"code": "rgb_nonfinite", "message": "RGB contains non-finite values"})

        bbox_valid = (
            np.isfinite(bboxes).all(axis=-1)
            & (bboxes[..., 2] > bboxes[..., 0])
            & (bboxes[..., 3] > bboxes[..., 1])
        )
        bbox_in_bounds = (
            bbox_valid
            & (bboxes[..., 0] >= 0.0)
            & (bboxes[..., 1] >= 0.0)
            # MuJoCo's raster bbox uses an exclusive right/bottom edge, so
            # x1==width and y1==height are valid clipped boxes.
            & (bboxes[..., 2] <= width)
            & (bboxes[..., 3] <= height)
        )
        bbox_valid_count = int(bbox_valid.sum())
        bbox_out_of_bounds_count = int((bbox_valid & ~bbox_in_bounds).sum())
        if bbox_valid_count == 0:
            errors.append(
                {"code": "no_bbox_labels", "message": "no valid bounding-box labels were found"}
            )
        elif bbox_out_of_bounds_count:
            errors.append(
                {
                    "code": "bbox_out_of_bounds",
                    "message": f"{bbox_out_of_bounds_count} valid boxes lie outside image bounds",
                }
            )

        depth_stats: dict[str, Any] = {"present": "depth_m" in archive.files}
        if "depth_m" in archive.files:
            depth = np.asarray(archive["depth_m"], dtype=np.float64)
            depth_finite = np.isfinite(depth)
            depth_positive = depth_finite & (depth > 0.0)
            depth_stats.update(
                {
                    "shape": list(depth.shape),
                    "finite_fraction": _finite_fraction(depth),
                    "positive_fraction": _fraction(int(depth_positive.sum()), depth.size),
                    "min_positive_m": (
                        float(np.min(depth[depth_positive])) if bool(depth_positive.any()) else None
                    ),
                    "max_positive_m": (
                        float(np.max(depth[depth_positive])) if bool(depth_positive.any()) else None
                    ),
                }
            )
            if not bool(depth_positive.any()):
                errors.append(
                    {"code": "no_depth", "message": "depth_m has no positive finite samples"}
                )
        elif require_depth:
            errors.append(
                {"code": "depth_required", "message": "RGB-D audit requested but depth_m is absent"}
            )
        else:
            warnings.append(
                {
                    "code": "depth_absent",
                    "message": "archive is RGB-only; RGB-D training is unavailable",
                }
            )

        optional_stats: dict[str, Any] = {}
        if "instance_mask" in archive.files:
            mask = np.asarray(archive["instance_mask"])
            optional_stats["mask_nonzero_fraction"] = _fraction(
                int(np.count_nonzero(mask)), mask.size
            )
        if "keypoint_xy" in archive.files:
            keypoints = np.asarray(archive["keypoint_xy"], dtype=np.float64)
            keypoint_valid = np.isfinite(keypoints).all(axis=-1)
            keypoint_in_bounds = (
                keypoint_valid
                & (keypoints[..., 0] >= 0.0)
                & (keypoints[..., 0] <= width - 1)
                & (keypoints[..., 1] >= 0.0)
                & (keypoints[..., 1] <= height - 1)
            )
            optional_stats.update(
                {
                    "keypoint_valid_count": int(keypoint_valid.sum()),
                    "keypoint_in_bounds_fraction": _fraction(
                        int(keypoint_in_bounds.sum()), int(keypoint_valid.sum())
                    ),
                }
            )
            visible_keypoint_out = keypoint_valid & bbox_valid & ~keypoint_in_bounds
            if bool(visible_keypoint_out.any()):
                errors.append(
                    {
                        "code": "keypoint_out_of_bounds",
                        "message": "visible finite keypoint labels exceed image bounds",
                    }
                )
        for name in ("robot_state", "action"):
            if name in archive.files:
                value = np.asarray(archive[name])
                finite = _finite_fraction(value)
                optional_stats[f"{name}_finite_fraction"] = finite
                if finite < 1.0:
                    errors.append(
                        {
                            "code": f"{name}_nonfinite",
                            "message": f"{name} contains non-finite values",
                        }
                    )
        if "action" in archive.files:
            action = np.asarray(archive["action"], dtype=np.float64)
            optional_stats["action_saturation_fraction"] = _fraction(
                int((np.abs(action) >= 0.999).sum()), action.size
            )

        episode_values, episode_counts = np.unique(episode_ids, return_counts=True)
        if episode_values.size < 2:
            warnings.append(
                {
                    "code": "too_few_episodes",
                    "message": "at least two episodes are needed for an episode-safe split",
                }
            )
        split = _episode_split(episode_ids, seed=split_seed)
        duplicate_hashes: set[str] = set()
        for sample_index in range(samples):
            digest = hashlib.sha256(np.ascontiguousarray(rgb[sample_index]).tobytes()).hexdigest()
            duplicate_hashes.add(digest)
        duplicate_fraction = _fraction(samples - len(duplicate_hashes), samples)
        if duplicate_fraction > max_duplicate_fraction:
            warnings.append(
                {
                    "code": "duplicate_rgb_samples",
                    "message": f"duplicate RGB sample fraction {duplicate_fraction:.3f} exceeds {max_duplicate_fraction:.3f}",
                }
            )

        class_visible_counts = {
            str(classes[index]): int(bbox_valid[..., index].sum()) for index in range(len(classes))
        }
        quality_score = float(
            np.mean(
                [
                    rgb_finite,
                    _fraction(int(bbox_in_bounds.sum()), int(bbox_valid.size)),
                    optional_stats.get("keypoint_in_bounds_fraction", 1.0),
                    depth_stats.get("positive_fraction", 1.0) if require_depth else 1.0,
                ]
            )
        )
        report: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "dataset": {"path": str(source.resolve()), "sha256": sha256_file(source)},
            "archive": shape_info,
            "modalities": {
                "rgb_finite_fraction": rgb_finite,
                "depth": depth_stats,
                **optional_stats,
            },
            "labels": {
                "valid_bbox_count": bbox_valid_count,
                "bbox_in_bounds_count": int(bbox_in_bounds.sum()),
                "bbox_in_bounds_fraction": _fraction(int(bbox_in_bounds.sum()), bbox_valid_count),
                "class_visible_counts": class_visible_counts,
            },
            "episodes": {
                "count": int(episode_values.size),
                "min_samples": int(episode_counts.min()) if episode_counts.size else 0,
                "max_samples": int(episode_counts.max()) if episode_counts.size else 0,
                "split_recommendation": split,
            },
            "deduplication": {
                "unique_rgb_sample_count": len(duplicate_hashes),
                "duplicate_fraction": duplicate_fraction,
            },
            "quality_gate": {
                "passed": not errors,
                "quality_score_0_to_1": max(0.0, min(1.0, quality_score)),
                "errors": errors,
                "warnings": warnings,
            },
            "source_manifest": _load_adjacent_manifest(source),
            "runtime": runtime_snapshot(),
        }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


__all__ = ["AUDIT_SCHEMA_VERSION", "audit_perception_dataset"]
