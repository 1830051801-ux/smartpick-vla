"""Qualitative episode media with mandatory provenance sidecars."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from smartpick_vla.evaluation.metrics import EVALUATION_SCHEMA_VERSION, EpisodeResult


def _uint8_frame(frame: np.ndarray | Image.Image, index: int) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError(f"frame {index} must have shape [H,W], [H,W,3], or [H,W,4]")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"frame {index} has an empty image dimension")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
            raise ValueError(f"floating frame {index} must contain finite values in [0,1]")
        return np.ascontiguousarray(np.rint(array * 255.0).astype(np.uint8))
    raise TypeError(f"frame {index} must use uint8 or floating-point pixels")


def save_episode_gif(
    frames: Iterable[np.ndarray | Image.Image],
    output_path: str | Path,
    result: EpisodeResult,
    *,
    fps: float = 10.0,
    loop: int = 0,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a GIF and a ``.gif.json`` sidecar tied to one episode result."""

    if not isinstance(result, EpisodeResult):
        raise TypeError("result must be an EpisodeResult")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if isinstance(loop, bool) or not isinstance(loop, int) or loop < 0:
        raise ValueError("loop must be a non-negative integer")
    arrays = tuple(_uint8_frame(frame, index) for index, frame in enumerate(frames))
    if not arrays:
        raise ValueError("at least one GIF frame is required")
    expected_shape = arrays[0].shape
    if any(array.shape != expected_shape for array in arrays[1:]):
        raise ValueError("all GIF frames must have the same shape")

    destination = Path(output_path)
    if destination.suffix.lower() != ".gif":
        raise ValueError("GIF output path must end in .gif")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    temporary_gif = destination.with_name(f".{destination.stem}.tmp.gif")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp")
    metadata: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "media_type": "qualitative_episode_gif",
        "gif_file": destination.name,
        "episode_id": result.episode_id,
        "suite": result.suite,
        "method": result.method,
        "seed": result.seed,
        "task_class": result.task_class,
        "success": result.success,
        "frame_count": len(arrays),
        "fps": float(fps),
    }
    if extra_metadata is not None:
        if not isinstance(extra_metadata, Mapping):
            raise TypeError("extra_metadata must be a mapping")
        reserved = set(metadata).intersection(extra_metadata)
        if reserved:
            raise ValueError(f"extra_metadata cannot replace provenance fields: {sorted(reserved)}")
        metadata.update(extra_metadata)
    try:
        metadata_text = json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("GIF metadata must be finite and JSON serializable") from error

    try:
        pil_frames = [Image.fromarray(array) for array in arrays]
        pil_frames[0].save(
            temporary_gif,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(1, round(1000.0 / fps)),
            loop=loop,
            optimize=False,
        )
        with temporary_sidecar.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(metadata_text)
            stream.write("\n")
        temporary_gif.replace(destination)
        temporary_sidecar.replace(sidecar)
    finally:
        temporary_gif.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return destination, sidecar
