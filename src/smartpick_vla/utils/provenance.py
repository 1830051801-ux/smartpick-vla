"""Hashes and runtime facts used to trace generated artifacts."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
import torch


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        snapshot["gpu"] = torch.cuda.get_device_name(0)
    return snapshot
