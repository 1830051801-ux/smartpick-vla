"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without silently changing later seeds."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def make_rng(seed: int | None) -> np.random.Generator:
    """Return an explicit NumPy generator used for task sampling."""

    return np.random.default_rng(seed)
