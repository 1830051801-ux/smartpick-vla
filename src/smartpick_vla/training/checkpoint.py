"""Versioned training checkpoints with optional optimizer and RNG state."""

from __future__ import annotations

import pickle
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim import Optimizer

CHECKPOINT_VERSION = 2
OptimizerCollection = Optimizer | Mapping[str, Optimizer]


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    step: int
    config: dict[str, Any]
    extra: dict[str, Any]


def _optimizer_state(optimizers: OptimizerCollection | None) -> dict[str, Any] | None:
    if optimizers is None:
        return None
    if isinstance(optimizers, Optimizer):
        return {"default": optimizers.state_dict()}
    return {name: optimizer.state_dict() for name, optimizer in optimizers.items()}


def _config_dict(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        return dict(asdict(config))
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dataclass, mapping, or None")


def _rng_state() -> dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    key_array = cast(npt.NDArray[np.uint32], keys)
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": bit_generator,
            "state": torch.from_numpy(key_array.copy()),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizers: OptimizerCollection | None = None,
    step: int = 0,
    config: Mapping[str, Any] | Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a trusted local training checkpoint."""

    if step < 0:
        raise ValueError("step must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_states": _optimizer_state(optimizers),
        "step": step,
        "config": _config_dict(config),
        "extra": dict(extra or {}),
        "rng_state": _rng_state(),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Safely load a versioned tensor-and-primitive checkpoint payload."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            "checkpoint cannot be loaded safely; PickSort-VLA requires a v2 "
            "weights-only checkpoint from a trusted release"
        ) from error
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported or malformed checkpoint")
    return dict(payload)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizers: OptimizerCollection | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = False,
) -> CheckpointInfo:
    """Safely load a v2 checkpoint created by this project."""

    source = Path(path)
    payload = load_checkpoint_payload(source, map_location=map_location)
    model.load_state_dict(payload["model_state"], strict=strict)
    stored_optimizers = payload.get("optimizer_states")
    if optimizers is not None:
        if not isinstance(stored_optimizers, dict):
            raise ValueError("checkpoint does not contain optimizer states")
        if isinstance(optimizers, Optimizer):
            optimizers.load_state_dict(stored_optimizers["default"])
        else:
            missing = set(optimizers).difference(stored_optimizers)
            if missing:
                raise KeyError(f"checkpoint is missing optimizers: {sorted(missing)}")
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(stored_optimizers[name])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return CheckpointInfo(
        path=source,
        step=int(payload["step"]),
        config=dict(payload.get("config", {})),
        extra=dict(payload.get("extra", {})),
    )


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or not isinstance(
        numpy_state.get("state"), torch.Tensor
    ):
        raise ValueError("checkpoint contains an invalid NumPy RNG state")
    keys = numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False)
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            keys,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
