"""One-step supervised updates for BC and action-chunk policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class SupervisedStepResult:
    loss: float
    gradient_norm: float
    batch_size: int


def action_imitation_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    loss_kind: Literal["mse", "smooth_l1"] = "mse",
) -> Tensor:
    """Compute a shape-safe action loss with an optional horizon mask."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {prediction.shape} and {target.shape}"
        )
    if prediction.ndim not in (2, 3):
        raise ValueError("actions must have shape [B,A] or [B,H,A]")
    if not torch.isfinite(target).all():
        raise ValueError("target action contains NaN or infinity")
    if loss_kind == "mse":
        element_loss = functional.mse_loss(prediction, target, reduction="none")
    elif loss_kind == "smooth_l1":
        element_loss = functional.smooth_l1_loss(prediction, target, reduction="none")
    else:
        raise ValueError(f"unsupported loss_kind: {loss_kind}")
    per_action = element_loss.mean(dim=-1)
    if mask is None:
        return per_action.mean()
    expected_shape = prediction.shape[:-1]
    if mask.shape != expected_shape:
        raise ValueError(f"mask must have shape {expected_shape}, got {mask.shape}")
    weights = mask.to(device=prediction.device, dtype=prediction.dtype)
    if (weights < 0).any():
        raise ValueError("mask values must be non-negative")
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("mask must select at least one action")
    return (per_action * weights).sum() / denominator


def supervised_train_step(
    model: nn.Module,
    optimizer: Optimizer,
    batch: Mapping[str, Any],
    *,
    gradient_clip_norm: float = 1.0,
    loss_kind: Literal["mse", "smooth_l1"] = "mse",
) -> SupervisedStepResult:
    """Run exactly one optimizer step on a BC or Compact-VLA batch.

    Required keys are ``rgb``, ``instruction``, ``robot_state``, and
    ``action``.  ``action_mask`` is optional for padded action chunks.
    """

    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    required = {"rgb", "instruction", "robot_state", "action"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"batch is missing keys: {sorted(missing)}")
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("model has no parameters")
    device = first_parameter.device
    rgb = _tensor_from_batch(batch["rgb"], "rgb").to(device)
    robot_state = _tensor_from_batch(batch["robot_state"], "robot_state").to(
        device=device, dtype=first_parameter.dtype
    )
    target = _tensor_from_batch(batch["action"], "action").to(
        device=device, dtype=first_parameter.dtype
    )
    instruction = batch["instruction"]
    if isinstance(instruction, str) or not isinstance(instruction, (Tensor, Sequence)):
        raise TypeError("instruction must be a token tensor or a sequence of strings")
    if isinstance(instruction, Tensor):
        instruction = instruction.to(device)
    mask_value = batch.get("action_mask")
    mask = None if mask_value is None else _tensor_from_batch(mask_value, "action_mask").to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(rgb=rgb, instruction=instruction, robot_state=robot_state)
    if not isinstance(prediction, Tensor):
        raise TypeError("model must return an action tensor")
    loss = action_imitation_loss(prediction, target, mask=mask, loss_kind=loss_kind)
    if not torch.isfinite(loss):
        raise FloatingPointError("supervised loss is not finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return SupervisedStepResult(
        loss=float(loss.detach().cpu()),
        gradient_norm=float(torch.as_tensor(gradient_norm).detach().cpu()),
        batch_size=int(target.shape[0]),
    )


def _tensor_from_batch(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
    return value
