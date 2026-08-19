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


def policy_forward_from_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Run either a single-frame or history-aware policy from one training batch."""

    required = {"rgb", "instruction", "robot_state", "action"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"batch is missing keys: {sorted(missing)}")
    instruction = batch["instruction"]
    if isinstance(instruction, str) or not isinstance(instruction, (Tensor, Sequence)):
        raise TypeError("instruction must be a token tensor or a sequence of strings")
    if isinstance(instruction, Tensor):
        instruction = instruction.to(device)

    target = _tensor_from_batch(batch["action"], "action").to(device=device, dtype=dtype)
    mask_value = batch.get("action_mask")
    action_mask = (
        None
        if mask_value is None
        else _tensor_from_batch(mask_value, "action_mask").to(device=device)
    )

    has_rgb_history = "rgb_history" in batch
    has_state_history = "robot_state_history" in batch
    if has_rgb_history != has_state_history:
        raise KeyError("temporal batches require both rgb_history and robot_state_history")
    if not has_rgb_history:
        rgb = _tensor_from_batch(batch["rgb"], "rgb").to(device)
        robot_state = _tensor_from_batch(batch["robot_state"], "robot_state").to(
            device=device, dtype=dtype
        )
        prediction = model(rgb=rgb, instruction=instruction, robot_state=robot_state)
    else:
        if "history_mask" not in batch:
            raise KeyError("temporal batches require history_mask")
        rgb_history = _tensor_from_batch(batch["rgb_history"], "rgb_history").to(device)
        robot_state_history = _tensor_from_batch(
            batch["robot_state_history"], "robot_state_history"
        ).to(device=device, dtype=dtype)
        history_mask = _tensor_from_batch(batch["history_mask"], "history_mask").to(device)
        if history_mask.dtype is not torch.bool:
            raise TypeError("history_mask must use bool dtype")
        prediction = model(
            rgb_history=rgb_history,
            instruction=instruction,
            robot_state_history=robot_state_history,
            history_mask=history_mask,
        )
    if not isinstance(prediction, Tensor):
        raise TypeError("model must return an action tensor")
    return prediction, target, action_mask


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
    ``action``. Temporal policies additionally receive ``rgb_history``,
    ``robot_state_history``, and ``history_mask``. ``action_mask`` is optional
    for padded action chunks.
    """

    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("model has no parameters")
    device = first_parameter.device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, target, mask = policy_forward_from_batch(
        model,
        batch,
        device=device,
        dtype=first_parameter.dtype,
    )
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
