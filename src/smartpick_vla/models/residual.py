"""Bounded residual-action networks and a frozen base-policy adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.distributions import Normal

ScalarOrVector = float | Sequence[float] | Tensor


def _as_action_vector(value: ScalarOrVector, action_dim: int, *, name: str) -> Tensor:
    vector = torch.as_tensor(value, dtype=torch.float32)
    if vector.ndim == 0:
        vector = vector.repeat(action_dim)
    if vector.shape != (action_dim,):
        raise ValueError(f"{name} must be scalar or have shape ({action_dim},)")
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def compose_bounded_action(
    base_action: Tensor,
    residual_logits: Tensor,
    *,
    residual_scale: ScalarOrVector,
    action_low: ScalarOrVector,
    action_high: ScalarOrVector,
) -> tuple[Tensor, Tensor]:
    """Compute `clip(base + scale * tanh(residual), low, high)` exactly."""

    if base_action.shape != residual_logits.shape or base_action.ndim < 1:
        raise ValueError("base_action and residual_logits must have the same non-scalar shape")
    if not base_action.is_floating_point() or not residual_logits.is_floating_point():
        raise TypeError("base_action and residual_logits must be floating-point tensors")
    if not torch.isfinite(base_action).all() or not torch.isfinite(residual_logits).all():
        raise ValueError("base_action and residual_logits must contain only finite values")
    action_dim = base_action.shape[-1]
    scale = _as_action_vector(residual_scale, action_dim, name="residual_scale").to(
        base_action.device, base_action.dtype
    )
    low = _as_action_vector(action_low, action_dim, name="action_low").to(
        base_action.device, base_action.dtype
    )
    high = _as_action_vector(action_high, action_dim, name="action_high").to(
        base_action.device, base_action.dtype
    )
    if (scale < 0).any():
        raise ValueError("residual_scale must be non-negative")
    if (high <= low).any():
        raise ValueError("action_high must be greater than action_low")
    residual_action = scale * torch.tanh(residual_logits)
    final_action = torch.minimum(torch.maximum(base_action + residual_action, low), high)
    return final_action, residual_action


class FrozenBasePolicy(nn.Module):
    """Expose a base action while permanently freezing the wrapped policy.

    Chunk policies are reduced to one action with ``action_index``.  Keeping
    this adapter separate from SAC makes it explicit that replay stores base
    actions and that RL never optimizes the VLA/BC policy.
    """

    def __init__(self, base_policy: nn.Module, *, action_index: int = 0) -> None:
        super().__init__()
        if action_index < 0:
            raise ValueError("action_index must be non-negative")
        self.base_policy = base_policy
        self.action_index = action_index
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()
        super().train(False)

    def train(self, mode: bool = True) -> Self:
        del mode
        super().train(False)
        self.base_policy.eval()
        return self

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        output = self.base_policy(*args, **kwargs)
        if not isinstance(output, Tensor):
            raise TypeError("base policy must return a tensor")
        if output.ndim == 3:
            if self.action_index >= output.shape[1]:
                raise IndexError("action_index exceeds the base action chunk")
            output = output[:, self.action_index]
        if output.ndim != 2:
            raise ValueError("base policy output must have shape [B,A] or [B,H,A]")
        return output.detach()


@dataclass(frozen=True, slots=True)
class ResidualActionSample:
    """One SAC actor sample and its bounded composition with the base action."""

    final_action: Tensor
    residual_action: Tensor
    normalized_residual: Tensor
    log_probability: Tensor


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    if not hidden_dims or any(width < 1 for width in hidden_dims):
        raise ValueError("hidden_dims must contain positive widths")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.SiLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class GaussianResidualActor(nn.Module):
    """Gaussian tanh policy whose residual is added to a supplied base action."""

    residual_scale: Tensor
    action_low: Tensor
    action_high: Tensor

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        residual_scale: ScalarOrVector = 0.2,
        action_low: ScalarOrVector = -1.0,
        action_high: ScalarOrVector = 1.0,
        log_std_bounds: tuple[float, float] = (-5.0, 1.0),
    ) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if log_std_bounds[0] >= log_std_bounds[1]:
            raise ValueError("invalid log_std_bounds")
        scale = _as_action_vector(residual_scale, action_dim, name="residual_scale")
        low = _as_action_vector(action_low, action_dim, name="action_low")
        high = _as_action_vector(action_high, action_dim, name="action_high")
        if (scale < 0).any():
            raise ValueError("residual_scale must be non-negative")
        if (high <= low).any():
            raise ValueError("action_high must be greater than action_low")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.log_std_bounds = log_std_bounds
        self.backbone = _build_mlp(observation_dim + action_dim, hidden_dims, 2 * action_dim)
        output_layer = self.backbone[-1]
        if isinstance(output_layer, nn.Linear):
            # Start from a zero-mean correction; RL must earn any deviation
            # from the frozen imitation policy instead of perturbing it at step 0.
            nn.init.normal_(output_layer.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(output_layer.bias)
        self.register_buffer("residual_scale", scale)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def distribution_parameters(
        self, observation: Tensor, base_action: Tensor
    ) -> tuple[Tensor, Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(
                f"observation must have shape [B,{self.observation_dim}], got {tuple(observation.shape)}"
            )
        if base_action.ndim != 2 or base_action.shape != (observation.shape[0], self.action_dim):
            raise ValueError(
                f"base_action must have shape [B,{self.action_dim}], got {tuple(base_action.shape)}"
            )
        if not torch.isfinite(observation).all() or not torch.isfinite(base_action).all():
            raise ValueError("observation and base_action must contain only finite values")
        if ((base_action < self.action_low) | (base_action > self.action_high)).any():
            raise ValueError("base_action is outside configured action bounds")
        mean, raw_log_std = self.backbone(torch.cat((observation, base_action), dim=-1)).chunk(
            2, dim=-1
        )
        minimum, maximum = self.log_std_bounds
        bounded = torch.tanh(raw_log_std)
        log_std = minimum + 0.5 * (maximum - minimum) * (bounded + 1.0)
        return mean, log_std

    def sample(
        self,
        observation: Tensor,
        base_action: Tensor,
        *,
        deterministic: bool = False,
    ) -> ResidualActionSample:
        mean, log_std = self.distribution_parameters(observation, base_action)
        distribution = Normal(mean, log_std.exp())
        residual_logits = mean if deterministic else distribution.rsample()
        normalized_residual = torch.tanh(residual_logits)
        # Entropy is defined in normalized residual space.  The subsequent
        # base addition and safety clip are deterministic constraints.
        log_probability = distribution.log_prob(residual_logits)
        correction = 2.0 * (
            torch.log(torch.tensor(2.0, device=residual_logits.device, dtype=residual_logits.dtype))
            - residual_logits
            - functional.softplus(-2.0 * residual_logits)
        )
        log_probability = (log_probability - correction).sum(dim=-1, keepdim=True)
        residual_action = self.residual_scale * normalized_residual
        final_action = torch.minimum(
            torch.maximum(base_action + residual_action, self.action_low), self.action_high
        )
        return ResidualActionSample(
            final_action=final_action,
            residual_action=residual_action,
            normalized_residual=normalized_residual,
            log_probability=log_probability,
        )


class ResidualQNetwork(nn.Module):
    """Critic over encoded observation and the executed final action."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.network = _build_mlp(observation_dim + action_dim, hidden_dims, 1)

    def forward(self, observation: Tensor, final_action: Tensor) -> Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError("invalid critic observation shape")
        if final_action.shape != (observation.shape[0], self.action_dim):
            raise ValueError("invalid critic action shape")
        return self.network(torch.cat((observation, final_action), dim=-1))
