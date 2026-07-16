"""A small Soft Actor-Critic trainer for bounded residual actions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

from smartpick_vla.models.residual import (
    GaussianResidualActor,
    ResidualQNetwork,
    ScalarOrVector,
)


@dataclass(frozen=True, slots=True)
class ResidualSACConfig:
    """Hyperparameters for encoded-state residual SAC."""

    observation_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)
    residual_scale: ScalarOrVector = 0.2
    action_low: ScalarOrVector = -1.0
    action_high: ScalarOrVector = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    initial_temperature: float = 0.1
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float | None = None
    automatic_entropy_tuning: bool = True
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if self.observation_dim < 1 or self.action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if (
            min(
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.temperature_learning_rate,
                self.initial_temperature,
                self.gradient_clip_norm,
            )
            <= 0
        ):
            raise ValueError("learning rates, temperature, and gradient clip must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0,1]")


class ResidualSAC(nn.Module):
    """Twin-Q SAC that learns only a constrained correction to a base action.

    Replay batches contain encoded observations and the already-computed base
    action.  The executed ``action`` is the clipped final action, never a raw
    residual.  This keeps VLA inference outside the optimizer and makes the
    frozen-base contract auditable.
    """

    update_count: Tensor

    def __init__(self, config: ResidualSACConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = GaussianResidualActor(
            config.observation_dim,
            config.action_dim,
            hidden_dims=config.hidden_dims,
            residual_scale=config.residual_scale,
            action_low=config.action_low,
            action_high=config.action_high,
        )
        self.critic_one = ResidualQNetwork(
            config.observation_dim, config.action_dim, hidden_dims=config.hidden_dims
        )
        self.critic_two = ResidualQNetwork(
            config.observation_dim, config.action_dim, hidden_dims=config.hidden_dims
        )
        self.target_critic_one = copy.deepcopy(self.critic_one).requires_grad_(False)
        self.target_critic_two = copy.deepcopy(self.critic_two).requires_grad_(False)
        self.log_temperature = nn.Parameter(
            torch.tensor(config.initial_temperature).log(),
            requires_grad=config.automatic_entropy_tuning,
        )
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = Adam(
            [*self.critic_one.parameters(), *self.critic_two.parameters()],
            lr=config.critic_learning_rate,
        )
        self.temperature_optimizer: Adam | None = None
        if config.automatic_entropy_tuning:
            self.temperature_optimizer = Adam(
                [self.log_temperature], lr=config.temperature_learning_rate
            )

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    @property
    def optimizers(self) -> dict[str, Optimizer]:
        optimizers: dict[str, Optimizer] = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
        }
        if self.temperature_optimizer is not None:
            optimizers["temperature"] = self.temperature_optimizer
        return optimizers

    @torch.no_grad()
    def select_action(
        self,
        observation: Tensor,
        base_action: Tensor,
        *,
        deterministic: bool = True,
    ) -> Tensor:
        """Return the bounded final action for evaluation or collection."""

        return self.actor.sample(observation, base_action, deterministic=deterministic).final_action

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Run one residual-SAC update from a replay batch.

        Required keys: ``observation``, ``base_action``, executed ``action``,
        ``reward``, ``next_observation``, ``next_base_action``, and ``done``.
        """

        tensors = self._prepare_batch(batch)
        observation = tensors["observation"]
        base_action = tensors["base_action"]
        action = tensors["action"]
        reward = tensors["reward"]
        next_observation = tensors["next_observation"]
        next_base_action = tensors["next_base_action"]
        done = tensors["done"]

        with torch.no_grad():
            next_sample = self.actor.sample(next_observation, next_base_action)
            next_q = torch.minimum(
                self.target_critic_one(next_observation, next_sample.final_action),
                self.target_critic_two(next_observation, next_sample.final_action),
            )
            entropy_adjusted_q = next_q - self.temperature.detach() * next_sample.log_probability
            target_q = reward + self.config.gamma * (1.0 - done) * entropy_adjusted_q

        predicted_q_one = self.critic_one(observation, action)
        predicted_q_two = self.critic_two(observation, action)
        critic_loss = functional.mse_loss(predicted_q_one, target_q) + functional.mse_loss(
            predicted_q_two, target_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*self.critic_one.parameters(), *self.critic_two.parameters()],
            self.config.gradient_clip_norm,
        )
        self.critic_optimizer.step()

        for critic in (self.critic_one, self.critic_two):
            critic.requires_grad_(False)
        try:
            sample = self.actor.sample(observation, base_action)
            policy_q = torch.minimum(
                self.critic_one(observation, sample.final_action),
                self.critic_two(observation, sample.final_action),
            )
            actor_loss = (self.temperature.detach() * sample.log_probability - policy_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.gradient_clip_norm
            )
            self.actor_optimizer.step()
        finally:
            for critic in (self.critic_one, self.critic_two):
                critic.requires_grad_(True)

        temperature_loss = torch.zeros((), device=observation.device)
        if self.temperature_optimizer is not None:
            target_entropy = (
                -float(self.config.action_dim)
                if self.config.target_entropy is None
                else self.config.target_entropy
            )
            temperature_loss = -(
                self.log_temperature * (sample.log_probability.detach() + target_entropy)
            ).mean()
            self.temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.temperature_optimizer.step()

        self._soft_update_targets()
        self.update_count.add_(1)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "temperature_loss": float(temperature_loss.detach().cpu()),
            "temperature": float(self.temperature.detach().cpu()),
            "mean_absolute_residual": float(sample.residual_action.detach().abs().mean().cpu()),
            "critic_gradient_norm": float(torch.as_tensor(critic_gradient_norm).detach().cpu()),
            "actor_gradient_norm": float(torch.as_tensor(actor_gradient_norm).detach().cpu()),
        }

    def _prepare_batch(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        required = {
            "observation",
            "base_action",
            "action",
            "reward",
            "next_observation",
            "next_base_action",
            "done",
        }
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"batch is missing keys: {sorted(missing)}")
        device = next(self.actor.parameters()).device
        result: dict[str, Tensor] = {}
        for key in required:
            value = batch[key]
            if not isinstance(value, Tensor):
                raise TypeError(f"batch[{key!r}] must be a torch.Tensor")
            result[key] = value.to(device=device, dtype=torch.float32)
            if not torch.isfinite(result[key]).all():
                raise ValueError(f"batch[{key!r}] contains NaN or infinity")
        batch_size = result["observation"].shape[0]
        expected_observation = (batch_size, self.config.observation_dim)
        expected_action = (batch_size, self.config.action_dim)
        for key in ("observation", "next_observation"):
            if result[key].shape != expected_observation:
                raise ValueError(f"{key} must have shape {expected_observation}")
        for key in ("base_action", "action", "next_base_action"):
            if result[key].shape != expected_action:
                raise ValueError(f"{key} must have shape {expected_action}")
        for key in ("reward", "done"):
            if result[key].shape == (batch_size,):
                result[key] = result[key].unsqueeze(-1)
            elif result[key].shape != (batch_size, 1):
                raise ValueError(f"{key} must have shape [B] or [B,1]")
        if ((result["done"] < 0) | (result["done"] > 1)).any():
            raise ValueError("done must be in [0,1]")
        low = self.actor.action_low
        high = self.actor.action_high
        if ((result["action"] < low) | (result["action"] > high)).any():
            raise ValueError("executed action is outside configured action bounds")
        for key in ("base_action", "next_base_action"):
            if ((result[key] < low) | (result[key] > high)).any():
                raise ValueError(f"{key} is outside configured action bounds")
        tolerance = 16.0 * torch.finfo(result["action"].dtype).eps
        if (
            (result["action"] - result["base_action"]).abs() > self.actor.residual_scale + tolerance
        ).any():
            raise ValueError("executed action exceeds configured residual bounds")
        return result

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        for source, target in (
            (self.critic_one, self.target_critic_one),
            (self.critic_two, self.target_critic_two),
        ):
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.lerp_(source_parameter, self.config.tau)
