"""Compact latent world model for six-axis manipulation rollouts.

This is a predictive model for planning and data diagnostics, not a claim of
physical-world understanding.  It predicts the next proprioceptive state and
task outcomes from an observation/action window, allowing imagined rollouts to
be compared with the MuJoCo ground truth before a policy is promoted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from smartpick_vla.models.encoders import ByteTextEncoder, CompactVisionEncoder, count_parameters


@dataclass(frozen=True, slots=True)
class WorldModelConfig:
    """Architecture and contract settings for the predictive transformer."""

    robot_state_dim: int = 29
    action_dim: int = 6
    observation_horizon: int = 4
    d_model: int = 128
    nhead: int = 4
    temporal_layers: int = 3
    language_layers: int = 1
    feedforward_dim: int = 256
    language_max_length: int = 64
    vision_grid_size: int = 4
    dropout: float = 0.0
    min_state_std: float = 0.02

    def __post_init__(self) -> None:
        values = (
            self.robot_state_dim,
            self.action_dim,
            self.observation_horizon,
            self.d_model,
            self.nhead,
            self.temporal_layers,
            self.language_layers,
            self.feedforward_dim,
            self.language_max_length,
            self.vision_grid_size,
        )
        if any(value < 1 for value in values):
            raise ValueError("world-model dimensions must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.min_state_std <= 0.0:
            raise ValueError("min_state_std must be positive")


class WorldModelPrediction:
    """Typed output bundle for one-step and imagined-rollout prediction."""

    def __init__(
        self,
        next_robot_state: Tensor,
        reward: Tensor,
        terminated_logit: Tensor,
        truncated_logit: Tensor,
        collision_logit: Tensor,
        wrong_pick_logit: Tensor,
        wrong_bin_logit: Tensor,
        next_robot_state_std: Tensor,
    ) -> None:
        self.next_robot_state = next_robot_state
        self.reward = reward
        self.terminated_logit = terminated_logit
        self.truncated_logit = truncated_logit
        self.collision_logit = collision_logit
        self.wrong_pick_logit = wrong_pick_logit
        self.next_robot_state_std = next_robot_state_std
        self.wrong_bin_logit = wrong_bin_logit

    def outcome_probabilities(self) -> dict[str, Tensor]:
        return {
            "terminated": torch.sigmoid(self.terminated_logit),
            "truncated": torch.sigmoid(self.truncated_logit),
            "collision": torch.sigmoid(self.collision_logit),
            "wrong_pick": torch.sigmoid(self.wrong_pick_logit),
            "wrong_bin": torch.sigmoid(self.wrong_bin_logit),
        }


class WorldModelTransformer(nn.Module):
    """Predictive transformer over visual/proprioceptive/action history.

    The image encoder is intentionally compact and locally trainable.  Each
    timestep is represented by one pooled visual token, one state token, and
    one action token; a temporal transformer models the transition context and
    emits both continuous dynamics and calibrated event heads.
    """

    requires_observation_history = True

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        d_model = self.config.d_model
        self.vision_encoder = CompactVisionEncoder(d_model, grid_size=self.config.vision_grid_size)
        self.language_encoder = ByteTextEncoder(
            d_model,
            max_length=self.config.language_max_length,
            num_layers=self.config.language_layers,
            nhead=self.config.nhead,
            feedforward_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.config.robot_state_dim),
            nn.Linear(self.config.robot_state_dim, d_model),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.config.action_dim),
            nn.Linear(self.config.action_dim, d_model),
            nn.SiLU(),
        )
        self.planned_action_encoder = nn.Sequential(
            nn.LayerNorm(self.config.action_dim),
            nn.Linear(self.config.action_dim, d_model),
            nn.SiLU(),
        )
        self.time_embedding = nn.Embedding(self.config.observation_horizon, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.config.temporal_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        # Predict a residual from the last observed state.  This gives the
        # dynamics head a stable identity path and makes short imagined
        # rollouts less sensitive to the absolute state scale.
        self.state_delta_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * self.config.robot_state_dim),
        )
        self.reward_head = nn.Linear(d_model, 1)
        self.event_head = nn.Linear(d_model, 5)

    def _check_inputs(
        self,
        rgb_history: Tensor,
        robot_state_history: Tensor,
        action_history: Tensor,
        history_mask: Tensor | None,
    ) -> Tensor:
        if rgb_history.ndim != 5:
            raise ValueError("rgb_history must have shape [B,T,3,H,W]")
        if robot_state_history.ndim != 3 or action_history.ndim != 3:
            raise ValueError("state/action history must have shape [B,T,D]")
        batch, steps, channels, _, _ = rgb_history.shape
        if channels != 3 or steps != self.config.observation_horizon:
            raise ValueError("RGB history shape does not match the world-model contract")
        if robot_state_history.shape != (batch, steps, self.config.robot_state_dim):
            raise ValueError("robot_state_history shape does not match configuration")
        if action_history.shape != (batch, steps, self.config.action_dim):
            raise ValueError("action_history shape does not match configuration")
        if history_mask is None:
            history_mask = torch.ones((batch, steps), dtype=torch.bool, device=rgb_history.device)
        if history_mask.shape != (batch, steps) or history_mask.dtype is not torch.bool:
            raise ValueError("history_mask must be bool with shape [B,T]")
        if not bool(history_mask.any(dim=1).all()):
            raise ValueError("each sample needs at least one valid history frame")
        # The current dataset uses left-padding.  Reject holes so a malformed
        # ROS history cannot silently select the wrong "last" observation.
        expected = torch.arange(steps, device=history_mask.device).unsqueeze(0)
        valid_count = history_mask.sum(dim=1, keepdim=True)
        contiguous = expected >= (steps - valid_count)
        if not bool(torch.equal(history_mask, contiguous.expand_as(history_mask))):
            raise ValueError("history_mask must be left-padded and contiguous")
        return history_mask

    def _check_planned_action(self, planned_action: Tensor, batch: int) -> Tensor:
        if planned_action.shape != (batch, self.config.action_dim):
            raise ValueError(f"planned_action must have shape [{batch},{self.config.action_dim}]")
        if not bool(torch.isfinite(planned_action).all()):
            raise ValueError("planned_action contains NaN or infinity")
        return planned_action

    def forward(
        self,
        rgb_history: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state_history: Tensor,
        action_history: Tensor,
        history_mask: Tensor | None = None,
        planned_action: Tensor | None = None,
    ) -> WorldModelPrediction:
        """Predict state and event outcomes after the last supplied action."""

        history_mask = self._check_inputs(
            rgb_history, robot_state_history, action_history, history_mask
        )
        batch, steps, channels, height, width = rgb_history.shape
        flattened = rgb_history.reshape(batch * steps, channels, height, width)
        visual = self.vision_encoder(flattened).mean(dim=1).reshape(batch, steps, -1)
        state = self.state_encoder(robot_state_history.to(device=visual.device, dtype=visual.dtype))
        action = self.action_encoder(action_history.to(device=visual.device, dtype=visual.dtype))
        time_index = torch.arange(steps, device=visual.device)
        tokens = visual + state + action + self.time_embedding(time_index).unsqueeze(0)
        encoded = self.temporal_encoder(
            tokens, src_key_padding_mask=~history_mask.to(device=visual.device)
        )
        last_index = history_mask.to(device=encoded.device).sum(dim=1).clamp_min(1) - 1
        context = encoded[torch.arange(batch, device=encoded.device), last_index]
        state_device = robot_state_history.device
        last_state = robot_state_history[
            torch.arange(batch, device=state_device),
            last_index.to(device=state_device),
        ].to(device=visual.device, dtype=visual.dtype)
        if planned_action is None:
            planned_action = action_history[
                torch.arange(batch, device=action_history.device),
                last_index.to(device=action_history.device),
            ]
        planned_action = self._check_planned_action(
            planned_action.to(device=visual.device, dtype=visual.dtype), batch
        )
        # A language summary is fused only after temporal encoding so text does
        # not alter the episode-safe masking semantics of the state sequence.
        language, language_mask = self.language_encoder(instruction)
        language_valid = (~language_mask).unsqueeze(-1).to(language.dtype)
        language_summary = (language * language_valid).sum(dim=1) / language_valid.sum(
            dim=1
        ).clamp_min(1.0)
        context = context + language_summary + self.planned_action_encoder(planned_action)
        query = self.query.to(device=context.device, dtype=context.dtype).expand(batch, -1, -1)
        # One query layer gives a stable bottleneck for deployment and keeps the
        # output contract independent of sequence length.
        query = query + context.unsqueeze(1)
        state_outputs = self.state_delta_head(query[:, 0])
        state_delta, raw_std = state_outputs.chunk(2, dim=-1)
        next_state = last_state + state_delta
        next_state_std = torch.nn.functional.softplus(raw_std) + self.config.min_state_std
        reward = self.reward_head(query[:, 0]).squeeze(-1)
        events = self.event_head(query[:, 0])
        return WorldModelPrediction(
            next_robot_state=next_state,
            reward=reward,
            terminated_logit=events[:, 0],
            truncated_logit=events[:, 1],
            collision_logit=events[:, 2],
            wrong_pick_logit=events[:, 3],
            wrong_bin_logit=events[:, 4],
            next_robot_state_std=next_state_std,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    @torch.no_grad()
    def imagine(
        self,
        rgb_history: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state_history: Tensor,
        action_history: Tensor,
        history_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-friendly one-step risk summary for preview tooling."""

        prediction = self(
            rgb_history,
            instruction,
            robot_state_history,
            action_history,
            history_mask,
        )
        probabilities = prediction.outcome_probabilities()
        return {
            "next_robot_state": prediction.next_robot_state[0].detach().cpu().tolist(),
            "next_robot_state_std": prediction.next_robot_state_std[0].detach().cpu().tolist(),
            "reward": float(prediction.reward[0].detach().cpu()),
            "event_probabilities": {
                name: float(value[0].detach().cpu()) for name, value in probabilities.items()
            },
        }

    @torch.no_grad()
    def rollout_risk(
        self,
        rgb_history: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state_history: Tensor,
        action_history: Tensor,
        history_mask: Tensor | None = None,
        *,
        horizon: int = 4,
        collision_threshold: float = 0.5,
        wrong_pick_threshold: float = 0.7,
        wrong_bin_threshold: float = 0.7,
        termination_threshold: float = 0.9,
        uncertainty_threshold: float = 0.75,
        planned_actions: Tensor | None = None,
    ) -> dict[str, Any]:
        """Run a short imagined rollout for conservative event-risk preview.

        The latest image is held constant because this compact model has no
        learned pixel renderer. This is a model-prediction diagnostic, not a
        complete visual world simulator or a hardware safety certification.
        """

        thresholds = (
            collision_threshold,
            wrong_pick_threshold,
            wrong_bin_threshold,
            termination_threshold,
        )
        if horizon < 1 or any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("horizon must be positive and risk thresholds must be in [0,1]")
        if uncertainty_threshold <= 0.0:
            raise ValueError("uncertainty_threshold must be positive")
        model_device = next(self.parameters()).device
        current_rgb = rgb_history.to(model_device)[:, -1:].expand(
            -1, self.config.observation_horizon, -1, -1, -1
        )
        current_state = robot_state_history.to(model_device).clone()
        current_actions = action_history.to(model_device).clone()
        current_mask = None if history_mask is None else history_mask.to(model_device)
        if planned_actions is not None:
            planned_actions = planned_actions.to(model_device)
            if planned_actions.ndim == 2:
                planned_actions = planned_actions.unsqueeze(0)
            if planned_actions.ndim != 3 or planned_actions.shape[0] != current_state.shape[0]:
                raise ValueError("planned_actions must have shape [B,H,action_dim]")
            if planned_actions.shape[2] != self.config.action_dim:
                raise ValueError("planned_actions action dimension does not match configuration")
            if planned_actions.shape[1] < horizon:
                raise ValueError("planned_actions must cover the requested rollout horizon")
        steps: list[dict[str, Any]] = []
        for step in range(horizon):
            step_action = (
                current_actions[:, -1] if planned_actions is None else planned_actions[:, step]
            )
            prediction = self(
                current_rgb,
                instruction,
                current_state,
                current_actions,
                current_mask,
                planned_action=step_action,
            )
            probabilities = prediction.outcome_probabilities()
            collision_probability = float(probabilities["collision"].max().cpu())
            wrong_pick_probability = float(probabilities["wrong_pick"].max().cpu())
            wrong_bin_probability = float(probabilities["wrong_bin"].max().cpu())
            termination_probability = float(probabilities["terminated"].max().cpu())
            state_std = float(prediction.next_robot_state_std.max().cpu())
            reasons: list[str] = []
            if collision_probability >= collision_threshold:
                reasons.append("collision")
            if wrong_pick_probability >= wrong_pick_threshold:
                reasons.append("wrong_pick")
            if wrong_bin_probability >= wrong_bin_threshold:
                reasons.append("wrong_bin")
            if termination_probability >= termination_threshold:
                reasons.append("termination")
            if state_std >= uncertainty_threshold:
                reasons.append("state_uncertainty")
            steps.append(
                {
                    "step": step,
                    "reward": float(prediction.reward.mean().cpu()),
                    "collision_probability": collision_probability,
                    "terminated_probability": termination_probability,
                    "wrong_pick_probability": wrong_pick_probability,
                    "wrong_bin_probability": wrong_bin_probability,
                    "max_state_std": state_std,
                    "risk_reasons": reasons,
                    "unsafe": bool(reasons),
                }
            )
            next_state = prediction.next_robot_state.detach()
            current_state = torch.cat((current_state[:, 1:], next_state.unsqueeze(1)), dim=1)
            current_actions = torch.cat((current_actions[:, 1:], step_action.unsqueeze(1)), dim=1)
            current_rgb = torch.cat((current_rgb[:, 1:], current_rgb[:, -1:]), dim=1)
            current_mask = torch.ones_like(current_mask) if current_mask is not None else None
        reasons = sorted({reason for item in steps for reason in item["risk_reasons"]})
        return {
            "horizon": horizon,
            "collision_threshold": collision_threshold,
            "wrong_pick_threshold": wrong_pick_threshold,
            "wrong_bin_threshold": wrong_bin_threshold,
            "termination_threshold": termination_threshold,
            "uncertainty_threshold": uncertainty_threshold,
            "unsafe": any(bool(item["unsafe"]) for item in steps),
            "risk_reasons": reasons,
            "steps": steps,
            "physical_robot_execution": False,
        }


__all__ = ["WorldModelConfig", "WorldModelPrediction", "WorldModelTransformer"]
