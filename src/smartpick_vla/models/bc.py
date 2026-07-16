"""Single-step language-conditioned behavior-cloning baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from smartpick_vla.models.encoders import (
    ByteTextEncoder,
    CompactVisionEncoder,
    count_parameters,
    masked_mean,
)


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    """Configuration for the deliberately small one-action BC baseline."""

    robot_state_dim: int = 24
    action_dim: int = 5
    d_model: int = 96
    hidden_dim: int = 192
    language_max_length: int = 64
    language_layers: int = 1
    nhead: int = 4
    dropout: float = 0.0
    squash_actions: bool = True

    def __post_init__(self) -> None:
        if self.robot_state_dim < 1 or self.action_dim < 1:
            raise ValueError("state and action dimensions must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")


class BehaviorCloningPolicy(nn.Module):
    """Predict one normalized continuous action from RGB, text, and state."""

    def __init__(self, config: BehaviorCloningConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = CompactVisionEncoder(config.d_model)
        self.language_encoder = ByteTextEncoder(
            config.d_model,
            max_length=config.language_max_length,
            num_layers=config.language_layers,
            nhead=config.nhead,
            dropout=config.dropout,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.robot_state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(3 * config.d_model, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(
        self,
        rgb: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state: Tensor,
    ) -> Tensor:
        if robot_state.ndim != 2 or robot_state.shape[1] != self.config.robot_state_dim:
            raise ValueError(
                "robot_state must have shape "
                f"[B,{self.config.robot_state_dim}], got {tuple(robot_state.shape)}"
            )
        vision = self.vision_encoder(rgb).mean(dim=1)
        language_tokens, language_padding = self.language_encoder(instruction)
        language = masked_mean(language_tokens, language_padding)
        state = self.state_encoder(robot_state.to(dtype=vision.dtype, device=vision.device))
        if not (vision.shape[0] == language.shape[0] == state.shape[0]):
            raise ValueError("rgb, instruction, and robot_state batch sizes must match")
        action = self.action_head(torch.cat((vision, language, state), dim=-1))
        return torch.tanh(action) if self.config.squash_actions else action

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)
