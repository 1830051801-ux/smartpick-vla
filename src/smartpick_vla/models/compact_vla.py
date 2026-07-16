"""ACT-style compact vision-language-action policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from smartpick_vla.models.encoders import ByteTextEncoder, CompactVisionEncoder, count_parameters


@dataclass(frozen=True, slots=True)
class CompactVLAConfig:
    """Configuration for a laptop-scale action-query policy."""

    robot_state_dim: int = 24
    action_dim: int = 5
    action_horizon: int = 8
    d_model: int = 128
    nhead: int = 4
    decoder_layers: int = 2
    language_layers: int = 1
    feedforward_dim: int = 256
    language_max_length: int = 64
    vision_grid_size: int = 4
    dropout: float = 0.0
    squash_actions: bool = True
    freeze_vision: bool = False
    freeze_language: bool = False

    def __post_init__(self) -> None:
        if self.robot_state_dim < 1 or self.action_dim < 1 or self.action_horizon < 1:
            raise ValueError("state, action, and action_horizon dimensions must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.decoder_layers < 1 or self.language_layers < 1:
            raise ValueError("encoder and decoder layer counts must be positive")


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class CompactVLAPolicy(nn.Module):
    """Predict an action chunk with learned parallel action queries.

    This is a compact VLA, not a large pretrained foundation model.  Visual
    grid tokens, byte-language tokens, and one robot-state token form decoder
    memory.  Learned action queries decode all horizon positions in parallel,
    following the central action-chunking idea of ACT.
    """

    def __init__(self, config: CompactVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = CompactVisionEncoder(
            config.d_model, grid_size=config.vision_grid_size
        )
        self.language_encoder = ByteTextEncoder(
            config.d_model,
            max_length=config.language_max_length,
            num_layers=config.language_layers,
            nhead=config.nhead,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.robot_state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.action_queries = nn.Embedding(config.action_horizon, config.d_model)
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.action_dim),
        )
        if config.freeze_vision:
            _freeze(self.vision_encoder)
        if config.freeze_language:
            _freeze(self.language_encoder)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self.config.freeze_vision:
            self.vision_encoder.eval()
        if self.config.freeze_language:
            self.language_encoder.eval()
        return self

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
        vision_tokens = self.vision_encoder(rgb)
        language_tokens, language_padding = self.language_encoder(instruction)
        state_token = self.state_encoder(
            robot_state.to(dtype=vision_tokens.dtype, device=vision_tokens.device)
        ).unsqueeze(1)
        batch_size = vision_tokens.shape[0]
        if language_tokens.shape[0] != batch_size or state_token.shape[0] != batch_size:
            raise ValueError("rgb, instruction, and robot_state batch sizes must match")

        memory = torch.cat((vision_tokens, language_tokens, state_token), dim=1)
        always_valid = torch.zeros(
            (batch_size, vision_tokens.shape[1] + 1),
            dtype=torch.bool,
            device=memory.device,
        )
        memory_padding = torch.cat(
            (always_valid[:, : vision_tokens.shape[1]], language_padding, always_valid[:, -1:]),
            dim=1,
        )
        queries = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.action_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_padding,
        )
        action_chunk = self.action_head(decoded)
        return torch.tanh(action_chunk) if self.config.squash_actions else action_chunk

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    @torch.no_grad()
    def act(
        self,
        rgb: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state: Tensor,
    ) -> Tensor:
        """Return the first executable action while retaining chunk training."""

        return self(rgb, instruction, robot_state)[:, 0]
