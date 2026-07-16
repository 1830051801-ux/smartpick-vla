"""Compact visual and byte-level language encoders used by local policies."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    return next(groups for groups in (8, 4, 2, 1) if channels % groups == 0)


class CompactVisionEncoder(nn.Module):
    """A small CNN that returns a fixed grid of visual tokens.

    The encoder intentionally avoids an external pretrained backbone.  It keeps
    the default model small enough to train on a 4 GB laptop GPU while still
    exposing spatial tokens to the action-query decoder.
    """

    def __init__(self, d_model: int, *, grid_size: int = 4, in_channels: int = 3) -> None:
        super().__init__()
        if d_model < 16:
            raise ValueError("d_model must be at least 16")
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.in_channels = in_channels
        self.grid_size = grid_size
        mid_channels = max(16, d_model // 2)
        self.convolutions = nn.Sequential(
            # Explicit normalized image coordinates make small-object spatial
            # grounding learnable at smoke-scale data budgets (CoordConv).
            nn.Conv2d(in_channels + 2, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(d_model), d_model),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.position = nn.Parameter(torch.empty(1, grid_size * grid_size, d_model))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4:
            raise ValueError(f"rgb must have shape [B,C,H,W], got {tuple(rgb.shape)}")
        if rgb.shape[1] != self.in_channels:
            raise ValueError(
                f"rgb channel dimension must be {self.in_channels}, got {rgb.shape[1]}"
            )
        if rgb.dtype == torch.uint8:
            image = rgb.to(device=self.position.device, dtype=self.position.dtype).div(255.0)
        elif rgb.is_floating_point():
            image = rgb.to(device=self.position.device, dtype=self.position.dtype)
        else:
            raise TypeError("rgb must be uint8 or a floating-point tensor")
        height, width = image.shape[-2:]
        y_coordinates = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        x_coordinates = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        coordinates = torch.stack((x_grid, y_grid)).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
        features = self.convolutions(torch.cat((image, coordinates), dim=1))
        tokens = features.flatten(2).transpose(1, 2)
        return tokens + self.position


class ByteTextEncoder(nn.Module):
    """Encode UTF-8 instructions without a downloaded tokenizer or vocabulary."""

    PAD_TOKEN = 0
    BYTE_OFFSET = 1
    BOS_TOKEN = 257
    VOCAB_SIZE = 258

    def __init__(
        self,
        d_model: int,
        *,
        max_length: int = 64,
        num_layers: int = 1,
        nhead: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.max_length = max_length
        self.embedding = nn.Embedding(self.VOCAB_SIZE, d_model, padding_idx=self.PAD_TOKEN)
        self.position = nn.Parameter(torch.empty(1, max_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim or 2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.position, std=0.02)

    def tokenize(self, instructions: Sequence[str], *, device: torch.device) -> Tensor:
        """Return padded UTF-8 byte tokens with a mandatory BOS token."""

        batch_size = len(instructions)
        tokens = torch.full(
            (batch_size, self.max_length),
            self.PAD_TOKEN,
            dtype=torch.long,
            device=device,
        )
        if batch_size == 0:
            return tokens
        tokens[:, 0] = self.BOS_TOKEN
        for row, instruction in enumerate(instructions):
            if not isinstance(instruction, str):
                raise TypeError("all instructions must be strings")
            encoded = instruction.encode("utf-8")[: self.max_length - 1]
            if encoded:
                byte_tokens = torch.tensor(
                    [value + self.BYTE_OFFSET for value in encoded],
                    dtype=torch.long,
                    device=device,
                )
                tokens[row, 1 : 1 + len(encoded)] = byte_tokens
        return tokens

    def forward(self, instruction: Sequence[str] | Tensor) -> tuple[Tensor, Tensor]:
        """Return encoded tokens and a padding mask (`True` means padding)."""

        device = self.embedding.weight.device
        if isinstance(instruction, Tensor):
            if instruction.ndim != 2:
                raise ValueError("token tensor must have shape [B,L]")
            if instruction.shape[1] < 1:
                raise ValueError("token tensor must contain at least one position")
            if instruction.shape[1] > self.max_length:
                raise ValueError(
                    f"token sequence length {instruction.shape[1]} exceeds {self.max_length}"
                )
            tokens = instruction.to(device=device, dtype=torch.long)
            if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.VOCAB_SIZE):
                raise ValueError("token ids are outside the byte vocabulary")
            all_padding = tokens.eq(self.PAD_TOKEN).all(dim=1)
            if all_padding.any():
                tokens = tokens.clone()
                tokens[all_padding, 0] = self.BOS_TOKEN
        else:
            tokens = self.tokenize(instruction, device=device)
        padding_mask = tokens.eq(self.PAD_TOKEN)
        embeddings = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        return encoded, padding_mask


def masked_mean(tokens: Tensor, padding_mask: Tensor) -> Tensor:
    """Pool a token sequence without allowing padding to affect the result."""

    valid = (~padding_mask).unsqueeze(-1).to(dtype=tokens.dtype)
    denominator = valid.sum(dim=1).clamp_min(1.0)
    return (tokens * valid).sum(dim=1) / denominator


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count scalar parameters, optionally excluding frozen parameters."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )
