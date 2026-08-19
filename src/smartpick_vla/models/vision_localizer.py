"""Compact RGB target-localization networks for the six-axis vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from smartpick_vla.models.encoders import count_parameters


@dataclass(frozen=True, slots=True)
class VisionLocalizerConfig:
    """Architecture settings for class-conditioned image-plane localization."""

    quality_classes: int = 3
    width: int = 32
    feature_grid_size: int = 16
    hidden_dim: int = 128
    architecture: str = "spatial_heatmap_v1"

    def __post_init__(self) -> None:
        if self.quality_classes < 1:
            raise ValueError("quality_classes must be positive")
        if self.width < 8 or self.feature_grid_size < 2 or self.hidden_dim < 16:
            raise ValueError("vision localizer dimensions are too small")
        if self.architecture not in {"global_regression_v0", "spatial_heatmap_v1"}:
            raise ValueError("unsupported vision localizer architecture")


@dataclass(frozen=True, slots=True)
class VisionLocalizationPrediction:
    """Per-class normalized image centers, visibility, and optional heatmaps."""

    centers_normalized: Tensor
    visibility_logits: Tensor
    heatmap_logits: Tensor | None = None

    def visibility_probabilities(self) -> Tensor:
        return torch.sigmoid(self.visibility_logits)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(),
    )


class VisionLocalizer(nn.Module):
    """Locate class-conditioned targets from RGB pixels.

    ``spatial_heatmap_v1`` retains a two-dimensional feature grid, adds explicit
    normalized coordinate channels, and recovers sub-grid centers with a spatial
    soft-argmax. It avoids the spatial collapse of the historical global
    regression head. ``global_regression_v0`` remains loadable for existing
    experimental checkpoints.

    The model receives pixels only. It does not consume simulator object poses,
    detector labels, language strings, or robot state. Language-to-class routing
    is handled by the transparent instruction parser.
    """

    def __init__(self, config: VisionLocalizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or VisionLocalizerConfig()
        width = self.config.width
        channels = width * 4
        if self.config.architecture == "global_regression_v0":
            self.backbone = nn.Sequential(
                _conv_block(3, width, stride=2),
                _conv_block(width, width, stride=1),
                _conv_block(width, width * 2, stride=2),
                _conv_block(width * 2, width * 2, stride=1),
                _conv_block(width * 2, channels, stride=2),
                _conv_block(channels, channels, stride=1),
                nn.AdaptiveAvgPool2d(
                    (self.config.feature_grid_size, self.config.feature_grid_size)
                ),
            )
            flattened = channels * self.config.feature_grid_size * self.config.feature_grid_size
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened, self.config.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.hidden_dim, self.config.quality_classes * 3),
            )
        else:
            self.backbone = nn.Sequential(
                _conv_block(3, width, stride=2),
                _conv_block(width, width, stride=1),
                _conv_block(width, width * 2, stride=2),
                _conv_block(width * 2, width * 2, stride=1),
                _conv_block(width * 2, channels, stride=1),
                _conv_block(channels, channels, stride=1),
                nn.AdaptiveAvgPool2d(
                    (self.config.feature_grid_size, self.config.feature_grid_size)
                ),
            )
            self.heatmap_head = nn.Sequential(
                nn.Conv2d(channels + 2, self.config.hidden_dim, kernel_size=1),
                nn.GroupNorm(_group_count(self.config.hidden_dim), self.config.hidden_dim),
                nn.SiLU(),
                nn.Conv2d(self.config.hidden_dim, self.config.quality_classes, kernel_size=1),
            )
            self.visibility_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, self.config.quality_classes),
            )

    @staticmethod
    def _coordinate_channels(features: Tensor) -> Tensor:
        """Create stable [x, y] channels for the spatial localization head."""

        _, _, height, width = features.shape
        y = torch.linspace(0.0, 1.0, height, dtype=features.dtype, device=features.device)
        x = torch.linspace(0.0, 1.0, width, dtype=features.dtype, device=features.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(features.shape[0], -1, -1, -1)

    @staticmethod
    def _soft_argmax(heatmap_logits: Tensor) -> Tensor:
        """Map class heatmaps to normalized centers without discretizing outputs."""

        _, _, height, width = heatmap_logits.shape
        probabilities = torch.softmax(heatmap_logits.flatten(2), dim=-1).reshape_as(heatmap_logits)
        y = torch.linspace(
            0.0, 1.0, height, dtype=heatmap_logits.dtype, device=heatmap_logits.device
        )
        x = torch.linspace(
            0.0, 1.0, width, dtype=heatmap_logits.dtype, device=heatmap_logits.device
        )
        center_x = (probabilities * x.view(1, 1, 1, width)).sum(dim=(2, 3))
        center_y = (probabilities * y.view(1, 1, height, 1)).sum(dim=(2, 3))
        return torch.stack((center_x, center_y), dim=-1)

    def forward(self, rgb: Tensor) -> VisionLocalizationPrediction:
        """Return class-indexed centers in ``[0, 1]`` and visibility logits."""

        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        if rgb.shape[2] < 16 or rgb.shape[3] < 16:
            raise ValueError("rgb height and width must be at least 16")
        if not torch.isfinite(rgb).all():
            raise ValueError("rgb contains NaN or infinity")
        pixels = rgb.to(dtype=next(self.parameters()).dtype)
        if rgb.dtype == torch.uint8 or float(pixels.detach().amax().cpu()) > 1.5:
            pixels = pixels / 255.0
        features = self.backbone(pixels)
        if self.config.architecture == "global_regression_v0":
            output = self.head(features).reshape(pixels.shape[0], self.config.quality_classes, 3)
            return VisionLocalizationPrediction(
                centers_normalized=torch.sigmoid(output[..., :2]),
                visibility_logits=output[..., 2],
            )
        heatmap_logits = self.heatmap_head(
            torch.cat((features, self._coordinate_channels(features)), dim=1)
        )
        return VisionLocalizationPrediction(
            centers_normalized=self._soft_argmax(heatmap_logits),
            visibility_logits=self.visibility_head(features),
            heatmap_logits=heatmap_logits,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)
