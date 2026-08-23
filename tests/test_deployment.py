"""Deployment-boundary tests for the learned visual localizer."""

from pathlib import Path

import pytest

from smartpick_vla.deployment.vision_onnx import export_vision_localizer_onnx
from smartpick_vla.models.vision_localizer import VisionLocalizer, VisionLocalizerConfig
from smartpick_vla.training.checkpoint import save_checkpoint


@pytest.mark.parametrize("architecture", ["spatial_heatmap_v1", "global_regression_v0"])
def test_vision_localizer_onnx_export_matches_pytorch(tmp_path: Path, architecture: str) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = VisionLocalizerConfig(
        width=8,
        feature_grid_size=2,
        hidden_dim=16,
        architecture=architecture,
    )
    model = VisionLocalizer(config)
    checkpoint = save_checkpoint(
        tmp_path / "best.pt",
        model,
        config=config,
        extra={
            "model_kind": "vision_localizer",
            "model_config": {
                "quality_classes": 3,
                "width": 8,
                "feature_grid_size": 2,
                "hidden_dim": 16,
                "architecture": architecture,
            },
            "quality_classes": ["accepted", "scratch", "unknown"],
            "camera": "top",
            "image_size": [32, 32],
        },
    )
    output = tmp_path / "vision.onnx"
    manifest = export_vision_localizer_onnx(checkpoint, output, image_size=32)
    assert output.is_file()
    assert Path(str(output).replace(".onnx", ".manifest.json")).is_file()
    assert manifest["verification_passed"] is True
    assert manifest["max_center_abs_error"] <= 1e-4
    assert manifest["max_visibility_abs_error"] <= 1e-4
