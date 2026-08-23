"""Export and verify the RGB localizer used by the six-axis visual pipeline.

The simulator trains and evaluates the PyTorch model. This module adds a
deployment boundary: the exact checkpoint is exported to ONNX, checked with
the ONNX graph validator, and compared against ONNX Runtime on the same input.
It intentionally exports only the perception head, not simulator state or
privileged object poses.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from smartpick_vla.training.vision import load_vision_localizer
from smartpick_vla.utils.provenance import runtime_snapshot, sha256_file


class _VisionLocalizerExportWrapper(nn.Module):
    """Turn the dataclass model output into stable tensor outputs for ONNX."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        # Make preprocessing explicit before tracing. The PyTorch model also
        # accepts [0, 1], but tracing a data-dependent range branch would make
        # the exported graph disagree with runtime inputs.
        rgb = rgb.to(dtype=torch.float32) / 255.0
        prediction = self.model(rgb)
        centers = getattr(prediction, "centers_normalized", None)
        visibility = getattr(prediction, "visibility_logits", None)
        if not isinstance(centers, Tensor) or not isinstance(visibility, Tensor):
            raise TypeError("vision localizer must return tensor center and visibility outputs")
        return centers, visibility


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "ONNX export verification requires the optional dependencies; "
            "install with `pip install -e .[onnx]`"
        ) from error


def export_vision_localizer_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    image_size: int | None = None,
    opset_version: int = 17,
    device: str | torch.device = "cpu",
    verify: bool = True,
) -> dict[str, Any]:
    """Export one trusted localizer checkpoint and return an audit manifest.

    The exported input is float32 ``[B,3,H,W]`` in the explicit [0, 255] pixel
    range. The wrapper performs the [0, 255] -> [0, 1] conversion inside the
    graph. Dynamic batch axes are retained; spatial dimensions are fixed to
    the checkpoint's training image size for predictable edge deployment.
    """

    if opset_version < 13:
        raise ValueError("opset_version must be at least 13")
    model, metadata = load_vision_localizer(checkpoint_path, device=device)
    configured_shape = metadata.get("image_size")
    configured_size: int | None = None
    if isinstance(configured_shape, (list, tuple)) and len(configured_shape) == 2:
        if int(configured_shape[0]) != int(configured_shape[1]):
            raise ValueError("vision checkpoint image_size must be square for ONNX export")
        configured_size = int(configured_shape[0])
    size = int(image_size or configured_size or 96)
    if size < 16:
        raise ValueError("image_size must be at least 16")

    destination = Path(output_path)
    if destination.suffix.lower() != ".onnx":
        raise ValueError("output_path must use the .onnx suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _VisionLocalizerExportWrapper(model).eval()
    example = torch.zeros((1, 3, size, size), dtype=torch.float32, device=device)
    torch.onnx.export(
        wrapper,
        (example,),
        str(destination),
        input_names=["rgb"],
        output_names=["centers_normalized", "visibility_logits"],
        dynamic_axes={
            "rgb": {0: "batch"},
            "centers_normalized": {0: "batch"},
            "visibility_logits": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_version: str | None = None
    onnxruntime_version: str | None = None
    max_center_error = 0.0
    max_visibility_error = 0.0
    verification_passed = False
    if verify:
        onnx = _optional_module("onnx")
        onnxruntime = _optional_module("onnxruntime")
        graph = onnx.load(str(destination))
        onnx.checker.check_model(graph)
        onnx_version = str(getattr(onnx, "__version__", "unknown"))
        onnxruntime_version = str(getattr(onnxruntime, "__version__", "unknown"))
        rng = np.random.default_rng(20260822)
        sample = rng.uniform(0.0, 255.0, size=(2, 3, size, size)).astype(np.float32)
        with torch.no_grad():
            torch_centers, torch_visibility = wrapper(torch.from_numpy(sample).to(device))
        session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
        ort_centers, ort_visibility = session.run(
            ["centers_normalized", "visibility_logits"], {"rgb": sample}
        )
        max_center_error = float(np.max(np.abs(torch_centers.detach().cpu().numpy() - ort_centers)))
        max_visibility_error = float(
            np.max(np.abs(torch_visibility.detach().cpu().numpy() - ort_visibility))
        )
        verification_passed = max(max_center_error, max_visibility_error) <= 1e-4
        if not verification_passed:
            raise RuntimeError(
                "ONNX Runtime output diverges from PyTorch: "
                f"center={max_center_error:.6g}, visibility={max_visibility_error:.6g}"
            )

    manifest = {
        "schema_version": "picksort-vision-onnx/v1",
        "scope": "perception-only deployment artifact",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_file": destination.name,
        "onnx_sha256": sha256_file(destination),
        "image_size": [size, size],
        "input": {
            "name": "rgb",
            "dtype": "float32",
            "value_range": [0.0, 255.0],
            "shape": ["batch", 3, size, size],
        },
        "outputs": {
            "centers_normalized": ["batch", 3, 2],
            "visibility_logits": ["batch", 3],
        },
        "opset_version": opset_version,
        "verify_requested": verify,
        "verification_passed": verification_passed if verify else None,
        "max_center_abs_error": max_center_error if verify else None,
        "max_visibility_abs_error": max_visibility_error if verify else None,
        "onnx_version": onnx_version,
        "onnxruntime_version": onnxruntime_version,
        "runtime": runtime_snapshot(),
        "simulation_only": True,
        "physical_robot_execution": False,
    }
    _write_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest
