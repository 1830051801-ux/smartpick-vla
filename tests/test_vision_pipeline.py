"""Regression coverage for the six-axis RGB calibration and localization path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from smartpick_vla.envs import SmartPickEnv
from smartpick_vla.evaluation.vision_controller import (
    VisionControllerConfig,
    VisionGuidedSixAxisController,
    build_simulated_planar_calibration,
)
from smartpick_vla.models.vision_localizer import VisionLocalizer, VisionLocalizerConfig
from smartpick_vla.training.vision import (
    VisionTrainingConfig,
    load_vision_localizer,
    train_vision_localizer,
)
from smartpick_vla.vision.instructions import infer_quality_class


def test_instruction_parser_resolves_declared_quality_templates() -> None:
    assert infer_quality_class("place the accepted part") == "accepted"
    assert infer_quality_class("sort the scratched workpiece") == "scratch"
    assert infer_quality_class("move the unknown item to manual review") == "unknown"
    with pytest.raises(ValueError, match="unique quality class"):
        infer_quality_class("place the accepted scratched part")


def test_vision_localizer_output_contract() -> None:
    model = VisionLocalizer(VisionLocalizerConfig(width=8, feature_grid_size=2, hidden_dim=16))
    prediction = model(torch.zeros((2, 3, 32, 32), dtype=torch.uint8))
    assert prediction.centers_normalized.shape == (2, 3, 2)
    assert prediction.visibility_logits.shape == (2, 3)
    assert prediction.heatmap_logits is not None
    assert prediction.heatmap_logits.shape == (2, 3, 2, 2)
    assert bool(
        ((prediction.centers_normalized >= 0.0) & (prediction.centers_normalized <= 1.0)).all()
    )


@pytest.mark.mujoco
def test_calibration_and_controller_keep_world_state_outside_runtime() -> None:
    environment = SmartPickEnv(image_size=64, six_axis=True)
    try:
        observation, _ = environment.reset(seed=31, options={"task_class": "accepted"})
        calibration = build_simulated_planar_calibration(environment)
        for object_index in range(3):
            pixel = environment.project_world_to_pixel(environment.object_position(object_index))
            restored_xy = calibration.pixels_to_base_xy(pixel)
            np.testing.assert_allclose(
                restored_xy,
                environment.object_position(object_index)[:2],
                atol=1e-8,
            )
        controller = VisionGuidedSixAxisController(
            VisionLocalizer(VisionLocalizerConfig(width=8, feature_grid_size=2, hidden_dim=16)),
            calibration=calibration,
            bin_base_xy_m={
                quality_class: environment.bin_position(quality_class)[:2]
                for quality_class in environment.QUALITY_CLASSES
            },
            config=VisionControllerConfig(
                minimum_detection_confidence=0.0,
                confirmation_frames=1,
            ),
        )
        decision = controller.act(observation)
        assert decision.action.shape == (6,)
        assert np.isfinite(decision.action).all()
        assert not hasattr(controller, "environment")
        assert controller.last_diagnostics is not None
        assert controller.last_diagnostics.detection_used
    finally:
        environment.close()


@pytest.mark.mujoco
def test_randomized_camera_calibration_matches_rendered_scene() -> None:
    """Keep the simulated camera, calibration fixture, and renderer aligned."""

    environment = SmartPickEnv(
        image_size=96,
        six_axis=True,
        domain_randomization={
            "enabled": True,
            "camera_position_std_m": 0.006,
            "camera_fovy_delta_deg": 1.5,
        },
    )
    try:
        environment.reset(seed=914, options={"task_class": "unknown", "ood_layout": True})
        calibration = build_simulated_planar_calibration(environment)
        rendered = environment.render_perception("top")
        for object_index, instance in enumerate(rendered["instances"]):
            projected = environment.project_world_to_pixel(
                environment.object_position(object_index)
            )
            restored = calibration.pixels_to_base_xy(projected)
            np.testing.assert_allclose(
                restored,
                environment.object_position(object_index)[:2],
                atol=5e-7,
            )
            centroid = instance["centroid_xy"]
            assert centroid is not None
            assert float(np.linalg.norm(projected - np.asarray(centroid))) < 4.0
    finally:
        environment.close()


@pytest.mark.mujoco
def test_six_axis_home_pose_keeps_seed_909_target_visible() -> None:
    """Regression-test the camera-clear reset pose used by the release run."""

    environment = SmartPickEnv(image_size=96, six_axis=True)
    try:
        _, info = environment.reset(
            seed=909,
            options={
                "instruction_split": "paraphrase",
                "ood_layout": True,
                "task_class": "accepted",
            },
        )
        target_instance = environment.render_perception("top", include_depth=False)["instances"][
            environment.target_object_index
        ]
        assert info["collision_steps"] == 0
        assert target_instance["visible_pixels"] >= 20
    finally:
        environment.close()


def test_vision_training_saves_loadable_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_path = tmp_path / "perception.npz"
    samples = 8
    image_size = 32
    rgb = np.zeros((samples, 1, image_size, image_size, 3), dtype=np.uint8)
    bboxes = np.zeros((samples, 1, 3, 4), dtype=np.int16)
    for sample_index in range(samples):
        for class_index in range(3):
            left = 3 + class_index * 8 + sample_index % 2
            top = 5 + class_index * 4
            rgb[sample_index, 0, top : top + 5, left : left + 5, class_index] = 255
            bboxes[sample_index, 0, class_index] = [left, top, left + 5, top + 5]
    np.savez_compressed(
        dataset_path,
        rgb=rgb,
        bbox_xyxy=bboxes,
        episode_id=np.arange(samples, dtype=np.int32),
        camera_names=np.asarray(["top"]),
        quality_classes=np.asarray(["accepted", "scratch", "unknown"]),
    )
    output = tmp_path / "vision_run"
    manifest = train_vision_localizer(
        dataset_path,
        output,
        training_config=VisionTrainingConfig(
            epochs=1,
            batch_size=2,
            validation_fraction=0.25,
            device="cpu",
            seed=99,
        ),
        model_options={"width": 8, "feature_grid_size": 2, "hidden_dim": 16},
    )
    loaded, metadata = load_vision_localizer(output / "best.pt")
    assert manifest["best_epoch"] == 1
    assert metadata["model_kind"] == "vision_localizer"
    assert loaded.config.quality_classes == 3
