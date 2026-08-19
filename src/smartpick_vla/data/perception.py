"""Deterministic synthetic RGB-D and instance-label generation from MuJoCo."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smartpick_vla.data.expert import IKWaypointExpert
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.envs.smartpick_env import CameraName, SmartPickEnv
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import runtime_snapshot, sha256_file


@dataclass(frozen=True, slots=True)
class PerceptionGenerationConfig:
    """Small, repeatable synthetic perception collection configuration.

    The generated archive is intended for detector/segmentation prototyping and
    perception-regression checks. It is not labelled as real-camera data.
    """

    episodes: int = 8
    frames_per_episode: int = 4
    capture_stride: int = 12
    seed: int = 41
    image_size: int = 96
    max_episode_steps: int = 180
    include_depth: bool = True
    cameras: tuple[CameraName, ...] = ("top", "oblique", "wrist")
    domain_randomization: bool = True
    noisy_expert_detection: bool = False
    grasp_assist: bool = True
    six_axis: bool = False
    ood_layout: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if self.frames_per_episode < 1:
            raise ValueError("frames_per_episode must be positive")
        if self.capture_stride < 1:
            raise ValueError("capture_stride must be positive")
        if self.image_size < 32 or self.image_size > 192:
            raise ValueError("image_size must be in [32,192]")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not self.cameras:
            raise ValueError("at least one camera is required")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("camera names must be unique")
        invalid = set(self.cameras).difference({"top", "oblique", "wrist"})
        if invalid:
            raise ValueError(f"unsupported cameras: {sorted(invalid)}")


def _save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _capture_labels(
    env: SmartPickEnv,
    cameras: tuple[CameraName, ...],
    *,
    include_depth: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Capture multi-view pixels plus fixed-shape detection labels."""

    rgb_views: list[np.ndarray] = []
    depth_views: list[np.ndarray] = []
    mask_views: list[np.ndarray] = []
    bboxes = np.full((len(cameras), len(env.QUALITY_CLASSES), 4), -1, dtype=np.int16)
    visible_pixels = np.zeros((len(cameras), len(env.QUALITY_CLASSES)), dtype=np.int32)
    # The projected free-joint origin is the simulated grasp keypoint. It is a
    # more faithful supervision target than the centre of a partially occluded
    # bounding box, while the visibility mask still prevents hidden objects
    # from contributing localization loss.
    keypoints = np.full((len(cameras), len(env.QUALITY_CLASSES), 2), np.nan, dtype=np.float32)
    for camera_index, camera in enumerate(cameras):
        frame = env.render_perception(camera, include_depth=include_depth)
        rgb_views.append(np.asarray(frame["rgb"], dtype=np.uint8))
        mask_views.append(np.asarray(frame["instance_mask"], dtype=np.int16))
        depth = frame["depth_m"]
        if include_depth:
            if depth is None:
                raise RuntimeError("depth was requested but not rendered")
            depth_views.append(np.asarray(depth, dtype=np.float32))
        for instance in frame["instances"]:
            object_index = int(instance["object_index"])
            bbox = instance["bbox_xyxy"]
            if bbox is not None:
                bboxes[camera_index, object_index] = np.asarray(bbox, dtype=np.int16)
            visible_pixels[camera_index, object_index] = int(instance["visible_pixels"])
            if bbox is not None:
                with suppress(ValueError):
                    keypoints[camera_index, object_index] = env.project_world_to_pixel(
                        env.object_position(object_index), camera=camera
                    ).astype(np.float32)
    return (
        np.stack(rgb_views),
        np.stack(depth_views) if include_depth else None,
        np.stack(mask_views),
        bboxes,
        visible_pixels,
        keypoints,
    )


def generate_synthetic_perception_dataset(
    output_path: str | Path,
    *,
    config: PerceptionGenerationConfig,
    domain_config: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    """Create a labelled multi-view synthetic dataset with an auditable manifest."""

    randomization = domain_config or DomainRandomizationConfig(enabled=config.domain_randomization)
    scene_path = Path(__file__).resolve().parents[1] / "envs" / "assets" / "smartpick_scene.xml"
    env = SmartPickEnv(
        image_size=config.image_size,
        max_episode_steps=config.max_episode_steps,
        domain_randomization=randomization,
        grasp_assist=config.grasp_assist,
        six_axis=config.six_axis,
    )
    records: dict[str, list[Any]] = {
        "rgb": [],
        "depth_m": [],
        "instance_mask": [],
        "bbox_xyxy": [],
        "visible_pixels": [],
        "keypoint_xy": [],
        "robot_state": [],
        "action": [],
        "instruction": [],
        "episode_id": [],
        "step_index": [],
        "target_object_index": [],
    }
    episode_summaries: list[dict[str, Any]] = []
    try:
        for episode_id in range(config.episodes):
            seed = config.seed + episode_id
            observation, reset_info = env.reset(
                seed=seed, options={"ood_layout": config.ood_layout}
            )
            expert = IKWaypointExpert(env, use_noisy_detection=config.noisy_expert_detection)
            expert.reset()
            captured = 0
            final_info: dict[str, Any] = reset_info
            episode_return = 0.0
            for step_index in range(config.max_episode_steps):
                action, _ = expert.act()
                if step_index % config.capture_stride == 0 and captured < config.frames_per_episode:
                    rgb, depth, mask, bboxes, visible_pixels, keypoints = _capture_labels(
                        env,
                        config.cameras,
                        include_depth=config.include_depth,
                    )
                    records["rgb"].append(rgb)
                    if depth is not None:
                        records["depth_m"].append(depth)
                    records["instance_mask"].append(mask)
                    records["bbox_xyxy"].append(bboxes)
                    records["visible_pixels"].append(visible_pixels)
                    records["keypoint_xy"].append(keypoints)
                    records["robot_state"].append(observation["robot_state"])
                    records["action"].append(action)
                    records["instruction"].append(observation["instruction"])
                    records["episode_id"].append(episode_id)
                    records["step_index"].append(step_index)
                    records["target_object_index"].append(env.target_object_index)
                    captured += 1

                observation, reward, terminated, truncated, final_info = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            episode_summaries.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "captured_frames": captured,
                    "task_class": env.task.target_class,
                    "instruction_template_id": env.task.template_id,
                    "success": bool(final_info.get("success", False)),
                    "steps": step_index + 1,
                    "return": episode_return,
                    "randomization": final_info.get("randomization", reset_info["randomization"]),
                }
            )
    finally:
        env.close()

    if not records["rgb"]:
        raise RuntimeError("synthetic perception collection produced no frames")
    arrays: dict[str, np.ndarray] = {
        "rgb": np.asarray(records["rgb"], dtype=np.uint8),
        "instance_mask": np.asarray(records["instance_mask"], dtype=np.int16),
        "bbox_xyxy": np.asarray(records["bbox_xyxy"], dtype=np.int16),
        "visible_pixels": np.asarray(records["visible_pixels"], dtype=np.int32),
        "keypoint_xy": np.asarray(records["keypoint_xy"], dtype=np.float32),
        "robot_state": np.asarray(records["robot_state"], dtype=np.float32),
        "action": np.asarray(records["action"], dtype=np.float32),
        "instruction": np.asarray(records["instruction"], dtype=np.str_),
        "episode_id": np.asarray(records["episode_id"], dtype=np.int32),
        "step_index": np.asarray(records["step_index"], dtype=np.int32),
        "target_object_index": np.asarray(records["target_object_index"], dtype=np.int8),
        "camera_names": np.asarray(config.cameras, dtype=np.str_),
        "quality_classes": np.asarray(env.QUALITY_CLASSES, dtype=np.str_),
    }
    if config.include_depth:
        arrays["depth_m"] = np.asarray(records["depth_m"], dtype=np.float32)
    destination = _save_npz(output_path, arrays)
    manifest = {
        "schema_version": "picksort-synthetic-perception/v2",
        "dataset_file": destination.name,
        "dataset_sha256": sha256_file(destination),
        "scene_file": scene_path.name,
        "scene_sha256": sha256_file(scene_path),
        "generation": asdict(config),
        "samples": int(arrays["rgb"].shape[0]),
        "views": list(config.cameras),
        "rgb_shape": list(arrays["rgb"].shape),
        "instance_mask_shape": list(arrays["instance_mask"].shape),
        "bbox_shape": list(arrays["bbox_xyxy"].shape),
        "keypoint_shape": list(arrays["keypoint_xy"].shape),
        "localization_label": "camera_projected_object_origin",
        "robot_state_dim": int(arrays["robot_state"].shape[1]),
        "action_dim": int(arrays["action"].shape[1]),
        "episodes": episode_summaries,
        "runtime": runtime_snapshot(),
        "disclosure": {
            "synthetic_data": True,
            "physical_robot_data": False,
            "privileged_ik_expert": True,
            "grasp_assist": config.grasp_assist,
            "arm_variant": env.arm_variant,
        },
    }
    atomic_write_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest
