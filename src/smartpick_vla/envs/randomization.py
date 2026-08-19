"""Episode-level physics, sensing, and control domain randomization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True, slots=True)
class DomainRandomizationConfig:
    enabled: bool = False
    object_mass_scale: tuple[float, float] = (0.65, 1.45)
    friction_scale: tuple[float, float] = (0.55, 1.55)
    camera_position_std_m: float = 0.012
    camera_fovy_delta_deg: float = 3.0
    light_intensity_scale: tuple[float, float] = (0.72, 1.28)
    object_color_jitter: float = 0.10
    robot_state_noise_std: float = 0.002
    detection_noise_std_m: float = 0.004
    control_delay_steps: tuple[int, int] = (0, 3)
    image_noise_std_px: float = 0.0
    image_occlusion_probability: float = 0.0
    image_occlusion_max_fraction: float = 0.0
    vision_latency_frames: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if (
            self.object_mass_scale[0] <= 0.0
            or self.object_mass_scale[0] > self.object_mass_scale[1]
        ):
            raise ValueError("object_mass_scale must be positive and ordered")
        if self.friction_scale[0] <= 0.0 or self.friction_scale[0] > self.friction_scale[1]:
            raise ValueError("friction_scale must be positive and ordered")
        if (
            self.light_intensity_scale[0] < 0.0
            or self.light_intensity_scale[0] > self.light_intensity_scale[1]
        ):
            raise ValueError("light_intensity_scale must be non-negative and ordered")
        if self.camera_position_std_m < 0.0 or self.camera_fovy_delta_deg < 0.0:
            raise ValueError("camera randomization values must be non-negative")
        if self.object_color_jitter < 0.0 or self.robot_state_noise_std < 0.0:
            raise ValueError("state and color randomization values must be non-negative")
        if self.detection_noise_std_m < 0.0 or self.image_noise_std_px < 0.0:
            raise ValueError("detection and image noise values must be non-negative")
        if not 0.0 <= self.image_occlusion_probability <= 1.0:
            raise ValueError("image_occlusion_probability must be in [0, 1]")
        if not 0.0 <= self.image_occlusion_max_fraction <= 1.0:
            raise ValueError("image_occlusion_max_fraction must be in [0, 1]")
        if (
            self.control_delay_steps[0] < 0
            or self.control_delay_steps[0] > self.control_delay_steps[1]
        ):
            raise ValueError("control_delay_steps must be non-negative and ordered")
        if (
            self.vision_latency_frames[0] < 0
            or self.vision_latency_frames[0] > self.vision_latency_frames[1]
        ):
            raise ValueError("vision_latency_frames must be non-negative and ordered")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DomainRandomizationConfig:
        if not data:
            return cls()
        normalized = dict(data)
        tuple_fields = (
            "object_mass_scale",
            "friction_scale",
            "light_intensity_scale",
            "control_delay_steps",
            "vision_latency_frames",
        )
        for name in tuple_fields:
            if name in normalized:
                normalized[name] = tuple(normalized[name])
        return cls(**normalized)


class DomainRandomizer:
    """Mutate a compiled model from a captured nominal snapshot each reset."""

    def __init__(self, model: mujoco.MjModel, config: DomainRandomizationConfig) -> None:
        self.model = model
        self.config = config
        self._body_mass = model.body_mass.copy()
        self._geom_friction = model.geom_friction.copy()
        self._geom_rgba = model.geom_rgba.copy()
        self._cam_pos = model.cam_pos.copy()
        self._cam_fovy = model.cam_fovy.copy()
        self._light_diffuse = model.light_diffuse.copy()
        self._object_body_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(3)]
        )
        self._object_geom_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obj{i}_geom") for i in range(3)]
        )

    def reset_nominal(self) -> None:
        self.model.body_mass[:] = self._body_mass
        self.model.geom_friction[:] = self._geom_friction
        self.model.geom_rgba[:] = self._geom_rgba
        self.model.cam_pos[:] = self._cam_pos
        self.model.cam_fovy[:] = self._cam_fovy
        self.model.light_diffuse[:] = self._light_diffuse

    def apply(self, rng: np.random.Generator) -> dict[str, Any]:
        self.reset_nominal()
        if not self.config.enabled:
            return {
                "enabled": False,
                "mass_scales": [1.0, 1.0, 1.0],
                "friction_scale": 1.0,
                "camera_offset_m": [0.0, 0.0, 0.0],
                "camera_fovy_delta_deg": 0.0,
                "light_scale": 1.0,
                "control_delay_steps": 0,
                "robot_state_noise_std": 0.0,
                "detection_noise_std_m": 0.0,
                "image_noise_std_px": 0.0,
                "image_occlusion_probability": 0.0,
                "image_occlusion_max_fraction": 0.0,
                "vision_latency_frames": 0,
            }

        cfg = self.config
        mass_scales = rng.uniform(*cfg.object_mass_scale, size=3)
        self.model.body_mass[self._object_body_ids] = (
            self._body_mass[self._object_body_ids] * mass_scales
        )
        friction_scale = float(rng.uniform(*cfg.friction_scale))
        self.model.geom_friction[:, 0] = np.clip(
            self._geom_friction[:, 0] * friction_scale, 0.05, 3.0
        )

        top_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
        camera_offset = rng.normal(0.0, cfg.camera_position_std_m, size=3)
        self.model.cam_pos[top_camera_id] = self._cam_pos[top_camera_id] + camera_offset
        fovy_delta = float(rng.uniform(-cfg.camera_fovy_delta_deg, cfg.camera_fovy_delta_deg))
        self.model.cam_fovy[top_camera_id] = self._cam_fovy[top_camera_id] + fovy_delta

        light_scale = float(rng.uniform(*cfg.light_intensity_scale))
        self.model.light_diffuse[:] = np.clip(self._light_diffuse * light_scale, 0.0, 1.5)
        for geom_id in self._object_geom_ids:
            jitter = rng.uniform(-cfg.object_color_jitter, cfg.object_color_jitter, size=3)
            self.model.geom_rgba[geom_id, :3] = np.clip(
                self._geom_rgba[geom_id, :3] + jitter, 0.05, 0.95
            )

        delay = int(rng.integers(cfg.control_delay_steps[0], cfg.control_delay_steps[1] + 1))
        vision_latency = int(
            rng.integers(cfg.vision_latency_frames[0], cfg.vision_latency_frames[1] + 1)
        )
        return {
            "enabled": True,
            "mass_scales": mass_scales.round(6).tolist(),
            "friction_scale": friction_scale,
            "camera_offset_m": camera_offset.round(6).tolist(),
            "camera_fovy_delta_deg": fovy_delta,
            "light_scale": light_scale,
            "control_delay_steps": delay,
            "robot_state_noise_std": cfg.robot_state_noise_std,
            "detection_noise_std_m": cfg.detection_noise_std_m,
            "image_noise_std_px": cfg.image_noise_std_px,
            "image_occlusion_probability": cfg.image_occlusion_probability,
            "image_occlusion_max_fraction": cfg.image_occlusion_max_fraction,
            "vision_latency_frames": vision_latency,
            "config": asdict(cfg),
        }
