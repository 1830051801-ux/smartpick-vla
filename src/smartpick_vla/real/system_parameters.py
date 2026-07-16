"""Measured/estimated cell parameters used to seed simulation profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SystemParameters:
    object_mass_kg: float = 0.07
    object_friction: float = 1.10
    table_friction: float = 0.90
    joint_damping_scale: float = 1.0
    actuator_gain_scale: float = 1.0
    control_delay_s: float = 0.04
    detection_noise_std_m: float = 0.004
    camera_position_std_m: float = 0.012

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.object_mass_kg,
                self.object_friction,
                self.table_friction,
                self.joint_damping_scale,
                self.actuator_gain_scale,
                self.control_delay_s,
                self.detection_noise_std_m,
                self.camera_position_std_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("system parameters must be finite and non-negative")
        if (
            min(
                self.object_mass_kg,
                self.object_friction,
                self.table_friction,
                self.joint_damping_scale,
                self.actuator_gain_scale,
            )
            <= 0.0
        ):
            raise ValueError("mass, friction, damping, and gain scales must be positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> SystemParameters:
        return cls(**dict(payload or {}))

    def control_delay_steps(self, control_dt_s: float) -> int:
        if not np.isfinite(control_dt_s) or control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and positive")
        return max(0, round(self.control_delay_s / control_dt_s))
