"""Simulator-only predictive action shield for PickSort rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from smartpick_vla.envs.smartpick_env import SmartPickEnv


@dataclass(frozen=True, slots=True)
class PredictiveSafetyFilterConfig:
    """Bounded copied-state rollout parameters for the action shield."""

    lookahead_control_steps: int = 2
    minimum_motion_scale: float = 0.125
    scale_decay: float = 0.5
    allow_finger_table_contact: bool = True

    def __post_init__(self) -> None:
        if self.lookahead_control_steps < 1:
            raise ValueError("lookahead_control_steps must be positive")
        if not 0.0 < self.minimum_motion_scale <= 1.0:
            raise ValueError("minimum_motion_scale must be in (0,1]")
        if not 0.0 < self.scale_decay < 1.0:
            raise ValueError("scale_decay must be in (0,1)")
        if not isinstance(self.allow_finger_table_contact, bool):
            raise TypeError("allow_finger_table_contact must be a bool")

    def candidate_scales(self) -> tuple[float, ...]:
        """Return deterministic full-to-minimum geometric action scales."""

        scales = [1.0]
        current = 1.0
        while current * self.scale_decay > self.minimum_motion_scale + 1e-9:
            current *= self.scale_decay
            scales.append(current)
        if scales[-1] != self.minimum_motion_scale:
            scales.append(self.minimum_motion_scale)
        return tuple(scales)


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """One auditable filter outcome prior to an environment step."""

    action: np.ndarray
    intervened: bool
    motion_scale: float
    predicted_collision: bool
    predicted_contact_count: int
    rejected_all_motion: bool


class PredictiveSafetyFilter:
    """Project candidate controls through a copied MuJoCo state before execution.

    This class has no hardware I/O. It uses the current simulation state, the
    same IK/control mapping as :class:`SmartPickEnv`, and a short forward
    rollout. A collision with a tray, table, or non-target part causes a
    geometric reduction of Cartesian/orientation motion. If every candidate is
    unsafe, the filter emits a no-motion fallback while preserving the most
    recent gripper state.
    """

    def __init__(
        self,
        environment: SmartPickEnv,
        *,
        config: PredictiveSafetyFilterConfig | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or PredictiveSafetyFilterConfig()
        self._rollout_data = mujoco.MjData(environment.model)

    def filter(self, action: np.ndarray) -> SafetyDecision:
        """Return the first copied-state rollout candidate with no collision."""

        normalized = self.environment.normalize_action(action)
        predicted_collision = False
        predicted_contact_count = 0
        for motion_scale in self.config.candidate_scales():
            candidate = normalized.copy()
            candidate[:-1] *= motion_scale
            collision, contacts = self._would_collide(candidate)
            if motion_scale == 1.0:
                predicted_collision = collision
                predicted_contact_count = contacts
            if not collision:
                return SafetyDecision(
                    action=candidate,
                    intervened=motion_scale != 1.0,
                    motion_scale=motion_scale,
                    predicted_collision=predicted_collision,
                    predicted_contact_count=predicted_contact_count,
                    rejected_all_motion=False,
                )

        fallback = np.zeros(self.environment.action_dim, dtype=np.float32)
        fallback[-1] = float(self.environment._last_action[-1])
        return SafetyDecision(
            action=fallback,
            intervened=True,
            motion_scale=0.0,
            predicted_collision=predicted_collision,
            predicted_contact_count=predicted_contact_count,
            rejected_all_motion=True,
        )

    def _would_collide(self, action: np.ndarray) -> tuple[bool, int]:
        targets = self.environment.preview_control_targets(action)
        mujoco.mj_copyData(self._rollout_data, self.environment.model, self.environment.data)
        self._rollout_data.ctrl[self.environment._arm_actuator_ids] = targets.arm_qpos
        self._rollout_data.ctrl[self.environment._finger_actuator_ids] = targets.finger_position
        contact_count = 0
        for _ in range(self.config.lookahead_control_steps):
            mujoco.mj_step(
                self.environment.model,
                self._rollout_data,
                nstep=self.environment.frame_skip,
            )
            collision, contacts = self.environment.detect_collision(
                self._rollout_data,
                allow_finger_table_contact=self.config.allow_finger_table_contact,
            )
            contact_count += contacts
            if collision:
                return True, contact_count
        return False, contact_count
