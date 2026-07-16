"""Privileged waypoint expert used only for demonstrations and upper bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from smartpick_vla.envs.smartpick_env import SmartPickEnv

ExpertStage = Literal[
    "pregrasp",
    "descend",
    "close",
    "lift",
    "transfer",
    "lower",
    "release",
    "retreat",
    "done",
]


@dataclass(slots=True)
class ExpertDiagnostics:
    stage: ExpertStage
    stage_step: int
    position_error_m: float


class IKWaypointExpert:
    """Closed-loop Cartesian waypoint expert with privileged object poses.

    The expert emits exactly the same normalized five-dimensional action as a
    learned policy. It never receives hidden state through the observation; it
    accesses environment poses explicitly and is therefore reported as a
    privileged IK upper-bound, not as a deployable vision policy.
    """

    def __init__(
        self,
        env: SmartPickEnv,
        *,
        use_noisy_detection: bool = False,
        waypoint_tolerance_m: float = 0.018,
    ) -> None:
        self.env = env
        self.use_noisy_detection = use_noisy_detection
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.stage: ExpertStage = "pregrasp"
        self.stage_step = 0
        self._object_xy = np.zeros(2, dtype=np.float64)
        self._bin_xy = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.stage = "pregrasp"
        self.stage_step = 0
        self._object_xy = self.env.target_position(noisy=self.use_noisy_detection)[:2]
        self._bin_xy = self.env.bin_position(self.env.task.target_class)[:2]

    def act(self) -> tuple[np.ndarray, ExpertDiagnostics]:
        if self.stage == "done":
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), self._diag(0.0)

        gripper_position = self.env.gripper_position()
        target, gripper_command, minimum_steps = self._stage_target()
        error = target - gripper_position
        distance = float(np.linalg.norm(error))
        action = np.zeros(5, dtype=np.float32)
        action[:3] = np.clip(error / self.env.max_translation_m, -1.0, 1.0)
        action[4] = gripper_command
        diagnostics = self._diag(distance)

        arrived = distance <= self.waypoint_tolerance_m
        grasped = self.env._grasped_object == self.env.target_object_index
        if self.stage == "close":
            arrived = grasped and self.stage_step >= minimum_steps
        elif self.stage == "release":
            arrived = self.env._grasped_object is None and self.stage_step >= minimum_steps

        self.stage_step += 1
        if arrived and self.stage_step >= minimum_steps:
            self._advance()
        return action, diagnostics

    def _stage_target(self) -> tuple[np.ndarray, float, int]:
        if self.stage == "pregrasp":
            return np.array([*self._object_xy, 0.19]), 1.0, 1
        if self.stage == "descend":
            # The open fingers stop above the object; the explicit proximity
            # latch is engaged only after the subsequent close command.
            return np.array([*self._object_xy, 0.100]), 1.0, 1
        if self.stage == "close":
            return np.array([*self._object_xy, 0.100]), -1.0, 3
        if self.stage == "lift":
            return np.array([*self._object_xy, 0.16]), -1.0, 1
        if self.stage == "transfer":
            return np.array([*self._bin_xy, 0.16]), -1.0, 1
        if self.stage == "lower":
            return np.array([*self._bin_xy, 0.105]), -1.0, 1
        if self.stage == "release":
            return np.array([*self._bin_xy, 0.105]), 1.0, 3
        if self.stage == "retreat":
            return np.array([*self._bin_xy, 0.16]), 1.0, 1
        return self.env.gripper_position(), 1.0, 1

    def _advance(self) -> None:
        transitions: dict[ExpertStage, ExpertStage] = {
            "pregrasp": "descend",
            "descend": "close",
            "close": "lift",
            "lift": "transfer",
            "transfer": "lower",
            "lower": "release",
            "release": "retreat",
            "retreat": "done",
            "done": "done",
        }
        self.stage = transitions[self.stage]
        self.stage_step = 0

    def _diag(self, distance: float) -> ExpertDiagnostics:
        return ExpertDiagnostics(
            stage=self.stage,
            stage_step=self.stage_step,
            position_error_m=distance,
        )
