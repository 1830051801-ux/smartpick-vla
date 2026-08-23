"""Optional ROS-independent world-model risk monitor.

The monitor consumes the latest camera frame, language instruction, robot
feedback, and paired action chunk. It only returns a prediction object; the
ROS node decides whether that prediction is advisory or a configurable dry-run
block. No transport, serial port, CAN channel, or motor API is opened here.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from smartpick_vla.real import ActionChunk, RobotState
from smartpick_vla.training.world_model import load_world_model
from smartpick_vla.utils.provenance import sha256_file
from smartpick_vla_ros2.core import PredictiveRisk


class PredictiveRiskMonitor:
    """Maintain an episode-safe history and query a six-axis world model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        horizon: int = 4,
        collision_threshold: float = 0.65,
        wrong_pick_threshold: float = 0.75,
        wrong_bin_threshold: float = 0.75,
        termination_threshold: float = 0.95,
        uncertainty_threshold: float = 0.75,
        image_size: int = 64,
    ) -> None:
        thresholds = (
            collision_threshold,
            wrong_pick_threshold,
            wrong_bin_threshold,
            termination_threshold,
        )
        if horizon < 1 or any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("predictive horizon/risk thresholds are invalid")
        if uncertainty_threshold <= 0.0:
            raise ValueError("uncertainty_threshold must be positive")
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.model, metadata = load_world_model(self.checkpoint_path, device=self.device)
        if self.model.config.robot_state_dim != 29 or self.model.config.action_dim != 6:
            raise ValueError("predictive ROS monitor requires the six-axis world-model contract")
        self.history_horizon = self.model.config.observation_horizon
        self.horizon = horizon
        self.collision_threshold = collision_threshold
        self.wrong_pick_threshold = wrong_pick_threshold
        self.wrong_bin_threshold = wrong_bin_threshold
        self.termination_threshold = termination_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.image_size = image_size
        self.model_id = (
            f"{metadata.get('model_kind', 'world_model_transformer')}"
            f"@{sha256_file(self.checkpoint_path)[:12]}"
        )
        self._rgb: deque[np.ndarray] = deque(maxlen=self.history_horizon)
        self._states: deque[np.ndarray] = deque(maxlen=self.history_horizon)
        self._actions: deque[np.ndarray] = deque(maxlen=self.history_horizon)

    @property
    def available(self) -> bool:
        return True

    def reset(self) -> None:
        self._rgb.clear()
        self._states.clear()
        self._actions.clear()

    @staticmethod
    def _state_vector(state: RobotState) -> np.ndarray:
        """Map the ROS feedback subset into the six-axis simulator state shape.

        The physical profile may provide fewer joint values during bring-up;
        refusing that input is safer than silently inventing a joint mapping.
        Velocities are zero-filled only because the current ROS state message
        does not carry velocity feedback; the report remains preview-only.
        """

        if len(state.joint_positions_rad) < 6:
            raise ValueError("six-axis predictive preview requires six joint positions")
        values = np.asarray(
            (*state.joint_positions_rad[:6], state.gripper, state.gripper),
            dtype=np.float32,
        )
        if (
            not np.isfinite(values).all()
            or not np.isfinite(state.tcp_position_m).all()
            or not np.isfinite(state.tcp_yaw_rad)
        ):
            raise ValueError("robot state contains NaN or infinity")
        qpos = values
        qvel = np.zeros(8, dtype=np.float32)
        yaw = float(state.tcp_yaw_rad)
        orientation = np.asarray((np.sin(yaw), np.cos(yaw), 0.0, 1.0), dtype=np.float32)
        return np.concatenate(
            (qpos, qvel, np.asarray(state.tcp_position_m, dtype=np.float32), orientation)
        )

    @staticmethod
    def _action_vector(chunk: ActionChunk) -> np.ndarray:
        action = np.asarray(chunk.actions[0].as_array(dtype=np.float32), dtype=np.float32)
        # The ROS safety contract remains five-channel for compatibility.  The
        # six-axis model inserts the absent tool-roll command before gripper.
        return PredictiveRiskMonitor._action_with_roll_zero(action)

    @classmethod
    def _action_plan(cls, chunk: ActionChunk, horizon: int) -> np.ndarray:
        values = np.stack(
            [
                cls._action_with_roll_zero(action.as_array(dtype=np.float32))
                for action in chunk.actions
            ]
        )
        if values.shape[0] < horizon:
            values = np.concatenate(
                (values, np.repeat(values[-1:], horizon - values.shape[0], axis=0)),
                axis=0,
            )
        return values

    @staticmethod
    def _action_with_roll_zero(action: np.ndarray) -> np.ndarray:
        if action.shape != (5,):
            raise ValueError("ROS action must have shape (5,)")
        return np.concatenate((action[:4], np.zeros(1, dtype=np.float32), action[4:]))

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Convert variable ROS camera sizes to the checkpoint input size."""

        if image.shape[:2] == (self.image_size, self.image_size):
            return image.copy()
        y = np.linspace(0, image.shape[0] - 1, self.image_size).round().astype(np.int64)
        x = np.linspace(0, image.shape[1] - 1, self.image_size).round().astype(np.int64)
        return image[np.ix_(y, x)].copy()

    def update(
        self,
        rgb: np.ndarray,
        instruction: str,
        state: RobotState,
        chunk: ActionChunk,
    ) -> PredictiveRisk:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("predictive RGB frame must have shape [H,W,3]")
        image = self._prepare_image(image)
        action_plan = self._action_plan(chunk, self.horizon)
        action_vector = action_plan[0]
        # The simulator state ends with the previous normalized six-channel
        # action. At the ROS boundary the current command is the only available
        # compatible action history, so it is recorded explicitly in that slot.
        state_vector = np.concatenate((self._state_vector(state), action_vector))
        self._rgb.append(image.copy())
        self._states.append(state_vector)
        self._actions.append(action_vector)
        rgb_history = np.zeros((self.history_horizon, *image.shape), dtype=np.uint8)
        state_history = np.zeros((self.history_horizon, 29), dtype=np.float32)
        action_history = np.zeros((self.history_horizon, 6), dtype=np.float32)
        mask = np.zeros(self.history_horizon, dtype=bool)
        offset = self.history_horizon - len(self._rgb)
        for index, (frame, state_row, action_row) in enumerate(
            zip(self._rgb, self._states, self._actions, strict=True), start=offset
        ):
            rgb_history[index] = frame
            state_history[index] = state_row
            action_history[index] = action_row
            mask[index] = True
        with torch.no_grad():
            result = self.model.rollout_risk(
                torch.from_numpy(rgb_history).permute(0, 3, 1, 2).unsqueeze(0),
                [instruction],
                torch.from_numpy(state_history).unsqueeze(0),
                torch.from_numpy(action_history).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
                horizon=self.horizon,
                collision_threshold=self.collision_threshold,
                wrong_pick_threshold=self.wrong_pick_threshold,
                wrong_bin_threshold=self.wrong_bin_threshold,
                termination_threshold=self.termination_threshold,
                uncertainty_threshold=self.uncertainty_threshold,
                planned_actions=torch.from_numpy(action_plan).unsqueeze(0),
            )
        steps = result["steps"]
        collision = max(float(step["collision_probability"]) for step in steps)
        termination = max(float(step["terminated_probability"]) for step in steps)
        wrong_pick = max(float(step["wrong_pick_probability"]) for step in steps)
        wrong_bin = max(float(step["wrong_bin_probability"]) for step in steps)
        max_state_std = max(float(step["max_state_std"]) for step in steps)
        return PredictiveRisk(
            collision_probability=collision,
            termination_probability=termination,
            wrong_bin_probability=wrong_bin,
            horizon=self.horizon,
            blocked=bool(result["unsafe"]),
            model_id=self.model_id,
            wrong_pick_probability=wrong_pick,
            max_state_std=max_state_std,
            risk_reasons=tuple(str(reason) for reason in result["risk_reasons"]),
        )


__all__ = ["PredictiveRiskMonitor"]
