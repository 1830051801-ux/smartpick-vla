"""Compact NPZ demonstration format and PyTorch action-chunk dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_ARRAYS = {
    "rgb",
    "robot_state",
    "action",
    "episode_id",
    "step_index",
    "instruction",
    "task_class",
    "episode_success",
}


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    transitions: int
    episodes: int
    successful_episodes: int
    image_shape: tuple[int, int, int]
    robot_state_dim: int
    action_dim: int


def validate_trajectory_arrays(arrays: dict[str, np.ndarray]) -> DatasetStatistics:
    """Validate aligned transitions before writing or training."""

    missing = REQUIRED_ARRAYS.difference(arrays)
    if missing:
        raise ValueError(f"trajectory arrays missing keys: {sorted(missing)}")
    transition_count = int(arrays["action"].shape[0])
    if transition_count < 1:
        raise ValueError("trajectory dataset is empty")
    for key in REQUIRED_ARRAYS:
        if arrays[key].shape[0] != transition_count:
            raise ValueError(f"array {key!r} has inconsistent leading dimension")
    if arrays["rgb"].ndim != 4 or arrays["rgb"].shape[-1] != 3:
        raise ValueError("rgb must have shape [N,H,W,3]")
    if arrays["rgb"].dtype != np.uint8:
        raise ValueError("rgb must use uint8 storage")
    if arrays["robot_state"].ndim != 2 or arrays["robot_state"].shape[1] < 1:
        raise ValueError("robot_state must have shape [N,D] with D >= 1")
    if arrays["action"].ndim != 2 or arrays["action"].shape[1] < 1:
        raise ValueError("action must have shape [N,A] with A >= 1")
    state_dim = int(arrays["robot_state"].shape[1])
    action_dim = int(arrays["action"].shape[1])
    if (state_dim, action_dim) not in {
        (24, 5),  # Legacy five-axis PickSort environment.
        (29, 6),  # Optional six-axis control variant.
    }:
        raise ValueError(
            "action must have shape [N,5] with robot_state [N,24], or "
            "action must have shape [N,6] with robot_state [N,29]"
        )
    if not np.isfinite(arrays["robot_state"]).all() or not np.isfinite(arrays["action"]).all():
        raise ValueError("state/action arrays contain NaN or infinity")
    if np.max(np.abs(arrays["action"])) > 1.00001:
        raise ValueError("normalized actions must be in [-1,1]")

    episode_ids = np.asarray(arrays["episode_id"], dtype=np.int64)
    unique_episodes = np.unique(episode_ids)
    success_by_episode = {
        int(episode_id): bool(np.asarray(arrays["episode_success"])[episode_ids == episode_id][0])
        for episode_id in unique_episodes
    }
    return DatasetStatistics(
        transitions=transition_count,
        episodes=len(unique_episodes),
        successful_episodes=sum(success_by_episode.values()),
        image_shape=(
            int(arrays["rgb"].shape[1]),
            int(arrays["rgb"].shape[2]),
            int(arrays["rgb"].shape[3]),
        ),
        robot_state_dim=state_dim,
        action_dim=action_dim,
    )


def save_trajectory_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Validate and atomically replace a compressed trajectory archive."""

    validate_trajectory_arrays(arrays)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
    temporary.replace(destination)
    return destination


def load_trajectory_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_trajectory_arrays(arrays)
    return arrays


class TrajectoryDataset(Dataset[dict[str, Any]]):
    """Return BC actions or padded, episode-safe action chunks."""

    def __init__(
        self,
        path: str | Path,
        *,
        action_horizon: int = 1,
        observation_horizon: int = 1,
        successful_only: bool = True,
    ) -> None:
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if observation_horizon < 1:
            raise ValueError("observation_horizon must be positive")
        self.path = Path(path)
        self.arrays = load_trajectory_npz(self.path)
        self.action_horizon = action_horizon
        self.observation_horizon = observation_horizon
        if successful_only:
            self.indices = np.flatnonzero(self.arrays["episode_success"].astype(bool))
        else:
            self.indices = np.arange(self.arrays["action"].shape[0])
        if self.indices.size == 0:
            raise ValueError("no transitions remain after dataset filtering")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        transition_index = int(self.indices[item])
        rgb = torch.from_numpy(self.arrays["rgb"][transition_index].copy()).permute(2, 0, 1)
        state = torch.from_numpy(
            self.arrays["robot_state"][transition_index].astype(np.float32, copy=True)
        )
        sample: dict[str, Any] = {
            "rgb": rgb,
            "robot_state": state,
            "instruction": str(self.arrays["instruction"][transition_index]),
            "episode_id": int(self.arrays["episode_id"][transition_index]),
            "step_index": int(self.arrays["step_index"][transition_index]),
        }
        if self.observation_horizon > 1:
            sample.update(self._observation_history(transition_index))
        if self.action_horizon == 1:
            sample["action"] = torch.from_numpy(
                self.arrays["action"][transition_index].astype(np.float32, copy=True)
            )
            return sample

        actions = np.zeros((self.action_horizon, self.arrays["action"].shape[1]), dtype=np.float32)
        mask = np.zeros(self.action_horizon, dtype=bool)
        episode_id = self.arrays["episode_id"][transition_index]
        for offset in range(self.action_horizon):
            future_index = transition_index + offset
            if future_index >= self.arrays["action"].shape[0]:
                break
            if self.arrays["episode_id"][future_index] != episode_id:
                break
            actions[offset] = self.arrays["action"][future_index]
            mask[offset] = True
        sample["action"] = torch.from_numpy(actions)
        sample["action_mask"] = torch.from_numpy(mask)
        return sample

    def _observation_history(self, transition_index: int) -> dict[str, torch.Tensor]:
        """Return episode-safe left-padded visual/proprioceptive context."""

        image_shape = self.arrays["rgb"].shape[1:]
        rgb_history = np.zeros((self.observation_horizon, *image_shape), dtype=np.uint8)
        state_history = np.zeros(
            (self.observation_horizon, self.arrays["robot_state"].shape[1]), dtype=np.float32
        )
        history_mask = np.zeros(self.observation_horizon, dtype=bool)
        episode_id = self.arrays["episode_id"][transition_index]
        for destination_index in range(self.observation_horizon):
            source_index = transition_index - (self.observation_horizon - 1 - destination_index)
            if source_index < 0 or self.arrays["episode_id"][source_index] != episode_id:
                continue
            rgb_history[destination_index] = self.arrays["rgb"][source_index]
            state_history[destination_index] = self.arrays["robot_state"][source_index]
            history_mask[destination_index] = True
        if not history_mask[-1]:
            raise RuntimeError("current observation must be present in its temporal history")
        return {
            "rgb_history": torch.from_numpy(rgb_history).permute(0, 3, 1, 2),
            "robot_state_history": torch.from_numpy(state_history),
            "history_mask": torch.from_numpy(history_mask),
        }
