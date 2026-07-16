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
    if arrays["robot_state"].shape[1:] != (24,):
        raise ValueError("robot_state must have shape [N,24]")
    if arrays["action"].shape[1:] != (5,):
        raise ValueError("action must have shape [N,5]")
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
        successful_only: bool = True,
    ) -> None:
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.path = Path(path)
        self.arrays = load_trajectory_npz(self.path)
        self.action_horizon = action_horizon
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
        if self.action_horizon == 1:
            sample["action"] = torch.from_numpy(
                self.arrays["action"][transition_index].astype(np.float32, copy=True)
            )
            return sample

        actions = np.zeros((self.action_horizon, 5), dtype=np.float32)
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
