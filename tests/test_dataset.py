"""Demonstration storage and episode-safe chunking tests."""

from pathlib import Path

import numpy as np
import pytest

from smartpick_vla.data.dataset import TrajectoryDataset, load_trajectory_npz
from smartpick_vla.data.generate import GenerationConfig, generate_expert_dataset

pytestmark = pytest.mark.mujoco


def test_generated_dataset_retains_attempt_accounting_and_chunks(tmp_path: Path) -> None:
    destination = tmp_path / "tiny_demo.npz"
    manifest = generate_expert_dataset(
        destination,
        config=GenerationConfig(
            episodes=2,
            seed=31,
            image_size=32,
            max_episode_steps=180,
        ),
    )
    arrays = load_trajectory_npz(destination)
    dataset = TrajectoryDataset(destination, action_horizon=8)
    final_index_of_first = int(np.flatnonzero(arrays["episode_id"] == 0)[-1])
    sample = dataset[final_index_of_first]

    assert manifest["attempted_episodes"] == 2
    assert manifest["successful_episodes"] == 2
    assert arrays["rgb"].dtype == np.uint8
    assert sample["action"].shape == (8, 5)
    assert sample["action_mask"].sum() == 1
    assert not sample["action_mask"][1:].any()


def test_dataset_validation_rejects_cross_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        rgb=np.zeros((2, 16, 16, 3), dtype=np.uint8),
        robot_state=np.zeros((2, 24), dtype=np.float32),
        action=np.zeros((2, 4), dtype=np.float32),
        episode_id=np.zeros(2, dtype=np.int32),
        step_index=np.arange(2, dtype=np.int32),
        instruction=np.asarray(["a", "a"]),
        task_class=np.asarray(["accepted", "accepted"]),
        episode_success=np.ones(2, dtype=bool),
    )
    try:
        TrajectoryDataset(path)
    except ValueError as error:
        assert "action must have shape" in str(error)
    else:
        raise AssertionError("invalid dataset was accepted")
