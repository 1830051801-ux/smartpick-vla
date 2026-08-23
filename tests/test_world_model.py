"""Tests for predictive Transformer dynamics and data contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from smartpick_vla.data.dataset import save_trajectory_npz
from smartpick_vla.models.world_model import WorldModelConfig, WorldModelTransformer
from smartpick_vla.training.world_model import (
    WorldModelSequenceDataset,
    WorldModelTrainingConfig,
    train_world_model,
)


def _trajectory(path: Path) -> None:
    episodes = 3
    steps = 4
    count = episodes * steps
    episode_id = np.repeat(np.arange(episodes, dtype=np.int32), steps)
    step_index = np.tile(np.arange(steps, dtype=np.int32), episodes)
    save_trajectory_npz(
        path,
        {
            "rgb": np.zeros((count, 24, 24, 3), dtype=np.uint8),
            "robot_state": np.zeros((count, 29), dtype=np.float32),
            "action": np.zeros((count, 6), dtype=np.float32),
            "episode_id": episode_id,
            "step_index": step_index,
            "instruction": np.asarray(["sort accepted"] * count),
            "task_class": np.asarray(["accepted"] * count),
            "episode_success": np.ones(count, dtype=bool),
            "reward": np.arange(count, dtype=np.float32),
            "terminated": np.asarray([False, False, False, True] * episodes, dtype=bool),
            "truncated": np.zeros(count, dtype=bool),
            "collision": np.zeros(count, dtype=bool),
            "wrong_pick": np.zeros(count, dtype=bool),
            "wrong_bin": np.zeros(count, dtype=bool),
        },
    )


def test_world_model_predicts_state_and_risk_rollout() -> None:
    model = WorldModelTransformer(
        WorldModelConfig(
            robot_state_dim=29,
            action_dim=6,
            observation_horizon=3,
            d_model=32,
            nhead=4,
            temporal_layers=1,
            language_layers=1,
            feedforward_dim=64,
            language_max_length=16,
            vision_grid_size=2,
        )
    )
    prediction = model(
        torch.zeros((2, 3, 3, 24, 24), dtype=torch.uint8),
        ["sort accepted", "inspect unknown"],
        torch.zeros((2, 3, 29)),
        torch.zeros((2, 3, 6)),
        torch.ones((2, 3), dtype=torch.bool),
    )
    assert prediction.next_robot_state.shape == (2, 29)
    assert prediction.next_robot_state_std.shape == (2, 29)
    assert bool((prediction.next_robot_state_std > 0.0).all())
    assert prediction.reward.shape == (2,)
    assert all(value.shape == (2,) for value in prediction.outcome_probabilities().values())
    rollout = model.rollout_risk(
        torch.zeros((1, 3, 3, 24, 24), dtype=torch.uint8),
        ["sort accepted"],
        torch.zeros((1, 3, 29)),
        torch.zeros((1, 3, 6)),
        torch.ones((1, 3), dtype=torch.bool),
        horizon=3,
        planned_actions=torch.zeros((1, 3, 6)),
    )
    assert len(rollout["steps"]) == 3
    assert "risk_reasons" in rollout
    assert all("max_state_std" in step for step in rollout["steps"])
    assert rollout["physical_robot_execution"] is False


def test_world_model_rejects_non_contiguous_history_mask() -> None:
    model = WorldModelTransformer(
        WorldModelConfig(
            robot_state_dim=29,
            action_dim=6,
            observation_horizon=3,
            d_model=32,
            nhead=4,
            temporal_layers=1,
            language_layers=1,
            feedforward_dim=64,
            language_max_length=16,
            vision_grid_size=2,
        )
    )
    with pytest.raises(ValueError, match="left-padded"):
        model(
            torch.zeros((1, 3, 3, 24, 24), dtype=torch.uint8),
            ["sort accepted"],
            torch.zeros((1, 3, 29)),
            torch.zeros((1, 3, 6)),
            torch.tensor([[True, False, True]]),
        )


def test_world_model_dataset_keeps_terminal_transition(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory.npz"
    _trajectory(dataset_path)
    dataset = WorldModelSequenceDataset(dataset_path, observation_horizon=3)
    # Three consecutive pairs plus the terminal row per episode.
    assert len(dataset) == 3 * 4
    terminal_samples = [
        dataset[index] for index in range(len(dataset)) if dataset[index]["terminated"]
    ]
    assert len(terminal_samples) == 3
    assert all(sample["next_robot_state"].shape == (29,) for sample in terminal_samples)


def test_world_model_training_writes_auditable_checkpoint(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory.npz"
    _trajectory(dataset_path)
    output = tmp_path / "world_model"
    manifest = train_world_model(
        dataset_path,
        output,
        training_config=WorldModelTrainingConfig(
            epochs=1,
            batch_size=4,
            validation_fraction=0.34,
            device="cpu",
            seed=55,
        ),
        model_options={
            "robot_state_dim": 29,
            "action_dim": 6,
            "observation_horizon": 3,
            "d_model": 32,
            "nhead": 4,
            "temporal_layers": 1,
            "language_layers": 1,
            "feedforward_dim": 64,
            "language_max_length": 16,
            "vision_grid_size": 2,
        },
    )
    assert manifest["best_epoch"] == 1
    assert manifest["physical_robot_execution"] is False
    assert (output / "best.pt").is_file()
    assert (output / "manifest.json").is_file()
