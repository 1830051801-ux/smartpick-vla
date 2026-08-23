"""Training and evaluation for the compact latent world-model transformer."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from smartpick_vla.data.dataset import load_trajectory_npz
from smartpick_vla.models.world_model import (
    WorldModelConfig,
    WorldModelPrediction,
    WorldModelTransformer,
)
from smartpick_vla.training.checkpoint import load_checkpoint_payload, save_checkpoint
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import runtime_snapshot, sha256_file
from smartpick_vla.utils.seed import seed_everything


@dataclass(frozen=True, slots=True)
class WorldModelTrainingConfig:
    """Reproducible training settings for residual dynamics and risk heads."""

    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    seed: int = 20260822
    device: str = "auto"
    num_workers: int = 0
    gradient_clip_norm: float = 1.0
    state_loss_weight: float = 1.0
    reward_loss_weight: float = 0.25
    event_loss_weight: float = 0.25
    uncertainty_loss_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("training and loader settings must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings are invalid")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0,0.5)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if (
            min(
                self.state_loss_weight,
                self.reward_loss_weight,
                self.event_loss_weight,
                self.uncertainty_loss_weight,
            )
            <= 0.0
        ):
            raise ValueError("loss weights must be positive")


class WorldModelSequenceDataset(Dataset[dict[str, Any]]):
    """Episode-safe windows for one-step latent dynamics training."""

    def __init__(self, path: str | Path, *, observation_horizon: int = 4) -> None:
        if observation_horizon < 1:
            raise ValueError("observation_horizon must be positive")
        self.path = Path(path)
        self.arrays = load_trajectory_npz(self.path)
        self.observation_horizon = observation_horizon
        episode_ids = np.asarray(self.arrays["episode_id"], dtype=np.int64)
        step_indices = np.asarray(self.arrays["step_index"], dtype=np.int64)
        terminal = np.asarray(
            self.arrays.get("terminated", np.zeros(len(episode_ids), dtype=bool)), dtype=bool
        )
        truncated = np.asarray(
            self.arrays.get("truncated", np.zeros(len(episode_ids), dtype=bool)), dtype=bool
        )
        self.valid_indices: list[int] = []
        self.next_indices: dict[int, int] = {}
        for index in range(len(episode_ids)):
            has_next = index + 1 < len(episode_ids)
            same_episode = has_next and episode_ids[index + 1] == episode_ids[index]
            consecutive = has_next and step_indices[index + 1] == step_indices[index] + 1
            if same_episode and consecutive:
                next_index = index + 1
            elif terminal[index] or truncated[index]:
                # Preserve the terminal transition for event-risk learning. A
                # terminal state is its own stable target because no successor
                # observation exists in the archive.
                next_index = index
            else:
                continue
            self.valid_indices.append(index)
            self.next_indices[index] = next_index
        if not self.valid_indices:
            raise ValueError("trajectory archive contains no consecutive transition pairs")
        self.episode_ids = episode_ids

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        transition_index = self.valid_indices[item]
        next_index = self.next_indices[transition_index]
        episode_id = self.episode_ids[transition_index]
        rgb_shape = self.arrays["rgb"].shape[1:]
        state_dim = self.arrays["robot_state"].shape[1]
        action_dim = self.arrays["action"].shape[1]
        rgb_history = np.zeros((self.observation_horizon, *rgb_shape), dtype=np.uint8)
        state_history = np.zeros((self.observation_horizon, state_dim), dtype=np.float32)
        action_history = np.zeros((self.observation_horizon, action_dim), dtype=np.float32)
        history_mask = np.zeros(self.observation_horizon, dtype=bool)
        for destination in range(self.observation_horizon):
            source = transition_index - (self.observation_horizon - 1 - destination)
            if source < 0 or self.episode_ids[source] != episode_id:
                continue
            rgb_history[destination] = self.arrays["rgb"][source]
            state_history[destination] = self.arrays["robot_state"][source]
            action_history[destination] = self.arrays["action"][source]
            history_mask[destination] = True
        sample: dict[str, Any] = {
            "rgb_history": torch.from_numpy(rgb_history).permute(0, 3, 1, 2),
            "robot_state_history": torch.from_numpy(state_history),
            "action_history": torch.from_numpy(action_history),
            "history_mask": torch.from_numpy(history_mask),
            "instruction": str(self.arrays["instruction"][transition_index]),
            "next_robot_state": torch.from_numpy(
                self.arrays["robot_state"][next_index].astype(np.float32, copy=True)
            ),
            "reward": torch.tensor(
                float(
                    self.arrays.get(
                        "reward", np.zeros(self.arrays["action"].shape[0], dtype=np.float32)
                    )[transition_index]
                ),
                dtype=torch.float32,
            ),
            "terminated": torch.tensor(
                float(
                    self.arrays.get(
                        "terminated", np.zeros(self.arrays["action"].shape[0], dtype=bool)
                    )[transition_index]
                    or self.episode_ids[next_index] != episode_id
                ),
                dtype=torch.float32,
            ),
            "truncated": torch.tensor(
                float(
                    self.arrays.get(
                        "truncated", np.zeros(self.arrays["action"].shape[0], dtype=bool)
                    )[transition_index]
                ),
                dtype=torch.float32,
            ),
            "collision": torch.tensor(
                float(
                    self.arrays.get(
                        "collision", np.zeros(self.arrays["action"].shape[0], dtype=bool)
                    )[transition_index]
                ),
                dtype=torch.float32,
            ),
            "wrong_pick": torch.tensor(
                float(
                    self.arrays.get(
                        "wrong_pick", np.zeros(self.arrays["action"].shape[0], dtype=bool)
                    )[transition_index]
                ),
                dtype=torch.float32,
            ),
            "wrong_bin": torch.tensor(
                float(
                    self.arrays.get(
                        "wrong_bin", np.zeros(self.arrays["action"].shape[0], dtype=bool)
                    )[transition_index]
                ),
                dtype=torch.float32,
            ),
            "episode_id": int(episode_id),
        }
        return sample

    def episode_ids_for_positions(self, positions: list[int]) -> np.ndarray:
        return np.asarray(
            [self.episode_ids[self.valid_indices[position]] for position in positions]
        )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def world_model_loss(
    prediction: WorldModelPrediction,
    batch: dict[str, Any],
    *,
    state_weight: float = 1.0,
    reward_weight: float = 0.25,
    event_weight: float = 0.25,
    uncertainty_weight: float = 0.10,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute residual dynamics, uncertainty, and event-risk supervision.

    Huber remains the primary state objective for stable local training.  The
    auxiliary scale objective teaches the model to expose larger uncertainty
    on transitions with larger residual error.  This is a rollout diagnostic,
    not a formal probabilistic calibration claim.
    """

    target_state = batch["next_robot_state"]
    state_loss = functional.smooth_l1_loss(prediction.next_robot_state, target_state)
    target_scale = (
        (prediction.next_robot_state.detach() - target_state).abs().clamp_min(0.02).clamp_max(5.0)
    )
    uncertainty_loss = functional.smooth_l1_loss(
        prediction.next_robot_state_std,
        target_scale,
    )
    reward_loss = functional.smooth_l1_loss(prediction.reward, batch["reward"])
    event_targets = torch.stack(
        (
            batch["terminated"],
            batch["truncated"],
            batch["collision"],
            batch["wrong_pick"],
            batch["wrong_bin"],
        ),
        dim=1,
    )
    event_logits = torch.stack(
        (
            prediction.terminated_logit,
            prediction.truncated_logit,
            prediction.collision_logit,
            prediction.wrong_pick_logit,
            prediction.wrong_bin_logit,
        ),
        dim=1,
    )
    event_loss = functional.binary_cross_entropy_with_logits(event_logits, event_targets)
    total = (
        state_weight * state_loss
        + uncertainty_weight * uncertainty_loss
        + reward_weight * reward_loss
        + event_weight * event_loss
    )
    return total, {
        "state_loss": state_loss,
        "uncertainty_loss": uncertainty_loss,
        "reward_loss": reward_loss,
        "event_loss": event_loss,
    }


def _forward_batch(
    model: WorldModelTransformer, batch: dict[str, Any], device: torch.device
) -> WorldModelPrediction:
    return model(
        batch["rgb_history"].to(device),
        batch["instruction"],
        batch["robot_state_history"].to(device=device, dtype=torch.float32),
        batch["action_history"].to(device=device, dtype=torch.float32),
        batch["history_mask"].to(device),
    )


def _loader_metrics(
    model: WorldModelTransformer,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    config: WorldModelTrainingConfig,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "state_loss": 0.0,
        "uncertainty_loss": 0.0,
        "reward_loss": 0.0,
        "event_loss": 0.0,
    }
    state_squared = 0.0
    state_values = 0
    state_covered = 0
    state_std_total = 0.0
    event_correct = 0
    event_values = 0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            prediction = _forward_batch(model, batch, device)
            loss, components = world_model_loss(
                prediction,
                {
                    key: value.to(device) if isinstance(value, Tensor) else value
                    for key, value in batch.items()
                },
                state_weight=config.state_loss_weight,
                reward_weight=config.reward_loss_weight,
                event_weight=config.event_loss_weight,
                uncertainty_weight=config.uncertainty_loss_weight,
            )
            batch_size = int(batch["next_robot_state"].shape[0])
            samples += batch_size
            totals["loss"] += float(loss.cpu()) * batch_size
            for name, value in components.items():
                totals[name] += float(value.cpu()) * batch_size
            error = prediction.next_robot_state - batch["next_robot_state"].to(device)
            state_squared += float((error * error).sum().cpu())
            state_values += int(error.numel())
            state_std = prediction.next_robot_state_std
            state_covered += int((error.abs() <= 2.0 * state_std).sum().cpu())
            state_std_total += float(state_std.sum().cpu())
            probabilities = prediction.outcome_probabilities()
            predicted_events = torch.stack(
                tuple(
                    (probabilities[name] >= 0.5)
                    for name in ("terminated", "truncated", "collision", "wrong_pick", "wrong_bin")
                ),
                dim=1,
            )
            target_events = torch.stack(
                tuple(
                    batch[name].to(device) >= 0.5
                    for name in ("terminated", "truncated", "collision", "wrong_pick", "wrong_bin")
                ),
                dim=1,
            )
            event_correct += int((predicted_events == target_events).sum().cpu())
            event_values += int(target_events.numel())
    if samples == 0 or state_values == 0:
        raise RuntimeError("world-model loader is empty")
    return {
        **{name: value / samples for name, value in totals.items()},
        "next_state_rmse": float(np.sqrt(state_squared / state_values)),
        "mean_state_std": state_std_total / state_values,
        "state_coverage_2sigma": state_covered / max(1, state_values),
        "event_accuracy": event_correct / max(1, event_values),
    }


def train_world_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    training_config: WorldModelTrainingConfig,
    model_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a world model with an episode-disjoint validation split."""

    seed_everything(training_config.seed)
    model_config = WorldModelConfig(**dict(model_options or {}))
    dataset = WorldModelSequenceDataset(
        dataset_path, observation_horizon=model_config.observation_horizon
    )
    unique_episodes = np.unique(dataset.episode_ids[dataset.valid_indices])
    if unique_episodes.size < 2:
        raise ValueError("at least two episodes are required for world-model validation")
    shuffled = np.random.default_rng(training_config.seed).permutation(unique_episodes)
    validation_count = max(1, round(unique_episodes.size * training_config.validation_fraction))
    validation_episodes = set(int(value) for value in shuffled[:validation_count])
    training_episodes = set(int(value) for value in shuffled[validation_count:])
    train_positions = [
        position
        for position in range(len(dataset))
        if int(dataset.episode_ids[dataset.valid_indices[position]]) in training_episodes
    ]
    validation_positions = [
        position
        for position in range(len(dataset))
        if int(dataset.episode_ids[dataset.valid_indices[position]]) in validation_episodes
    ]
    if not train_positions or not validation_positions:
        raise RuntimeError("episode split produced an empty world-model partition")
    device = _resolve_device(training_config.device)
    model = WorldModelTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        Subset(dataset, train_positions),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=training_config.num_workers,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_positions),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_epoch = 0
    global_step = 0
    for epoch in range(1, training_config.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_total = 0.0
        train_samples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward_batch(model, batch, device)
            moved = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
            loss, _ = world_model_loss(
                prediction,
                moved,
                state_weight=training_config.state_loss_weight,
                reward_weight=training_config.reward_loss_weight,
                event_weight=training_config.event_loss_weight,
                uncertainty_weight=training_config.uncertainty_loss_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("world-model loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            batch_size = int(batch["next_robot_state"].shape[0])
            train_total += float(loss.detach().cpu()) * batch_size
            train_samples += batch_size
            global_step += 1
        validation = _loader_metrics(
            model, validation_loader, device=device, config=training_config
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_total / train_samples,
            "validation_loss": validation["loss"],
            "validation_next_state_rmse": validation["next_state_rmse"],
            "validation_mean_state_std": validation["mean_state_std"],
            "validation_state_coverage_2sigma": validation["state_coverage_2sigma"],
            "validation_event_accuracy": validation["event_accuracy"],
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        common_extra = {
            "model_kind": "world_model_transformer",
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "dataset": str(Path(dataset_path)),
            "dataset_sha256": sha256_file(dataset_path),
            "training_episodes": sorted(training_episodes),
            "validation_episodes": sorted(validation_episodes),
            "history": history,
            "prediction_scope": "one_step_residual_dynamics_event_risk_and_uncertainty",
            "physical_robot_execution": False,
        }
        save_checkpoint(
            destination / "last.pt",
            model,
            optimizers=optimizer,
            step=global_step,
            config=model_config,
            extra=common_extra,
        )
        if validation["next_state_rmse"] < best_rmse:
            best_rmse = validation["next_state_rmse"]
            best_epoch = epoch
            save_checkpoint(
                destination / "best.pt",
                model,
                optimizers=optimizer,
                step=global_step,
                config=model_config,
                extra=common_extra,
            )
    with (destination / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    manifest = {
        "schema_version": "picksort-world-model-training/v2",
        "task": "residual_latent_dynamics_event_prediction_with_uncertainty",
        "dataset": {"path": str(Path(dataset_path)), "sha256": sha256_file(dataset_path)},
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "device": str(device),
        "runtime": runtime_snapshot(),
        "total_parameters": model.parameter_count(),
        "trainable_parameters": model.parameter_count(trainable_only=True),
        "training_episodes": sorted(training_episodes),
        "validation_episodes": sorted(validation_episodes),
        "valid_transition_pairs": len(dataset),
        "best_epoch": best_epoch,
        "best_validation_next_state_rmse": best_rmse,
        "best_validation_mean_state_std": history[best_epoch - 1]["validation_mean_state_std"],
        "best_validation_state_coverage_2sigma": history[best_epoch - 1][
            "validation_state_coverage_2sigma"
        ],
        "global_steps": global_step,
        "history": history,
        "physical_robot_execution": False,
        "checkpoints": {
            "best": {"path": "best.pt", "sha256": sha256_file(destination / "best.pt")},
            "last": {"path": "last.pt", "sha256": sha256_file(destination / "last.pt")},
        },
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def load_world_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[WorldModelTransformer, dict[str, Any]]:
    """Load a trusted world-model checkpoint and its construction metadata."""

    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    extra = payload.get("extra", {})
    model_config = extra.get("model_config")
    if extra.get("model_kind") != "world_model_transformer" or not isinstance(model_config, dict):
        raise ValueError("checkpoint lacks world-model construction metadata")
    model = WorldModelTransformer(WorldModelConfig(**model_config))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, dict(extra)


def evaluate_world_model(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate one-step prediction on a declared archive and write evidence."""

    model, metadata = load_world_model(checkpoint_path, device=device)
    dataset = WorldModelSequenceDataset(
        dataset_path, observation_horizon=model.config.observation_horizon
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    training_config = WorldModelTrainingConfig(device=str(device), epochs=1)
    metrics = _loader_metrics(model, loader, device=torch.device(device), config=training_config)
    report = {
        "schema_version": "picksort-world-model-evaluation/v2",
        "checkpoint": {"path": str(Path(checkpoint_path)), "sha256": sha256_file(checkpoint_path)},
        "dataset": {"path": str(Path(dataset_path)), "sha256": sha256_file(dataset_path)},
        "model_metadata": metadata,
        "samples": len(dataset),
        "metrics": metrics,
        "scope": "one_step_residual_prediction_and_uncertainty; not a physical-robot success metric",
        "physical_robot_execution": False,
        "runtime": runtime_snapshot(),
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


__all__ = [
    "WorldModelSequenceDataset",
    "WorldModelTrainingConfig",
    "evaluate_world_model",
    "load_world_model",
    "train_world_model",
    "world_model_loss",
]
