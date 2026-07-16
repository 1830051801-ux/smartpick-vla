"""Episode-split supervised training for BC and Compact-VLA policies."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from smartpick_vla.data.dataset import TrajectoryDataset
from smartpick_vla.models import (
    BehaviorCloningConfig,
    BehaviorCloningPolicy,
    CompactVLAConfig,
    CompactVLAPolicy,
)
from smartpick_vla.training.checkpoint import load_checkpoint_payload, save_checkpoint
from smartpick_vla.training.supervised import action_imitation_loss, supervised_train_step
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import sha256_file
from smartpick_vla.utils.seed import seed_everything

PolicyKind = Literal["bc", "compact_vla"]


@dataclass(frozen=True, slots=True)
class ImitationTrainingConfig:
    policy: PolicyKind
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    seed: int = 17
    device: str = "auto"
    num_workers: int = 0
    gradient_clip_norm: float = 1.0
    loss_kind: Literal["mse", "smooth_l1"] = "smooth_l1"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the installed PyTorch build cannot use it")
    return device


def _build_policy(
    policy: PolicyKind,
    model_options: dict[str, Any] | None,
) -> tuple[nn.Module, BehaviorCloningConfig | CompactVLAConfig]:
    options = dict(model_options or {})
    if policy == "bc":
        bc_config = BehaviorCloningConfig(**options)
        return BehaviorCloningPolicy(bc_config), bc_config
    vla_config = CompactVLAConfig(**options)
    return CompactVLAPolicy(vla_config), vla_config


def _episode_split(
    dataset: TrajectoryDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    episode_ids = dataset.arrays["episode_id"][dataset.indices]
    unique = np.unique(episode_ids)
    if unique.size < 2:
        raise ValueError("at least two successful episodes are required for train/validation split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    validation_count = max(1, round(unique.size * validation_fraction))
    validation_episodes = shuffled[:validation_count]
    training_episodes = shuffled[validation_count:]
    training_indices = np.flatnonzero(np.isin(episode_ids, training_episodes)).tolist()
    validation_indices = np.flatnonzero(np.isin(episode_ids, validation_episodes)).tolist()
    return (
        training_indices,
        validation_indices,
        training_episodes.astype(int).tolist(),
        validation_episodes.astype(int).tolist(),
    )


def train_imitation(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    training_config: ImitationTrainingConfig,
    model_options: dict[str, Any] | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train on successful expert episodes and retain the episode split."""

    seed_everything(training_config.seed)
    model, model_config = _build_policy(training_config.policy, model_options)
    if initial_checkpoint is not None:
        initial_payload = load_checkpoint_payload(initial_checkpoint)
        if "model_state" not in initial_payload:
            raise ValueError("initial checkpoint is malformed")
        initial_extra = initial_payload.get("extra", {})
        if initial_extra.get("policy_kind") != training_config.policy:
            raise ValueError("initial checkpoint policy kind does not match training config")
        model.load_state_dict(initial_payload["model_state"])
    if training_config.policy == "bc":
        action_horizon = 1
    else:
        if not isinstance(model_config, CompactVLAConfig):
            raise TypeError("compact_vla policy requires CompactVLAConfig")
        action_horizon = model_config.action_horizon
    dataset = TrajectoryDataset(
        dataset_path,
        action_horizon=action_horizon,
        successful_only=True,
    )
    train_indices, validation_indices, train_episodes, validation_episodes = _episode_split(
        dataset,
        validation_fraction=training_config.validation_fraction,
        seed=training_config.seed,
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=training_config.num_workers,
        drop_last=False,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    device = _resolve_device(training_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    global_step = 0

    for epoch in range(1, training_config.epochs + 1):
        started = time.perf_counter()
        train_losses: list[float] = []
        gradient_norms: list[float] = []
        for batch in train_loader:
            result = supervised_train_step(
                model,
                optimizer,
                batch,
                gradient_clip_norm=training_config.gradient_clip_norm,
                loss_kind=training_config.loss_kind,
            )
            train_losses.append(result.loss)
            gradient_norms.append(result.gradient_norm)
            global_step += 1
        validation_loss = _validation_loss(
            model, validation_loader, device=device, loss_kind=training_config.loss_kind
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        common_extra = {
            "policy_kind": training_config.policy,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "dataset": str(Path(dataset_path)),
            "initial_checkpoint": (
                None if initial_checkpoint is None else str(Path(initial_checkpoint))
            ),
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
            "history": history,
        }
        save_checkpoint(
            destination / "last.pt",
            model,
            optimizers=optimizer,
            step=global_step,
            config=model_config,
            extra=common_extra,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
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
        "schema_version": "smartpick-training/v1",
        "policy_kind": training_config.policy,
        "dataset": {
            "path": str(Path(dataset_path)),
            "sha256": sha256_file(dataset_path),
        },
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "initial_checkpoint": (
            None
            if initial_checkpoint is None
            else {
                "path": str(Path(initial_checkpoint)),
                "sha256": sha256_file(initial_checkpoint),
            }
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "training_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "global_steps": global_step,
        "history": history,
        "checkpoints": {
            "best": {
                "path": "best.pt",
                "sha256": sha256_file(destination / "best.pt"),
            },
            "last": {
                "path": "last.pt",
                "sha256": sha256_file(destination / "last.pt"),
            },
        },
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def _validation_loss(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    loss_kind: Literal["mse", "smooth_l1"],
) -> float:
    model.eval()
    losses: list[float] = []
    weights: list[int] = []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        state = batch["robot_state"].to(device)
        target = batch["action"].to(device)
        prediction = model(rgb, batch["instruction"], state)
        mask = batch.get("action_mask")
        if mask is not None:
            mask = mask.to(device)
        loss = action_imitation_loss(prediction, target, mask=mask, loss_kind=loss_kind)
        batch_size = int(rgb.shape[0])
        losses.append(float(loss.cpu()))
        weights.append(batch_size)
    return float(np.average(losses, weights=weights))


def load_trained_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Instantiate a policy from this project's weights-only v2 checkpoint."""

    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    extra = payload.get("extra", {})
    policy_kind = extra.get("policy_kind")
    model_config = extra.get("model_config")
    if policy_kind not in ("bc", "compact_vla") or not isinstance(model_config, dict):
        raise ValueError("checkpoint lacks policy construction metadata")
    model, _ = _build_policy(policy_kind, model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, dict(extra)
