"""Episode-split training and checkpointing for RGB target localization."""

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

from smartpick_vla.models.vision_localizer import (
    VisionLocalizationPrediction,
    VisionLocalizer,
    VisionLocalizerConfig,
)
from smartpick_vla.training.checkpoint import load_checkpoint_payload, save_checkpoint
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import sha256_file
from smartpick_vla.utils.seed import seed_everything


@dataclass(frozen=True, slots=True)
class VisionTrainingConfig:
    """Reproducible settings for synthetic RGB localization training."""

    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    seed: int = 271
    device: str = "auto"
    num_workers: int = 0
    gradient_clip_norm: float = 2.0
    camera: str = "top"
    center_loss_weight: float = 1.0
    visibility_loss_weight: float = 0.25
    heatmap_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs and batch settings must be valid")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer hyperparameters are invalid")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not self.camera:
            raise ValueError("camera must be non-empty")
        if (
            self.center_loss_weight <= 0.0
            or self.visibility_loss_weight <= 0.0
            or self.heatmap_loss_weight < 0.0
        ):
            raise ValueError(
                "center and visibility weights must be positive; heatmap weight non-negative"
            )


class PerceptionLocalizationDataset(Dataset[dict[str, Tensor]]):
    """One named RGB view with per-class grasp-keypoint supervision.

    Newer synthetic archives provide an exact camera-projected object-origin
    keypoint. Older archives remain loadable and fall back to bbox centres.
    """

    def __init__(self, archive_path: str | Path, *, camera: str = "top") -> None:
        source = Path(archive_path)
        with np.load(source, allow_pickle=False) as archive:
            required = {"rgb", "bbox_xyxy", "episode_id", "camera_names", "quality_classes"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"perception archive is missing: {sorted(missing)}")
            camera_names = [str(item) for item in archive["camera_names"].tolist()]
            if camera not in camera_names:
                raise ValueError(f"camera {camera!r} is not present in the perception archive")
            camera_index = camera_names.index(camera)
            rgb = np.asarray(archive["rgb"][:, camera_index], dtype=np.uint8)
            bboxes = np.asarray(archive["bbox_xyxy"][:, camera_index], dtype=np.float32)
            keypoints = (
                np.asarray(archive["keypoint_xy"][:, camera_index], dtype=np.float32)
                if "keypoint_xy" in archive.files
                else None
            )
            episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
            quality_classes = tuple(str(item) for item in archive["quality_classes"].tolist())
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape [N,H,W,3]")
        if bboxes.ndim != 3 or bboxes.shape[:2] != (rgb.shape[0], len(quality_classes)):
            raise ValueError("bbox_xyxy must have shape [N,C,4]")
        if bboxes.shape[2] != 4 or episode_ids.shape != (rgb.shape[0],):
            raise ValueError("perception archive has inconsistent sample dimensions")
        if len(quality_classes) != 3:
            raise ValueError("the six-axis sorting scene expects exactly three quality classes")
        visible = (bboxes[..., 0] >= 0.0) & (bboxes[..., 2] > bboxes[..., 0])
        if not bool(visible.any()):
            raise ValueError("perception archive contains no visible object labels")
        height, width = rgb.shape[1:3]
        centers = np.zeros((rgb.shape[0], len(quality_classes), 2), dtype=np.float32)
        centers[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / (2.0 * max(1, width - 1))
        centers[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / (2.0 * max(1, height - 1))
        self.localization_label_source = "bbox_center_legacy"
        if keypoints is not None:
            if keypoints.shape != (rgb.shape[0], len(quality_classes), 2):
                raise ValueError("keypoint_xy must have shape [N,C,2] for the requested camera")
            valid_keypoints = np.isfinite(keypoints).all(axis=-1)
            visible &= valid_keypoints
            centers = np.asarray(
                keypoints / np.asarray([max(1, width - 1), max(1, height - 1)], dtype=np.float32),
                dtype=np.float32,
            )
            self.localization_label_source = "camera_projected_object_origin"
        centers = np.clip(centers, 0.0, 1.0)
        self.archive_path = source
        self.camera = camera
        self.quality_classes = quality_classes
        self.image_size = (height, width)
        self.rgb = np.ascontiguousarray(rgb)
        self.centers = np.ascontiguousarray(centers)
        self.visible = np.ascontiguousarray(visible.astype(np.float32))
        self.episode_ids = np.ascontiguousarray(episode_ids)

    def __len__(self) -> int:
        return int(self.rgb.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "rgb": torch.from_numpy(self.rgb[index]).permute(2, 0, 1),
            "centers_normalized": torch.from_numpy(self.centers[index]),
            "visible": torch.from_numpy(self.visible[index]),
        }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the PyTorch build cannot use it")
    return device


def _episode_split(
    dataset: PerceptionLocalizationDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    unique_episodes = np.unique(dataset.episode_ids)
    if unique_episodes.size < 2:
        raise ValueError("at least two episodes are required for an episode-safe split")
    shuffled = np.random.default_rng(seed).permutation(unique_episodes)
    validation_count = max(1, round(unique_episodes.size * validation_fraction))
    validation_episodes = shuffled[:validation_count]
    training_episodes = shuffled[validation_count:]
    train_indices = np.flatnonzero(np.isin(dataset.episode_ids, training_episodes)).tolist()
    validation_indices = np.flatnonzero(np.isin(dataset.episode_ids, validation_episodes)).tolist()
    if not train_indices or not validation_indices:
        raise RuntimeError("episode split produced an empty partition")
    return (
        train_indices,
        validation_indices,
        training_episodes.astype(int).tolist(),
        validation_episodes.astype(int).tolist(),
    )


def localization_loss(
    prediction: VisionLocalizationPrediction,
    target_centers: Tensor,
    target_visible: Tensor,
    *,
    center_weight: float = 1.0,
    visibility_weight: float = 0.25,
    heatmap_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute center, visibility, and spatial heatmap supervision losses."""

    if prediction.centers_normalized.shape != target_centers.shape:
        raise ValueError("predicted and target centers must have matching shapes")
    if prediction.visibility_logits.shape != target_visible.shape:
        raise ValueError("visibility logits and targets must have matching shapes")
    visible = target_visible > 0.5
    if not bool(visible.any()):
        raise ValueError("a training batch must include at least one visible target")
    center_loss = functional.smooth_l1_loss(
        prediction.centers_normalized[visible], target_centers[visible]
    )
    visibility_loss = functional.binary_cross_entropy_with_logits(
        prediction.visibility_logits, target_visible
    )
    heatmap_loss = torch.zeros((), dtype=center_loss.dtype, device=center_loss.device)
    if prediction.heatmap_logits is not None:
        if prediction.heatmap_logits.shape[:2] != target_visible.shape:
            raise ValueError("heatmap class dimensions must match visibility targets")
        heatmap_height, heatmap_width = prediction.heatmap_logits.shape[-2:]
        centers = target_centers.clamp(0.0, 1.0)
        target_x = torch.round(centers[..., 0] * (heatmap_width - 1)).to(dtype=torch.long)
        target_y = torch.round(centers[..., 1] * (heatmap_height - 1)).to(dtype=torch.long)
        target_indices = target_y * heatmap_width + target_x
        heatmap_loss = functional.cross_entropy(
            prediction.heatmap_logits.flatten(2)[visible], target_indices[visible]
        )
    return (
        center_weight * center_loss
        + visibility_weight * visibility_loss
        + heatmap_weight * heatmap_loss,
        center_loss,
        visibility_loss,
        heatmap_loss,
    )


def _metrics_for_loader(
    model: VisionLocalizer,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    image_size: tuple[int, int],
    center_weight: float,
    visibility_weight: float,
    heatmap_weight: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    center_loss_sum = 0.0
    visibility_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    samples = 0
    visible_count = 0
    pixel_error_sum = 0.0
    visible_correct = 0
    visibility_total = 0
    confidence_sum = 0.0
    scale = torch.tensor(
        [max(1, image_size[1] - 1), max(1, image_size[0] - 1)],
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            centers = batch["centers_normalized"].to(device=device, dtype=torch.float32)
            visible = batch["visible"].to(device=device, dtype=torch.float32)
            prediction = model(rgb)
            total_loss, center_loss, visibility_loss, heatmap_loss = localization_loss(
                prediction,
                centers,
                visible,
                center_weight=center_weight,
                visibility_weight=visibility_weight,
                heatmap_weight=heatmap_weight,
            )
            batch_size = int(rgb.shape[0])
            loss_sum += float(total_loss.cpu()) * batch_size
            center_loss_sum += float(center_loss.cpu()) * batch_size
            visibility_loss_sum += float(visibility_loss.cpu()) * batch_size
            heatmap_loss_sum += float(heatmap_loss.cpu()) * batch_size
            samples += batch_size
            valid = visible > 0.5
            errors = torch.linalg.vector_norm(
                (prediction.centers_normalized - centers) * scale,
                dim=-1,
            )
            pixel_error_sum += float(errors[valid].sum().cpu())
            visible_count += int(valid.sum().cpu())
            probabilities = prediction.visibility_probabilities()
            visible_correct += int(((probabilities >= 0.5) == valid).sum().cpu())
            visibility_total += int(valid.numel())
            if bool(valid.any()):
                confidence_sum += float(probabilities[valid].sum().cpu())
    if samples < 1 or visible_count < 1 or visibility_total < 1:
        raise RuntimeError("validation loader did not yield valid localization samples")
    return {
        "loss": loss_sum / samples,
        "center_loss": center_loss_sum / samples,
        "visibility_loss": visibility_loss_sum / samples,
        "heatmap_loss": heatmap_loss_sum / samples,
        "mean_pixel_error": pixel_error_sum / visible_count,
        "visibility_accuracy": visible_correct / visibility_total,
        "mean_visible_confidence": confidence_sum / visible_count,
    }


def train_vision_localizer(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    training_config: VisionTrainingConfig,
    model_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train an RGB localizer with a disjoint episode-level validation split."""

    seed_everything(training_config.seed)
    dataset = PerceptionLocalizationDataset(dataset_path, camera=training_config.camera)
    train_indices, validation_indices, train_episodes, validation_episodes = _episode_split(
        dataset,
        validation_fraction=training_config.validation_fraction,
        seed=training_config.seed,
    )
    model_config = VisionLocalizerConfig(**dict(model_options or {}))
    if model_config.quality_classes != len(dataset.quality_classes):
        raise ValueError("model quality_classes must match the perception archive")
    device = _resolve_device(training_config.device)
    model = VisionLocalizer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=training_config.num_workers,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_validation_error = float("inf")
    best_epoch = 0
    global_step = 0
    for epoch in range(1, training_config.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        gradient_norms: list[float] = []
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            centers = batch["centers_normalized"].to(device=device, dtype=torch.float32)
            visible = batch["visible"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(rgb)
            loss, _, _, _ = localization_loss(
                prediction,
                centers,
                visible,
                center_weight=training_config.center_loss_weight,
                visibility_weight=training_config.visibility_loss_weight,
                heatmap_weight=training_config.heatmap_loss_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("vision localization loss is not finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            optimizer.step()
            batch_size = int(rgb.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_size
            train_samples += batch_size
            gradient_norms.append(float(torch.as_tensor(gradient_norm).detach().cpu()))
            global_step += 1
        validation = _metrics_for_loader(
            model,
            validation_loader,
            device=device,
            image_size=dataset.image_size,
            center_weight=training_config.center_loss_weight,
            visibility_weight=training_config.visibility_loss_weight,
            heatmap_weight=training_config.heatmap_loss_weight,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss_sum / train_samples,
            "validation_loss": validation["loss"],
            "validation_mean_pixel_error": validation["mean_pixel_error"],
            "validation_visibility_accuracy": validation["visibility_accuracy"],
            "validation_heatmap_loss": validation["heatmap_loss"],
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        common_extra = {
            "model_kind": "vision_localizer",
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "dataset": str(Path(dataset_path)),
            "camera": training_config.camera,
            "quality_classes": list(dataset.quality_classes),
            "localization_label_source": dataset.localization_label_source,
            "image_size": list(dataset.image_size),
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
        if validation["mean_pixel_error"] < best_validation_error:
            best_validation_error = validation["mean_pixel_error"]
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
        "schema_version": "picksort-vision-training/v1",
        "task": "synthetic_rgb_class_conditioned_localization",
        "dataset": {"path": str(Path(dataset_path)), "sha256": sha256_file(dataset_path)},
        "camera": training_config.camera,
        "quality_classes": list(dataset.quality_classes),
        "localization_label_source": dataset.localization_label_source,
        "image_size": list(dataset.image_size),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "device": str(device),
        "torch_version": torch.__version__,
        "total_parameters": model.parameter_count(),
        "trainable_parameters": model.parameter_count(trainable_only=True),
        "training_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "best_epoch": best_epoch,
        "best_validation_mean_pixel_error": best_validation_error,
        "global_steps": global_step,
        "history": history,
        "checkpoints": {
            "best": {"path": "best.pt", "sha256": sha256_file(destination / "best.pt")},
            "last": {"path": "last.pt", "sha256": sha256_file(destination / "last.pt")},
        },
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def load_vision_localizer(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[VisionLocalizer, dict[str, Any]]:
    """Load a trusted localizer checkpoint created by this project."""

    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    extra = payload.get("extra", {})
    model_config = extra.get("model_config")
    if extra.get("model_kind") != "vision_localizer" or not isinstance(model_config, dict):
        raise ValueError("checkpoint lacks vision localizer construction metadata")
    normalized_config = dict(model_config)
    # v0 checkpoints predate the architecture field and use the global head.
    normalized_config.setdefault("architecture", "global_regression_v0")
    model = VisionLocalizer(VisionLocalizerConfig(**normalized_config))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, dict(extra)
