"""Vision-only six-axis sorting controller and simulation evaluation workflow."""

from __future__ import annotations

import csv
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from smartpick_vla.data.schema import QualityClass
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.envs.smartpick_env import SmartPickEnv
from smartpick_vla.evaluation.controller import ActionDecision
from smartpick_vla.evaluation.media import save_episode_gif
from smartpick_vla.evaluation.metrics import EpisodeResult
from smartpick_vla.evaluation.safety_filter import (
    PredictiveSafetyFilter,
    PredictiveSafetyFilterConfig,
)
from smartpick_vla.training.vision import load_vision_localizer
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import runtime_snapshot, sha256_file
from smartpick_vla.vision.geometry import PlanarCalibration
from smartpick_vla.vision.instructions import infer_quality_class

ControllerStage = Literal[
    "search",
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

_QUALITY_CLASSES: tuple[QualityClass, ...] = ("accepted", "scratch", "unknown")


@dataclass(frozen=True, slots=True)
class VisionControllerConfig:
    """Static geometry and sequence parameters for six-axis pick-and-place."""

    minimum_detection_confidence: float = 0.45
    translation_step_m: float = 0.025
    waypoint_tolerance_m: float = 0.018
    pregrasp_z_m: float = 0.19
    grasp_z_m: float = 0.100
    transport_z_m: float = 0.16
    release_z_m: float = 0.105
    close_hold_steps: int = 3
    release_hold_steps: int = 3
    confirmation_frames: int = 3
    maximum_confirmation_spread_m: float = 0.030

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_detection_confidence <= 1.0:
            raise ValueError("minimum_detection_confidence must be in [0,1]")
        if self.translation_step_m <= 0.0 or self.waypoint_tolerance_m <= 0.0:
            raise ValueError("motion distances must be positive")
        if min(self.pregrasp_z_m, self.grasp_z_m, self.transport_z_m, self.release_z_m) <= 0.0:
            raise ValueError("waypoint heights must be positive")
        if self.close_hold_steps < 1 or self.release_hold_steps < 1:
            raise ValueError("gripper hold steps must be positive")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if self.maximum_confirmation_spread_m <= 0.0:
            raise ValueError("maximum_confirmation_spread_m must be positive")


@dataclass(frozen=True, slots=True)
class VisionDecisionDiagnostics:
    """Auditable state from one RGB-to-action decision."""

    stage: ControllerStage
    target_class: QualityClass
    pixel_xy: np.ndarray
    base_xy_m: np.ndarray
    detection_confidence: float
    detection_used: bool
    inference_latency_ms: float
    confirmation_sample_count: int
    confirmation_spread_m: float | None


class VisionGuidedSixAxisController:
    """Map RGB, robot state, language, and a stored calibration to six-axis actions.

    The controller deliberately owns no environment reference. Runtime action
    decisions use only the observation dictionary, the persisted image-to-base
    calibration, fixed bin coordinates, and learned RGB localization weights.
    Scene poses are read solely by the separate evaluation harness when it
    computes error metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        calibration: PlanarCalibration,
        bin_base_xy_m: Mapping[QualityClass, np.ndarray | list[float]],
        device: str | torch.device = "cpu",
        config: VisionControllerConfig | None = None,
    ) -> None:
        if calibration.camera_name != "top":
            raise ValueError("the RGB observation contract currently uses the top camera")
        if set(bin_base_xy_m) != set(_QUALITY_CLASSES):
            raise ValueError("bin_base_xy_m must contain each quality class exactly once")
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.calibration = calibration
        self.config = config or VisionControllerConfig()
        self.bin_base_xy_m: dict[QualityClass, np.ndarray] = {}
        for quality_class, xy in bin_base_xy_m.items():
            point = np.asarray(xy, dtype=np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("each bin coordinate must be a finite [x,y] point")
            self.bin_base_xy_m[quality_class] = point.copy()
        self._instruction: str | None = None
        self._target_class: QualityClass | None = None
        self._target_base_xy_m: np.ndarray | None = None
        self._stage: ControllerStage = "search"
        self._stage_step = 0
        self._last_diagnostics: VisionDecisionDiagnostics | None = None
        self._candidate_pixels: deque[np.ndarray] = deque(maxlen=self.config.confirmation_frames)
        self._candidate_base_xy: deque[np.ndarray] = deque(maxlen=self.config.confirmation_frames)
        self._candidate_confidences: deque[float] = deque(maxlen=self.config.confirmation_frames)

    @property
    def last_diagnostics(self) -> VisionDecisionDiagnostics | None:
        return self._last_diagnostics

    def reset(self) -> None:
        self._instruction = None
        self._target_class = None
        self._target_base_xy_m = None
        self._stage = "search"
        self._stage_step = 0
        self._last_diagnostics = None
        self._candidate_pixels.clear()
        self._candidate_base_xy.clear()
        self._candidate_confidences.clear()

    def _begin_instruction(self, instruction: str) -> None:
        self._instruction = instruction
        self._target_class = infer_quality_class(instruction)
        self._target_base_xy_m = None
        self._stage = "search"
        self._stage_step = 0
        self._candidate_pixels.clear()
        self._candidate_base_xy.clear()
        self._candidate_confidences.clear()

    def _current_gripper_position(self, robot_state: Any) -> np.ndarray:
        state = np.asarray(robot_state, dtype=np.float32)
        if state.shape != (SmartPickEnv.SIX_AXIS_ROBOT_STATE_DIM,):
            raise ValueError(
                "vision-guided control requires the six-axis robot_state shape "
                f"({SmartPickEnv.SIX_AXIS_ROBOT_STATE_DIM},)"
            )
        if not np.isfinite(state).all():
            raise ValueError("robot_state contains NaN or infinity")
        # Eight arm/finger positions plus eight velocities precede the gripper XYZ.
        return np.asarray(state[16:19], dtype=np.float64)

    @torch.no_grad()
    def _localize(
        self, rgb: Any, target_class: QualityClass
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        image = np.asarray(rgb, dtype=np.uint8)
        expected_shape = (self.calibration.image_size, self.calibration.image_size, 3)
        if image.shape != expected_shape:
            raise ValueError(f"rgb must have shape {expected_shape}, got {image.shape}")
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prediction = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        centers = getattr(prediction, "centers_normalized", None)
        logits = getattr(prediction, "visibility_logits", None)
        if not isinstance(centers, torch.Tensor) or not isinstance(logits, torch.Tensor):
            raise TypeError(
                "vision model must return centers_normalized and visibility_logits tensors"
            )
        class_index = _QUALITY_CLASSES.index(target_class)
        if centers.shape != (1, len(_QUALITY_CLASSES), 2) or logits.shape != (
            1,
            len(_QUALITY_CLASSES),
        ):
            raise ValueError(
                "vision model output does not match the three-class localization contract"
            )
        normalized = centers[0, class_index].detach().cpu().numpy().astype(np.float64)
        pixel_xy = normalized * float(self.calibration.image_size - 1)
        base_xy = self.calibration.pixels_to_base_xy(pixel_xy)
        confidence = float(torch.sigmoid(logits[0, class_index]).detach().cpu())
        return pixel_xy, np.asarray(base_xy, dtype=np.float64), confidence, latency_ms

    def _stage_target(self, current_position_m: np.ndarray) -> tuple[np.ndarray, float, int]:
        if self._target_class is None:
            raise RuntimeError("controller must resolve an instruction before planning")
        if self._target_base_xy_m is None:
            return current_position_m.copy(), 1.0, 1
        object_xy = self._target_base_xy_m
        bin_xy = self.bin_base_xy_m[self._target_class]
        if self._stage == "pregrasp":
            return np.array([*object_xy, self.config.pregrasp_z_m]), 1.0, 1
        if self._stage == "descend":
            return np.array([*object_xy, self.config.grasp_z_m]), 1.0, 1
        if self._stage == "close":
            return np.array([*object_xy, self.config.grasp_z_m]), -1.0, self.config.close_hold_steps
        if self._stage == "lift":
            return np.array([*object_xy, self.config.transport_z_m]), -1.0, 1
        if self._stage == "transfer":
            return np.array([*bin_xy, self.config.transport_z_m]), -1.0, 1
        if self._stage == "lower":
            return np.array([*bin_xy, self.config.release_z_m]), -1.0, 1
        if self._stage == "release":
            return np.array([*bin_xy, self.config.release_z_m]), 1.0, self.config.release_hold_steps
        if self._stage == "retreat":
            return np.array([*bin_xy, self.config.transport_z_m]), 1.0, 1
        return current_position_m.copy(), 1.0, 1

    def _confirm_target(
        self,
        pixel_xy: np.ndarray,
        base_xy: np.ndarray,
        confidence: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float | None, bool]:
        """Use a bounded temporal consensus before moving toward a target.

        The controller observes a few stationary frames first. This prevents a
        transient blur, synthetic occluder, or single inference outlier from
        becoming the irreversible grasp waypoint. It relies only on predicted
        pixels, the persisted homography, and model confidence.
        """

        if confidence >= self.config.minimum_detection_confidence:
            self._candidate_pixels.append(pixel_xy.copy())
            self._candidate_base_xy.append(base_xy.copy())
            self._candidate_confidences.append(confidence)
        if len(self._candidate_base_xy) < self.config.confirmation_frames:
            return pixel_xy, base_xy, confidence, None, False
        candidates = np.stack(tuple(self._candidate_base_xy))
        fused_base_xy = np.median(candidates, axis=0)
        fused_pixel_xy = np.median(np.stack(tuple(self._candidate_pixels)), axis=0)
        spread_m = float(np.max(np.linalg.norm(candidates - fused_base_xy, axis=1)))
        fused_confidence = float(np.mean(tuple(self._candidate_confidences)))
        is_consistent = spread_m <= self.config.maximum_confirmation_spread_m
        return fused_pixel_xy, fused_base_xy, fused_confidence, spread_m, is_consistent

    def _advance_stage(self) -> None:
        transitions: dict[ControllerStage, ControllerStage] = {
            "search": "pregrasp",
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
        self._stage = transitions[self._stage]
        self._stage_step = 0

    def act(self, observation: Mapping[str, Any]) -> ActionDecision:
        """Produce one normalized six-axis action from the public observation contract."""

        required = {"rgb", "robot_state", "instruction"}
        missing = required.difference(observation)
        if missing:
            raise KeyError(f"vision observation is missing: {sorted(missing)}")
        instruction = observation["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if instruction != self._instruction:
            self._begin_instruction(instruction)
        if self._target_class is None:
            raise RuntimeError("instruction parser did not set a target class")
        current_position = self._current_gripper_position(observation["robot_state"])
        pixel_xy, base_xy, confidence, inference_latency_ms = self._localize(
            observation["rgb"], self._target_class
        )
        confirmed_pixel_xy = pixel_xy
        confirmed_base_xy = base_xy
        confirmed_confidence = confidence
        confirmation_spread_m: float | None = None
        detection_used = False
        if self._target_base_xy_m is None:
            (
                confirmed_pixel_xy,
                confirmed_base_xy,
                confirmed_confidence,
                confirmation_spread_m,
                detection_used,
            ) = self._confirm_target(pixel_xy, base_xy, confidence)
        if detection_used:
            self._target_base_xy_m = confirmed_base_xy.copy()
            if self._stage == "search":
                self._stage = "pregrasp"
                self._stage_step = 0
        if self._target_base_xy_m is None:
            self._stage = "search"
            self._last_diagnostics = VisionDecisionDiagnostics(
                stage=self._stage,
                target_class=self._target_class,
                pixel_xy=confirmed_pixel_xy,
                base_xy_m=confirmed_base_xy,
                detection_confidence=confirmed_confidence,
                detection_used=False,
                inference_latency_ms=inference_latency_ms,
                confirmation_sample_count=len(self._candidate_base_xy),
                confirmation_spread_m=confirmation_spread_m,
            )
            action = np.zeros(SmartPickEnv.SIX_AXIS_ACTION_DIM, dtype=np.float32)
            action[-1] = 1.0
            return ActionDecision(action, inference_latency_ms, replanned=True)

        stage_before_action = self._stage
        target_position, gripper_command, minimum_steps = self._stage_target(current_position)
        error = target_position - current_position
        action = np.zeros(SmartPickEnv.SIX_AXIS_ACTION_DIM, dtype=np.float32)
        action[:3] = np.clip(error / self.config.translation_step_m, -1.0, 1.0)
        action[-1] = gripper_command
        reached = float(np.linalg.norm(error)) <= self.config.waypoint_tolerance_m
        if self._stage in ("close", "release"):
            reached = self._stage_step + 1 >= minimum_steps
        self._stage_step += 1
        if reached and self._stage_step >= minimum_steps:
            self._advance_stage()
        self._last_diagnostics = VisionDecisionDiagnostics(
            stage=stage_before_action,
            target_class=self._target_class,
            pixel_xy=confirmed_pixel_xy,
            base_xy_m=confirmed_base_xy,
            detection_confidence=confirmed_confidence,
            detection_used=detection_used,
            inference_latency_ms=inference_latency_ms,
            confirmation_sample_count=len(self._candidate_base_xy),
            confirmation_spread_m=confirmation_spread_m,
        )
        return ActionDecision(action, inference_latency_ms, replanned=True)


@dataclass(frozen=True, slots=True)
class VisionEvaluationConfig:
    """Six-axis visual closed-loop evaluation settings and disclosure fields."""

    seeds: tuple[int, ...] = (721, 722, 723)
    image_size: int = 64
    max_episode_steps: int = 240
    mission_length: int = 1
    instruction_split: str = "paraphrase"
    ood_layout: bool = True
    grasp_assist: bool = True
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=lambda: DomainRandomizationConfig(
            enabled=True,
            object_mass_scale=(0.8, 1.2),
            friction_scale=(0.8, 1.2),
            camera_position_std_m=0.004,
            camera_fovy_delta_deg=1.0,
            light_intensity_scale=(0.85, 1.15),
            object_color_jitter=0.05,
            robot_state_noise_std=0.001,
            detection_noise_std_m=0.0,
            control_delay_steps=(0, 1),
            image_noise_std_px=2.0,
            image_occlusion_probability=0.10,
            image_occlusion_max_fraction=0.06,
            vision_latency_frames=(0, 1),
        )
    )
    controller: VisionControllerConfig = field(default_factory=VisionControllerConfig)
    safety_filter: PredictiveSafetyFilterConfig | None = field(
        default_factory=PredictiveSafetyFilterConfig
    )
    calibration_reference_pixel_noise_std: float = 0.0
    gif_seeds: tuple[int, ...] = ()
    gif_frame_stride: int = 3
    device: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "gif_seeds", tuple(int(seed) for seed in self.gif_seeds))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("seeds must be unique non-negative integers")
        if self.image_size < 32 or self.image_size > 192:
            raise ValueError("image_size must be in [32,192]")
        if self.max_episode_steps < 1 or self.mission_length < 1 or self.mission_length > 3:
            raise ValueError("episode and mission settings are invalid")
        if self.instruction_split not in ("train", "paraphrase", "ood"):
            raise ValueError("instruction_split must be train, paraphrase, or ood")
        if self.calibration_reference_pixel_noise_std < 0.0 or self.gif_frame_stride < 1:
            raise ValueError("calibration noise and GIF stride must be non-negative/positive")
        if not set(self.gif_seeds).issubset(self.seeds):
            raise ValueError("gif_seeds must be contained in seeds")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisionEvaluationConfig:
        values = dict(payload)
        if "seeds" in values:
            values["seeds"] = tuple(values["seeds"])
        if "gif_seeds" in values:
            values["gif_seeds"] = tuple(values["gif_seeds"])
        if "domain_randomization" in values:
            values["domain_randomization"] = DomainRandomizationConfig.from_dict(
                values["domain_randomization"]
            )
        if "controller" in values:
            values["controller"] = VisionControllerConfig(**values["controller"])
        if "safety_filter" in values and values["safety_filter"] is not None:
            values["safety_filter"] = PredictiveSafetyFilterConfig(**values["safety_filter"])
        return cls(**values)


def build_simulated_planar_calibration(
    environment: SmartPickEnv,
    *,
    pixel_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> PlanarCalibration:
    """Fit the calibration artifact from a simulated nine-point fixture.

    This is the simulator analogue of photographing a fixed calibration board
    before a shift. The returned artifact, not environment state, is passed to
    the visual controller for runtime operation.
    """

    if pixel_noise_std < 0.0:
        raise ValueError("pixel_noise_std must be non-negative")
    plane_z_m = environment.calibration_plane_z_m
    reference_pixels, reference_base_xy = environment.planar_calibration_reference_points(
        camera="top", plane_z_m=plane_z_m
    )
    if pixel_noise_std > 0.0:
        generator = rng or np.random.default_rng()
        reference_pixels = reference_pixels + generator.normal(
            0.0, pixel_noise_std, size=reference_pixels.shape
        )
    return PlanarCalibration.fit(
        camera_name="top",
        image_size=environment.image_size,
        plane_z_m=plane_z_m,
        reference_pixels=reference_pixels,
        reference_base_xy_m=reference_base_xy,
    )


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _p95_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.percentile(values, 95.0))


def _write_episode_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id",
        "seed",
        "initial_task_class",
        "success",
        "collision",
        "wrong_pick",
        "wrong_bin",
        "timeout",
        "step_count",
        "cycle_time_s",
        "episode_return",
        "mean_vision_latency_ms",
        "p95_vision_latency_ms",
        "mean_localization_error_mm",
        "p95_localization_error_mm",
        "mean_detection_confidence",
        "task_selection_accuracy",
        "calibration_fit_rmse_mm",
        "safety_interventions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_vision_guided_evaluation(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: VisionEvaluationConfig,
) -> dict[str, Any]:
    """Evaluate learned RGB localization with a non-privileged six-axis controller."""

    device = _resolve_device(config.device)
    model, model_metadata = load_vision_localizer(checkpoint_path, device=device)
    if tuple(model_metadata.get("quality_classes", ())) != _QUALITY_CLASSES:
        raise ValueError("vision checkpoint quality-class order is incompatible with the simulator")
    if model_metadata.get("camera") != "top":
        raise ValueError("vision checkpoint must be trained on the top camera")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    calibration_directory = destination / "calibrations"
    calibration_directory.mkdir(parents=True, exist_ok=True)
    results: list[EpisodeResult] = []
    rows: list[dict[str, Any]] = []
    media: list[str] = []
    all_localization_errors_mm: list[float] = []
    all_confidences: list[float] = []
    all_selection: list[float] = []
    all_latencies_ms: list[float] = []
    calibration_rmses_mm: list[float] = []
    total_safety_interventions = 0
    for seed in config.seeds:
        environment = SmartPickEnv(
            image_size=config.image_size,
            max_episode_steps=config.max_episode_steps,
            domain_randomization=config.domain_randomization,
            grasp_assist=config.grasp_assist,
            six_axis=True,
            mission_length=config.mission_length,
        )
        try:
            observation, info = environment.reset(
                seed=seed,
                options={
                    "instruction_split": config.instruction_split,
                    "ood_layout": config.ood_layout,
                },
            )
            calibration = build_simulated_planar_calibration(
                environment,
                pixel_noise_std=config.calibration_reference_pixel_noise_std,
                rng=np.random.default_rng(seed + 10_000),
            )
            calibration_path = calibration.save(calibration_directory / f"seed_{seed}.json")
            calibration_rmses_mm.append(calibration.fit_rmse_mm)
            bins = {
                quality_class: environment.bin_position(quality_class)[:2]
                for quality_class in _QUALITY_CLASSES
            }
            controller = VisionGuidedSixAxisController(
                model,
                calibration=calibration,
                bin_base_xy_m=bins,
                device=device,
                config=config.controller,
            )
            controller.reset()
            safety = (
                None
                if config.safety_filter is None
                else PredictiveSafetyFilter(environment, config=config.safety_filter)
            )
            frames = [np.asarray(observation["rgb"], dtype=np.uint8).copy()]
            episode_errors_mm: list[float] = []
            episode_confidences: list[float] = []
            episode_selection: list[float] = []
            episode_latencies_ms: list[float] = []
            safety_interventions = 0
            episode_return = 0.0
            final_info = info
            terminated = False
            truncated = False
            for step_index in range(config.max_episode_steps):
                expected_task_class = str(final_info["task_class"])
                started = time.perf_counter()
                decision = controller.act(observation)
                planning_latency_ms = max(
                    (time.perf_counter() - started) * 1000.0,
                    decision.inference_latency_ms,
                )
                diagnostics = controller.last_diagnostics
                if diagnostics is None:
                    raise RuntimeError("vision controller did not retain decision diagnostics")
                episode_latencies_ms.append(planning_latency_ms)
                episode_confidences.append(diagnostics.detection_confidence)
                episode_selection.append(float(diagnostics.target_class == expected_task_class))
                # Ground truth is evaluator-only. It is never passed to the controller.
                if diagnostics.detection_used and diagnostics.stage in ("pregrasp", "descend"):
                    ground_truth_xy = environment.target_position()[:2]
                    error_mm = float(
                        np.linalg.norm(diagnostics.base_xy_m - ground_truth_xy) * 1000.0
                    )
                    episode_errors_mm.append(error_mm)
                action = decision.action
                if safety is not None:
                    safety_decision = safety.filter(action)
                    action = safety_decision.action
                    safety_interventions += int(safety_decision.intervened)
                observation, reward, terminated, truncated, final_info = environment.step(action)
                episode_return += float(reward)
                if seed in config.gif_seeds and (
                    (step_index + 1) % config.gif_frame_stride == 0 or terminated or truncated
                ):
                    frames.append(np.asarray(observation["rgb"], dtype=np.uint8).copy())
                if terminated or truncated:
                    break
            timeout = bool(truncated and not final_info["success"])
            episode_id = f"vision-guided-seed-{seed}"
            result = EpisodeResult(
                episode_id=episode_id,
                suite="vision_guided",
                method="rgb_homography_six_axis",
                seed=seed,
                task_class=str(info["task_class"]),
                success=bool(final_info["success"]),
                collision=bool(final_info["collision_steps"] > 0),
                cycle_time_s=float(final_info["cycle_time_s"]),
                episode_return=episode_return,
                inference_latency_ms=float(np.mean(episode_latencies_ms)),
                inference_latency_p95_ms=_p95_or_none(episode_latencies_ms),
                step_count=step_index + 1,
                wrong_pick=bool(final_info["wrong_pick"]),
                wrong_bin=bool(final_info["wrong_bin"]),
                timeout=timeout,
                instruction=str(info["instruction"]),
                instruction_template_id=str(info["instruction_template_id"]),
                metadata={
                    "arm_variant": "six_axis",
                    "controller_inputs": ["rgb", "robot_state", "instruction", "calibration"],
                    "calibration_file": str(calibration_path),
                    "calibration_fit_rmse_mm": calibration.fit_rmse_mm,
                    "domain_randomization": final_info["randomization"],
                    "safety_interventions": safety_interventions,
                    "localization_error_sample_count": len(episode_errors_mm),
                },
            )
            results.append(result)
            row = {
                "episode_id": episode_id,
                "seed": seed,
                "initial_task_class": str(info["task_class"]),
                "success": result.success,
                "collision": result.collision,
                "wrong_pick": result.wrong_pick,
                "wrong_bin": result.wrong_bin,
                "timeout": result.timeout,
                "step_count": result.step_count,
                "cycle_time_s": result.cycle_time_s,
                "episode_return": result.episode_return,
                "mean_vision_latency_ms": result.inference_latency_ms,
                "p95_vision_latency_ms": result.inference_latency_p95_ms,
                "mean_localization_error_mm": _mean_or_none(episode_errors_mm),
                "p95_localization_error_mm": _p95_or_none(episode_errors_mm),
                "mean_detection_confidence": _mean_or_none(episode_confidences),
                "task_selection_accuracy": _mean_or_none(episode_selection),
                "calibration_fit_rmse_mm": calibration.fit_rmse_mm,
                "safety_interventions": safety_interventions,
            }
            rows.append(row)
            all_localization_errors_mm.extend(episode_errors_mm)
            all_confidences.extend(episode_confidences)
            all_selection.extend(episode_selection)
            all_latencies_ms.extend(episode_latencies_ms)
            total_safety_interventions += safety_interventions
            if seed in config.gif_seeds:
                gif_path, _ = save_episode_gif(
                    frames,
                    destination / f"{episode_id}.gif",
                    result,
                    fps=10.0,
                    extra_metadata={
                        "controller_inputs": ["rgb", "robot_state", "instruction", "calibration"],
                        "calibration_file": str(calibration_path),
                    },
                )
                media.append(str(gif_path))
        finally:
            environment.close()
    episode_csv = destination / "episodes.csv"
    _write_episode_rows(episode_csv, rows)
    success_count = sum(result.success for result in results)
    collision_count = sum(result.collision for result in results)
    summary = {
        "schema_version": "picksort-vision-evaluation/v1",
        "method": "rgb_homography_six_axis",
        "checkpoint": {"path": str(Path(checkpoint_path)), "sha256": sha256_file(checkpoint_path)},
        "model_metadata": model_metadata,
        "config": asdict(config),
        "aggregate": {
            "episodes": len(results),
            "success_count": success_count,
            "success_rate": success_count / len(results),
            "collision_episode_count": collision_count,
            "collision_episode_rate": collision_count / len(results),
            "mean_localization_error_mm": _mean_or_none(all_localization_errors_mm),
            "p95_localization_error_mm": _p95_or_none(all_localization_errors_mm),
            "localization_error_sample_count": len(all_localization_errors_mm),
            "task_selection_accuracy": _mean_or_none(all_selection),
            "mean_detection_confidence": _mean_or_none(all_confidences),
            "mean_perception_to_action_latency_ms": _mean_or_none(all_latencies_ms),
            "p95_perception_to_action_latency_ms": _p95_or_none(all_latencies_ms),
            "mean_calibration_fit_rmse_mm": _mean_or_none(calibration_rmses_mm),
            "safety_interventions": total_safety_interventions,
        },
        "artifacts": {
            "episodes_csv": str(episode_csv),
            "calibration_directory": str(calibration_directory),
            "gifs": media,
        },
        "disclosure": {
            "arm_variant": "six_axis",
            "controller_inputs": ["rgb", "robot_state", "instruction", "calibration"],
            "calibration": "simulated nine-point fixture; controller consumes saved homography only",
            "ground_truth_usage": "scene poses are used only after action selection for evaluator metrics",
            "physical_hardware_execution": False,
        },
        "runtime": runtime_snapshot(),
    }
    atomic_write_json(destination / "summary.json", summary)
    return summary
