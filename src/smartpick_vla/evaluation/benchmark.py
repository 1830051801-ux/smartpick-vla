"""Paired fixed-seed benchmark runner for the four declared simulation suites."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from torch import nn

from smartpick_vla.data.expert import IKWaypointExpert
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.envs.smartpick_env import SmartPickEnv
from smartpick_vla.envs.tasks import InstructionSplit
from smartpick_vla.evaluation.artifacts import EvaluationArtifacts, write_evaluation_artifacts
from smartpick_vla.evaluation.controller import ActionDecision, LearnedPolicyController
from smartpick_vla.evaluation.media import save_episode_gif
from smartpick_vla.evaluation.metrics import EpisodeAccumulator, EpisodeResult

SuiteName = Literal["id", "paraphrase", "ood", "physics"]
ExperimentTier = Literal["smoke", "local-benchmark", "long"]


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """One controlled evaluation shift relative to nominal ID episodes."""

    name: SuiteName
    instruction_split: InstructionSplit = "train"
    ood_layout: bool = False
    physics_randomization: bool = False

    @property
    def reset_options(self) -> dict[str, str | bool]:
        return {
            "instruction_split": self.instruction_split,
            "ood_layout": self.ood_layout,
        }


BENCHMARK_SUITES: Mapping[SuiteName, BenchmarkSuite] = {
    "id": BenchmarkSuite("id"),
    # Paraphrase changes language only; layout and physics stay nominal.
    "paraphrase": BenchmarkSuite("paraphrase", instruction_split="paraphrase"),
    # OOD changes layout only; it intentionally retains training-language templates.
    "ood": BenchmarkSuite("ood", ood_layout=True),
    # Physics retains ID layout/language and enables declared domain randomization.
    "physics": BenchmarkSuite("physics", physics_randomization=True),
}


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Fixed evaluation manifest and resource settings shared by every method."""

    seeds: tuple[int, ...] = (1000, 1001, 1002)
    suites: tuple[SuiteName, ...] = ("id", "paraphrase", "ood", "physics")
    experiment_tier: ExperimentTier = "smoke"
    image_size: int = 96
    max_episode_steps: int = 180
    grasp_assist: bool = True
    device: str = "cpu"
    replan_interval: int = 1
    physics_randomization: DomainRandomizationConfig = field(
        default_factory=lambda: DomainRandomizationConfig(enabled=True)
    )
    gif_seeds: tuple[int, ...] = ()
    gif_fps: float = 25.0
    gif_frame_stride: int = 1
    summary_group_by: tuple[str, ...] = ("suite", "method")

    def __post_init__(self) -> None:
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not self.suites or any(name not in BENCHMARK_SUITES for name in self.suites):
            raise ValueError("suites must contain supported benchmark suite names")
        if len(set(self.suites)) != len(self.suites):
            raise ValueError("suites must be unique")
        if self.experiment_tier not in ("smoke", "local-benchmark", "long"):
            raise ValueError("experiment_tier is not supported")
        if self.image_size < 32 or self.image_size > 192:
            raise ValueError("image_size must be in [32,192]")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if self.replan_interval < 1:
            raise ValueError("replan_interval must be positive")
        if not isinstance(self.physics_randomization, DomainRandomizationConfig):
            raise TypeError("physics_randomization must be a DomainRandomizationConfig")
        if "physics" in self.suites and not self.physics_randomization.enabled:
            raise ValueError("the physics suite requires enabled domain randomization")
        if any(seed not in self.seeds for seed in self.gif_seeds):
            raise ValueError("gif_seeds must be a subset of evaluation seeds")
        if len(set(self.gif_seeds)) != len(self.gif_seeds):
            raise ValueError("gif_seeds must be unique")
        if not math.isfinite(self.gif_fps) or self.gif_fps <= 0:
            raise ValueError("gif_fps must be finite and positive")
        if self.gif_frame_stride < 1:
            raise ValueError("gif_frame_stride must be positive")
        if not self.summary_group_by:
            raise ValueError("summary_group_by must not be empty")
        invalid_groups = [
            name
            for name in self.summary_group_by
            if name not in EpisodeResult.__dataclass_fields__ or name == "metadata"
        ]
        if invalid_groups:
            raise ValueError(f"invalid summary grouping fields: {invalid_groups}")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "suites", tuple(self.suites))
        object.__setattr__(self, "gif_seeds", tuple(self.gif_seeds))
        object.__setattr__(self, "summary_group_by", tuple(self.summary_group_by))


class BenchmarkController(Protocol):
    """Minimal stateful controller contract accepted by the runner."""

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> ActionDecision | np.ndarray: ...


class BenchmarkEnvironment(Protocol):
    """Gymnasium subset used by the benchmark and fakeable in unit tests."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


EnvironmentFactory = Callable[[BenchmarkSuite, BenchmarkConfig], BenchmarkEnvironment]
MethodSource = nn.Module | BenchmarkController | None


@dataclass(frozen=True, slots=True)
class BenchmarkMedia:
    episode_id: str
    method: str
    suite: SuiteName
    seed: int
    gif_path: Path
    sidecar_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    artifacts: EvaluationArtifacts
    results: tuple[EpisodeResult, ...]
    seed_manifest: tuple[tuple[SuiteName, int], ...]
    media: tuple[BenchmarkMedia, ...]


def _default_environment_factory(
    suite: BenchmarkSuite,
    config: BenchmarkConfig,
) -> SmartPickEnv:
    randomization = (
        config.physics_randomization
        if suite.physics_randomization
        else DomainRandomizationConfig(enabled=False)
    )
    return SmartPickEnv(
        image_size=config.image_size,
        max_episode_steps=config.max_episode_steps,
        domain_randomization=randomization,
        grasp_assist=config.grasp_assist,
    )


def _validate_action(action: Any) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (5,):
        raise ValueError(f"controller action must have shape (5,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("controller action contains NaN or infinity")
    return np.clip(array, -1.0, 1.0)


def _resolve_controller(
    method: str,
    source: MethodSource,
    config: BenchmarkConfig,
) -> BenchmarkController | None:
    if not isinstance(method, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", method) is None:
        raise ValueError("method names must be filesystem-safe identifiers")
    if method == "ik_expert":
        if source is not None:
            raise ValueError("ik_expert is constructed from the environment and takes no source")
        return None
    if source is None:
        raise ValueError(f"learned method {method!r} requires a model or controller")
    if isinstance(source, nn.Module):
        return LearnedPolicyController(
            source,
            device=config.device,
            replan_interval=config.replan_interval,
        )
    if not callable(getattr(source, "reset", None)) or not callable(getattr(source, "act", None)):
        raise TypeError(f"method {method!r} source must be a torch model or controller")
    return source


def _controller_action(
    controller: BenchmarkController,
    observation: dict[str, Any],
) -> tuple[np.ndarray, float, np.ndarray | None]:
    started = time.perf_counter()
    decision = controller.act(observation)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if isinstance(decision, ActionDecision):
        action = decision.action
        if not math.isfinite(decision.inference_latency_ms) or decision.inference_latency_ms < 0:
            raise ValueError("controller returned an invalid inference latency")
        # The outer timer includes preprocessing and any device synchronization
        # incurred while materializing the action. This avoids under-reporting
        # asynchronous CUDA inference while retaining a controller's longer
        # internally synchronized measurement when one is supplied.
        latency_ms = max(elapsed_ms, decision.inference_latency_ms)
        residual = decision.residual_action
        if residual is not None:
            residual = np.asarray(residual, dtype=np.float32)
            if residual.shape != (5,) or not np.isfinite(residual).all():
                raise ValueError("controller returned an invalid residual action")
            if decision.base_action is not None:
                base = _validate_action(decision.base_action)
                expected = np.clip(base + residual, -1.0, 1.0)
                if not np.allclose(expected, _validate_action(action), atol=1e-5):
                    raise ValueError("final action does not match base plus residual")
    else:
        action = decision
        latency_ms = elapsed_ms
        residual = None
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("controller returned an invalid inference latency")
    return _validate_action(action), float(latency_ms), residual


def _episode_identity(suite: BenchmarkSuite, seed: int) -> str:
    return f"{suite.name}-seed-{seed}"


def _required_info_bool(info: Mapping[str, Any], name: str) -> bool:
    if name not in info or not isinstance(info[name], bool):
        raise ValueError(f"environment info field {name!r} must be a bool")
    return info[name]


def _run_episode(
    environment: BenchmarkEnvironment,
    *,
    suite: BenchmarkSuite,
    method: str,
    seed: int,
    controller: BenchmarkController | None,
    config: BenchmarkConfig,
    output_directory: Path,
) -> tuple[EpisodeResult, BenchmarkMedia | None]:
    reset_options = suite.reset_options
    observation, reset_info = environment.reset(seed=seed, options=dict(reset_options))
    if reset_info.get("instruction_split") != suite.instruction_split:
        raise RuntimeError("environment did not apply the benchmark instruction split")
    randomization = reset_info.get("randomization")
    if not isinstance(randomization, Mapping) or not isinstance(randomization.get("enabled"), bool):
        raise ValueError("environment reset info must contain randomization.enabled")
    if randomization["enabled"] is not suite.physics_randomization:
        raise RuntimeError("domain randomization must be enabled only for the physics suite")
    task_class = str(reset_info["task_class"])
    instruction = str(observation["instruction"])
    episode_id = _episode_identity(suite, seed)
    metadata: dict[str, Any] = {
        "experiment_tier": config.experiment_tier,
        "reset_options": reset_options,
        "suite_definition": {
            "instruction_split": suite.instruction_split,
            "ood_layout": suite.ood_layout,
            "physics_randomization": suite.physics_randomization,
        },
        "instruction_split": reset_info.get("instruction_split"),
        "randomization": dict(randomization),
        "grasp_assist": reset_info.get("grasp_assist", config.grasp_assist),
        "privileged_expert": method == "ik_expert",
    }
    accumulator = EpisodeAccumulator(
        episode_id=episode_id,
        suite=suite.name,
        method=method,
        seed=seed,
        task_class=task_class,
        instruction=instruction,
        instruction_template_id=str(reset_info.get("instruction_template_id", "")),
        metadata=metadata,
    )

    expert: IKWaypointExpert | None = None
    if method == "ik_expert":
        expert = IKWaypointExpert(
            environment,  # type: ignore[arg-type]
            use_noisy_detection=suite.physics_randomization,
        )
        expert.reset()
    else:
        if controller is None:
            raise RuntimeError("learned controller was not initialized")
        controller.reset()

    capture_media = seed in config.gif_seeds
    frames: list[np.ndarray] = []
    if capture_media:
        frames.append(np.asarray(observation["rgb"], dtype=np.uint8).copy())
    terminated = False
    truncated = False
    final_info = reset_info
    executed_actions: list[np.ndarray] = []
    residual_magnitudes: list[float] = []
    for step_index in range(config.max_episode_steps):
        if expert is not None:
            started = time.perf_counter()
            action, _ = expert.act()
            latency_ms = (time.perf_counter() - started) * 1000.0
            normalized_action = _validate_action(action)
            residual_action = None
        else:
            if controller is None:
                raise RuntimeError("learned controller was not initialized")
            normalized_action, latency_ms, residual_action = _controller_action(
                controller, observation
            )
        executed_actions.append(normalized_action.copy())
        if residual_action is not None:
            residual_magnitudes.append(float(np.linalg.norm(residual_action)))
        observation, reward, terminated, truncated, final_info = environment.step(normalized_action)
        accumulator.record_step(
            reward=float(reward),
            inference_latency_ms=latency_ms,
            collision=_required_info_bool(final_info, "collision"),
        )
        if capture_media and (
            (step_index + 1) % config.gif_frame_stride == 0 or terminated or truncated
        ):
            frames.append(np.asarray(observation["rgb"], dtype=np.uint8).copy())
        if terminated or truncated:
            break
    if not terminated and not truncated:
        truncated = True

    success = _required_info_bool(final_info, "success")
    metadata.update(
        {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "collision_steps": int(final_info.get("collision_steps", 0)),
            "contact_count": int(final_info.get("contact_count", 0)),
            "randomization": final_info.get("randomization", metadata["randomization"]),
            "control_dt": final_info.get("control_dt"),
        }
    )
    if "cycle_time_s" not in final_info:
        raise KeyError("environment info must contain cycle_time_s")
    smoothness = 0.0
    if len(executed_actions) > 1:
        smoothness = float(
            np.mean(
                [
                    np.linalg.norm(current - previous)
                    for previous, current in pairwise(executed_actions)
                ]
            )
        )
    result = accumulator.finalize(
        success=success,
        cycle_time_s=float(final_info["cycle_time_s"]),
        wrong_pick=_required_info_bool(final_info, "wrong_pick"),
        wrong_bin=_required_info_bool(final_info, "wrong_bin"),
        timeout=bool(truncated and not success),
        action_smoothness=smoothness,
        mean_residual_magnitude=(
            float(np.mean(residual_magnitudes)) if residual_magnitudes else None
        ),
        p95_residual_magnitude=(
            float(np.percentile(residual_magnitudes, 95)) if residual_magnitudes else None
        ),
    )

    media: BenchmarkMedia | None = None
    if capture_media:
        media_root = output_directory / "media" / method
        gif_path, sidecar_path = save_episode_gif(
            frames,
            media_root / f"{episode_id}.gif",
            result,
            fps=config.gif_fps,
            extra_metadata={
                "experiment_tier": config.experiment_tier,
                "reset_options": reset_options,
                "randomization": metadata["randomization"],
            },
        )
        media = BenchmarkMedia(
            episode_id=episode_id,
            method=method,
            suite=suite.name,
            seed=seed,
            gif_path=gif_path,
            sidecar_path=sidecar_path,
        )
    return result, media


def _seed_manifest(config: BenchmarkConfig) -> tuple[tuple[SuiteName, int], ...]:
    return tuple((suite, seed) for suite in config.suites for seed in config.seeds)


def _validate_equal_seed_budget(
    results: Sequence[EpisodeResult],
    method_names: Sequence[str],
    manifest: Sequence[tuple[SuiteName, int]],
) -> None:
    expected = {(suite, seed) for suite, seed in manifest}
    for method in method_names:
        rows = [row for row in results if row.method == method]
        observed = {(row.suite, row.seed) for row in rows}
        if len(rows) != len(manifest) or observed != expected:
            raise RuntimeError(f"method {method!r} did not complete the shared seed manifest")
    for suite, seed in manifest:
        paired = [row for row in results if row.suite == suite and row.seed == seed]
        task_conditions = {
            (row.task_class, row.instruction_template_id, row.instruction) for row in paired
        }
        randomization_conditions = {
            json.dumps(row.metadata["randomization"], sort_keys=True, allow_nan=False)
            for row in paired
        }
        if len(task_conditions) != 1 or len(randomization_conditions) != 1:
            raise RuntimeError(
                f"paired methods received different conditions for suite={suite!r}, seed={seed}"
            )


def run_benchmark(
    output_directory: str | Path,
    methods: Mapping[str, MethodSource],
    *,
    config: BenchmarkConfig | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> BenchmarkRun:
    """Evaluate every method on the same ordered suite/seed manifest.

    Use ``{"ik_expert": None}`` for the privileged upper bound. Learned
    entries accept either a ``torch.nn.Module`` or a stateful controller.
    Artifacts are written only after every requested method completes.
    """

    resolved_config = config or BenchmarkConfig()
    if not methods:
        raise ValueError("at least one benchmark method is required")
    if any(
        not isinstance(method, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", method) is None
        for method in methods
    ):
        raise ValueError("method names must be filesystem-safe identifiers")
    ordered_methods = tuple(sorted(methods))
    controllers = {
        method: _resolve_controller(method, methods[method], resolved_config)
        for method in ordered_methods
    }
    factory = environment_factory or _default_environment_factory
    destination = Path(output_directory)
    results: list[EpisodeResult] = []
    media: list[BenchmarkMedia] = []
    for method in ordered_methods:
        for suite_name in resolved_config.suites:
            suite = BENCHMARK_SUITES[suite_name]
            environment = factory(suite, resolved_config)
            try:
                for seed in resolved_config.seeds:
                    result, episode_media = _run_episode(
                        environment,
                        suite=suite,
                        method=method,
                        seed=seed,
                        controller=controllers[method],
                        config=resolved_config,
                        output_directory=destination,
                    )
                    results.append(result)
                    if episode_media is not None:
                        media.append(episode_media)
            finally:
                environment.close()

    manifest = _seed_manifest(resolved_config)
    _validate_equal_seed_budget(results, ordered_methods, manifest)
    artifacts = write_evaluation_artifacts(
        destination,
        results,
        group_by=resolved_config.summary_group_by,
    )
    return BenchmarkRun(
        artifacts=artifacts,
        results=tuple(results),
        seed_manifest=manifest,
        media=tuple(media),
    )


def evaluate_method(
    output_directory: str | Path,
    *,
    method: str,
    model: nn.Module | None = None,
    controller: BenchmarkController | None = None,
    config: BenchmarkConfig | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> BenchmarkRun:
    """Convenience wrapper for one IK-expert or learned method benchmark."""

    if model is not None and controller is not None:
        raise ValueError("provide either model or controller, not both")
    source: MethodSource = controller if controller is not None else model
    return run_benchmark(
        output_directory,
        {method: source},
        config=config,
        environment_factory=environment_factory,
    )
