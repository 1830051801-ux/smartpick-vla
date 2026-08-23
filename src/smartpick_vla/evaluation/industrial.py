"""Reproducible multi-task experiment orchestration for the six-axis platform.

The regular benchmark runner is intentionally small and paired: it evaluates a
single mission length against a shared seed manifest. This module adds the
experiment-management layer needed for a larger local study without changing
the physics or controller contracts. It owns task matrices, deterministic
worker sharding, resumable episode evidence, and per-task statistical reports.

The module never turns a simulation result into a physical-robot claim. The
metadata written beside every row records the mission, worker, randomization,
and source benchmark episode so that a result can be audited later.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from torch import nn

from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.evaluation.artifacts import load_episode_results, save_episode_results
from smartpick_vla.evaluation.benchmark import (
    BENCHMARK_SUITES,
    BenchmarkConfig,
    ExperimentTier,
    SuiteName,
    run_benchmark,
)
from smartpick_vla.evaluation.metrics import EpisodeResult, aggregate_episode_results
from smartpick_vla.training.imitation import load_trained_policy
from smartpick_vla.utils.provenance import runtime_snapshot

INDUSTRIAL_SCHEMA_VERSION = "smartpick-industrial/v1"


@dataclass(frozen=True, slots=True)
class IndustrialTask:
    """One mission family evaluated on the same six-axis scene."""

    task_id: str
    mission_length: int
    description: str = ""

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.task_id) is None:
            raise ValueError("task_id must be a filesystem-safe identifier")
        if self.mission_length not in (1, 2, 3):
            raise ValueError("mission_length must be in [1, 3]")


@dataclass(frozen=True, slots=True)
class IndustrialExperimentConfig:
    """Configuration for a deterministic two-task or multi-task study."""

    tasks: tuple[IndustrialTask, ...] = (
        IndustrialTask("single_sort", 1, "single-object language-conditioned sorting"),
        IndustrialTask("ordered_pair", 2, "two-object ordered sorting mission"),
    )
    suites: tuple[SuiteName, ...] = ("id", "paraphrase", "ood", "physics", "perception")
    seed_start: int = 72000
    episodes_per_task: int = 32
    workers: int = 1
    max_episode_steps: int = 200
    image_size: int = 96
    replan_interval: int = 2
    six_axis: bool = True
    grasp_assist: bool = True
    device: str = "cpu"
    retries: int = 1
    resume: bool = True
    physics_randomization: DomainRandomizationConfig = field(
        default_factory=lambda: DomainRandomizationConfig(
            enabled=True,
            object_mass_scale=(0.50, 1.70),
            friction_scale=(0.40, 1.80),
            camera_position_std_m=0.018,
            camera_fovy_delta_deg=5.0,
            light_intensity_scale=(0.60, 1.40),
            object_color_jitter=0.15,
            robot_state_noise_std=0.004,
            detection_noise_std_m=0.008,
            control_delay_steps=(1, 3),
        )
    )
    perception_randomization: DomainRandomizationConfig = field(
        default_factory=lambda: DomainRandomizationConfig(
            enabled=True,
            object_mass_scale=(1.0, 1.0),
            friction_scale=(1.0, 1.0),
            camera_position_std_m=0.0,
            camera_fovy_delta_deg=0.0,
            light_intensity_scale=(1.0, 1.0),
            object_color_jitter=0.0,
            robot_state_noise_std=0.0,
            detection_noise_std_m=0.0,
            control_delay_steps=(0, 0),
            image_noise_std_px=14.0,
            image_occlusion_probability=0.55,
            image_occlusion_max_fraction=0.20,
            vision_latency_frames=(1, 3),
        )
    )

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("at least one industrial task is required")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("task_id values must be unique")
        if not self.suites or any(name not in BENCHMARK_SUITES for name in self.suites):
            raise ValueError("suites contain an unsupported benchmark suite")
        if len(set(self.suites)) != len(self.suites):
            raise ValueError("suites must be unique")
        if self.seed_start < 0 or self.episodes_per_task < 1:
            raise ValueError("seed_start must be non-negative and episode count positive")
        if self.workers < 1 or self.retries < 0:
            raise ValueError("workers must be positive and retries must be non-negative")
        if self.max_episode_steps < 1 or self.replan_interval < 1:
            raise ValueError("episode and replanning budgets must be positive")
        if self.image_size < 32 or self.image_size > 192:
            raise ValueError("image_size must be in [32, 192]")
        if not self.six_axis:
            raise ValueError("industrial experiments are defined for the six-axis platform")
        if not isinstance(self.physics_randomization, DomainRandomizationConfig):
            raise TypeError("physics_randomization must be a DomainRandomizationConfig")
        if not isinstance(self.perception_randomization, DomainRandomizationConfig):
            raise TypeError("perception_randomization must be a DomainRandomizationConfig")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> IndustrialExperimentConfig:
        """Build and validate a config from YAML-friendly primitive values."""

        data = dict(payload)
        raw_tasks = data.pop("tasks", None)
        if raw_tasks is not None:
            if not isinstance(raw_tasks, list):
                raise ValueError("tasks must be a list of mappings")
            tasks = []
            for item in raw_tasks:
                if not isinstance(item, dict):
                    raise ValueError("each task must be a mapping")
                tasks.append(IndustrialTask(**item))
            data["tasks"] = tuple(tasks)
        if "suites" in data:
            data["suites"] = tuple(data["suites"])
        for field_name in ("physics_randomization", "perception_randomization"):
            if field_name in data:
                value = data[field_name]
                if not isinstance(value, dict):
                    raise ValueError(f"{field_name} must be a mapping")
                data[field_name] = DomainRandomizationConfig.from_dict(value)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = [asdict(task) for task in self.tasks]
        payload["suites"] = list(self.suites)
        return payload


@dataclass(frozen=True, slots=True)
class IndustrialManifestEntry:
    task_id: str
    mission_length: int
    suite: SuiteName
    seed: int
    shard: int

    @property
    def unit_key(self) -> tuple[str, str, int]:
        return self.task_id, self.suite, self.seed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IndustrialRun:
    episode_csv: Path
    summary_json: Path
    manifest_json: Path
    results: tuple[EpisodeResult, ...]
    manifest: tuple[IndustrialManifestEntry, ...]
    resumed_rows: int
    executed_rows: int
    worker_failures: tuple[str, ...] = ()


def build_industrial_manifest(
    config: IndustrialExperimentConfig,
) -> tuple[IndustrialManifestEntry, ...]:
    """Create a stable task/suite/seed manifest with deterministic sharding."""

    entries: list[IndustrialManifestEntry] = []
    index = 0
    for task in config.tasks:
        for seed_offset in range(config.episodes_per_task):
            seed = config.seed_start + seed_offset
            shard = index % config.workers
            index += 1
            for suite in config.suites:
                entries.append(
                    IndustrialManifestEntry(
                        task_id=task.task_id,
                        mission_length=task.mission_length,
                        suite=suite,
                        seed=seed,
                        shard=shard,
                    )
                )
    return tuple(entries)


def _config_hash(config: IndustrialExperimentConfig) -> str:
    encoded = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _benchmark_config(
    config: IndustrialExperimentConfig,
    *,
    seeds: tuple[int, ...],
    suites: tuple[SuiteName, ...],
    mission_length: int,
) -> BenchmarkConfig:
    tier: ExperimentTier = "smoke" if len(seeds) < 20 else "local-benchmark"
    return BenchmarkConfig(
        seeds=seeds,
        suites=suites,
        experiment_tier=tier,
        image_size=config.image_size,
        max_episode_steps=config.max_episode_steps,
        grasp_assist=config.grasp_assist,
        device=config.device,
        replan_interval=config.replan_interval,
        six_axis=True,
        mission_length=mission_length,
        physics_randomization=config.physics_randomization,
        perception_randomization=config.perception_randomization,
    )


def _decorate_results(
    results: tuple[EpisodeResult, ...],
    *,
    task: IndustrialTask,
    shard: int,
) -> tuple[EpisodeResult, ...]:
    decorated: list[EpisodeResult] = []
    for result in results:
        metadata = dict(result.metadata)
        metadata.update(
            {
                "industrial_schema_version": INDUSTRIAL_SCHEMA_VERSION,
                "industrial_task_id": task.task_id,
                "industrial_task_description": task.description,
                "industrial_mission_length": task.mission_length,
                "industrial_worker_shard": shard,
                "source_benchmark_episode_id": result.episode_id,
                "simulation_only": True,
            }
        )
        decorated.append(
            replace(
                result,
                episode_id=f"{task.task_id}::{result.episode_id}",
                metadata=metadata,
            )
        )
    return tuple(decorated)


def _run_task_sources(
    destination: Path,
    *,
    task: IndustrialTask,
    seeds: tuple[int, ...],
    methods: dict[str, Any],
    config: IndustrialExperimentConfig,
    shard: int,
) -> Path:
    """Run one task shard and write a canonical worker CSV."""

    task_root = destination / ".workers" / f"shard-{shard}" / task.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    benchmark = _benchmark_config(
        config,
        seeds=seeds,
        suites=config.suites,
        mission_length=task.mission_length,
    )
    run = run_benchmark(task_root / "benchmark", methods, config=benchmark)
    decorated = _decorate_results(run.results, task=task, shard=shard)
    path = task_root / "industrial_episodes.csv"
    save_episode_results(path, decorated)
    return path


def _worker_entry(request: dict[str, Any]) -> str:
    """Process-pool entry point; all arguments are JSON-like primitives."""

    config = IndustrialExperimentConfig.from_mapping(request["config"])
    task = IndustrialTask(**request["task"])
    methods: dict[str, Any] = {}
    for name, checkpoint in request["methods"]:
        if name == "ik_expert":
            if checkpoint is not None:
                raise ValueError("ik_expert must not have a checkpoint")
            methods[name] = None
        else:
            if checkpoint is None:
                raise ValueError(f"method {name!r} requires a checkpoint")
            model, _ = load_trained_policy(checkpoint, device=config.device)
            methods[name] = model
    path = _run_task_sources(
        Path(request["output"]),
        task=task,
        seeds=tuple(int(seed) for seed in request["seeds"]),
        methods=methods,
        config=config,
        shard=int(request["shard"]),
    )
    return str(path)


def _load_existing(path: Path, *, resume: bool) -> tuple[EpisodeResult, ...]:
    if not resume or not path.exists():
        return ()
    return load_episode_results(path)


def _unit_key_from_result(result: EpisodeResult) -> tuple[str, str, int]:
    task_id = result.metadata.get("industrial_task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"episode {result.episode_id!r} is missing industrial_task_id")
    return task_id, result.suite, result.seed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_summary(
    results: tuple[EpisodeResult, ...],
    *,
    config: IndustrialExperimentConfig,
    manifest: tuple[IndustrialManifestEntry, ...],
    resumed_rows: int,
    executed_rows: int,
    worker_failures: tuple[str, ...],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        task_id = result.metadata.get("industrial_task_id")
        if not isinstance(task_id, str):
            raise ValueError(f"episode {result.episode_id!r} has invalid task metadata")
        grouped[(task_id, result.suite, result.method)].append(result)
    groups = []
    for (task_id, suite, method), rows in sorted(grouped.items()):
        groups.append(
            aggregate_episode_results(
                rows,
                group={"task_id": task_id, "suite": suite, "method": method},
            ).to_dict()
        )
    task_totals: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        task_id = str(result.metadata["industrial_task_id"])
        task_totals[(task_id, result.method)].append(result)
    return {
        "schema_version": INDUSTRIAL_SCHEMA_VERSION,
        "episode_count": len(results),
        "manifest_count": len(manifest),
        "config_hash": _config_hash(config),
        "config": config.to_dict(),
        "confidence_interval": "Wilson score interval, 95%, episode-level",
        "simulation_only": True,
        "resumed_rows": resumed_rows,
        "executed_rows": executed_rows,
        "worker_failures": list(worker_failures),
        "groups": groups,
        "task_totals": [
            aggregate_episode_results(
                rows,
                group={"task_id": task_id, "method": method},
            ).to_dict()
            for (task_id, method), rows in sorted(task_totals.items())
        ],
    }


def _request_groups(
    config: IndustrialExperimentConfig,
    manifest: tuple[IndustrialManifestEntry, ...],
    complete_units: set[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    """Group pending seeds by task/shard so workers reuse one environment setup."""

    by_group: dict[tuple[int, str], set[int]] = defaultdict(set)
    task_lookup = {task.task_id: task for task in config.tasks}
    for entry in manifest:
        if entry.unit_key not in complete_units:
            by_group[(entry.shard, entry.task_id)].add(entry.seed)
    requests: list[dict[str, Any]] = []
    for (shard, task_id), seeds in sorted(by_group.items()):
        requests.append(
            {
                "config": config.to_dict(),
                "task": asdict(task_lookup[task_id]),
                "seeds": sorted(seeds),
                "methods": [],
                "output": "",
                "shard": shard,
            }
        )
    return requests


def _validate_method_specs(
    methods: dict[str, str | Path | None],
) -> tuple[tuple[str, str | None], ...]:
    if not methods:
        raise ValueError("at least one method is required")
    normalized: list[tuple[str, str | None]] = []
    for name in sorted(methods):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name) is None:
            raise ValueError("method names must be filesystem-safe identifiers")
        value = methods[name]
        if name == "ik_expert":
            if value is not None:
                raise ValueError("ik_expert must use None as its source")
            normalized.append((name, None))
            continue
        if value is None:
            raise ValueError(f"method {name!r} requires a checkpoint path")
        checkpoint = str(Path(value).resolve())
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        normalized.append((name, checkpoint))
    return tuple(normalized)


def run_industrial_from_checkpoints(
    output_directory: str | Path,
    methods: dict[str, str | Path | None],
    *,
    config: IndustrialExperimentConfig | None = None,
) -> IndustrialRun:
    """Run a resumable experiment using checkpoint paths and/or ``ik_expert``."""

    resolved = config or IndustrialExperimentConfig()
    method_specs = _validate_method_specs(methods)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    episode_csv = destination / "industrial_episodes.csv"
    existing = _load_existing(episode_csv, resume=resolved.resume)
    expected_methods = {name for name, _ in method_specs}
    observed: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in existing:
        observed[_unit_key_from_result(row)].add(row.method)
    complete_units = {unit for unit, names in observed.items() if names == expected_methods}

    manifest = build_industrial_manifest(resolved)
    requests = _request_groups(resolved, manifest, complete_units)
    for request in requests:
        request["methods"] = list(method_specs)
        request["output"] = str(destination.resolve())

    worker_paths: list[Path] = []
    failures: list[str] = []
    if requests and resolved.workers == 1:
        for request in requests:
            last_error: Exception | None = None
            for _attempt in range(resolved.retries + 1):
                try:
                    worker_paths.append(Path(_worker_entry(request)))
                    last_error = None
                    break
                except Exception as error:  # pragma: no cover - exercised by CLI failures
                    last_error = error
            if last_error is not None:
                failures.append(f"{request['task']['task_id']}: {last_error}")
    elif requests:
        with ProcessPoolExecutor(max_workers=resolved.workers) as executor:
            futures = {executor.submit(_worker_entry, request): request for request in requests}
            for future in as_completed(futures):
                request = futures[future]
                try:
                    worker_paths.append(Path(future.result()))
                except Exception as first_error:  # pragma: no cover - platform/runtime dependent
                    last_error = first_error
                    recovered = False
                    for _attempt in range(resolved.retries):
                        try:
                            worker_paths.append(Path(_worker_entry(request)))
                            recovered = True
                            break
                        except Exception as retry_error:
                            last_error = retry_error
                    if not recovered:
                        failures.append(f"{request['task']['task_id']}: {last_error}")

    new_results: list[EpisodeResult] = []
    for path in worker_paths:
        new_results.extend(load_episode_results(path))
    new_results_tuple = tuple(new_results)
    replaced_units = {_unit_key_from_result(row) for row in new_results_tuple}
    retained = [row for row in existing if _unit_key_from_result(row) not in replaced_units]
    combined = tuple(
        sorted(
            (*retained, *new_results_tuple),
            key=lambda row: (
                str(row.metadata.get("industrial_task_id", "")),
                row.suite,
                row.method,
                row.seed,
            ),
        )
    )
    if not combined:
        raise RuntimeError("industrial experiment produced no episode evidence")
    save_episode_results(episode_csv, combined)
    summary = _build_summary(
        combined,
        config=resolved,
        manifest=manifest,
        resumed_rows=len(retained),
        executed_rows=len(new_results_tuple),
        worker_failures=tuple(failures),
    )
    summary_json = destination / "industrial_summary.json"
    _write_json(summary_json, summary)
    manifest_json = destination / "industrial_manifest.json"
    _write_json(
        manifest_json,
        {
            "schema_version": INDUSTRIAL_SCHEMA_VERSION,
            "config_hash": _config_hash(resolved),
            "git_revision": _git_revision(),
            "runtime": runtime_snapshot(),
            "manifest": [entry.to_dict() for entry in manifest],
            "expected_methods": [name for name, _ in method_specs],
            "complete_unit_count": len(complete_units),
            "executed_row_count": len(new_results_tuple),
            "worker_failures": failures,
        },
    )
    return IndustrialRun(
        episode_csv=episode_csv,
        summary_json=summary_json,
        manifest_json=manifest_json,
        results=combined,
        manifest=manifest,
        resumed_rows=len(retained),
        executed_rows=len(new_results_tuple),
        worker_failures=tuple(failures),
    )


def run_industrial_experiment(
    output_directory: str | Path,
    methods: dict[str, nn.Module | None],
    *,
    config: IndustrialExperimentConfig | None = None,
) -> IndustrialRun:
    """Convenience API for live modules; uses one process for module safety."""

    resolved = config or IndustrialExperimentConfig()
    if resolved.workers != 1:
        raise ValueError("run_industrial_experiment accepts live modules only with workers=1")
    if not methods:
        raise ValueError("at least one method is required")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    episode_csv = destination / "industrial_episodes.csv"
    existing = _load_existing(episode_csv, resume=resolved.resume)
    expected_methods = set(methods)
    observed: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in existing:
        observed[_unit_key_from_result(row)].add(row.method)
    complete_units = {unit for unit, names in observed.items() if names == expected_methods}
    expected_unit_keys = {
        (task.task_id, suite, seed)
        for task in resolved.tasks
        for suite in resolved.suites
        for seed in range(resolved.seed_start, resolved.seed_start + resolved.episodes_per_task)
    }
    pending_units = expected_unit_keys - complete_units
    new_results: list[EpisodeResult] = []
    for task in resolved.tasks:
        seeds = tuple(
            seed
            for seed in range(resolved.seed_start, resolved.seed_start + resolved.episodes_per_task)
            if any((task.task_id, suite, seed) in pending_units for suite in resolved.suites)
        )
        if not seeds:
            continue
        task_root = destination / ".workers" / "shard-0" / task.task_id
        benchmark = _benchmark_config(
            resolved,
            seeds=seeds,
            suites=resolved.suites,
            mission_length=task.mission_length,
        )
        run = run_benchmark(task_root / "benchmark", methods, config=benchmark)
        new_results.extend(_decorate_results(run.results, task=task, shard=0))
    replaced_units = {_unit_key_from_result(row) for row in new_results}
    retained = [row for row in existing if _unit_key_from_result(row) not in replaced_units]
    combined = tuple(
        sorted(
            (*retained, *new_results),
            key=lambda row: (row.episode_id, row.method, row.seed),
        )
    )
    if not combined:
        raise RuntimeError("industrial experiment produced no episode evidence")
    save_episode_results(episode_csv, combined)
    summary = _build_summary(
        combined,
        config=resolved,
        manifest=build_industrial_manifest(resolved),
        resumed_rows=len(retained),
        executed_rows=len(new_results),
        worker_failures=(),
    )
    summary_json = destination / "industrial_summary.json"
    _write_json(summary_json, summary)
    manifest = build_industrial_manifest(resolved)
    manifest_json = destination / "industrial_manifest.json"
    _write_json(
        manifest_json,
        {
            "schema_version": INDUSTRIAL_SCHEMA_VERSION,
            "config_hash": _config_hash(resolved),
            "git_revision": _git_revision(),
            "runtime": runtime_snapshot(),
            "manifest": [entry.to_dict() for entry in manifest],
            "expected_methods": sorted(expected_methods),
            "complete_unit_count": len(complete_units),
            "executed_row_count": len(new_results),
            "worker_failures": [],
        },
    )
    return IndustrialRun(
        episode_csv=episode_csv,
        summary_json=summary_json,
        manifest_json=manifest_json,
        results=combined,
        manifest=manifest,
        resumed_rows=len(retained),
        executed_rows=len(new_results),
    )
