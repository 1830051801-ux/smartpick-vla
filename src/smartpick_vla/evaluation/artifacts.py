"""Lossless CSV and derived JSON artifacts for evaluation evidence."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartpick_vla.evaluation.metrics import (
    EVALUATION_SCHEMA_VERSION,
    EpisodeResult,
    group_episode_results,
)

EPISODE_CSV_FIELDS = (
    "schema_version",
    "episode_id",
    "suite",
    "method",
    "seed",
    "task_class",
    "success",
    "collision",
    "wrong_pick",
    "wrong_bin",
    "timeout",
    "cycle_time_s",
    "episode_return",
    "inference_latency_ms",
    "inference_latency_p95_ms",
    "step_count",
    "action_smoothness",
    "mean_residual_magnitude",
    "p95_residual_magnitude",
    "instruction",
    "instruction_template_id",
    "metadata_json",
)


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    episode_csv: Path
    summary_json: Path


def _optional_float(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _result_to_row(result: EpisodeResult) -> dict[str, str | int]:
    try:
        metadata_json = json.dumps(
            dict(result.metadata), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"episode {result.episode_id!r} metadata is not JSON serializable"
        ) from error
    return {
        "schema_version": result.schema_version,
        "episode_id": result.episode_id,
        "suite": result.suite,
        "method": result.method,
        "seed": result.seed,
        "task_class": result.task_class,
        "success": str(result.success).lower(),
        "collision": str(result.collision).lower(),
        "wrong_pick": str(result.wrong_pick).lower(),
        "wrong_bin": str(result.wrong_bin).lower(),
        "timeout": str(result.timeout).lower(),
        "cycle_time_s": repr(float(result.cycle_time_s)),
        "episode_return": repr(float(result.episode_return)),
        "inference_latency_ms": repr(float(result.inference_latency_ms)),
        "inference_latency_p95_ms": _optional_float(result.inference_latency_p95_ms),
        "step_count": result.step_count,
        "action_smoothness": _optional_float(result.action_smoothness),
        "mean_residual_magnitude": _optional_float(result.mean_residual_magnitude),
        "p95_residual_magnitude": _optional_float(result.p95_residual_magnitude),
        "instruction": result.instruction,
        "instruction_template_id": result.instruction_template_id,
        "metadata_json": metadata_json,
    }


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EPISODE_CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_episode_results(path: str | Path, results: Iterable[EpisodeResult]) -> Path:
    """Write canonical per-episode CSV, rejecting ambiguous duplicate rows."""

    destination = Path(path)
    materialized = tuple(results)
    if not materialized:
        raise ValueError("cannot save an empty evaluation")
    identities: set[tuple[str, str, int, str]] = set()
    rows: list[dict[str, str | int]] = []
    for result in materialized:
        if not isinstance(result, EpisodeResult):
            raise TypeError("results must contain EpisodeResult instances")
        identity = (result.suite, result.method, result.seed, result.episode_id)
        if identity in identities:
            raise ValueError(f"duplicate evaluation episode identity: {identity}")
        identities.add(identity)
        rows.append(_result_to_row(result))
    _atomic_write_csv(destination, rows)
    return destination


def _parse_bool(value: str, name: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"row {row_number}: {name} must be true or false")


def _parse_optional_float(value: str) -> float | None:
    return None if not value.strip() else float(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _row_to_result(row: Mapping[str, str], row_number: int) -> EpisodeResult:
    try:
        metadata = json.loads(row["metadata_json"], parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"row {row_number}: metadata_json is invalid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"row {row_number}: metadata_json must encode an object")
    try:
        json.dumps(metadata, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"row {row_number}: metadata_json contains non-finite values") from error
    if row["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"row {row_number}: unsupported schema_version {row['schema_version']!r}")
    try:
        return EpisodeResult(
            schema_version=row["schema_version"],
            episode_id=row["episode_id"],
            suite=row["suite"],
            method=row["method"],
            seed=int(row["seed"]),
            task_class=row["task_class"],
            success=_parse_bool(row["success"], "success", row_number),
            collision=_parse_bool(row["collision"], "collision", row_number),
            wrong_pick=_parse_bool(row["wrong_pick"], "wrong_pick", row_number),
            wrong_bin=_parse_bool(row["wrong_bin"], "wrong_bin", row_number),
            timeout=_parse_bool(row["timeout"], "timeout", row_number),
            cycle_time_s=float(row["cycle_time_s"]),
            episode_return=float(row["episode_return"]),
            inference_latency_ms=float(row["inference_latency_ms"]),
            inference_latency_p95_ms=_parse_optional_float(row["inference_latency_p95_ms"]),
            step_count=int(row["step_count"]),
            action_smoothness=_parse_optional_float(row["action_smoothness"]),
            mean_residual_magnitude=_parse_optional_float(row["mean_residual_magnitude"]),
            p95_residual_magnitude=_parse_optional_float(row["p95_residual_magnitude"]),
            instruction=row["instruction"],
            instruction_template_id=row["instruction_template_id"],
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: invalid episode result: {error}") from error


def load_episode_results(path: str | Path) -> tuple[EpisodeResult, ...]:
    """Load and validate a canonical per-episode CSV artifact."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = set(EPISODE_CSV_FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(f"episode CSV is missing columns: {sorted(missing)}")
        results = tuple(_row_to_result(row, number) for number, row in enumerate(reader, start=2))
    if not results:
        raise ValueError("episode CSV contains no episode rows")
    identities = [(row.suite, row.method, row.seed, row.episode_id) for row in results]
    if len(set(identities)) != len(identities):
        raise ValueError("episode CSV contains duplicate evaluation episode identities")
    return results


def build_evaluation_summary(
    results: Sequence[EpisodeResult],
    *,
    group_by: Sequence[str] = ("suite", "method"),
    source_episode_csv: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic aggregate that remains linked to episode rows."""

    aggregates = group_episode_results(results, group_by=group_by)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "source_episode_csv": source_episode_csv,
        "episode_count": len(results),
        "group_by": list(group_by),
        "confidence_interval": "Wilson score interval, 95%, episode-level",
        "groups": [aggregate.to_dict() for aggregate in aggregates],
    }


def save_summary_json(path: str | Path, summary: Mapping[str, Any]) -> Path:
    """Atomically write a finite, machine-readable summary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                summary, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def summarize_episode_csv(
    episode_csv: str | Path,
    summary_json: str | Path,
    *,
    group_by: Sequence[str] = ("suite", "method"),
) -> Path:
    """Regenerate ``summary.json`` directly from raw episode CSV evidence."""

    source = Path(episode_csv)
    results = load_episode_results(source)
    summary = build_evaluation_summary(
        results,
        group_by=group_by,
        source_episode_csv=source.name,
    )
    return save_summary_json(summary_json, summary)


def write_evaluation_artifacts(
    output_directory: str | Path,
    results: Iterable[EpisodeResult],
    *,
    group_by: Sequence[str] = ("suite", "method"),
) -> EvaluationArtifacts:
    """Write ``eval_episodes.csv`` first, then derive ``summary.json`` from it."""

    directory = Path(output_directory)
    episode_csv = save_episode_results(directory / "eval_episodes.csv", results)
    summary_json = summarize_episode_csv(
        episode_csv,
        directory / "summary.json",
        group_by=group_by,
    )
    return EvaluationArtifacts(episode_csv=episode_csv, summary_json=summary_json)
