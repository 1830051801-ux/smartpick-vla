"""Episode-level evaluation records and statistically honest aggregates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any

EVALUATION_SCHEMA_VERSION = "1.0"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite(value: float, name: str, *, non_negative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_optional_non_negative(value: float | None, name: str) -> None:
    if value is not None:
        _require_finite(value, name, non_negative=True)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One immutable row of rollout evidence.

    ``cycle_time_s`` is the full episode duration. Aggregation follows the
    project protocol and averages cycle time over successful episodes only.
    ``metadata`` is where sampled domain-randomization values and other
    per-episode conditions are retained.
    """

    episode_id: str
    suite: str
    method: str
    seed: int
    task_class: str
    success: bool
    collision: bool
    cycle_time_s: float
    episode_return: float
    inference_latency_ms: float
    step_count: int
    wrong_pick: bool = False
    wrong_bin: bool = False
    timeout: bool = False
    inference_latency_p95_ms: float | None = None
    action_smoothness: float | None = None
    mean_residual_magnitude: float | None = None
    p95_residual_magnitude: float | None = None
    instruction: str = ""
    instruction_template_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "suite", "method", "task_class", "schema_version"):
            _require_text(getattr(self, name), name)
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema version: {self.schema_version!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.step_count, bool) or not isinstance(self.step_count, Integral):
            raise TypeError("step_count must be an integer")
        if self.step_count < 1:
            raise ValueError("step_count must be positive")
        for name in ("success", "collision", "wrong_pick", "wrong_bin", "timeout"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.success and self.timeout:
            raise ValueError("an episode cannot be both successful and timed out")
        _require_finite(self.cycle_time_s, "cycle_time_s", non_negative=True)
        _require_finite(self.episode_return, "episode_return")
        _require_finite(self.inference_latency_ms, "inference_latency_ms", non_negative=True)
        for name in (
            "inference_latency_p95_ms",
            "action_smoothness",
            "mean_residual_magnitude",
            "p95_residual_magnitude",
        ):
            _require_optional_non_negative(getattr(self, name), name)
        if not isinstance(self.instruction, str) or not isinstance(
            self.instruction_template_id, str
        ):
            raise TypeError("instruction fields must be strings")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "step_count", int(self.step_count))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(slots=True)
class EpisodeAccumulator:
    """Accumulate step measurements and finalize exactly one episode row."""

    episode_id: str
    suite: str
    method: str
    seed: int
    task_class: str
    instruction: str = ""
    instruction_template_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _rewards: list[float] = field(default_factory=list, init=False, repr=False)
    _latencies_ms: list[float] = field(default_factory=list, init=False, repr=False)
    _collision: bool = field(default=False, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)

    def record_step(
        self,
        *,
        reward: float,
        inference_latency_ms: float,
        collision: bool = False,
    ) -> None:
        """Record one environment transition and its policy latency."""

        if self._finalized:
            raise RuntimeError("cannot add a step after the episode was finalized")
        _require_finite(reward, "reward")
        _require_finite(inference_latency_ms, "inference_latency_ms", non_negative=True)
        if not isinstance(collision, bool):
            raise TypeError("collision must be a bool")
        self._rewards.append(float(reward))
        self._latencies_ms.append(float(inference_latency_ms))
        self._collision = self._collision or collision

    def finalize(
        self,
        *,
        success: bool,
        cycle_time_s: float,
        wrong_pick: bool = False,
        wrong_bin: bool = False,
        timeout: bool = False,
        action_smoothness: float | None = None,
        mean_residual_magnitude: float | None = None,
        p95_residual_magnitude: float | None = None,
    ) -> EpisodeResult:
        """Freeze accumulated evidence into one :class:`EpisodeResult`."""

        if self._finalized:
            raise RuntimeError("episode was already finalized")
        if not self._rewards:
            raise ValueError("cannot finalize an episode with no recorded steps")
        latency_mean = math.fsum(self._latencies_ms) / len(self._latencies_ms)
        result = EpisodeResult(
            episode_id=self.episode_id,
            suite=self.suite,
            method=self.method,
            seed=self.seed,
            task_class=self.task_class,
            success=success,
            collision=self._collision,
            cycle_time_s=cycle_time_s,
            episode_return=math.fsum(self._rewards),
            inference_latency_ms=latency_mean,
            inference_latency_p95_ms=_linear_percentile(self._latencies_ms, 0.95),
            step_count=len(self._rewards),
            wrong_pick=wrong_pick,
            wrong_bin=wrong_bin,
            timeout=timeout,
            action_smoothness=action_smoothness,
            mean_residual_magnitude=mean_residual_magnitude,
            p95_residual_magnitude=p95_residual_magnitude,
            instruction=self.instruction,
            instruction_template_id=self.instruction_template_id,
            metadata=self.metadata,
        )
        self._finalized = True
        return result


def wilson_interval(
    positive_count: int,
    total_count: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score interval for an episode-level Bernoulli rate."""

    if isinstance(positive_count, bool) or isinstance(total_count, bool):
        raise TypeError("counts must be integers")
    if not isinstance(positive_count, Integral) or not isinstance(total_count, Integral):
        raise TypeError("counts must be integers")
    if total_count < 1:
        raise ValueError("total_count must be positive")
    if not 0 <= positive_count <= total_count:
        raise ValueError("positive_count must be between zero and total_count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")

    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = positive_count / total_count
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / total_count
    center = (proportion + z_squared / (2.0 * total_count)) / denominator
    radius = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / total_count
            + z_squared / (4.0 * total_count * total_count)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True, slots=True)
class RateEstimate:
    positive_count: int
    total_count: int
    rate: float
    wilson95_low: float
    wilson95_high: float

    @classmethod
    def from_count(cls, positive_count: int, total_count: int) -> RateEstimate:
        lower, upper = wilson_interval(positive_count, total_count)
        return cls(
            positive_count=positive_count,
            total_count=total_count,
            rate=positive_count / total_count,
            wilson95_low=lower,
            wilson95_high=upper,
        )

    def as_prefixed_dict(self, name: str) -> dict[str, int | float]:
        return {
            f"{name}_count": self.positive_count,
            f"{name}_rate": self.rate,
            f"{name}_rate_wilson95_low": self.wilson95_low,
            f"{name}_rate_wilson95_high": self.wilson95_high,
        }


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Aggregate metrics for one explicit episode group."""

    group: Mapping[str, str | int]
    episode_count: int
    success: RateEstimate
    collision: RateEstimate
    wrong_pick: RateEstimate
    wrong_bin: RateEstimate
    timeout: RateEstimate
    successful_cycle_count: int
    mean_success_cycle_time_s: float | None
    mean_episode_return: float
    mean_inference_latency_ms: float
    mean_inference_latency_p95_ms: float | None
    mean_action_smoothness: float | None
    mean_residual_magnitude: float | None
    mean_p95_residual_magnitude: float | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "group": dict(self.group),
            "episode_count": self.episode_count,
            "successful_cycle_count": self.successful_cycle_count,
            "mean_success_cycle_time_s": self.mean_success_cycle_time_s,
            "mean_episode_return": self.mean_episode_return,
            "mean_inference_latency_ms": self.mean_inference_latency_ms,
            "mean_inference_latency_p95_ms": self.mean_inference_latency_p95_ms,
            "mean_action_smoothness": self.mean_action_smoothness,
            "mean_residual_magnitude": self.mean_residual_magnitude,
            "mean_p95_residual_magnitude": self.mean_p95_residual_magnitude,
        }
        for name in ("success", "collision", "wrong_pick", "wrong_bin", "timeout"):
            payload.update(getattr(self, name).as_prefixed_dict(name))
        return payload


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _optional_mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else _mean(present)


def aggregate_episode_results(
    results: Sequence[EpisodeResult],
    *,
    group: Mapping[str, str | int] | None = None,
) -> AggregateMetrics:
    """Aggregate a non-empty group without treating frames as samples."""

    if not results:
        raise ValueError("at least one episode result is required")
    total = len(results)

    def rate(attribute: str) -> RateEstimate:
        return RateEstimate.from_count(sum(bool(getattr(row, attribute)) for row in results), total)

    successful_cycles = [row.cycle_time_s for row in results if row.success]
    return AggregateMetrics(
        group=dict(group or {}),
        episode_count=total,
        success=rate("success"),
        collision=rate("collision"),
        wrong_pick=rate("wrong_pick"),
        wrong_bin=rate("wrong_bin"),
        timeout=rate("timeout"),
        successful_cycle_count=len(successful_cycles),
        mean_success_cycle_time_s=None if not successful_cycles else _mean(successful_cycles),
        mean_episode_return=_mean([row.episode_return for row in results]),
        mean_inference_latency_ms=_mean([row.inference_latency_ms for row in results]),
        mean_inference_latency_p95_ms=_optional_mean(
            [row.inference_latency_p95_ms for row in results]
        ),
        mean_action_smoothness=_optional_mean([row.action_smoothness for row in results]),
        mean_residual_magnitude=_optional_mean([row.mean_residual_magnitude for row in results]),
        mean_p95_residual_magnitude=_optional_mean([row.p95_residual_magnitude for row in results]),
    )


def group_episode_results(
    results: Sequence[EpisodeResult],
    *,
    group_by: Sequence[str] = ("suite", "method"),
) -> tuple[AggregateMetrics, ...]:
    """Group episode rows by named :class:`EpisodeResult` fields."""

    if not results:
        raise ValueError("at least one episode result is required")
    fields = EpisodeResult.__dataclass_fields__
    if not group_by:
        return (aggregate_episode_results(results),)
    invalid = [name for name in group_by if name not in fields or name == "metadata"]
    if invalid:
        raise ValueError(f"invalid grouping fields: {invalid}")

    grouped: dict[tuple[str | int, ...], list[EpisodeResult]] = defaultdict(list)
    for row in results:
        key_values: list[str | int] = []
        for name in group_by:
            value = getattr(row, name)
            if not isinstance(value, (str, int)):
                raise ValueError(f"field {name!r} cannot be used for grouping")
            key_values.append(value)
        grouped[tuple(key_values)].append(row)

    aggregates: list[AggregateMetrics] = []
    for key in sorted(grouped, key=lambda values: tuple(str(value) for value in values)):
        group = dict(zip(group_by, key, strict=True))
        aggregates.append(aggregate_episode_results(grouped[key], group=group))
    return tuple(aggregates)
