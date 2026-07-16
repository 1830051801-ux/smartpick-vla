"""Dry-run-first command routing with explicit hardware interlocks."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any

from .safety import SafetyReport, SafetySupervisor, SafetyViolation
from .types import ActionChunk, ComposedActionChunk, RobotState


class ExecutionMode(StrEnum):
    DRY_RUN = "dry_run"
    HARDWARE = "hardware"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Two independent software gates plus runtime interlocks.

    Hardware output is possible only when ``dry_run`` is explicitly false and
    ``hardware_enabled`` is explicitly true.  Controller readiness, a fresh
    heartbeat, and a released emergency stop are additional requirements.
    """

    dry_run: bool = True
    hardware_enabled: bool = False
    max_heartbeat_age_s: float = 0.25
    future_tolerance_s: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.dry_run, bool) or not isinstance(self.hardware_enabled, bool):
            raise ValueError("dry_run and hardware_enabled must be booleans")
        if not isfinite(self.max_heartbeat_age_s) or self.max_heartbeat_age_s <= 0.0:
            raise ValueError("max_heartbeat_age_s must be finite and positive")
        if not isfinite(self.future_tolerance_s) or self.future_tolerance_s < 0.0:
            raise ValueError("future_tolerance_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ExecutionSignals:
    controller_ready: bool = False
    emergency_stop_active: bool = False
    last_heartbeat_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.controller_ready, bool) or not isinstance(
            self.emergency_stop_active, bool
        ):
            raise ValueError("controller and emergency-stop signals must be booleans")


@dataclass(frozen=True, slots=True)
class GateDecision:
    mode: ExecutionMode
    reasons: tuple[str, ...] = ()

    @property
    def hardware_allowed(self) -> bool:
        return self.mode is ExecutionMode.HARDWARE


class ExecutionGate:
    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()

    def decide(self, signals: ExecutionSignals, *, now_s: float) -> GateDecision:
        config = self.config
        if config.dry_run:
            return GateDecision(ExecutionMode.DRY_RUN, ("dry_run=true",))

        reasons: list[str] = []
        if not config.hardware_enabled:
            reasons.append("hardware_enabled=false")
        if signals.emergency_stop_active:
            reasons.append("emergency_stop_active")
        if not signals.controller_ready:
            reasons.append("controller_not_ready")
        if not isfinite(now_s):
            reasons.append("clock_nonfinite")
        if signals.last_heartbeat_s is None:
            reasons.append("controller_heartbeat_missing")
        elif not isfinite(signals.last_heartbeat_s):
            reasons.append("controller_heartbeat_nonfinite")
        else:
            heartbeat_age = now_s - signals.last_heartbeat_s
            if (
                heartbeat_age > config.max_heartbeat_age_s
                or heartbeat_age < -config.future_tolerance_s
            ):
                reasons.append(f"controller_heartbeat_stale:{heartbeat_age:.3f}s")
        if reasons:
            return GateDecision(ExecutionMode.BLOCKED, tuple(reasons))
        return GateDecision(ExecutionMode.HARDWARE)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    mode: ExecutionMode
    task_id: str
    sequence_id: int
    safety_report: SafetyReport
    gate_reasons: tuple[str, ...]
    final_chunk: ActionChunk | None
    hardware_published: bool


class ActionAuditLogger:
    """Append-only JSONL audit sink for preview and hardware decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


ChunkSink = Callable[[ActionChunk], None]


class SafeCommandPipeline:
    """Safety-check and route a policy command without depending on ROS 2."""

    def __init__(
        self,
        *,
        supervisor: SafetySupervisor | None = None,
        gate: ExecutionGate | None = None,
        preview_sink: ChunkSink | None = None,
        hardware_sink: ChunkSink | None = None,
        audit_logger: ActionAuditLogger | None = None,
    ) -> None:
        self.supervisor = supervisor or SafetySupervisor()
        self.gate = gate or ExecutionGate()
        self.preview_sink = preview_sink
        self.hardware_sink = hardware_sink
        self.audit_logger = audit_logger

    def submit(
        self,
        base: ActionChunk,
        residual: ActionChunk,
        state: RobotState,
        signals: ExecutionSignals,
        *,
        now_s: float,
    ) -> PipelineResult:
        composed: ComposedActionChunk | None = None
        try:
            composed = self.supervisor.compose(base, residual)
            report = self.supervisor.inspect_final(composed.final, state, now_s=now_s)
            if not report.ok:
                raise SafetyViolation(report)
        except SafetyViolation as error:
            result = PipelineResult(
                mode=ExecutionMode.BLOCKED,
                task_id=base.task_id,
                sequence_id=base.sequence_id,
                safety_report=error.report,
                gate_reasons=("safety_rejected",),
                final_chunk=composed.final if composed is not None else None,
                hardware_published=False,
            )
            self._audit(result, composed, now_s)
            return result

        final_chunk = composed.final
        if self.preview_sink is not None:
            self.preview_sink(final_chunk)

        decision = self.gate.decide(signals, now_s=now_s)
        hardware_published = False
        if decision.hardware_allowed:
            if self.hardware_sink is None:
                decision = GateDecision(ExecutionMode.BLOCKED, ("hardware_sink_missing",))
            else:
                self.hardware_sink(final_chunk)
                hardware_published = True

        result = PipelineResult(
            mode=decision.mode,
            task_id=base.task_id,
            sequence_id=base.sequence_id,
            safety_report=report,
            gate_reasons=decision.reasons,
            final_chunk=final_chunk,
            hardware_published=hardware_published,
        )
        self._audit(result, composed, now_s)
        return result

    def _audit(
        self,
        result: PipelineResult,
        composed: ComposedActionChunk | None,
        now_s: float,
    ) -> None:
        if self.audit_logger is None:
            return
        record: dict[str, Any] = {
            "timestamp_s": now_s,
            "mode": result.mode.value,
            "task_id": result.task_id,
            "sequence_id": result.sequence_id,
            "safe": result.safety_report.ok,
            "safety_issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "step_index": issue.step_index,
                }
                for issue in result.safety_report.issues
            ],
            "gate_reasons": list(result.gate_reasons),
            "hardware_published": result.hardware_published,
        }
        if composed is not None:
            record["base"] = composed.base.to_dict()
            record["requested_residual"] = composed.requested_residual.to_dict()
            record["bounded_residual"] = composed.bounded_residual.to_dict()
            record["final"] = composed.final.to_dict()
        self.audit_logger.append(record)


def gate_config_from_mapping(values: dict[str, Any]) -> GateConfig:
    """Load only known gate fields and reject accidental truthy strings."""

    dry_run = values.get("dry_run", True)
    hardware_enabled = values.get("hardware_enabled", False)
    if not isinstance(dry_run, bool) or not isinstance(hardware_enabled, bool):
        raise ValueError("dry_run and hardware_enabled must be YAML booleans")
    return GateConfig(
        dry_run=dry_run,
        hardware_enabled=hardware_enabled,
        max_heartbeat_age_s=float(values.get("max_heartbeat_age_s", 0.25)),
        future_tolerance_s=float(values.get("future_tolerance_s", 0.05)),
    )
