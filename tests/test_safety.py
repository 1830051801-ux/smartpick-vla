"""Tests for the ROS-independent final-action safety boundary."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from smartpick_vla.real import (
    ActionAuditLogger,
    ActionChunk,
    ExecutionGate,
    ExecutionMode,
    ExecutionSignals,
    GateConfig,
    RobotState,
    SafeCommandPipeline,
    SafetyCode,
    SafetyLimits,
    SafetySupervisor,
    SafetyViolation,
)
from smartpick_vla.real.execution import gate_config_from_mapping


def _chunk(
    values: list[list[float]],
    *,
    stamp_s: float = 10.0,
    policy_id: str = "policy",
    sequence_id: int = 4,
) -> ActionChunk:
    return ActionChunk.from_array(
        values,
        task_id="sort-accepted",
        sequence_id=sequence_id,
        stamp_s=stamp_s,
        dt_s=0.05,
        policy_id=policy_id,
    )


def _state(*, stamp_s: float = 10.0) -> RobotState:
    return RobotState(
        stamp_s=stamp_s,
        tcp_position_m=(0.40, 0.0, 0.25),
        tcp_yaw_rad=0.0,
        joint_positions_rad=(0.0, 0.2, -0.4),
        gripper=0.0,
    )


def _armed_signals(*, heartbeat_s: float = 10.0) -> ExecutionSignals:
    return ExecutionSignals(
        controller_ready=True,
        emergency_stop_active=False,
        last_heartbeat_s=heartbeat_s,
    )


def test_default_pipeline_is_dry_run_and_never_publishes_hardware() -> None:
    preview: list[ActionChunk] = []
    hardware: list[ActionChunk] = []
    pipeline = SafeCommandPipeline(preview_sink=preview.append, hardware_sink=hardware.append)

    result = pipeline.submit(
        _chunk([[0.005, 0.0, 0.0, 0.0, 0.0]], policy_id="vla"),
        _chunk([[0.0, 0.0, 0.0, 0.0, 0.0]], policy_id="residual"),
        _state(),
        _armed_signals(),
        now_s=10.0,
    )

    assert result.mode is ExecutionMode.DRY_RUN
    assert result.safety_report.ok
    assert result.gate_reasons == ("dry_run=true",)
    assert len(preview) == 1
    assert hardware == []
    assert not result.hardware_published


def test_hardware_requires_both_flags_and_all_runtime_interlocks() -> None:
    armed = _armed_signals()
    dry_run = ExecutionGate(GateConfig(dry_run=True, hardware_enabled=True))
    disabled = ExecutionGate(GateConfig(dry_run=False, hardware_enabled=False))
    enabled = ExecutionGate(GateConfig(dry_run=False, hardware_enabled=True))

    assert dry_run.decide(armed, now_s=10.0).mode is ExecutionMode.DRY_RUN
    assert disabled.decide(armed, now_s=10.0).mode is ExecutionMode.BLOCKED
    assert enabled.decide(armed, now_s=10.0).mode is ExecutionMode.HARDWARE

    not_ready = ExecutionSignals(last_heartbeat_s=10.0)
    decision = enabled.decide(not_ready, now_s=10.0)
    assert decision.mode is ExecutionMode.BLOCKED
    assert "controller_not_ready" in decision.reasons

    stopped = ExecutionSignals(
        controller_ready=True,
        emergency_stop_active=True,
        last_heartbeat_s=10.0,
    )
    assert enabled.decide(stopped, now_s=10.0).mode is ExecutionMode.BLOCKED
    assert enabled.decide(armed, now_s=math.nan).mode is ExecutionMode.BLOCKED


def test_hardware_publish_occurs_only_after_final_action_check() -> None:
    preview: list[ActionChunk] = []
    hardware: list[ActionChunk] = []
    pipeline = SafeCommandPipeline(
        gate=ExecutionGate(GateConfig(dry_run=False, hardware_enabled=True)),
        preview_sink=preview.append,
        hardware_sink=hardware.append,
    )
    result = pipeline.submit(
        _chunk([[0.005, 0.0, 0.0, 0.0, 0.1]], policy_id="vla"),
        _chunk([[0.001, 0.0, 0.0, 0.0, -0.05]], policy_id="residual"),
        _state(),
        _armed_signals(),
        now_s=10.0,
    )

    assert result.mode is ExecutionMode.HARDWARE
    assert result.hardware_published
    assert len(preview) == len(hardware) == 1
    np.testing.assert_allclose(hardware[0].as_array()[0], [0.006, 0.0, 0.0, 0.0, 0.05])


def test_unsafe_composed_final_action_is_not_previewed_or_published() -> None:
    preview: list[ActionChunk] = []
    hardware: list[ActionChunk] = []
    pipeline = SafeCommandPipeline(
        gate=ExecutionGate(GateConfig(dry_run=False, hardware_enabled=True)),
        preview_sink=preview.append,
        hardware_sink=hardware.append,
    )

    # The base and residual are individually within their respective limits,
    # but their 23 mm final translation exceeds the 20 mm final-action bound.
    result = pipeline.submit(
        _chunk([[0.019, 0.0, 0.0, 0.0, 0.0]], policy_id="vla"),
        _chunk([[0.004, 0.0, 0.0, 0.0, 0.0]], policy_id="residual"),
        _state(),
        _armed_signals(),
        now_s=10.0,
    )

    assert result.mode is ExecutionMode.BLOCKED
    assert SafetyCode.ACTION_RATE.value in result.safety_report.reason_codes
    assert preview == []
    assert hardware == []


def test_nonfinite_residual_is_rejected_before_bounding() -> None:
    pipeline = SafeCommandPipeline()
    result = pipeline.submit(
        _chunk([[0.0, 0.0, 0.0, 0.0, 0.0]], policy_id="vla"),
        _chunk([[math.inf, 0.0, 0.0, 0.0, 0.0]], policy_id="residual"),
        _state(),
        _armed_signals(),
        now_s=10.0,
    )

    assert result.mode is ExecutionMode.BLOCKED
    assert result.safety_report.reason_codes == (SafetyCode.NONFINITE.value,)


def test_chunk_identity_includes_source_timestamp() -> None:
    supervisor = SafetySupervisor()
    with pytest.raises(SafetyViolation) as error:
        supervisor.compose(
            _chunk([[0.0, 0.0, 0.0, 0.0, 0.0]], stamp_s=10.0),
            _chunk([[0.0, 0.0, 0.0, 0.0, 0.0]], stamp_s=10.1),
        )
    assert "stamp_s" in str(error.value)


def test_gate_config_rejects_truthy_strings_and_invalid_timing() -> None:
    with pytest.raises(ValueError, match="YAML booleans"):
        gate_config_from_mapping({"dry_run": "false", "hardware_enabled": True})
    with pytest.raises(ValueError, match="finite and positive"):
        GateConfig(max_heartbeat_age_s=math.inf)
    with pytest.raises(ValueError, match="workspace minimum"):
        SafetyLimits(workspace_min_m=(1.0, 0.0, 0.0), workspace_max_m=(0.0, 1.0, 1.0))


def test_audit_log_records_bounded_residual_and_decision(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    pipeline = SafeCommandPipeline(audit_logger=ActionAuditLogger(audit_path))
    pipeline.submit(
        _chunk([[0.0, 0.0, 0.0, 0.0, 0.0]], policy_id="vla"),
        _chunk([[1.0, 0.0, 0.0, 0.0, 0.0]], policy_id="residual"),
        _state(),
        _armed_signals(),
        now_s=10.0,
    )

    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["mode"] == "dry_run"
    assert record["bounded_residual"]["actions"][0]["dx_m"] == pytest.approx(0.004)
    assert record["hardware_published"] is False


def test_ros_core_imports_without_rclpy_and_pairs_chunks() -> None:
    package_source = Path(__file__).parents[1] / "ros2_ws" / "src" / "smartpick_vla_ros2"
    sys.path.insert(0, str(package_source))
    try:
        core = importlib.import_module("smartpick_vla_ros2.core")
        buffer = core.ChunkPairBuffer(max_pending=2)
        base = _chunk([[0.0, 0.0, 0.0, 0.0, 0.0]], policy_id="vla")
        residual = _chunk([[0.0, 0.0, 0.0, 0.0, 0.0]], policy_id="residual")
        assert buffer.add_base(base) is None
        pair = buffer.add_residual(residual)
        assert pair is not None
        assert pair.base is base and pair.residual is residual
        assert core.seconds_to_stamp_parts(1.9999999996) == (2, 0)
        assert core.stamp_to_seconds(2, 500_000_000) == 2.5
    finally:
        sys.path.remove(str(package_source))
