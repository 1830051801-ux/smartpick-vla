"""Safety checks for language-conditioned action chunks.

This module is intentionally independent of ROS 2.  It is the final authority
between a policy (including residual RL) and any executor: the residual is
bounded first, the final command is composed, and only that final command is
checked against timing, frame, rate, workspace, joint-feedback, and numeric
constraints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import hypot
from typing import Any

import numpy as np

from .types import ACTION_DIM, ActionChunk, ComposedActionChunk, RobotState


class SafetyCode(StrEnum):
    METADATA_MISMATCH = "metadata_mismatch"
    INVALID_FRAME = "invalid_frame"
    INVALID_TIMING = "invalid_timing"
    STALE_COMMAND = "stale_command"
    STALE_STATE = "stale_state"
    INVALID_HORIZON = "invalid_horizon"
    NONFINITE = "nonfinite"
    ACTION_RATE = "action_rate"
    GRIPPER_RANGE = "gripper_range"
    WORKSPACE = "workspace"
    JOINT_LIMIT = "joint_limit"


@dataclass(frozen=True, slots=True)
class SafetyIssue:
    code: SafetyCode
    message: str
    step_index: int | None = None


@dataclass(frozen=True, slots=True)
class SafetyReport:
    issues: tuple[SafetyIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(issue.code.value for issue in self.issues))


class SafetyViolation(RuntimeError):
    """Raised when a composed command must not reach an executor."""

    def __init__(self, report: SafetyReport) -> None:
        self.report = report
        detail = "; ".join(issue.message for issue in report.issues)
        super().__init__(detail or "action chunk rejected")


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Conservative software limits; hardware limits remain authoritative."""

    required_frame: str = "base_link"
    workspace_min_m: tuple[float, float, float] = (-0.70, -0.45, 0.04)
    workspace_max_m: tuple[float, float, float] = (0.75, 0.45, 0.65)
    min_radius_m: float = 0.10
    max_radius_m: float = 0.75
    max_translation_step_m: float = 0.02
    max_yaw_step_rad: float = 0.08
    gripper_min: float = -1.0
    gripper_max: float = 1.0
    min_dt_s: float = 0.01
    max_dt_s: float = 0.20
    max_horizon: int = 16
    max_command_age_s: float = 0.50
    max_state_age_s: float = 0.25
    future_tolerance_s: float = 0.05
    residual_max_abs: tuple[float, float, float, float, float] = (
        0.004,
        0.004,
        0.004,
        0.02,
        0.15,
    )
    joint_lower_rad: tuple[float, ...] = ()
    joint_upper_rad: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.required_frame.strip():
            raise ValueError("required_frame must not be empty")
        if len(self.workspace_min_m) != 3 or len(self.workspace_max_m) != 3:
            raise ValueError("workspace bounds must each contain three values")
        if len(self.residual_max_abs) != ACTION_DIM:
            raise ValueError(f"residual_max_abs must contain {ACTION_DIM} values")
        if bool(self.joint_lower_rad) != bool(self.joint_upper_rad):
            raise ValueError("joint lower and upper limits must be configured together")
        if self.joint_lower_rad and len(self.joint_lower_rad) != len(self.joint_upper_rad):
            raise ValueError("joint lower and upper limits must have equal length")

        workspace_min = np.asarray(self.workspace_min_m, dtype=np.float64)
        workspace_max = np.asarray(self.workspace_max_m, dtype=np.float64)
        residual_max = np.asarray(self.residual_max_abs, dtype=np.float64)
        scalar_limits = np.asarray(
            (
                self.min_radius_m,
                self.max_radius_m,
                self.max_translation_step_m,
                self.max_yaw_step_rad,
                self.gripper_min,
                self.gripper_max,
                self.min_dt_s,
                self.max_dt_s,
                self.max_command_age_s,
                self.max_state_age_s,
                self.future_tolerance_s,
            ),
            dtype=np.float64,
        )
        if not (
            np.isfinite(workspace_min).all()
            and np.isfinite(workspace_max).all()
            and np.isfinite(residual_max).all()
            and np.isfinite(scalar_limits).all()
        ):
            raise ValueError("safety limits must be finite")
        if np.any(workspace_min >= workspace_max):
            raise ValueError("workspace minimum must be below maximum on every axis")
        if not 0.0 <= self.min_radius_m < self.max_radius_m:
            raise ValueError("radial workspace limits must satisfy 0 <= min < max")
        if self.max_translation_step_m <= 0.0 or self.max_yaw_step_rad <= 0.0:
            raise ValueError("action-step limits must be positive")
        if self.gripper_min >= self.gripper_max:
            raise ValueError("gripper minimum must be below maximum")
        if not 0.0 < self.min_dt_s <= self.max_dt_s:
            raise ValueError("timing limits must satisfy 0 < min_dt_s <= max_dt_s")
        if isinstance(self.max_horizon, bool) or not isinstance(self.max_horizon, int):
            raise ValueError("max_horizon must be an integer")
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if self.max_command_age_s < 0.0 or self.max_state_age_s < 0.0:
            raise ValueError("maximum data ages must be non-negative")
        if self.future_tolerance_s < 0.0:
            raise ValueError("future_tolerance_s must be non-negative")
        if np.any(residual_max < 0.0):
            raise ValueError("residual bounds must be non-negative")

        if self.joint_lower_rad:
            joint_lower = np.asarray(self.joint_lower_rad, dtype=np.float64)
            joint_upper = np.asarray(self.joint_upper_rad, dtype=np.float64)
            if not np.isfinite(joint_lower).all() or not np.isfinite(joint_upper).all():
                raise ValueError("joint limits must be finite")
            if np.any(joint_lower >= joint_upper):
                raise ValueError("every joint lower limit must be below its upper limit")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SafetyLimits:
        """Construct limits from the ``safety`` section of a YAML mapping."""

        kwargs = dict(values)
        tuple_fields = {
            "workspace_min_m",
            "workspace_max_m",
            "residual_max_abs",
            "joint_lower_rad",
            "joint_upper_rad",
        }
        for field in tuple_fields.intersection(kwargs):
            kwargs[field] = tuple(float(value) for value in kwargs[field])
        return cls(**kwargs)


class SafetySupervisor:
    """Compose bounded residual actions and validate the resulting chunk."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()

    def compose(self, base: ActionChunk, residual: ActionChunk) -> ComposedActionChunk:
        """Bound the residual and compose it with the base policy output.

        Structural identity checks happen before arithmetic.  Numeric safety is
        intentionally deferred to :meth:`inspect_final`, which evaluates the
        actual final command rather than trusting either input independently.
        """

        metadata_issues = self._metadata_issues(base, residual)
        if metadata_issues:
            raise SafetyViolation(SafetyReport(tuple(metadata_issues)))

        base_values = base.as_array().astype(np.float64)
        requested = residual.as_array().astype(np.float64)
        if not np.isfinite(base_values).all() or not np.isfinite(requested).all():
            raise SafetyViolation(
                SafetyReport(
                    (SafetyIssue(SafetyCode.NONFINITE, "base or residual action is non-finite"),)
                )
            )
        residual_bounds = np.asarray(self.limits.residual_max_abs, dtype=np.float64)
        bounded = np.clip(requested, -residual_bounds, residual_bounds)
        final_values = base_values + bounded

        bounded_chunk = ActionChunk.from_array(
            bounded,
            task_id=residual.task_id,
            sequence_id=residual.sequence_id,
            stamp_s=residual.stamp_s,
            dt_s=residual.dt_s,
            frame_id=residual.frame_id,
            policy_id=f"{residual.policy_id}:bounded",
        )
        final_chunk = ActionChunk.from_array(
            final_values,
            task_id=base.task_id,
            sequence_id=base.sequence_id,
            stamp_s=max(base.stamp_s, residual.stamp_s),
            dt_s=base.dt_s,
            frame_id=base.frame_id,
            policy_id=f"{base.policy_id}+{residual.policy_id}",
        )
        return ComposedActionChunk(base, residual, bounded_chunk, final_chunk)

    def inspect_final(self, chunk: ActionChunk, state: RobotState, *, now_s: float) -> SafetyReport:
        """Inspect a final action chunk without mutating or silently clipping it."""

        issues: list[SafetyIssue] = []
        limits = self.limits

        if chunk.frame_id != limits.required_frame:
            issues.append(
                SafetyIssue(
                    SafetyCode.INVALID_FRAME,
                    f"command frame {chunk.frame_id!r} is not {limits.required_frame!r}",
                )
            )
        if state.frame_id != limits.required_frame:
            issues.append(
                SafetyIssue(
                    SafetyCode.INVALID_FRAME,
                    f"robot-state frame {state.frame_id!r} is not {limits.required_frame!r}",
                )
            )

        timing_values = np.asarray((now_s, chunk.stamp_s, chunk.dt_s, state.stamp_s), dtype=float)
        if not np.isfinite(timing_values).all() or chunk.dt_s <= 0.0:
            issues.append(
                SafetyIssue(
                    SafetyCode.INVALID_TIMING,
                    "timestamps and dt must be finite; dt must be positive",
                )
            )
        else:
            command_age = now_s - chunk.stamp_s
            state_age = now_s - state.stamp_s
            if command_age > limits.max_command_age_s or command_age < -limits.future_tolerance_s:
                issues.append(
                    SafetyIssue(
                        SafetyCode.STALE_COMMAND,
                        f"command age {command_age:.3f}s is outside the accepted window",
                    )
                )
            if state_age > limits.max_state_age_s or state_age < -limits.future_tolerance_s:
                issues.append(
                    SafetyIssue(
                        SafetyCode.STALE_STATE,
                        f"robot-state age {state_age:.3f}s is outside the accepted window",
                    )
                )
            if not limits.min_dt_s <= chunk.dt_s <= limits.max_dt_s:
                issues.append(
                    SafetyIssue(
                        SafetyCode.INVALID_TIMING,
                        f"chunk dt {chunk.dt_s:.3f}s is outside [{limits.min_dt_s}, {limits.max_dt_s}]",
                    )
                )

        if chunk.horizon > limits.max_horizon:
            issues.append(
                SafetyIssue(
                    SafetyCode.INVALID_HORIZON,
                    f"horizon {chunk.horizon} exceeds {limits.max_horizon}",
                )
            )

        state_values = np.asarray(
            (*state.tcp_position_m, state.tcp_yaw_rad, *state.joint_positions_rad, state.gripper),
            dtype=float,
        )
        if not np.isfinite(state_values).all():
            issues.append(SafetyIssue(SafetyCode.NONFINITE, "robot state contains NaN or infinity"))
        elif not limits.gripper_min <= state.gripper <= limits.gripper_max:
            issues.append(
                SafetyIssue(
                    SafetyCode.GRIPPER_RANGE,
                    f"robot-state gripper {state.gripper:.3f} is outside "
                    f"[{limits.gripper_min}, {limits.gripper_max}]",
                )
            )

        action_values = chunk.as_array().astype(np.float64)
        if not np.isfinite(action_values).all():
            issues.append(
                SafetyIssue(SafetyCode.NONFINITE, "final action contains NaN or infinity")
            )
            return SafetyReport(tuple(issues))

        tcp = np.asarray(state.tcp_position_m, dtype=np.float64).copy()
        lower = np.asarray(limits.workspace_min_m, dtype=np.float64)
        upper = np.asarray(limits.workspace_max_m, dtype=np.float64)
        for index, action in enumerate(action_values):
            translation = action[:3]
            norm = float(np.linalg.norm(translation))
            if norm > limits.max_translation_step_m:
                issues.append(
                    SafetyIssue(
                        SafetyCode.ACTION_RATE,
                        f"translation step {norm:.4f}m exceeds {limits.max_translation_step_m:.4f}m",
                        index,
                    )
                )
            if abs(float(action[3])) > limits.max_yaw_step_rad:
                issues.append(
                    SafetyIssue(
                        SafetyCode.ACTION_RATE,
                        f"yaw step {action[3]:.4f}rad exceeds {limits.max_yaw_step_rad:.4f}rad",
                        index,
                    )
                )
            if not limits.gripper_min <= float(action[4]) <= limits.gripper_max:
                issues.append(
                    SafetyIssue(
                        SafetyCode.GRIPPER_RANGE,
                        f"gripper command {action[4]:.3f} is outside [{limits.gripper_min}, {limits.gripper_max}]",
                        index,
                    )
                )

            tcp += translation
            radial = hypot(float(tcp[0]), float(tcp[1]))
            if (
                np.any(tcp < lower)
                or np.any(tcp > upper)
                or not limits.min_radius_m <= radial <= limits.max_radius_m
            ):
                issues.append(
                    SafetyIssue(
                        SafetyCode.WORKSPACE,
                        f"predicted TCP {tcp.tolist()} is outside the configured workspace",
                        index,
                    )
                )

        if limits.joint_lower_rad:
            if len(state.joint_positions_rad) != len(limits.joint_lower_rad):
                issues.append(
                    SafetyIssue(
                        SafetyCode.JOINT_LIMIT,
                        "joint-state length does not match configured joint limits",
                    )
                )
            else:
                for index, (value, lower_value, upper_value) in enumerate(
                    zip(
                        state.joint_positions_rad,
                        limits.joint_lower_rad,
                        limits.joint_upper_rad,
                        strict=True,
                    )
                ):
                    if not lower_value <= value <= upper_value:
                        issues.append(
                            SafetyIssue(
                                SafetyCode.JOINT_LIMIT,
                                f"joint {index} value {value:.4f}rad is outside configured limits",
                            )
                        )

        return SafetyReport(tuple(issues))

    def compose_and_validate(
        self,
        base: ActionChunk,
        residual: ActionChunk,
        state: RobotState,
        *,
        now_s: float,
    ) -> ComposedActionChunk:
        """Return the composed action only when the *final* chunk is safe."""

        composed = self.compose(base, residual)
        report = self.inspect_final(composed.final, state, now_s=now_s)
        if not report.ok:
            raise SafetyViolation(report)
        return composed

    @staticmethod
    def _metadata_issues(base: ActionChunk, residual: ActionChunk) -> list[SafetyIssue]:
        differences: list[str] = []
        if base.task_id != residual.task_id:
            differences.append("task_id")
        if base.sequence_id != residual.sequence_id:
            differences.append("sequence_id")
        if base.frame_id != residual.frame_id:
            differences.append("frame_id")
        if base.horizon != residual.horizon:
            differences.append("horizon")
        if not np.isclose(base.dt_s, residual.dt_s, rtol=0.0, atol=1e-9):
            differences.append("dt_s")
        if not np.isclose(base.stamp_s, residual.stamp_s, rtol=0.0, atol=1e-6):
            differences.append("stamp_s")
        if differences:
            return [
                SafetyIssue(
                    SafetyCode.METADATA_MISMATCH,
                    "base and residual chunks differ in " + ", ".join(differences),
                )
            ]
        return []
