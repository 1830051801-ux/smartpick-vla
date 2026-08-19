"""Planning-only bridge from XiaoU camera detections to six-axis target poses.

The module intentionally stops at a ROS 2-compatible pose preview. It can read
the project homography and per-object grasp profiles, but it never opens a
serial, CAN, or hardware executor. A real arm must still satisfy the separate
ROS 2 and hardware safety gates before any motion is considered.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, isfinite, sin
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from smartpick_vla.utils.io import atomic_write_json

XiaoUPhase = Literal["pregrasp", "grasp", "lift"]
XIAOU_HARDWARE_PROFILE_SCHEMA = "picksort-xiaou-hardware-profile/v1"
XIAOU_JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _required_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"{name} must be an integer")
    return parsed


def _float_tuple(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value))


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(_required_text(item, f"{name}[{index}]") for index, item in enumerate(value))


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


@dataclass(frozen=True, slots=True)
class XiaoUHardwareProfile:
    """Validated XiaoU hardware facts and the deliberate execution boundary.

    Values are loaded from a versioned project profile derived from the supplied
    technical note.  This object describes an interface contract; it does not
    open a transport or certify that the latest physical assembly is ready.
    """

    schema_version: str
    source_document: str
    source_sha256: str
    joint_count: int
    link_lengths_mm: tuple[float, ...]
    tcp_axis_center_mm: float
    tcp_tip_mm: float
    joint_limits_deg: tuple[tuple[float, float], ...]
    uart_device: str
    uart_baud: int
    uart_data_bits: int
    uart_parity: str
    uart_stop_bits: int
    uart_logic_voltage_v: float
    can_transport: str
    can_bitrate_bps: int
    can_id_bits: int
    can_dlc: int
    can_id_policy: str
    trajectory_angle_encoding: str
    trajectory_angle_semantics: str
    trajectory_angle_unit: str
    trajectory_duration_encoding: str
    trajectory_duration_unit: str
    trajectory_payload_bytes: int
    trajectory_interpolation_period_ms: int
    trajectory_sync_field: str
    admission_requirements: tuple[str, ...]
    yolo_trained_classes: tuple[str, ...]
    yolo_untrained_classes: tuple[str, ...]
    decision_output_fields: tuple[str, ...]
    forbidden_low_level_outputs: tuple[str, ...]
    moveit_mode: str
    ros2_dds_verified: bool
    real_motion_ready: bool
    hardware_execution_enabled: bool
    latest_firmware_validated: bool
    six_axis_feedback_validated: bool
    emergency_stop_validated: bool

    def __post_init__(self) -> None:
        if self.schema_version != XIAOU_HARDWARE_PROFILE_SCHEMA:
            raise ValueError(f"unsupported XiaoU hardware profile {self.schema_version!r}")
        if self.joint_count != len(XIAOU_JOINT_NAMES):
            raise ValueError("XiaoU hardware profile must describe six joints")
        if len(self.link_lengths_mm) != 5 or any(value <= 0.0 for value in self.link_lengths_mm):
            raise ValueError("XiaoU link lengths must contain five positive values")
        if self.tcp_axis_center_mm <= 0.0 or self.tcp_tip_mm < self.tcp_axis_center_mm:
            raise ValueError("XiaoU TCP dimensions must be positive and ordered")
        if len(self.joint_limits_deg) != self.joint_count:
            raise ValueError("XiaoU joint limits must contain six pairs")
        for lower, upper in self.joint_limits_deg:
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError("XiaoU joint limits must be finite and ordered")
        if self.uart_baud <= 0 or self.uart_data_bits <= 0 or self.uart_stop_bits <= 0:
            raise ValueError("XiaoU UART parameters must be positive")
        if self.uart_logic_voltage_v <= 0.0:
            raise ValueError("XiaoU UART logic voltage must be positive")
        if (
            self.uart_device,
            self.uart_baud,
            self.uart_data_bits,
            self.uart_parity,
            self.uart_stop_bits,
        ) != ("/dev/serial0", 115200, 8, "none", 1):
            raise ValueError("XiaoU UART baseline must be /dev/serial0 at 115200 8N1")
        if self.uart_logic_voltage_v != 3.3:
            raise ValueError("XiaoU UART baseline must use 3.3 V TTL logic")
        if self.can_bitrate_bps <= 0 or self.can_id_bits <= 0 or self.can_dlc <= 0:
            raise ValueError("XiaoU CAN parameters must be positive")
        if (self.can_transport, self.can_bitrate_bps, self.can_id_bits, self.can_dlc) != (
            "classic_can",
            1000000,
            11,
            8,
        ):
            raise ValueError("XiaoU CAN baseline must be Classic CAN, 11-bit, DLC 8 at 1 Mbps")
        if self.can_id_policy != "externally_versioned_not_declared_here":
            raise ValueError("XiaoU CAN IDs must remain externally versioned and undeclared")
        if (
            self.trajectory_angle_encoding,
            self.trajectory_angle_semantics,
            self.trajectory_angle_unit,
            self.trajectory_duration_encoding,
            self.trajectory_duration_unit,
            self.trajectory_sync_field,
        ) != (
            "little_endian_float32_x6",
            "absolute_joint_angle",
            "deg",
            "little_endian_uint16",
            "ms",
            "duration_ms",
        ):
            raise ValueError("XiaoU trajectory encoding does not match the six-axis baseline")
        if self.trajectory_payload_bytes != self.joint_count * 4 + 2:
            raise ValueError(
                "XiaoU trajectory payload does not match six float32 angles plus uint16"
            )
        if self.trajectory_interpolation_period_ms <= 0:
            raise ValueError("XiaoU interpolation period must be positive")
        if self.moveit_mode != "review_only":
            raise ValueError("XiaoU MoveIt mode must remain review_only")
        if self.hardware_execution_enabled and not self.real_motion_ready:
            raise ValueError("hardware execution cannot be enabled before real_motion_ready")
        if self.real_motion_ready and not all(
            (
                self.latest_firmware_validated,
                self.six_axis_feedback_validated,
                self.emergency_stop_validated,
                self.ros2_dds_verified,
            )
        ):
            raise ValueError("real_motion_ready requires all hardware validation gates")

    @property
    def joint_limit_map(self) -> dict[str, tuple[float, float]]:
        return dict(zip(XIAOU_JOINT_NAMES, self.joint_limits_deg, strict=True))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe normalized summary for audit artifacts."""

        return {
            "schema_version": self.schema_version,
            "source": {
                "document": self.source_document,
                "sha256": self.source_sha256,
            },
            "mechanical": {
                "joint_count": self.joint_count,
                "link_lengths_mm": list(self.link_lengths_mm),
                "tcp_axis_center_mm": self.tcp_axis_center_mm,
                "tcp_tip_mm": self.tcp_tip_mm,
                "joint_limits_deg": {
                    name: list(self.joint_limit_map[name]) for name in XIAOU_JOINT_NAMES
                },
            },
            "interfaces": {
                "pi_f407_uart": {
                    "device": self.uart_device,
                    "baud": self.uart_baud,
                    "data_bits": self.uart_data_bits,
                    "parity": self.uart_parity,
                    "stop_bits": self.uart_stop_bits,
                    "logic_voltage_v": self.uart_logic_voltage_v,
                },
                "f407_can": {
                    "transport": self.can_transport,
                    "bitrate_bps": self.can_bitrate_bps,
                    "id_bits": self.can_id_bits,
                    "dlc": self.can_dlc,
                    "id_policy": self.can_id_policy,
                },
            },
            "trajectory": {
                "angle_encoding": self.trajectory_angle_encoding,
                "angle_semantics": self.trajectory_angle_semantics,
                "angle_unit": self.trajectory_angle_unit,
                "duration_encoding": self.trajectory_duration_encoding,
                "duration_unit": self.trajectory_duration_unit,
                "payload_bytes": self.trajectory_payload_bytes,
                "interpolation_period_ms": self.trajectory_interpolation_period_ms,
                "sync_field": self.trajectory_sync_field,
                "admission_requirements": list(self.admission_requirements),
            },
            "software_boundary": {
                "yolo_trained_classes": list(self.yolo_trained_classes),
                "yolo_untrained_classes": list(self.yolo_untrained_classes),
                "decision_output_fields": list(self.decision_output_fields),
                "forbidden_low_level_outputs": list(self.forbidden_low_level_outputs),
                "moveit_mode": self.moveit_mode,
                "ros2_dds_verified": self.ros2_dds_verified,
            },
            "execution": {
                "real_motion_ready": self.real_motion_ready,
                "hardware_execution_enabled": self.hardware_execution_enabled,
                "latest_firmware_validated": self.latest_firmware_validated,
                "six_axis_feedback_validated": self.six_axis_feedback_validated,
                "emergency_stop_validated": self.emergency_stop_validated,
            },
        }


def load_xiaou_hardware_profile(path: str | Path) -> XiaoUHardwareProfile:
    """Load the versioned XiaoU six-axis hardware baseline without enabling it."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("XiaoU hardware profile must contain a mapping")
    schema_version = _required_text(payload.get("schema_version"), "schema_version")
    if schema_version != XIAOU_HARDWARE_PROFILE_SCHEMA:
        raise ValueError(f"unsupported XiaoU hardware profile {schema_version!r}")

    source_section = _mapping(payload.get("source"), "source")
    source_document = _required_text(source_section.get("document"), "source.document")
    source_sha256 = _required_text(source_section.get("sha256"), "source.sha256").lower()
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("source.sha256 must be a 64-character hexadecimal digest")

    mechanical = _mapping(payload.get("mechanical"), "mechanical")
    joint_count = _required_int(mechanical.get("joint_count"), "mechanical.joint_count")
    link_lengths_mm = _float_tuple(
        mechanical.get("link_lengths_mm"), "mechanical.link_lengths_mm", 5
    )
    joint_limits_value = _mapping(mechanical.get("joint_limits_deg"), "mechanical.joint_limits_deg")
    joint_limits: list[tuple[float, float]] = []
    for joint_name in XIAOU_JOINT_NAMES:
        pair = _float_tuple(
            joint_limits_value.get(joint_name),
            f"mechanical.joint_limits_deg.{joint_name}",
            2,
        )
        joint_limits.append((pair[0], pair[1]))

    interfaces = _mapping(payload.get("interfaces"), "interfaces")
    uart = _mapping(interfaces.get("pi_f407_uart"), "interfaces.pi_f407_uart")
    uart_device = _required_text(uart.get("device"), "interfaces.pi_f407_uart.device")
    uart_baud = _required_int(uart.get("baud"), "interfaces.pi_f407_uart.baud")
    uart_data_bits = _required_int(uart.get("data_bits"), "interfaces.pi_f407_uart.data_bits")
    uart_parity = _required_text(uart.get("parity"), "interfaces.pi_f407_uart.parity").lower()
    uart_stop_bits = _required_int(uart.get("stop_bits"), "interfaces.pi_f407_uart.stop_bits")
    if uart_parity not in {"none", "even", "odd"}:
        raise ValueError("interfaces.pi_f407_uart.parity must be none, even, or odd")
    uart_logic_voltage_v = _finite_float(
        uart.get("logic_voltage_v"), "interfaces.pi_f407_uart.logic_voltage_v"
    )

    can = _mapping(interfaces.get("f407_can"), "interfaces.f407_can")
    can_transport = _required_text(can.get("transport"), "interfaces.f407_can.transport").lower()
    can_bitrate_bps = _required_int(can.get("bitrate_bps"), "interfaces.f407_can.bitrate_bps")
    can_id_bits = _required_int(can.get("id_bits"), "interfaces.f407_can.id_bits")
    can_dlc = _required_int(can.get("dlc"), "interfaces.f407_can.dlc")
    can_id_policy = _required_text(can.get("id_policy"), "interfaces.f407_can.id_policy")

    trajectory = _mapping(payload.get("trajectory"), "trajectory")
    trajectory_angle_encoding = _required_text(
        trajectory.get("angle_encoding"), "trajectory.angle_encoding"
    )
    trajectory_angle_semantics = _required_text(
        trajectory.get("angle_semantics"), "trajectory.angle_semantics"
    )
    trajectory_angle_unit = _required_text(trajectory.get("angle_unit"), "trajectory.angle_unit")
    trajectory_duration_encoding = _required_text(
        trajectory.get("duration_encoding"), "trajectory.duration_encoding"
    )
    trajectory_duration_unit = _required_text(
        trajectory.get("duration_unit"), "trajectory.duration_unit"
    )
    trajectory_payload_bytes = _required_int(
        trajectory.get("payload_bytes"), "trajectory.payload_bytes"
    )
    trajectory_interpolation_period_ms = _required_int(
        trajectory.get("interpolation_period_ms"), "trajectory.interpolation_period_ms"
    )
    trajectory_sync_field = _required_text(trajectory.get("sync_field"), "trajectory.sync_field")
    admission = _text_tuple(
        trajectory.get("admission_requirements"), "trajectory.admission_requirements"
    )
    required_admission = {
        "crc_valid",
        "sequence_valid",
        "fifo_free_slots",
        "feedback_fresh",
        "zero_valid",
        "joint_limits_valid",
        "stop_clear",
        "estop_clear",
    }
    if not required_admission.issubset(admission):
        missing = ", ".join(sorted(required_admission.difference(admission)))
        raise ValueError(f"trajectory.admission_requirements missing: {missing}")

    software = _mapping(payload.get("software_boundary"), "software_boundary")
    yolo_trained_classes = _text_tuple(
        software.get("yolo_trained_classes"), "software_boundary.yolo_trained_classes"
    )
    yolo_untrained_classes = _text_tuple(
        software.get("yolo_untrained_classes"), "software_boundary.yolo_untrained_classes"
    )
    decision_output_fields = _text_tuple(
        software.get("decision_output_fields"), "software_boundary.decision_output_fields"
    )
    forbidden_low_level_outputs = _text_tuple(
        software.get("forbidden_low_level_outputs"),
        "software_boundary.forbidden_low_level_outputs",
    )
    if not {"joint_angles", "pwm", "raw_can_uart_bytes"}.issubset(forbidden_low_level_outputs):
        raise ValueError("software boundary must forbid direct low-level motor outputs")
    moveit_mode = _required_text(software.get("moveit_mode"), "software_boundary.moveit_mode")
    ros2_dds_verified = software.get("ros2_dds_verified")
    if not isinstance(ros2_dds_verified, bool):
        raise ValueError("software_boundary.ros2_dds_verified must be a YAML boolean")

    execution = _mapping(payload.get("execution"), "execution")
    execution_values: list[bool] = []
    for name in (
        "real_motion_ready",
        "hardware_execution_enabled",
        "latest_firmware_validated",
        "six_axis_feedback_validated",
        "emergency_stop_validated",
    ):
        value = execution.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"execution.{name} must be a YAML boolean")
        execution_values.append(value)

    return XiaoUHardwareProfile(
        schema_version=schema_version,
        source_document=source_document,
        source_sha256=source_sha256,
        joint_count=joint_count,
        link_lengths_mm=link_lengths_mm,
        tcp_axis_center_mm=_finite_float(
            mechanical.get("tcp_axis_center_mm"), "mechanical.tcp_axis_center_mm"
        ),
        tcp_tip_mm=_finite_float(mechanical.get("tcp_tip_mm"), "mechanical.tcp_tip_mm"),
        joint_limits_deg=tuple(joint_limits),
        uart_device=uart_device,
        uart_baud=uart_baud,
        uart_data_bits=uart_data_bits,
        uart_parity=uart_parity,
        uart_stop_bits=uart_stop_bits,
        uart_logic_voltage_v=uart_logic_voltage_v,
        can_transport=can_transport,
        can_bitrate_bps=can_bitrate_bps,
        can_id_bits=can_id_bits,
        can_dlc=can_dlc,
        can_id_policy=can_id_policy,
        trajectory_angle_encoding=trajectory_angle_encoding,
        trajectory_angle_semantics=trajectory_angle_semantics,
        trajectory_angle_unit=trajectory_angle_unit,
        trajectory_duration_encoding=trajectory_duration_encoding,
        trajectory_duration_unit=trajectory_duration_unit,
        trajectory_payload_bytes=trajectory_payload_bytes,
        trajectory_interpolation_period_ms=trajectory_interpolation_period_ms,
        trajectory_sync_field=trajectory_sync_field,
        admission_requirements=admission,
        yolo_trained_classes=yolo_trained_classes,
        yolo_untrained_classes=yolo_untrained_classes,
        decision_output_fields=decision_output_fields,
        forbidden_low_level_outputs=forbidden_low_level_outputs,
        moveit_mode=moveit_mode,
        ros2_dds_verified=ros2_dds_verified,
        real_motion_ready=execution_values[0],
        hardware_execution_enabled=execution_values[1],
        latest_firmware_validated=execution_values[2],
        six_axis_feedback_validated=execution_values[3],
        emergency_stop_validated=execution_values[4],
    )


@dataclass(frozen=True, slots=True)
class XiaoUHomography:
    """Validated pixel-to-robot-base planar calibration from the XiaoU workflow."""

    matrix: tuple[tuple[float, float, float], ...]
    mean_error_mm: float
    max_error_mm: float
    source_frame: str = "camera_pixels"
    target_frame: str = "base_link"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("homography matrix must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("homography matrix must be invertible")
        errors = np.asarray((self.mean_error_mm, self.max_error_mm), dtype=np.float64)
        if not np.isfinite(errors).all() or np.any(errors < 0.0):
            raise ValueError("homography errors must be finite and non-negative")
        if not self.source_frame.strip() or self.target_frame != "base_link":
            raise ValueError("homography must map a named camera frame into base_link")

    @property
    def matrix_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def project_pixel(self, u_px: float, v_px: float) -> tuple[float, float]:
        """Project pixel coordinates into base-link metres."""

        pixel = np.asarray((_finite_float(u_px, "u_px"), _finite_float(v_px, "v_px"), 1.0))
        homogeneous = self.matrix_array @ pixel
        if abs(float(homogeneous[2])) < 1e-12:
            raise ValueError("homography projection reached the line at infinity")
        point_mm = homogeneous[:2] / homogeneous[2]
        if not np.isfinite(point_mm).all():
            raise ValueError("homography projection is non-finite")
        return float(point_mm[0] / 1000.0), float(point_mm[1] / 1000.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "homography": [list(row) for row in self.matrix],
            "mean_error_mm": self.mean_error_mm,
            "max_error_mm": self.max_error_mm,
        }


@dataclass(frozen=True, slots=True)
class XiaoUDetection:
    """One stable camera detection supplied by the vision system."""

    label: str
    u_px: float
    v_px: float
    confidence: float
    stamp_s: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("detection label must not be empty")
        for name in ("u_px", "v_px", "confidence", "stamp_s"):
            _finite_float(getattr(self, name), name)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")
        if self.stamp_s < 0.0:
            raise ValueError("detection timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class XiaoUGraspProfile:
    """Measured or simulation-only vertical grasp contract for one object label."""

    label: str
    grasp_height_m: float
    approach_height_m: float
    lift_height_m: float
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("grasp profile label must not be empty")
        values = np.asarray(
            (self.grasp_height_m, self.approach_height_m, self.lift_height_m, self.yaw_rad),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("grasp profile values must be finite")
        if self.grasp_height_m < 0.0:
            raise ValueError("grasp height must be non-negative")
        if not self.grasp_height_m < self.approach_height_m <= self.lift_height_m:
            raise ValueError("grasp, approach, and lift heights must be strictly ordered")


@dataclass(frozen=True, slots=True)
class XiaoUPoseTarget:
    phase: XiaoUPhase
    position_m: tuple[float, float, float]
    yaw_rad: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("pose target position must be a finite 3-vector")
        if not isfinite(self.yaw_rad):
            raise ValueError("pose target yaw must be finite")

    def to_ros2_pose_stamped(self, *, stamp_s: float) -> dict[str, Any]:
        """Return a JSON-safe ``geometry_msgs/PoseStamped``-compatible preview."""

        half_yaw = self.yaw_rad * 0.5
        return {
            "header": {"frame_id": "base_link", "stamp_s": stamp_s},
            "pose": {
                "position": {
                    "x": self.position_m[0],
                    "y": self.position_m[1],
                    "z": self.position_m[2],
                },
                "orientation": {"x": 0.0, "y": 0.0, "z": sin(half_yaw), "w": cos(half_yaw)},
            },
        }


@dataclass(frozen=True, slots=True)
class XiaoUPlanPreview:
    """Auditable planning-only output for XiaoU's six-axis MoveIt target node."""

    task_id: str
    label: str
    detection_confidence: float
    detection_stamp_s: float
    calibration_max_error_mm: float
    targets: tuple[XiaoUPoseTarget, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.label.strip():
            raise ValueError("task_id and label must not be empty")
        if len(self.targets) != 3:
            raise ValueError("XiaoU plan preview must contain pregrasp, grasp, and lift targets")
        if tuple(target.phase for target in self.targets) != ("pregrasp", "grasp", "lift"):
            raise ValueError("XiaoU targets must use pregrasp, grasp, lift ordering")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "picksort-xiaou-plan-preview/v1",
            "planning_only": True,
            "real_motion_authorized": False,
            "task_id": self.task_id,
            "label": self.label,
            "detection_confidence": self.detection_confidence,
            "detection_stamp_s": self.detection_stamp_s,
            "calibration_max_error_mm": self.calibration_max_error_mm,
            "targets": [
                {
                    "phase": target.phase,
                    "position_m": list(target.position_m),
                    "yaw_rad": target.yaw_rad,
                    "ros2_pose_stamped_preview": target.to_ros2_pose_stamped(
                        stamp_s=self.detection_stamp_s
                    ),
                }
                for target in self.targets
            ],
        }


def load_xiaou_homography(path: str | Path) -> XiaoUHomography:
    """Load the established XiaoU ``workspace_homography.yaml`` contract."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("XiaoU homography file must contain a mapping")
    if str(payload.get("type", "")) != "pixel_to_robot_base_mm":
        raise ValueError("XiaoU homography type must be pixel_to_robot_base_mm")
    matrix = payload.get("homography")
    if not isinstance(matrix, list):
        raise ValueError("XiaoU homography must contain a 3x3 matrix")
    rows: list[tuple[float, ...]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)):
            raise ValueError("XiaoU homography rows must be sequences")
        values = tuple(_finite_float(value, "homography value") for value in row)
        if len(values) != 3:
            raise ValueError("XiaoU homography must contain exactly three columns")
        rows.append(values)
    if len(rows) != 3:
        raise ValueError("XiaoU homography must contain exactly three rows")
    return XiaoUHomography(
        matrix=cast(tuple[tuple[float, float, float], ...], tuple(rows)),
        mean_error_mm=_finite_float(payload.get("mean_error_mm"), "mean_error_mm"),
        max_error_mm=_finite_float(payload.get("max_error_mm"), "max_error_mm"),
    )


def load_xiaou_grasp_profiles(path: str | Path) -> dict[str, XiaoUGraspProfile]:
    """Load only complete grasp profiles and reject unknown measured values.

    The current XiaoU hardware profile deliberately stores unmeasured heights
    as ``null``. This function refuses those entries rather than assigning a
    generic grasp height, which keeps the planning bridge safe by default.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payload = (
        yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    )
    if not isinstance(payload, Mapping):
        raise ValueError("XiaoU grasp profile file must contain a mapping")
    profiles_value = payload.get("profiles", payload.get("classes"))
    if not isinstance(profiles_value, Mapping):
        raise ValueError("XiaoU grasp profile file must contain profiles/classes")
    profiles: dict[str, XiaoUGraspProfile] = {}
    incomplete: list[str] = []
    for label, raw_profile in profiles_value.items():
        if not isinstance(label, str) or not isinstance(raw_profile, Mapping):
            raise ValueError("XiaoU grasp profiles must map labels to objects")
        required = ("grasp_height_m", "approach_height_m")
        if any(raw_profile.get(name) is None for name in required):
            incomplete.append(label)
            continue
        lift_value = raw_profile.get("lift_height_m", raw_profile.get("approach_height_m"))
        if lift_value is None:
            incomplete.append(label)
            continue
        profiles[label] = XiaoUGraspProfile(
            label=label,
            grasp_height_m=_finite_float(raw_profile["grasp_height_m"], "grasp_height_m"),
            approach_height_m=_finite_float(raw_profile["approach_height_m"], "approach_height_m"),
            lift_height_m=_finite_float(lift_value, "lift_height_m"),
            yaw_rad=_finite_float(raw_profile.get("yaw_rad", 0.0), "yaw_rad"),
        )
    if incomplete:
        raise ValueError(
            "XiaoU grasp profiles are incomplete for: " + ", ".join(sorted(incomplete))
        )
    if not profiles:
        raise ValueError("XiaoU grasp profile file contains no complete profiles")
    return profiles


def build_xiaou_plan_preview(
    detection: XiaoUDetection,
    *,
    homography: XiaoUHomography,
    profiles: Mapping[str, XiaoUGraspProfile],
    task_id: str,
    minimum_confidence: float = 0.55,
    maximum_calibration_error_mm: float = 2.0,
) -> XiaoUPlanPreview:
    """Build a planning-only six-axis target sequence from a camera detection."""

    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if maximum_calibration_error_mm <= 0.0 or not isfinite(maximum_calibration_error_mm):
        raise ValueError("maximum_calibration_error_mm must be finite and positive")
    if detection.confidence < minimum_confidence:
        raise ValueError(
            f"detection confidence {detection.confidence:.3f} is below {minimum_confidence:.3f}"
        )
    if homography.max_error_mm > maximum_calibration_error_mm:
        raise ValueError(
            "homography max error "
            f"{homography.max_error_mm:.3f}mm exceeds {maximum_calibration_error_mm:.3f}mm"
        )
    profile = profiles.get(detection.label)
    if profile is None:
        raise ValueError(f"no complete XiaoU grasp profile exists for {detection.label!r}")
    x_m, y_m = homography.project_pixel(detection.u_px, detection.v_px)
    targets = (
        XiaoUPoseTarget("pregrasp", (x_m, y_m, profile.approach_height_m), profile.yaw_rad),
        XiaoUPoseTarget("grasp", (x_m, y_m, profile.grasp_height_m), profile.yaw_rad),
        XiaoUPoseTarget("lift", (x_m, y_m, profile.lift_height_m), profile.yaw_rad),
    )
    return XiaoUPlanPreview(
        task_id=task_id,
        label=detection.label,
        detection_confidence=detection.confidence,
        detection_stamp_s=detection.stamp_s,
        calibration_max_error_mm=homography.max_error_mm,
        targets=targets,
    )


def save_xiaou_plan_preview(preview: XiaoUPlanPreview, path: str | Path) -> Path:
    """Write a JSON planning artifact, never a hardware command."""

    destination = Path(path)
    atomic_write_json(destination, preview.to_dict())
    return destination
