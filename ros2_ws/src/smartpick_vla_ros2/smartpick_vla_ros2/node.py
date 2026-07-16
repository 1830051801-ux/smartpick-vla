"""ROS 2 safety bridge for paired VLA and residual action chunks.

This module is imported only by the installed executable.  Keeping all ROS 2
imports here means the package's pairing logic remains testable without
``rclpy`` on the host Python environment.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty

from smartpick_vla.real import (
    ActionAuditLogger,
    ActionChunk,
    CartesianDelta,
    ExecutionGate,
    ExecutionMode,
    ExecutionSignals,
    GateConfig,
    PipelineResult,
    RobotState,
    SafeCommandPipeline,
    SafetyLimits,
    SafetySupervisor,
    load_real_config,
)
from smartpick_vla_ros2.core import (
    ChunkPair,
    ChunkPairBuffer,
    seconds_to_stamp_parts,
    stamp_to_seconds,
)
from smartpick_vla_ros2.msg import (
    ActionChunk as ActionChunkMsg,
)
from smartpick_vla_ros2.msg import (
    CartesianDelta as CartesianDeltaMsg,
)
from smartpick_vla_ros2.msg import (
    ExecutionStatus as ExecutionStatusMsg,
)
from smartpick_vla_ros2.msg import (
    RobotState as RobotStateMsg,
)


class SmartPickSafetyBridge(Node):
    """Compose, inspect, preview, and conditionally publish final actions."""

    def __init__(self) -> None:
        super().__init__("smartpick_vla_safety_bridge")

        self.declare_parameter("config_path", "")
        config_path = str(self.get_parameter("config_path").value).strip()
        runtime = load_real_config(Path(config_path)) if config_path else None
        gate_defaults = runtime.execution if runtime is not None else GateConfig()
        safety_limits = runtime.safety if runtime is not None else SafetyLimits()
        robot_config = runtime.robot if runtime is not None else {}

        # These two independent values are captured at startup.  Hardware
        # publication requires dry_run=false AND hardware_enabled=true.
        self.declare_parameter("dry_run", gate_defaults.dry_run)
        self.declare_parameter("hardware_enabled", gate_defaults.hardware_enabled)
        self.declare_parameter("max_pending_chunks", 32)
        self.declare_parameter("base_action_topic", "/smartpick/base_action_chunk")
        self.declare_parameter("residual_action_topic", "/smartpick/residual_action_chunk")
        self.declare_parameter("preview_topic", "/smartpick/action_preview")
        self.declare_parameter(
            "command_topic",
            str(robot_config.get("command_topic", "/smartpick/safe_action_chunk")),
        )
        self.declare_parameter(
            "robot_state_topic",
            str(robot_config.get("state_topic", "/smartpick/robot_state")),
        )
        self.declare_parameter("controller_ready_topic", "/smartpick/controller_ready")
        self.declare_parameter("emergency_stop_topic", "/smartpick/emergency_stop")
        self.declare_parameter(
            "controller_heartbeat_topic",
            str(
                robot_config.get(
                    "controller_heartbeat_topic",
                    "/smartpick/controller_heartbeat",
                )
            ),
        )
        self.declare_parameter("status_topic", "/smartpick/execution_status")
        self.declare_parameter(
            "audit_jsonl",
            runtime.audit_jsonl if runtime is not None and runtime.audit_jsonl else "",
        )

        gate_config = GateConfig(
            dry_run=bool(self.get_parameter("dry_run").value),
            hardware_enabled=bool(self.get_parameter("hardware_enabled").value),
            max_heartbeat_age_s=gate_defaults.max_heartbeat_age_s,
            future_tolerance_s=gate_defaults.future_tolerance_s,
        )
        max_pending = int(self.get_parameter("max_pending_chunks").value)
        self._pairs = ChunkPairBuffer(max_pending=max_pending)
        self._latest_state: RobotState | None = None
        self._controller_ready = False
        self._emergency_stop_active = False
        self._last_heartbeat_s: float | None = None

        self._preview_publisher = self.create_publisher(
            ActionChunkMsg,
            str(self.get_parameter("preview_topic").value),
            10,
        )
        self._hardware_publisher = self.create_publisher(
            ActionChunkMsg,
            str(self.get_parameter("command_topic").value),
            10,
        )
        self._status_publisher = self.create_publisher(
            ExecutionStatusMsg,
            str(self.get_parameter("status_topic").value),
            10,
        )
        audit_path = str(self.get_parameter("audit_jsonl").value).strip()
        audit_logger = ActionAuditLogger(audit_path) if audit_path else None
        self._pipeline = SafeCommandPipeline(
            supervisor=SafetySupervisor(safety_limits),
            gate=ExecutionGate(gate_config),
            preview_sink=self._publish_preview,
            hardware_sink=self._publish_hardware,
            audit_logger=audit_logger,
        )

        self.create_subscription(
            ActionChunkMsg,
            str(self.get_parameter("base_action_topic").value),
            self._on_base,
            10,
        )
        self.create_subscription(
            ActionChunkMsg,
            str(self.get_parameter("residual_action_topic").value),
            self._on_residual,
            10,
        )
        self.create_subscription(
            RobotStateMsg,
            str(self.get_parameter("robot_state_topic").value),
            self._on_robot_state,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("controller_ready_topic").value),
            self._on_controller_ready,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("emergency_stop_topic").value),
            self._on_emergency_stop,
            10,
        )
        self.create_subscription(
            Empty,
            str(self.get_parameter("controller_heartbeat_topic").value),
            self._on_heartbeat,
            10,
        )

        self.get_logger().info(
            "PickSort safety bridge started: "
            f"dry_run={gate_config.dry_run}, hardware_enabled={gate_config.hardware_enabled}"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_base(self, message: ActionChunkMsg) -> None:
        try:
            pair = self._pairs.add_base(action_chunk_from_message(message))
        except ValueError as error:
            self.get_logger().error(f"rejected malformed base chunk: {error}")
            return
        if pair is not None:
            self._process_pair(pair)

    def _on_residual(self, message: ActionChunkMsg) -> None:
        try:
            pair = self._pairs.add_residual(action_chunk_from_message(message))
        except ValueError as error:
            self.get_logger().error(f"rejected malformed residual chunk: {error}")
            return
        if pair is not None:
            self._process_pair(pair)

    def _on_robot_state(self, message: RobotStateMsg) -> None:
        try:
            self._latest_state = robot_state_from_message(message)
        except ValueError as error:
            self._latest_state = None
            self.get_logger().error(f"rejected malformed robot state: {error}")

    def _on_controller_ready(self, message: Bool) -> None:
        self._controller_ready = bool(message.data)

    def _on_emergency_stop(self, message: Bool) -> None:
        self._emergency_stop_active = bool(message.data)

    def _on_heartbeat(self, _message: Empty) -> None:
        self._last_heartbeat_s = self._now_s()

    def _process_pair(self, pair: ChunkPair) -> None:
        state = self._latest_state
        if state is None:
            self._publish_blocked(pair.base, "robot_state_missing")
            return
        now_s = self._now_s()
        result = self._pipeline.submit(
            pair.base,
            pair.residual,
            state,
            ExecutionSignals(
                controller_ready=self._controller_ready,
                emergency_stop_active=self._emergency_stop_active,
                last_heartbeat_s=self._last_heartbeat_s,
            ),
            now_s=now_s,
        )
        self._publish_result(result, now_s=now_s)

    def _publish_preview(self, chunk: ActionChunk) -> None:
        self._preview_publisher.publish(action_chunk_to_message(chunk))

    def _publish_hardware(self, chunk: ActionChunk) -> None:
        self._hardware_publisher.publish(action_chunk_to_message(chunk))

    def _publish_blocked(self, chunk: ActionChunk, reason: str) -> None:
        message = ExecutionStatusMsg()
        _set_header(message.header, self._now_s(), chunk.frame_id)
        message.task_id = chunk.task_id
        message.sequence_id = chunk.sequence_id
        message.mode = ExecutionMode.BLOCKED.value
        message.safe = False
        message.hardware_published = False
        message.reason_codes = [reason]
        message.detail = reason
        self._status_publisher.publish(message)

    def _publish_result(self, result: PipelineResult, *, now_s: float) -> None:
        message = ExecutionStatusMsg()
        frame_id = result.final_chunk.frame_id if result.final_chunk is not None else "base_link"
        _set_header(message.header, now_s, frame_id)
        message.task_id = result.task_id
        message.sequence_id = result.sequence_id
        message.mode = result.mode.value
        message.safe = result.safety_report.ok
        message.hardware_published = result.hardware_published
        message.reason_codes = list(
            dict.fromkeys((*result.safety_report.reason_codes, *result.gate_reasons))
        )
        details = [issue.message for issue in result.safety_report.issues]
        details.extend(result.gate_reasons)
        message.detail = "; ".join(details)
        self._status_publisher.publish(message)


def action_chunk_from_message(message: ActionChunkMsg) -> ActionChunk:
    """Convert a generated ROS message to the ROS-independent command type."""

    stamp_s = stamp_to_seconds(message.header.stamp.sec, message.header.stamp.nanosec)
    return ActionChunk(
        task_id=str(message.task_id),
        sequence_id=int(message.sequence_id),
        stamp_s=stamp_s,
        dt_s=float(message.dt_s),
        actions=tuple(
            CartesianDelta(
                dx_m=float(action.dx_m),
                dy_m=float(action.dy_m),
                dz_m=float(action.dz_m),
                dyaw_rad=float(action.dyaw_rad),
                gripper=float(action.gripper),
            )
            for action in message.actions
        ),
        frame_id=str(message.header.frame_id),
        policy_id=str(message.policy_id),
    )


def robot_state_from_message(message: RobotStateMsg) -> RobotState:
    """Convert a generated ROS state message to the safety-core type."""

    stamp_s = stamp_to_seconds(message.header.stamp.sec, message.header.stamp.nanosec)
    return RobotState(
        stamp_s=stamp_s,
        tcp_position_m=(
            float(message.tcp_position_m.x),
            float(message.tcp_position_m.y),
            float(message.tcp_position_m.z),
        ),
        tcp_yaw_rad=float(message.tcp_yaw_rad),
        joint_positions_rad=tuple(float(value) for value in message.joint_positions_rad),
        gripper=float(message.gripper),
        frame_id=str(message.header.frame_id),
    )


def action_chunk_to_message(chunk: ActionChunk) -> ActionChunkMsg:
    """Convert a safety-approved command to its generated ROS message."""

    message = ActionChunkMsg()
    _set_header(message.header, chunk.stamp_s, chunk.frame_id)
    message.task_id = chunk.task_id
    message.sequence_id = chunk.sequence_id
    message.policy_id = chunk.policy_id
    message.dt_s = chunk.dt_s
    actions: list[CartesianDeltaMsg] = []
    for action in chunk.actions:
        action_message = CartesianDeltaMsg()
        action_message.dx_m = action.dx_m
        action_message.dy_m = action.dy_m
        action_message.dz_m = action.dz_m
        action_message.dyaw_rad = action.dyaw_rad
        action_message.gripper = action.gripper
        actions.append(action_message)
    message.actions = actions
    return message


def _set_header(header: object, stamp_s: float, frame_id: str) -> None:
    sec, nanosec = seconds_to_stamp_parts(stamp_s)
    header.stamp.sec = sec  # type: ignore[attr-defined]
    header.stamp.nanosec = nanosec  # type: ignore[attr-defined]
    header.frame_id = frame_id  # type: ignore[attr-defined]


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SmartPickSafetyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
