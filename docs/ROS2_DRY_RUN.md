# ROS 2 dry-run boundary

## Safety intent

The ROS 2 integration is an inspection boundary between learned policy output
and a future robot-specific executor. The default launch configuration is:

```text
dry_run=true
hardware_enabled=false
```

It validates, composes, limits, and publishes action previews. It does not
publish to a vendor motor-controller topic and is not a certified safety
controller. Changing a parameter does not supply the missing hardware executor
or safety case.

## Package and launch

The ROS package is `smartpick_vla_ros2`. In an environment with ROS 2 and a
built workspace, launch the preview path with:

```bash
ros2 launch smartpick_vla_ros2 safety_bridge.launch.py
```

`rclpy` and ROS message-generation dependencies come from the selected ROS 2
distribution, not from the core package's PyPI extras. The core simulator and
model tests must remain usable without ROS 2 installed.

## Message contract

The package defines these messages:

- `CartesianDelta`: one canonical physical action;
- `ActionChunk`: an ordered sequence of Cartesian deltas plus metadata;
- `RobotState`: state, timestamp, and frame information used by the preview
  gate;
- `ExecutionStatus`: accepted/rejected status and reason codes.
- `PredictiveRisk`: optional world-model probabilities, uncertainty, model
  identity, and risk reasons.

The canonical order is:

```text
[dx_m, dy_m, dz_m, dyaw_rad, gripper]
```

Translation is metres, yaw is radians, gripper is normalized to `[-1,1]`, and
`frame_id` must be `base_link`. A chunk has shape `[H,5]`; empty chunks,
non-finite values, mismatched horizons, unknown frames, and stale timestamps are
rejected.

The generated ROS message uses named `CartesianDelta` fields, so wire order is
not inferred from an untyped numeric array. Compatibility across a future
message-schema change requires a new interface/package version; the current
message does not carry a separate schema-version field.

## Topics

| Direction | Topic | Message | Purpose |
| --- | --- | --- | --- |
| input | `/smartpick/base_action_chunk` | `ActionChunk` | base BC/VLA proposal |
| input | `/smartpick/residual_action_chunk` | `ActionChunk` | requested residual proposal |
| input | `/smartpick/robot_state` | `RobotState` | current state for freshness and workspace checks |
| input | `/smartpick/controller_ready` | `std_msgs/Bool` | controller-ready interlock |
| input | `/smartpick/emergency_stop` | `std_msgs/Bool` | emergency-stop interlock |
| input | `/smartpick/controller_heartbeat` | `std_msgs/Empty` | liveness gate |
| input | `/smartpick/camera/rgb` | `sensor_msgs/Image` | latest RGB frame for predictive preview |
| input | `/smartpick/instruction` | `std_msgs/String` | latest language task for predictive preview |
| output | `/smartpick/action_preview` | `ActionChunk` | composed action before authorization |
| gated output | `/smartpick/safe_action_chunk` | `ActionChunk` | hardware-mode output only after all gates; silent in default dry-run |
| output | `/smartpick/execution_status` | `ExecutionStatus` | decision and rejection reason |
| output | `/smartpick/predictive_risk` | `PredictiveRisk` | imagined rollout risk preview |

Topic names are launch parameters in deployments, but remapping must preserve
the dry-run separation from motor topics.

## Predictive world-model preview

Set `world_model_checkpoint` to a trusted six-axis checkpoint to enable the
optional monitor. It maintains an episode-bounded RGB/state/action history and
passes the complete incoming action chunk as the imagined action plan. Since
the legacy ROS action message has five channels, the adapter inserts
`droll=0` before `gripper` for the six-channel world-model contract.

The monitor publishes collision, termination, wrong-pick, wrong-bin, maximum
state standard deviation, model hash, and a list of risk reasons. The default
`predictive_risk_blocking=false` makes this advisory. When set to `true`, the
bridge fails closed if the model is unavailable, the camera has not produced a
valid frame, or a prediction exceeds any configured risk/uncertainty threshold.
This is an additional software gate, not a certified safety controller.

Example preview launch:

```bash
ros2 launch smartpick_vla_ros2 safety_bridge.launch.py \
  world_model_checkpoint:=/path/to/world_model_six_axis_smoke_v2/best.pt \
  predictive_risk_blocking:=false
```

The current `RobotState` interface lacks joint velocities and wrist-roll
feedback. The adapter fills those fields with explicit zeros and therefore
keeps this path preview-only until a versioned feedback interface and measured
time/calibration contract are validated.

## Composition

The base and residual chunks must share task/sequence identity, frame, horizon,
control period, and timestamp. For each step and dimension:

```text
bounded_residual = clip(residual, -residual_limit, residual_limit)
composed         = base + bounded_residual
decision         = inspect(composed, robot_state, limits)
```

Residual limits are per dimension in physical ROS units. Gripper composition
is not silently clipped: an out-of-range final gripper or motion is rejected.
The optional JSONL audit record retains the original base, requested residual,
bounded residual, composed action, final decision, and reason codes.

The current bridge requires matching base and residual chunks. It has no
missing-residual-to-zero mode. Unpaired chunks stay in a bounded pairing buffer
and the oldest entry is discarded when that buffer exceeds its configured
capacity; this is not equivalent to executing a zero residual.

## Validation gates

Before publishing a safe preview, the node checks at least:

1. matching task, sequence, frame, horizon, timestamp, and control period for
   the base/residual pair;
2. exact `base_link` frame;
3. finite numeric values and a non-empty typed action sequence;
4. command and state age;
5. robot-state freshness and validity;
6. heartbeat freshness;
7. per-step Cartesian, yaw, and gripper residual limits;
8. accumulated TCP workspace bounds across the chunk;
9. configured workspace and optional current-joint bounds;
10. dry-run/hardware mode consistency.

Any failed mandatory check rejects the full chunk. Partial execution after a
validation failure is not permitted. Rejection publishes a machine-readable
status and retains enough context for diagnosis without exposing credentials or
sensitive image data.

## Watchdog behavior

On each paired command, the bridge fails closed when the base/residual command,
robot state, or heartbeat is stale. It emits a rejected status and does not
publish that chunk to the hardware-output topic. The current node does not run
a periodic timer that emits a separate hold command after traffic stops; a real
controller must provide its own independent watchdog and safe-stop behavior.

ROS time and wall time must not be mixed without an explicit clock policy.
Simulation with `/use_sim_time` requires a live clock and a separate timeout
strategy for a paused clock.

## Dry-run verification

Before any hardware-specific work, exercise these cases with tests or recorded
messages:

- valid zero-residual and bounded-residual chunks;
- NaN/Inf, empty/wrong action count, and wrong frame;
- excessive per-step and cumulative motion;
- stale base, residual, robot state, and heartbeat;
- horizon/timestamp mismatch;
- workspace boundary and gripper saturation;
- dropped/reordered pairs and bounded pending-buffer behavior;
- `hardware_enabled=true` while the release remains dry-run-only.

Save input/output message fixtures and `ExecutionStatus` reason codes as
artifacts. A successful dry-run means the declared software checks behaved as
tested; it does not validate robot dynamics or physical safety.

## Python-side preview

ROS-independent tests and log replay may use the Python API under
`smartpick_vla.real` to construct canonical chunks, apply residual composition,
and obtain dry-run status objects. This keeps safety semantics testable on
Windows systems without ROS 2. It is not a back door to hardware execution.

## Hardware extension boundary

A future robot adapter belongs downstream of `/smartpick/safe_action_chunk` and
requires its own package, review, and commissioning evidence. At minimum it
must add robot-model joint/workspace validation, controller-mode checks,
velocity/acceleration/jerk and force/torque limits, hardware watchdog, guarded
enable procedure, emergency-stop integration, and operator supervision.

Do not modify this dry-run node to directly call a vendor SDK merely to shorten
the path. Keeping preview and execution separate is an intentional safety and
audit boundary.
