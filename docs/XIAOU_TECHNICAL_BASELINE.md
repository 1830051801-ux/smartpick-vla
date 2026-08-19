# XiaoU six-axis technical baseline

This document connects the user-supplied XiaoU technical note to PickSort-VLA
without overstating what has been verified in this repository. The machine-readable
source of truth is
[`configs/real/xiaou_hardware_profile.yaml`](../configs/real/xiaou_hardware_profile.yaml).

The profile is an inspection and simulation-mapping baseline. It is not a ROS
driver, CAN configuration, calibration package, or permission to move the arm.

## Source and use

| Item | Value |
| --- | --- |
| Source document | `小U具身智能桌面服务机器人_项目技术说明.pdf` |
| SHA-256 | `f1a776eb8d9d961dbe2d9a01f03bfc1268698b97de083ee2ca72aba8b29489b0` |
| Profile schema | `picksort-xiaou-hardware-profile/v1` |
| Repository role | Hardware facts, software boundary, and readiness gates |

The PDF is not committed to this public repository. Its checksum lets a local
source document be matched to this profile without publishing the report or
implying that every statement in it has been re-tested here.

Inspect the normalized profile without opening serial, CAN, ROS, or motor
transport:

```powershell
.\.venv\Scripts\python.exe -m smartpick_vla xiaou-hardware-profile
```

The output always reports `hardware_transport_opened: false`.

## Mapped hardware facts

### Mechanical baseline

The profile records a six-revolute-joint layout with CAD/URDF geometry values
in millimetres:

| Segment | Length |
| --- | ---: |
| J1-J2 | 156.0 |
| J2-J3 | 180.0 |
| J3-J4 | 180.0 |
| J4-J5 | 93.0 |
| J5-J6 | 106.0 |
| J6-axis-center TCP | 207.5 |
| Tool-tip TCP | 239.9 |

Software working limits are J1 `[-165, 165]`, J2 `[-125, 125]`, J3
`[-135, 135]`, J4 `[-175, 175]`, J5 `[-85, 115]`, and J6 `[-175, 175]`
degrees. They are a baseline for cross-checking a future URDF, firmware, and
MoveIt configuration, not a replacement for encoder zeroing, actual hard-stop
measurements, payload testing, or a certified collision model.

### Pi, F407, and CAN boundary

The declared Pi-to-F407 link is `/dev/serial0`, 115200 baud, 8N1, 3.3 V TTL.
The F407-to-joint link is Classic CAN with 11-bit IDs, DLC 8, and a 1 Mbps
engineering bitrate. CAN IDs are intentionally absent: the technical note does
not establish a versioned identifier map and this repository must not invent one.

A trajectory point is little-endian `6 x float32` absolute joint angles in
degrees plus little-endian `uint16 duration_ms`, for a 26-byte payload.
`duration_ms` is the shared synchronization field. The F407-side design target
is 10 ms interpolation and a FIFO/admission boundary; this repository records
that contract but implements no serial or CAN transport.

Before a physical FIFO could accept a trajectory, the contract requires CRC,
sequence, free-slot, feedback-freshness, zero-valid, joint-limit, STOP-clear,
and ESTOP-clear checks. That is deliberately stricter than having a valid packet.

## Vision, decision, and planning boundary

The note establishes four current YOLO training classes: `pen`, `bottle`,
`cola`, and `earphone`. `cup` and `tissue` are explicitly recorded as untrained
for the real dataset; no repository content should claim their real-camera
recognition generalizes.

The Transformer/VLA boundary is task-level only. Its allowed fields are
`task_candidate`, `phase`, `action_parameters`, and `risk`; it may not directly
output joint angles, PWM, or raw CAN/UART bytes. The intended chain remains:

```text
camera detection -> calibrated base_link fact -> task proposal ->
MoveIt/ROS review -> deterministic trajectory -> F407 admission -> CAN
```

MoveIt remains `review_only` here. The current ROS 2/DDS path is not verified,
so `ros2_dds_verified: false` is explicit.

## Relationship to PickSort-VLA simulation

PickSort-VLA's six-axis MuJoCo scene is a generic simulator contract with a
sixth tool-roll joint. It supports perception, language-conditioned control,
trajectory behavior, and safety-filter experiments. It is not a calibrated
dynamic replica of XiaoU merely because both expose six joints.

| Repository component | XiaoU relationship | Not established |
| --- | --- | --- |
| Six-axis simulator action/state | Same arm-joint count; planning research surface | Encoder offset, gear ratio, inertia, friction, motor gains |
| `xiaou-preview` adapter | Pixel-to-`base_link` target-preview contract | Current camera calibration, TCP/gripper height, live motion |
| Real2Sim2Real types | Traceable frame, timing, limit, and dry-run gates | Digital-twin identification or transfer success |
| ROS 2 package | Typed dry-run and review architecture | Current ROS 2/DDS build, hardware driver, physical execution |

The generic policy uses Cartesian metres/radians. The XiaoU payload uses six
absolute joint angles in degrees plus duration. A validated planner and
explicit conversion layer must sit between them; this update does not add a
model-to-payload conversion.

## Evidence ledger

| Evidence level | What can be said | What must not be said |
| --- | --- | --- |
| Current repository | The loader validates geometry, trajectory shape, low-level-output boundary, and disabled execution. | It has communicated with an F407, CAN bus, or physical arm. |
| User technical note | CAD/URDF/POE/DH, protocol design, ONNX/Transformer assets, and historical/offline checks exist in the supplied material. | Those records are a current PickSort-VLA benchmark or latest hardware acceptance. |
| Historical/offline figures | They guide follow-up work and evidence review. | They prove current real grasp rate, Pi FPS/RSS, firmware behavior, or safety certification. |
| Future acceptance | Fresh feedback, calibration, STOP/ESTOP, and admission logs can support a new execution claim. | Any claim before a versioned record exists. |

Source-note figures such as the historical single-point angular error, local
`8/8` side-grasp regression, multi-object offline regression, YOLO dataset
audit, and Transformer holdout result are not copied into PickSort-VLA result
tables. Their provenance differs and some source simulations used a different
MuJoCo version than the repository's recorded runs.

## Required acceptance before real motion

All execution flags remain `false`. A new, versioned hardware acceptance record
must show the following before any higher-level code is described as ready for
real motion:

1. Current F407 firmware and six-axis feedback with timestamps and feedback age.
2. Physical ESTOP/STOP, FIFO clearing, and CAN-watchdog checks.
3. Measured encoder zero, joint direction/limits, TCP, gripper geometry, table
   height, and camera-to-`base_link` calibration.
4. Actual CAN IDs, payload framing, CRC, sequence handling, and `free_slots`
   flow control against the current firmware.
5. Target-environment ROS 2/DDS, MoveIt collision review, and execution-adapter
   validation.
6. Logged empty-space, single-axis, synchronized six-axis, and real-object
   trials with hashes, failure labels, and operator authorization.

Until then, use simulator evaluation, real-log replay, `xiaou-preview`, and
hardware-profile inspection only.
