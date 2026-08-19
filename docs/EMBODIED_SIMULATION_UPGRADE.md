# Embodied Simulation Upgrade

This document describes the August 2026 simulator upgrade. It separates
implemented behavior, local evidence, and deliberately unclaimed capability.
No path in this upgrade publishes a serial, CAN, ROS motor command, or enables
XiaoU hardware motion.

## What changed

### Configurable six-axis control

`SmartPickEnv(six_axis=True)` activates a six-axis desktop-arm variant while
keeping the original five-axis interface as the default. The two contracts are:

| Variant | Robot state | Normalized action |
| --- | ---: | --- |
| Legacy five-axis | 24 | `[dx, dy, dz, dyaw, gripper]` |
| Optional six-axis | 29 | `[dx, dy, dz, dyaw, droll, gripper]` |

The sixth joint is a tool-roll axis. Existing five-axis datasets, checkpoints,
and the external XiaoU action contract stay unchanged. A six-axis policy must
use a dataset and model configuration with `robot_state_dim: 29` and
`action_dim: 6`.

### Ordered multi-object missions

`mission_length` selects one to three unique quality classes in a single
MuJoCo episode. After a correct place event, the environment changes its active
language instruction to the next class and keeps the previously sorted part in
its tray. Episode success means every requested subtask finished in order.

The privileged `IKWaypointExpert` tracks task switches only for synthetic
demonstrations and upper-bound evaluation. It is not a vision-only result.

### Multi-view synthetic perception data

`picksort generate-perception` exports a compressed, simulator-labelled archive
from top, oblique, and wrist cameras. Each sample can contain:

- RGB images, metric depth maps, and instance masks;
- per-view `bbox_xyxy` and visible-pixel counts for all three parts;
- current robot state, expert action, language instruction, episode, and step;
- camera names and quality-class ordering inside the archive;
- a JSON manifest with scene/data SHA-256 hashes, randomization values, and a
  `synthetic_data: true` disclosure.

Masks and boxes come from MuJoCo segmentation rendering, so occluded pixels are
not guessed from privileged object poses. The stored world pose is diagnostic
metadata only and is not an input to the VLA policies.

### Predictive simulator safety shield

`PredictiveSafetyFilter` takes a normalized candidate action, derives the same
IK and position-control target used by the environment, copies the MuJoCo
state, and rolls it forward for a bounded number of control steps. It uses a
geometric motion-scale search before falling back to no Cartesian motion.

The benchmark writes the filter configuration and a per-episode report into
`metadata_json`: intervention count/rate, rejected-all-motion count, average
motion scale, and actions whose full-scale rollout predicted a collision.

This is a simulation guard, not a real-machine safety system. The default
filter can ignore finger-to-table brushes caused by the simplified
proximity-grasp geometry, but it still treats tray-wall, arm/palm, and
non-target-object contacts as unsafe. Its constraints have not been validated
on XiaoU hardware.

### RGB-guided six-axis release path

The release path combines the upgrade components into one auditable loop:

```text
top RGB -> spatial heatmap localization -> three-frame confirmation
         -> nine-point homography -> language task parsing
         -> six-axis waypoint/IK control -> predictive safety filter -> MuJoCo
```

The top-camera home keyframe was selected by a fixed-seed pose sweep and then
regression-tested. The release pose is
`[1.5, -0.2, -1.8, -1.142, 0, 0]` radians; it keeps the target visible on the
previously failing seed 909 while remaining collision-free in the release
seed set.

Recorded release evidence:

| Artifact | Result |
|---|---:|
| `vision_six_axis_release_v2.npz` | 3,600 synthetic RGB frames from 360 episodes |
| `vision_six_axis_release_v2/best.pt` | 305,702 parameters; 1.606 px validation error |
| `vision_guided_six_axis_release_v2` | 15/15 success, 0/15 collision episodes |
| Closed-loop metrics | 12.213 mm mean localization error; 4.279 ms mean perception-to-action latency |

The run used a simulated calibration fixture and `grasp_assist=true`. It does
not claim physical-camera, real-arm, or industrial inspection performance.

## Reproduce the verified local artifacts

The upgrade smoke commands below were run on 2026-08-15 with the repository
virtual environment, Python 3.13, CPU PyTorch, and MuJoCo 3.10. The release
vision run was regenerated on 2026-08-19 with the same MuJoCo version and CUDA
PyTorch on an RTX 3050 Laptop GPU.

```powershell
$py = ".\.venv\Scripts\python.exe"

# Multi-view RGB-D, masks, boxes, state, action, and language labels.
& $py -m smartpick_vla generate-perception `
  --config configs/data/perception_multiview_smoke.yaml `
  --output results/upgrade_20260815/perception_multiview_smoke.npz

# Three ordered parts in a six-axis simulator using the privileged IK expert.
& $py -m smartpick_vla evaluate `
  --config configs/eval/mission_six_axis_expert_smoke.yaml `
  --include-expert `
  --output results/upgrade_20260815/mission_six_axis_expert_smoke

# Separate six-axis safety-filter smoke run.
& $py -m smartpick_vla evaluate `
  --config configs/eval/safety_six_axis_smoke.yaml `
  --include-expert `
  --output results/upgrade_20260815/safety_six_axis_smoke

# Exercise the six-axis temporal action-chunk training path.
& $py -m smartpick_vla generate `
  --config configs/train/data_mission_six_axis_smoke.yaml `
  --output datasets/generated/upgrade_20260815/mission_six_axis_expert.npz

& $py -m smartpick_vla train `
  --config configs/train/temporal_vla_mission_six_axis_smoke.yaml `
  --dataset datasets/generated/upgrade_20260815/mission_six_axis_expert.npz `
  --output checkpoints/upgrade_20260815/temporal_vla_mission_six_axis
```

Recorded smoke evidence:

| Artifact | Recorded result | Interpretation |
| --- | --- | --- |
| `perception_multiview_smoke.npz` | 9 labelled synthetic multi-view samples | Verifies the export contract, not detector accuracy. |
| `mission_six_axis_expert_smoke` | 6/6 three-subtask episodes over ID/OOD/physics suites | Privileged IK upper bound with the configured grasp assist. |
| `safety_six_axis_smoke` | 8/8 single-subtask expert episodes with filter enabled | Nominal expert path did not require an intervention; not a shield-effectiveness score. |
| `temporal_vla_mission_six_axis` | 2/2 expert demonstrations, 521 transitions, 16 updates, 552,150 parameters | Training-path smoke only; no policy rollout metric is claimed. |

For a visual smoke artifact, run:

```powershell
& $py -m smartpick_vla evaluate `
  --config configs/eval/mission_six_axis_showcase.yaml `
  --include-expert --gif `
  --output results/upgrade_20260815/mission_six_axis_showcase
```

To reproduce the release RGB-guided six-axis result:

```powershell
& $py -m smartpick_vla generate-perception `
  --config configs/data/vision_six_axis.yaml `
  --output datasets/generated/vision_six_axis_release_v2.npz

& $py -m smartpick_vla train-vision `
  --config configs/train/vision_localizer_six_axis.yaml `
  --dataset datasets/generated/vision_six_axis_release_v2.npz `
  --output checkpoints/vision_six_axis_release_v2

& $py -m smartpick_vla evaluate-vision `
  --config configs/eval/vision_guided_six_axis.yaml `
  --checkpoint checkpoints/vision_six_axis_release_v2/best.pt `
  --output results/vision_guided_six_axis_release_v2
```

## Deliberate boundaries

- The arm is a self-contained MuJoCo primitive model, not a calibrated URDF or
  named industrial robot.
- The release RGB-guided result is a one-object-per-episode visual benchmark;
  the three-object mission remains a privileged IK result. A learned
  multi-mission vision-policy score is not claimed.
- Synthetic segmentation labels do not make the project an industrial defect
  detector, and they should not be mixed with real factory images without a
  declared data split and provenance.
- The XiaoU bridge remains planning-only. It converts camera coordinates into
  auditable pose previews and rejects incomplete profiles rather than moving
  the robot.
