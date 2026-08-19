# XiaoU six-axis planning-preview bridge

## Scope

This module connects the XiaoU vision contract to the PickSort-VLA simulation
and Real2Sim2Real workflow without creating a motor-control route. It accepts a
stable 2D detector result, projects it with XiaoU's pixel-to-`base_link`
homography, applies a complete object grasp profile, and emits three target
poses for the six-axis planner:

```text
YOLO center (u, v, confidence)
  -> workspace_homography.yaml
  -> base_link (x, y)
  -> object profile (grasp / approach / lift heights)
  -> pregrasp -> grasp -> lift PoseStamped previews
  -> ROS 2 / MoveIt planning-only boundary
```

The output is a JSON preview. It does not connect to UART, CAN, a robot driver,
or a MoveIt execution service.

## Inputs and safety checks

`picksort xiaou-preview` validates all of the following before producing a
target:

- a finite, invertible 3x3 homography with `type: pixel_to_robot_base_mm`;
- homography `max_error_mm` below the configured threshold (default 2 mm);
- a finite detection timestamp and confidence at or above the configured
  threshold (default 0.55);
- a complete per-label grasp profile with ordered
  `grasp_height_m < approach_height_m <= lift_height_m` values;
- named source and `base_link` target frames.

The current XiaoU hardware profile intentionally leaves several physical
heights, gripper values, and placements as `null` until they are measured. The
bridge rejects such incomplete profiles. It does not fill in guesses.

## Simulation-only example

The committed assets are explicitly synthetic:

- `configs/real/xiaou_demo_homography.yaml`
- `configs/real/xiaou_simulated_profiles.yaml`

They are useful for validating coordinate order and planner message shape, but
they are not a camera calibration or grasp configuration for a real arm.

```powershell
.\.venv\Scripts\python.exe -m smartpick_vla xiaou-preview `
  --homography configs/real/xiaou_demo_homography.yaml `
  --profiles configs/real/xiaou_simulated_profiles.yaml `
  --label cola --u-px 1030 --v-px 490 --confidence 0.92 `
  --output results/examples/xiaou_plan_preview.json
```

The output has `planning_only: true` and `real_motion_authorized: false`.
Each target contains a `geometry_msgs/PoseStamped`-compatible preview with
`frame_id: base_link` and a yaw-only unit quaternion.

## Integrating measured XiaoU data

When measured data is available, pass the actual `workspace_homography.yaml`
and a profile file whose height values are populated from physical calibration.
The bridge is only one input to the existing safety chain. Before hardware
motion, the downstream XiaoU stack must separately validate:

1. ROS 2 / MoveIt build and collision scene.
2. TCP zero, joint direction, limits, and feedback.
3. Fresh camera and robot timestamps.
4. F407 protocol, CAN IDs, watchdog, and emergency-stop behavior.
5. A planning-only review followed by explicit execution authorization.

No result from this module is evidence of physical grasp success.
