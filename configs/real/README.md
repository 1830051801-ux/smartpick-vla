# Real2Sim2Real configuration

`default.yaml` is a dry-run preparation profile. It defines the canonical
`base_link` action order, software workspace/rate checks, bounded residual,
controller heartbeat, replay timing and an explicitly uncalibrated camera
transform.

Before hardware output, copy the profile and replace the workspace, joint
limits, controller timing and camera-to-base transform with measured values.
Changing only `dry_run` is insufficient: hardware publication also requires
`hardware_enabled: true`, a ready controller, a fresh heartbeat and no active
emergency stop.

For the XiaoU-specific six-axis baseline, see
[`xiaou_hardware_profile.yaml`](xiaou_hardware_profile.yaml) and
[`docs/XIAOU_TECHNICAL_BASELINE.md`](../../docs/XIAOU_TECHNICAL_BASELINE.md).
That profile is inspection-only: it records source-traced geometry and
Pi/F407/CAN interface facts, but it cannot authorize hardware motion or replace
measured calibration and firmware acceptance.
