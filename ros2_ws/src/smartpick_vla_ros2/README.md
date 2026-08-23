# PickSort-VLA ROS 2 safety bridge

This package converts typed action chunks into a dry-run preview and, only
after all interlocks pass, a hardware command topic. It does not claim to be a
certified robot safety controller.

The default launch values are `dry_run:=true` and
`hardware_enabled:=false`. Hardware publication requires both
`dry_run:=false` and `hardware_enabled:=true`, plus a ready controller, a fresh
heartbeat, a released emergency stop, fresh robot feedback, and a final action
chunk that passes the Python safety supervisor.

An optional six-axis world-model Transformer can be attached by setting
`world_model_checkpoint`. It publishes predictive risk on
`/smartpick/predictive_risk`; the preview includes collision, task-outcome,
uncertainty, model-hash, and risk-reason fields. The complete incoming action
chunk is used for the imagined plan. The legacy five-channel ROS action is
adapted to the six-axis model by inserting `droll=0` before `gripper`.

`predictive_risk_blocking` defaults to `false` and is advisory. When enabled,
missing model/camera input and prediction errors fail closed in addition to the
existing deterministic safety gates. This does not make the package a
certified controller.

The ROS environment must also be able to import the `smartpick-vla` Python
package from this repository. Build and run from a ROS 2 shell:

```bash
python3 -m pip install -e /path/to/picksort-vla
cd /path/to/picksort-vla/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch smartpick_vla_ros2 safety_bridge.launch.py
```

Do not enable hardware output until the real profile contains measured
workspace, joint, timing, and camera calibration values and the cell has been
commissioned independently.
