# SmartPick-VLA ROS 2 safety bridge

This package converts typed action chunks into a dry-run preview and, only
after all interlocks pass, a hardware command topic. It does not claim to be a
certified robot safety controller.

The default launch values are `dry_run:=true` and
`hardware_enabled:=false`. Hardware publication requires both
`dry_run:=false` and `hardware_enabled:=true`, plus a ready controller, a fresh
heartbeat, a released emergency stop, fresh robot feedback, and a final action
chunk that passes the Python safety supervisor.

The ROS environment must also be able to import the `smartpick-vla` Python
package from this repository. Build and run from a ROS 2 shell:

```bash
python3 -m pip install -e /path/to/smartpick-vla
cd /path/to/smartpick-vla/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch smartpick_vla_ros2 safety_bridge.launch.py
```

Do not enable hardware output until the real profile contains measured
workspace, joint, timing, and camera calibration values and the cell has been
commissioned independently.
