"""Launch the SmartPick-VLA bridge in dry-run mode unless explicitly overridden."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    config_path = LaunchConfiguration("config_path")
    dry_run = LaunchConfiguration("dry_run")
    hardware_enabled = LaunchConfiguration("hardware_enabled")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("hardware_enabled", default_value="false"),
            Node(
                package="smartpick_vla_ros2",
                executable="smartpick_vla_safety_bridge",
                name="smartpick_vla_safety_bridge",
                output="screen",
                parameters=[
                    {
                        "config_path": config_path,
                        "dry_run": ParameterValue(dry_run, value_type=bool),
                        "hardware_enabled": ParameterValue(
                            hardware_enabled,
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
