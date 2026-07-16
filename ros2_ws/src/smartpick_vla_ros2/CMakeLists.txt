cmake_minimum_required(VERSION 3.16)
project(smartpick_vla_ros2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CartesianDelta.msg"
  "msg/ActionChunk.msg"
  "msg/RobotState.msg"
  "msg/ExecutionStatus.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/smartpick_vla_safety_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
