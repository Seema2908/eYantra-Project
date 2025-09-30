cmake_minimum_required(VERSION 3.5)
project(pymoveit2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(urdf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/MoveToPose.srv"
)


# Install Python example scripts
set(EXAMPLES_DIR examples)
install(PROGRAMS
    ${EXAMPLES_DIR}/twist_servo_cal.py
    ${EXAMPLES_DIR}/kdl_parser_py.py
    DESTINATION lib/${PROJECT_NAME}
)

install(FILES
    ${EXAMPLES_DIR}/ur5_arm.urdf
    DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
    srv
    DESTINATION share/${PROJECT_NAME}
)


ament_package()

