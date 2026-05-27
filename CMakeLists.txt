cmake_minimum_required(VERSION 3.16)
project(global_path_deformer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(global_path_deformer_node
  src/global_path_deformer_node.cpp
)
target_compile_features(global_path_deformer_node PUBLIC cxx_std_17)

ament_target_dependencies(global_path_deformer_node
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
)

install(TARGETS
  global_path_deformer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
