# global_path_deformer

ROS 2 Jazzy package for Ubuntu 24.04. It locally deforms a global path using a realtime `nav_msgs/OccupancyGrid` map.

## Environment

- Ubuntu 24.04
- ROS 2 Jazzy
- C++17

## Inputs

- `/global_path` (`nav_msgs/Path`): global path. `pose.position.z` is preserved as velocity.
- `/map` (`nav_msgs/OccupancyGrid`): realtime occupancy/cost map.
- `/odom`, `/static_obstacle`, `/dynamic_obstacle`, `/obj_flag`: subscribed for interface compatibility.

## Outputs

- `/Path` (`nav_msgs/Path`): corrected path.
- `/local_path` (`visualization_msgs/Marker`): corrected window marker.

## Vehicle launch

```bash
ros2 launch global_path_deformer local_planner.launch.py
```

## Rosbag test launch

```bash
ros2 launch global_path_deformer test_rosbag.launch.py bag_path:=/path/to/rosbag2
```
# global_path_deformer
