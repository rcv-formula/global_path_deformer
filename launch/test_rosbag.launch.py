import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bag_path = LaunchConfiguration('bag_path')
    play_bag = LaunchConfiguration('play_bag')
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = os.path.join(
        get_package_share_directory('global_path_deformer'),
        'config',
        'global_path_deformer.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument('bag_path', default_value='', description='Path to rosbag2 directory'),
        DeclareLaunchArgument('play_bag', default_value='false', description='Play rosbag2 when true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path, '--clock'],
            condition=IfCondition(play_bag),
            output='screen',
        ),

        Node(
            package='global_path_deformer',
            executable='global_path_deformer_node',
            name='global_path_deformer',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': use_sim_time},
            ]
        )
    ])
