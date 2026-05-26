import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('global_path_deformer'),
        'config',
        'global_path_deformer.yaml',
    )

    return LaunchDescription([
        Node(
            package='global_path_deformer',
            executable='global_path_deformer_node',
            name='global_path_deformer',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': False},
            ]
        )
    ])
