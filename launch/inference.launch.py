##launch file
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    configs = [
        os.path.join(
            get_package_share_directory("roboto_inference"),
            "config",
            "inference.yaml",
        ),
    ]

    return LaunchDescription(
        [
            Node(
                package="roboto_inference",
                executable="roboto_inference_node",
                name="inference_node",
                parameters=configs,
                output="screen",
                # prefix=["xterm -e gdb -ex run --args"],
            ),
        ]
    )
