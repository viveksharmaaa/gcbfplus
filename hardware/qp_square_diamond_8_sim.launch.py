import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory("crazyflie_vicon_bringup")
    gcbf_share = get_package_share_directory("gcbfplus")
    square_yaml = os.path.join(
        bringup_share, "config", "crazyflies_8_sim_square_diamond.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value=EnvironmentVariable("ROS_DOMAIN_ID", default_value="88"),
            ),
            DeclareLaunchArgument("rviz", default_value="True"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("rate_hz", default_value="50.0"),
            DeclareLaunchArgument("print_latency", default_value="false"),
            DeclareLaunchArgument("max_runtime", default_value="60.0"),
            DeclareLaunchArgument("policy_test_duration", default_value="0.0"),
            DeclareLaunchArgument("acceleration_bound", default_value="2.5"),
            DeclareLaunchArgument("velocity_bound", default_value="1.0"),
            DeclareLaunchArgument("communication_radius", default_value="0.8"),
            DeclareLaunchArgument("car_radius", default_value="0.08"),
            DeclareLaunchArgument("qp_car_radius", default_value="0.10"),
            DeclareLaunchArgument("sim_command_delay", default_value="0.0"),
            DeclareLaunchArgument("sim_command_delay_jitter", default_value="0.0"),
            DeclareLaunchArgument("sim_position_noise_std", default_value="0.0"),
            DeclareLaunchArgument("sim_noise_seed", default_value="2026"),
            DeclareLaunchArgument("velocity_filter_window", default_value="0.0"),
            DeclareLaunchArgument("save_error_plots", default_value="true"),
            DeclareLaunchArgument("save_animation", default_value="true"),
            DeclareLaunchArgument(
                "simulation_crazyflies_yaml",
                default_value=square_yaml,
            ),
            DeclareLaunchArgument("qp_alpha", default_value="1.0"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", LaunchConfiguration("ros_domain_id")),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(bringup_share, "launch", "bringup_8.launch.py")
                ),
                launch_arguments={
                    "backend": "sim",
                    "mocap": "False",
                    "gui": LaunchConfiguration("gui"),
                    "rviz": LaunchConfiguration("rviz"),
                    "simulation_crazyflies_yaml": LaunchConfiguration(
                        "simulation_crazyflies_yaml"
                    ),
                    "ros_domain_id": LaunchConfiguration("ros_domain_id"),
                    "mass": "0.033",
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gcbf_share, "launch", "gcbf_crazyswarm_nodes.launch.py")
                ),
                launch_arguments={
                    "mode": "sim",
                    "controller": "centralized_qp",
                    "simulation_crazyflies_yaml": LaunchConfiguration(
                        "simulation_crazyflies_yaml"
                    ),
                    "ros_domain_id": LaunchConfiguration("ros_domain_id"),
                    "rate_hz": LaunchConfiguration("rate_hz"),
                    "lookahead_dt": "0.03",
                    "max_runtime": LaunchConfiguration("max_runtime"),
                    "policy_test_duration": LaunchConfiguration(
                        "policy_test_duration"
                    ),
                    "car_radius": LaunchConfiguration("car_radius"),
                    "qp_car_radius": LaunchConfiguration("qp_car_radius"),
                    "mass": "0.033",
                    "qp_acceleration_bound": LaunchConfiguration("acceleration_bound"),
                    "qp_velocity_bound": LaunchConfiguration("velocity_bound"),
                    "qp_communication_radius": LaunchConfiguration("communication_radius"),
                    "sim_command_delay": LaunchConfiguration("sim_command_delay"),
                    "sim_command_delay_jitter": LaunchConfiguration("sim_command_delay_jitter"),
                    "sim_position_noise_std": LaunchConfiguration("sim_position_noise_std"),
                    "sim_noise_seed": LaunchConfiguration("sim_noise_seed"),
                    "velocity_filter_window": LaunchConfiguration("velocity_filter_window"),
                    "velocity_filter_min_samples": "3",
                    "qp_alpha": LaunchConfiguration("qp_alpha"),
                    "qp_relaxation_penalty": "1000.0",
                    "acceleration_component_limit": LaunchConfiguration("acceleration_bound"),
                    "velocity_limit": LaunchConfiguration("velocity_bound"),
                    "max_planar_acceleration": "0.0",
                    "dynamic_reference_generator": "false",
                    "policy_diagnostics": "true",
                    "print_latency": LaunchConfiguration("print_latency"),
                    "save_error_plots": LaunchConfiguration("save_error_plots"),
                    "save_animation": LaunchConfiguration("save_animation"),
                }.items(),
            ),
        ]
    )
