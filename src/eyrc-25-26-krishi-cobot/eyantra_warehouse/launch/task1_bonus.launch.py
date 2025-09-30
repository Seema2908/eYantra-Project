#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 
*****************************************************************************************
*  Filename:       world.launch.py
*  Description:    Launch Ignition Gazebo Fortress world
*  Modified by:    Sahil
*  Author:         e-Yantra Team
*****************************************************************************************
'''

import os
from os.path import join
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    eyantra_warehouse = get_package_share_directory("eyantra_warehouse")

    world_file = LaunchConfiguration(
        "world_file", 
        default=join(eyantra_warehouse, "worlds", "eyantra_warehouse_task1_bonus.world")
    )

    gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": PythonExpression(["'", world_file, " -r'"])
        }.items(),
    )

    return LaunchDescription([
        # Set resource paths for Gazebo
        AppendEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=join(eyantra_warehouse, "worlds")
        ),
        AppendEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=join(eyantra_warehouse, "models")
        ),
        # Declare launch arguments
        DeclareLaunchArgument("world_file", default_value=world_file),

        gz_sim,
    ])
