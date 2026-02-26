import os
import time
from typing import Any

import numpy as np
import trimesh
from fire import Fire

from autolife_planning.config.robot_config import autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import motion_planning
from autolife_planning.planning.validation import valid_config
from autolife_planning.types import RobotConfiguration
from autolife_planning.utils.vamp_utils import create_planning_context

POINT_RADIUS = 0.01

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))


def sample_valid(context):
    """Sample a valid (collision-free) configuration."""
    while True:
        config = context.sampler.next()
        if valid_config(config, context):
            return config


def main(planner="rrtc"):
    # 1. Load and Process Pointcloud
    assets_dir = os.path.join(project_root, "assets", "envs", "rls_env", "pcd")
    table_pcd_path = os.path.join(assets_dir, "table.ply")

    table_pcd: Any = trimesh.load(table_pcd_path)
    points = np.array(table_pcd.vertices)

    # Rotate 90 degrees around Z axis
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = points @ rotation_matrix.T

    # Translate to middle of the scene
    translation = np.array([0.5, 3.0, 0.0])
    points += translation

    # 2. Create Planning Context
    planning_context = create_planning_context(
        "autolife", points, planner, POINT_RADIUS
    )

    # 3. Setup Visualization Environment
    env = PyBulletEnv(autolife_robot_config, visualize=True)
    env.add_pointcloud(points)

    # 4. Interactive Configuration Sampling
    def sample_and_choose(name):
        while True:
            config = sample_valid(planning_context)
            env.set_joint_states(RobotConfiguration.from_array(config))

            # Wait for user input via PyBullet window
            while True:
                keys = env.sim.client.getKeyboardEvents()
                if ord("y") in keys and (
                    keys[ord("y")] & env.sim.client.KEY_WAS_TRIGGERED
                ):
                    print(f"Accepted {name}")
                    return RobotConfiguration.from_array(config)
                if ord("n") in keys and (
                    keys[ord("n")] & env.sim.client.KEY_WAS_TRIGGERED
                ):
                    print("Rejected, sampling again...")
                    break

    print("Press 'y' to accept the configuration, 'n' to reject and resample")

    start = sample_and_choose("start")
    goal = sample_and_choose("goal")

    print("Start:", start)
    print("Goal:", goal)

    # 5. Plan Path
    plan = motion_planning.plan_motion(start, goal, planning_context)

    if plan is not None:
        print("Animating...")
        for i in range(len(plan)):
            env.set_joint_states(RobotConfiguration.from_array(plan[i]))
            time.sleep(1.0 / 60.0)
    else:
        print("Failed to find a path.")


if __name__ == "__main__":
    Fire(main)
