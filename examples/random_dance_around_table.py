import os
import time
from typing import Any

import numpy as np
import trimesh
from fire import Fire

from autolife_planning.config.robot_config import autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

POINT_RADIUS = 0.01

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))


def main(planner_name="rrtc"):
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

    # 2. Create Planner
    planner = create_planner(
        "autolife",
        config=PlannerConfig(planner_name=planner_name, point_radius=POINT_RADIUS),
        pointcloud=points,
    )

    # 3. Setup Visualization Environment
    env = PyBulletEnv(autolife_robot_config, visualize=True)
    env.add_pointcloud(points)

    # 4. Interactive Configuration Sampling
    def sample_and_choose(name):
        while True:
            config = planner.sample_valid()
            env.set_configuration(config)

            # Wait for user input via PyBullet window
            while True:
                keys = env.sim.client.getKeyboardEvents()
                if ord("y") in keys and (
                    keys[ord("y")] & env.sim.client.KEY_WAS_TRIGGERED
                ):
                    print(f"Accepted {name}")
                    return config
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
    result = planner.plan(start, goal)

    if result.success:
        print(
            f"Path found! {result.path.shape[0]} waypoints, "
            f"{result.planning_time_ns / 1e6:.1f}ms"
        )
        for i in range(result.path.shape[0]):
            env.set_configuration(result.path[i])
            time.sleep(1.0 / 60.0)
    else:
        print(f"Planning failed: {result.status.value}")


if __name__ == "__main__":
    Fire(main)
