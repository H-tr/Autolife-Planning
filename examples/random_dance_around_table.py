import os

import numpy as np
import trimesh
import vamp
from fire import Fire
from vamp import pybullet_interface as vpb

POINT_RADIUS = 0.01

script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))


def sample_valid(vamp_module, rng, env):
    while True:
        config = rng.next()
        if vamp_module.validate(config, env):
            return config


def main(robot="autolife", planner="rrtc", n_samples=10000):
    # 1. Setup Paths
    assets_dir = os.path.join(project_root, "assets", "envs", "rls_env", "pcd")
    table_pcd_path = os.path.join(assets_dir, "table.ply")

    # 2. Configure Robot and Planner
    print(f"Configuring {robot} with {planner}...")
    (
        vamp_module,
        planner_func,
        plan_settings,
        simp_settings,
    ) = vamp.configure_robot_and_planner_with_kwargs(robot, planner)

    # 3. Load and Process Pointcloud
    table_pcd = trimesh.load(table_pcd_path)
    points = np.array(table_pcd.vertices)

    # Rotate 90 degrees around Z axis
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = points @ rotation_matrix.T

    # Translate to middle of the scene
    translation = np.array([0.5, 3.0, 0.0])
    points += translation

    # 4. Setup Environment (Collision)
    env = vamp.Environment()

    # Get robot radii for pointcloud addition
    r_min, r_max = vamp_module.min_max_radii()

    env.add_pointcloud(points, r_min, r_max, POINT_RADIUS)

    # 5. Setup Simulation (Visualization)
    resources_dir = os.path.join(project_root, "third_party", "vamp", "resources")
    urdf_path = os.path.join(resources_dir, robot, f"{robot}_spherized.urdf")

    sim = vpb.PyBulletSimulator(urdf_path, vamp_module.joint_names(), visualize=True)

    # Draw pointcloud
    sim.draw_pointcloud(points)

    # 6. Define Start and Goal Configurations
    sampler = getattr(vamp_module, "halton")()

    def sample_and_choose(name, env):
        while True:
            config = sample_valid(vamp_module, sampler, env)

            # Visualize
            sim.set_joint_positions(config)

            # Wait for user input via PyBullet window
            while True:
                keys = sim.client.getKeyboardEvents()
                if ord("y") in keys and (keys[ord("y")] & sim.client.KEY_WAS_TRIGGERED):
                    print(f"Accepted {name}")
                    return config
                if ord("n") in keys and (keys[ord("n")] & sim.client.KEY_WAS_TRIGGERED):
                    print("Rejected, sampling again...")
                    break

    print("Press 'y' to accept the configuration, 'n' to reject and resample")

    start = sample_and_choose("start (hands under table)", env)
    goal = sample_and_choose("goal (hands on table)", env)

    print("Start:", start)
    print("Goal:", goal)

    # 7. Plan Path
    result = planner_func(start, goal, env, plan_settings, sampler)

    if result.solved:
        print("Path found! Simplifying...")
        simplify = vamp_module.simplify(result.path, env, simp_settings, sampler)
        plan = simplify.path

        # Interpolate to robot resolution for smooth animation
        plan.interpolate_to_resolution(vamp_module.resolution())

        print("Animating...")
        sim.animate(plan)
    else:
        print("Failed to find a path.")


if __name__ == "__main__":
    Fire(main)
