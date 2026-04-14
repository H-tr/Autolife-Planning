"""Motion planning around a table: plan a collision-free arm path.

Loads the bundled ``table.ply`` point cloud, places it in front of
the robot, and asks a 7-DOF left-arm planner to weave a path from
HOME to a randomly sampled collision-free goal.  The C++ collision
checker sees every point as a small sphere (``point_radius``), so
the planner has to curve the arm around the table to reach the
sampled goal.

    pixi run python examples/planning/motion.py
    pixi run python examples/planning/motion.py --planner_name bitstar --time_limit 3
"""

from pathlib import Path

import numpy as np
import trimesh
from fire import Fire

import autolife_planning
from autolife_planning.autolife import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig


def load_table(distance: float = 0.85, height: float = 0.35) -> np.ndarray:
    """Load ``table.ply`` and place it in front of the robot.

    The bundled scan lives a few metres away from the origin in its
    own coordinate frame, so we recentre it and push it forward along
    ``+x``.  Returns an ``(N, 3) float32`` point cloud ready to hand
    to the planner and to ``env.add_pointcloud``.
    """
    pkg_root = Path(autolife_planning.__file__).parent
    pcd = trimesh.load(str(pkg_root / "resources" / "envs" / "pcd" / "table.ply"))
    pts = np.asarray(pcd.vertices, dtype=np.float32)
    pts = pts - pts.mean(axis=0)
    # Rotate 90° around +z so the long side lines up with the robot's y axis.
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    pts = pts @ rot.T
    pts[:, 0] += float(distance)
    pts[:, 2] += float(height)
    return pts


def main(
    planner_name: str = "rrtc",
    time_limit: float = 2.0,
    point_radius: float = 0.012,
) -> None:
    env = PyBulletEnv(autolife_robot_config, visualize=True)

    cloud = load_table()
    env.add_pointcloud(cloud, pointsize=3)

    planner = create_planner(
        "autolife_left_arm",
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            point_radius=point_radius,
        ),
        base_config=HOME_JOINTS.copy(),
        pointcloud=cloud,
    )

    start = planner.extract_config(HOME_JOINTS)
    goal = planner.sample_valid()

    result = planner.plan(start, goal)
    n = result.path.shape[0] if result.path is not None else 0
    print(
        f"── motion planning: left arm around a table ──\n"
        f"  planner={planner_name}  cloud={len(cloud)} pts  "
        f"radius={point_radius}\n"
        f"  {result.status.value}: {n} waypoints in "
        f"{result.planning_time_ns / 1e6:.0f} ms, cost {result.path_cost:.2f}"
    )

    if result.success and result.path is not None:
        env.animate_path(planner.embed_path(result.path), fps=60)
    else:
        env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
