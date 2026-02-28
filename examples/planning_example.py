"""Minimal motion planning example — no visualization, no PyBullet."""

import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS
from autolife_planning.planning import create_planner


def main():
    # Create a planner with an empty environment (no obstacles)
    planner = create_planner("autolife")

    # HOME_JOINTS is the full 24 DOF config (3 base + 21 joints)
    start = HOME_JOINTS.copy()
    goal = planner.sample_valid()

    print(f"Start: {np.round(start, 4)}")
    print(f"Goal:  {np.round(goal, 4)}")

    # Plan motion
    result = planner.plan(start, goal)

    if result.success:
        print(f"Path found with {result.path.shape[0]} waypoints.")
        print(f"  first waypoint: {np.round(result.path[0], 4)}")
        print(f"  last waypoint:  {np.round(result.path[-1], 4)}")
        print(f"  planning time:  {result.planning_time_ns / 1e6:.1f}ms")
        print(f"  path cost:      {result.path_cost:.4f}")
    else:
        print(f"Planning failed: {result.status.value}")


if __name__ == "__main__":
    main()
