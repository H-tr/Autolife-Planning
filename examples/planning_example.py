"""Minimal motion planning example — no visualization, no PyBullet."""

import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS
from autolife_planning.planning.motion_planning import plan_motion
from autolife_planning.planning.validation import valid_config
from autolife_planning.types import RobotConfiguration
from autolife_planning.utils.vamp_utils import create_planning_context


def sample_valid_config(context):
    """Sample a random collision-free configuration."""
    while True:
        config = context.sampler.next()
        if valid_config(config, context):
            return config


def main():
    # Create a planning context with an empty environment (no obstacles)
    empty_points = np.zeros((0, 3))
    context = create_planning_context("autolife", empty_points, planner_name="rrtc")

    # Start from home, plan to a random valid goal
    start = RobotConfiguration.from_array(HOME_JOINTS)
    goal_array = sample_valid_config(context)
    goal = RobotConfiguration.from_array(goal_array)

    print(f"Start: {np.round(start.to_array(), 4)}")
    print(f"Goal:  {np.round(goal.to_array(), 4)}")

    # Plan motion
    path = plan_motion(start, goal, context, interpolate=True)

    if path is not None:
        print(f"Path found with {len(path)} waypoints.")
        print(f"  first waypoint: {np.round(np.array(path[0]), 4)}")
        print(f"  last waypoint:  {np.round(np.array(path[len(path) - 1]), 4)}")
    else:
        print("Planning failed.")


if __name__ == "__main__":
    main()
