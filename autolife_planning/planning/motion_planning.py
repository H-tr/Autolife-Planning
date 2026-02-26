from typing import Any

from autolife_planning.types import PlanningContext, RobotConfiguration


def plan_motion(
    start_config: RobotConfiguration,
    goal_config: RobotConfiguration,
    context: PlanningContext,
    interpolate: bool = True,
) -> Any | None:
    """
    Plan a motion from start to goal using the provided context.
    Input:
        start_config: 18 DoF configuration as the start configuration
        goal_config: 18 DoF configuration as the target configuration
        context: the vamp module manager
        interpolate: Boolean, whether to interpolate the plan to robot resolution
    Output:
        plan: the plan as a Path object
    """
    # Ensure inputs are numpy arrays
    start = start_config.to_array()
    goal = goal_config.to_array()

    # Plan
    result = context.planner_func(
        start, goal, context.env, context.plan_settings, context.sampler
    )

    if result.solved:
        print("Path found! Simplifying...")
        simplify = context.vamp_module.simplify(
            result.path, context.env, context.simp_settings, context.sampler
        )
        plan = simplify.path

        if interpolate:
            # Interpolate to robot resolution for smooth animation
            plan.interpolate_to_resolution(context.vamp_module.resolution())

        # Return the plan object directly to avoid segfaults during list conversion
        # and to match the datatype expected by the underlying C++ module.
        return plan
    else:
        print("Failed to find a path.")
        return None
