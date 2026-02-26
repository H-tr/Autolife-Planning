from typing import Any

from autolife_planning.types import ArmConfiguration, PlanningContext


def plan_arm_motion(
    start_config: ArmConfiguration,
    goal_config: ArmConfiguration,
    context: PlanningContext,
    interpolate: bool = True,
    side: str = "left",  # ["left", "right"]
) -> Any | None:
    """
    Arm planning only
    Input:
        start_config: start configuration
        goal_config: target configuration
        context: the vamp module manager
        interpolate: Boolean, whether to interpolate the plan to robot resolution
        side: "left" or "right"
    Output:
        plan: the plan as a Path object
    """
    if start_config.side != side:
        raise ValueError(
            f"Start configuration side {start_config.side} does not match requested side {side}"
        )
    if goal_config.side != side:
        raise ValueError(
            f"Goal configuration side {goal_config.side} does not match requested side {side}"
        )

    # TODO: to be implemented
    ...
    print("Arm planning not implemented yet")
