import numpy as np
from typing import Any
from autolife_planning.dataclass.planning_context import PlanningContext
from autolife_planning.dataclass.robot_configuration import BaseConfiguration

def navigation(
    start_base: BaseConfiguration,
    goal_base: BaseConfiguration,
    context: PlanningContext,
    interpolate: bool = True,
) -> Any | None:
    """
    Arm planning only
    Input:
        start_base: BaseConfiguration start base position and orientation
        goal_base: BaseConfiguration target base position and orientation
        context: the vamp module manager
        interpolate: Boolean, whether to interpolate the plan to robot resolution
    Output:
        plan: the plan as a Path object
    """
    # TODO: to be implemented
    ...
    print("Navigation not implemented yet")