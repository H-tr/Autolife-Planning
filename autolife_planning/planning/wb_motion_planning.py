import numpy as np
from typing import Any
from autolife_planning.dataclass.planning_context import PlanningContext
from autolife_planning.dataclass.robot_configuration import WholeBodyConfiguration

def plan_wb_motion(
    start_config: WholeBodyConfiguration,
    goal_config: WholeBodyConfiguration,
    context: PlanningContext,
    interpolate: bool = True
) -> Any | None:
    """
    Plan a whole body motion from start to goal using the provided context.
    Input:
        start_config: WholeBodyConfiguration
        goal_config: WholeBodyConfiguration
        context: the vamp module manager
        interpolate: Boolean, whether to interpolate the plan to robot resolution
    Output:
        plan: the plan as a Path object
    """
    # TODO: to be implemented
    ...
    print("Whole body motion planning not implemented yet")
