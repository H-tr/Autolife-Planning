import numpy as np

from autolife_planning.types import (
    ArmConfiguration,
    BaseConfiguration,
    PlanningContext,
    RobotConfiguration,
    WholeBodyConfiguration,
)


def valid_config(
    config: BaseConfiguration
    | RobotConfiguration
    | ArmConfiguration
    | WholeBodyConfiguration
    | np.ndarray
    | list[float],
    context: PlanningContext,
) -> bool:
    """
    Check if a configuration is valid (collision-free) using the provided context.
    """
    # TODO: need to match the planning context with the configuraiton
    if hasattr(config, "to_array"):
        cfg_array = config.to_array()
    else:
        cfg_array = np.array(config) if not isinstance(config, np.ndarray) else config

    return context.vamp_module.validate(cfg_array, context.env)
