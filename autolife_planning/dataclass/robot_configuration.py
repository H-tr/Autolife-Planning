# Re-export from autolife_planning.types for backwards compatibility
from autolife_planning.types.robot import (
    ArmConfiguration,
    BaseConfiguration,
    RobotConfiguration,
    WholeBodyConfiguration,
)

__all__ = [
    "BaseConfiguration",
    "RobotConfiguration",
    "ArmConfiguration",
    "WholeBodyConfiguration",
]
