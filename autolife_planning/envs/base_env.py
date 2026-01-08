from abc import ABC, abstractmethod
from typing import Any

from autolife_planning.dataclass.robot_configuration import (
    BaseConfiguration,
    RobotConfiguration,
)


class BaseEnv(ABC):
    @abstractmethod
    def get_joint_states(self) -> RobotConfiguration:
        """Get current joint positions."""
        raise NotImplementedError

    @abstractmethod
    def set_joint_states(self, config: RobotConfiguration):
        """Set joint positions."""

    @abstractmethod
    def get_localization(self) -> BaseConfiguration:
        """Get robot localization (e.g. base position and orientation)."""
        raise NotImplementedError

    @abstractmethod
    def get_obs(self) -> Any:
        """Get observations."""
        raise NotImplementedError
