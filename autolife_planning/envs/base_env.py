from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEnv(ABC):
    @abstractmethod
    def get_joint_states(self) -> np.ndarray:
        """Get current joint positions."""
        raise NotImplementedError

    @abstractmethod
    def set_joint_states(self, config: np.ndarray):
        """Set joint positions."""

    @abstractmethod
    def get_localization(self) -> np.ndarray:
        """Get robot localization as [x, y, theta]."""
        raise NotImplementedError

    @abstractmethod
    def get_obs(self) -> Any:
        """Get observations."""
        raise NotImplementedError
