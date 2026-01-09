from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class BehaviorStatus(Enum):
    IDLE = 0
    RUNNING = 1
    SUCCESS = 2
    FAILURE = 3


class BaseBehavior(ABC):
    """
    Abstract base class for all behaviors (skills).
    """

    def __init__(self, name: str):
        self.name = name
        self.status = BehaviorStatus.IDLE
        self.status_message = ""

    @abstractmethod
    def plan(self, context: Any) -> bool:
        """
        Perform geometry/motion planning.
        Returns True if a valid plan is found.
        """

    @abstractmethod
    def execute(self, env: Any, context: Any) -> BehaviorStatus:
        """
        Execute one step of the behavior.
        Called strictly in the update loop.

        should return:
        - BehaviorStatus.RUNNING if continuing
        - BehaviorStatus.SUCCESS if finished
        - BehaviorStatus.FAILURE if failed
        """

    def stop(self):
        """
        Gracefully stop execution.
        """
        self.status = BehaviorStatus.FAILURE
        self.status_message = "Stopped by user/manager."

    def reset(self):
        self.status = BehaviorStatus.IDLE
        self.status_message = ""
