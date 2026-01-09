from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from autolife_planning.dataclass.robot_configuration import (
    BaseConfiguration,
    RobotConfiguration,
)


@dataclass
class ContextState:
    """
    Maintains the global state of the robot and environment for the orchestrator.
    """

    # Robot State
    robot_config: Optional[RobotConfiguration] = None
    base_pose: Optional[BaseConfiguration] = None

    # World State
    # Dictionary mapping object_name -> Object details (pose, id, etc.)
    known_objects: Dict[str, Any] = field(default_factory=dict)

    # Task State
    current_mode: str = "IDLE"  # IDLE, PLANNING, EXECUTING, ERROR
    last_command: str = ""

    def update_robot_state(self, config: RobotConfiguration, base: BaseConfiguration):
        self.robot_config = config
        self.base_pose = base

    def register_object(self, name: str, info: Any):
        self.known_objects[name] = info

    def get_object(self, name: str) -> Optional[Any]:
        return self.known_objects.get(name)
