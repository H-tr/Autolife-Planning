import logging

from autolife_planning.behaviors.base_behavior import BaseBehavior
from autolife_planning.dataclass.commands import PointCommand, TextCommand
from autolife_planning.dataclass.state import ContextState
from autolife_planning.envs.base_env import BaseEnv

logger = logging.getLogger("Orchestrator")


class Orchestrator:
    """
    Central brain of the system.
    Connects Infrastructure (Env), Interface (Web), and Logic (Agents/Behaviors).
    """

    def __init__(self, env: BaseEnv):
        self.env = env
        self.state = ContextState()

        # Initialize internal state
        self.running: bool = True
        self.available_behaviors: set[BaseBehavior] = {}

    def submit_task(self, context: TextCommand | None, point: PointCommand | None):
        """
        Handle the behaviors from the instruction
        TODO: to be implemented
        """
        # Process context and point

        # Plan the behaviors

        # Execute the behaviors
        ...

    def shutdown(self):
        self.running = False
