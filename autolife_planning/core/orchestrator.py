import logging
import os

import numpy as np
import trimesh

from autolife_planning.agents.chat_agent import chat_agent
from autolife_planning.behaviors import (
    BaseBehavior,
    BehaviorStatus,
    RandomDanceBehavior,
)
from autolife_planning.dataclass.commands import PointCommand, TextCommand
from autolife_planning.dataclass.planning_context import PlanningContext
from autolife_planning.dataclass.state import ContextState
from autolife_planning.envs.base_env import BaseEnv
from autolife_planning.utils.vamp_utils import create_planning_context

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
        self.current_behavior: BaseBehavior | None = None
        self.planning_context: PlanningContext | None = None

    def _init_planning(self):
        if self.planning_context is None:
            # Load environment pointcloud
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            assets_dir = os.path.join(project_root, "assets", "envs", "rls_env", "pcd")

            mesh_names = [
                "open_kitchen",
                "rls_2",
                "table",
                "wall",
                "workstation",
                "sofa",
                "coffee_table",
            ]

            all_points = []

            for name in mesh_names:
                pcd_path = os.path.join(assets_dir, f"{name}.ply")
                logger.info(f"Loading pointcloud from {pcd_path}")
                pcd = trimesh.load(pcd_path)
                all_points.append(np.array(pcd.vertices))

            points = np.vstack(all_points)

            self.planning_context = create_planning_context(
                robot_name="autolife", points=points
            )

    def update(self):
        """
        Main update loop called by the server.
        """
        # Step the environment
        self.env.step()

        # Execute active behavior
        if self.current_behavior:
            status = self.current_behavior.execute(self.env, self.state)
            if status != BehaviorStatus.RUNNING:
                logger.info(
                    f"Behavior {self.current_behavior.name} finished with status {status}"
                )
                self.current_behavior = None

    def submit_task(self, context: TextCommand | None, point: PointCommand | None):
        """
        Handle the behaviors from the instruction
        TODO: to be implemented
        """
        # Process context and point
        if context:
            cmd = chat_agent.process_chat_command(context.text)

            if cmd == "RANDOM_DANCE":
                logger.info("Initializing Random Dance Behavior")
                self._init_planning()
                behavior = RandomDanceBehavior()

                # Get current robot configuration for planning start
                start_config = self.env.get_joint_states()

                if behavior.plan(self.planning_context, start_config):
                    self.current_behavior = behavior
                    logger.info("Random Dance Behavior Planned and Started")
                else:
                    logger.error("Failed to plan Random Dance")

    def shutdown(self):
        self.running = False
