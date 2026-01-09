from typing import Any

from autolife_planning.behaviors.base_behavior import BaseBehavior, BehaviorStatus
from autolife_planning.dataclass.planning_context import PlanningContext
from autolife_planning.dataclass.robot_configuration import RobotConfiguration
from autolife_planning.planning import motion_planning


class RandomDanceBehavior(BaseBehavior):
    def __init__(self, name: str = "random_dance"):
        super().__init__(name)
        self.interpolated_plan: Any | None = None
        self.current_step = 0

    def _sample_valid(self, context: PlanningContext):
        while True:
            config = context.sampler.next()
            if context.vamp_module.validate(config, context.env):
                return config

    def plan(
        self, context: PlanningContext, start_config: RobotConfiguration | None = None
    ) -> bool:
        """
        Plan a random dance motion.
        """
        if start_config is None:
            return False

        goal_array = self._sample_valid(context)
        goal_config = RobotConfiguration.from_array(goal_array)

        print(f"Planning from {start_config} to {goal_config}")

        # Plan Path using motion_planning module
        plan = motion_planning.plan_motion(
            start_config, goal_config, context, interpolate=True
        )

        if plan:
            print("Planning successful")
            self.interpolated_plan = plan
            self.current_step = 0
            self.status = BehaviorStatus.RUNNING
            return True
        else:
            self.status = BehaviorStatus.FAILURE
            return False

    def execute(self, env, context) -> BehaviorStatus:
        if self.status != BehaviorStatus.RUNNING or self.interpolated_plan is None:
            return self.status

        # Convert plan to list of configs if not already
        # The 'plan' object from vamp seems to be iterable or has specific access
        # vamp.Path usually is a list of configs

        path_configs = self.interpolated_plan

        if self.current_step < len(path_configs):
            config = path_configs[self.current_step]

            # Execute on environment
            # env is expected to be PyBulletEnv

            # Assuming config is a list or numpy array
            robot_config = RobotConfiguration.from_array(config)
            env.set_joint_states(robot_config)

            self.current_step += 1
            return BehaviorStatus.RUNNING
        else:
            self.status = BehaviorStatus.SUCCESS
            return BehaviorStatus.SUCCESS
