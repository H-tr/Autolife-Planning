"""VAMP-based motion planner.

Wraps the VAMP C++ library internally. No vamp types leak to the caller.
Mirrors the TracIKSolver pattern: configure -> create -> planner.plan() -> result.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from autolife_planning.types import PlannerConfig, PlanningResult, PlanningStatus


@runtime_checkable
class MotionPlannerBase(Protocol):
    """Protocol for motion planner backends."""

    @property
    def robot_name(self) -> str:
        ...

    @property
    def num_dof(self) -> int:
        ...

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        ...

    def validate(self, configuration: np.ndarray) -> bool:
        ...


class MotionPlanner:
    """Motion planner wrapping the VAMP C++ library.

    All vamp internals are private. The public API accepts and returns
    only numpy arrays.
    """

    def __init__(
        self,
        robot_name: str,
        config: PlannerConfig | None = None,
        pointcloud: np.ndarray | None = None,
    ) -> None:
        if config is None:
            config = PlannerConfig()

        import vamp

        self._config = config
        self._robot_name = robot_name

        (
            self._vamp_module,
            self._planner_func,
            self._plan_settings,
            self._simp_settings,
        ) = vamp.configure_robot_and_planner_with_kwargs(
            robot_name, config.planner_name
        )

        self._env = vamp.Environment()
        self._sampler = self._vamp_module.halton()
        self._ndof: int = self._vamp_module.dimension()

        if pointcloud is not None:
            r_min, r_max = self._vamp_module.min_max_radii()
            self._env.add_pointcloud(
                np.asarray(pointcloud, dtype=np.float32),
                r_min,
                r_max,
                config.point_radius,
            )

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def num_dof(self) -> int:
        return self._ndof

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        """Plan a collision-free path from start to goal.

        Input:
            start: Joint configuration array of length num_dof
            goal: Joint configuration array of length num_dof
        Output:
            PlanningResult with status, path as (N, ndof) numpy array
        """
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if len(start) != self._ndof:
            raise ValueError(f"start has {len(start)} DOF, expected {self._ndof}")
        if len(goal) != self._ndof:
            raise ValueError(f"goal has {len(goal)} DOF, expected {self._ndof}")

        # Validate start and goal
        if not self._vamp_module.validate(start, self._env):
            return PlanningResult(
                status=PlanningStatus.INVALID_START,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )
        if not self._vamp_module.validate(goal, self._env):
            return PlanningResult(
                status=PlanningStatus.INVALID_GOAL,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )

        result = self._planner_func(
            start, goal, self._env, self._plan_settings, self._sampler
        )

        if not result.solved:
            return PlanningResult(
                status=PlanningStatus.FAILED,
                path=None,
                planning_time_ns=result.nanoseconds,
                iterations=result.iterations,
                path_cost=float("inf"),
            )

        path = result.path

        if self._config.simplify:
            simplified = self._vamp_module.simplify(
                path, self._env, self._simp_settings, self._sampler
            )
            path = simplified.path

        if self._config.interpolate:
            path.interpolate_to_resolution(self._vamp_module.resolution())

        path_cost = float(path.cost())
        path_np = np.array(path.numpy())

        return PlanningResult(
            status=PlanningStatus.SUCCESS,
            path=path_np,
            planning_time_ns=result.nanoseconds,
            iterations=result.iterations,
            path_cost=path_cost,
        )

    def validate(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free.

        Input:
            configuration: Joint configuration array of length num_dof
        Output:
            True if collision-free
        """
        configuration = np.asarray(configuration, dtype=np.float64)
        return self._vamp_module.validate(configuration, self._env)

    def sample_valid(self) -> np.ndarray:
        """Sample a random collision-free configuration."""
        while True:
            config = self._sampler.next()
            if self._vamp_module.validate(config, self._env):
                return np.asarray(config, dtype=np.float64)


def create_planner(
    robot_name: str = "autolife",
    config: PlannerConfig | None = None,
    pointcloud: np.ndarray | None = None,
) -> MotionPlanner:
    """Factory function to create a motion planner.

    Input:
        robot_name: VAMP robot name (e.g. "autolife")
        config: Planner configuration (uses defaults if None)
        pointcloud: (N, 3) obstacle point cloud (optional)
    Output:
        MotionPlanner instance

    Examples:
        create_planner("autolife")
        create_planner("autolife", pointcloud=points)
        create_planner("autolife", config=PlannerConfig(planner_name="prm"))
    """
    return MotionPlanner(robot_name, config, pointcloud)
