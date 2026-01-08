import numpy as np
import vamp

from autolife_planning.dataclass.planning_context import PlanningContext


def create_planning_context(
    robot_name: str,
    points: np.ndarray,
    planner_name: str = "rrtc",
    point_radius: float = 0.01,
) -> PlanningContext:
    """
    Initialize and return a PlanningContext containing the vamp module, environment, and planner.
    This avoids re-initializing the environment for every call.
    """
    print(f"Configuring {robot_name} with {planner_name}...")
    (
        vamp_module,
        planner_func,
        plan_settings,
        simp_settings,
    ) = vamp.configure_robot_and_planner_with_kwargs(robot_name, planner_name)

    # Setup environment
    env = vamp.Environment()
    r_min, r_max = vamp_module.min_max_radii()
    env.add_pointcloud(points, r_min, r_max, point_radius)

    sampler = getattr(vamp_module, "halton")()

    return PlanningContext(
        vamp_module=vamp_module,
        planner_func=planner_func,
        plan_settings=plan_settings,
        simp_settings=simp_settings,
        env=env,
        sampler=sampler,
        robot_name=robot_name,
        planner_name=planner_name,
    )
