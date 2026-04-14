"""Plane cost + sphere obstacle: arm bends around, gripper prefers flat.

Soft counterpart to ``constrained_planning/plane_with_obstacle.py``.
The height-pinning residual ``tcp_z - z0`` is squared into a scalar
cost that RRT* integrates along every motion, and a solid red sphere
sits directly in the gripper's preferred sweep.  Collision avoidance
is still a hard constraint (handled by VAMP); only the height
preference is soft — so the planner bends the arm around the sphere
while keeping the gripper as close to the plane as the optimiser can
manage.

The plane shows up as a translucent blue plate, the obstacle as a
translucent red sphere, and the sphere is also discretised into a
volumetric point cloud so VAMP's SIMD collision checker sees it.

    pixi run python examples/planning/cost/plane_with_obstacle.py
"""

import casadi as ca
import numpy as np
from _shared import SUBGROUP, ee_position, ee_translation, run_demo, setup
from fire import Fire

from autolife_planning.planning import Cost, create_planner
from autolife_planning.types import PlannerConfig

# Same known-good joint seed used in the constrained version: after
# projection onto z = z0 the TCP lands ~18 cm in front of HOME, so a
# small sphere fits cleanly between start and goal without clipping
# the HOME arm pose.
_GOAL_SEED = np.array([0.95, -0.882, -0.881, 1.593, 0.209, 0.082, -0.418])


def _sample_ball(center: np.ndarray, radius: float, n: int, seed: int = 0):
    """Uniform volumetric point cloud inside a ball (float32, Nx3)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = radius * rng.random(n) ** (1.0 / 3.0)
    return (dirs * radii[:, None] + center).astype(np.float32)


def main(
    planner_name: str = "rrtstar",
    time_limit: float = 10.0,
    weight: float = 50.0,
):
    env, ctx, start = setup()
    p0 = ee_position(ctx, start)

    residual = ee_translation(ctx)[2] - float(p0[2])
    plane_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="plane_with_obs_cost",
        weight=weight,
    )

    # Project the pinned seed onto the residual's zero set so the goal
    # itself sits exactly on the soft manifold — same trick the
    # constrained version uses to produce the reliable geometry.
    goal = ctx.project(_GOAL_SEED, residual)

    p_goal = ee_position(ctx, goal)
    sphere_center = np.array(
        [
            float(0.5 * (p0[0] + p_goal[0])),
            float(0.5 * (p0[1] + p_goal[1])),
            float(p0[2]) - 0.03,
        ]
    )
    sphere_radius = 0.06

    env.draw_plane(
        center=[
            float(sphere_center[0]),
            float(sphere_center[1]),
            float(p0[2]) - 0.05,
        ],
        half_sizes=(0.65, 0.65),
        color=(0.15, 0.55, 0.95, 0.22),
    )
    env.draw_sphere(center=sphere_center, radius=sphere_radius)

    cloud = _sample_ball(sphere_center, sphere_radius, n=800)
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            point_radius=0.012,
            # OMPL's default simplifier is distance-based and would
            # erase the cost-shaped trajectory RRT* built, so leave it off.
            simplify=False,
        ),
        pointcloud=cloud,
        costs=[plane_cost],
    )
    run_demo(
        env,
        planner,
        start,
        goal,
        "plane cost + sphere: arm bends around the obstacle, gripper prefers the plane",
    )


if __name__ == "__main__":
    Fire(main)
