"""Plane constraint + sphere obstacle: arm bends around, gripper stays flat.

Same plane manifold as ``plane.py`` (left gripper z pinned to its
home value), but with a solid red sphere planted squarely in the
gripper's swept arc.  The planner has to curve the arm around the
obstacle while the gripper keeps sliding across the plane — the
classic curobo-style demo: end-effector constraint + collision
avoidance, handled by a single ``ProjectedStateSpace`` planner.

The plane shows up as a translucent blue plate, the obstacle as a
translucent red sphere, and the sphere is also discretised into a
volumetric point cloud so VAMP's SIMD collision checker sees it.

    pixi run python examples/constrained_planning/plane_with_obstacle.py
"""

import numpy as np
from _shared import SUBGROUP, ee_position, ee_translation, run_demo, setup
from fire import Fire

from autolife_planning.planning import Constraint, create_planner
from autolife_planning.types import PlannerConfig

# Known-good joint seed for the left-arm subgroup: after projection
# onto the z=z0 manifold it lands with the TCP ~18 cm in front of
# HOME, far enough that a small sphere fits cleanly in the swept
# midpoint without intersecting the arm's HOME configuration.  All
# other demos use the random find_goal search; this one needs a
# specific geometry so the sphere actually blocks the direct sweep.
_GOAL_SEED = np.array([0.95, -0.882, -0.881, 1.593, 0.209, 0.082, -0.418])


def _sample_ball(center: np.ndarray, radius: float, n: int, seed: int = 0):
    """Uniform volumetric point cloud inside a ball (float32, Nx3)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = radius * rng.random(n) ** (1.0 / 3.0)
    return (dirs * radii[:, None] + center).astype(np.float32)


def main(planner_name: str = "rrtc", time_limit: float = 10.0):
    env, ctx, start = setup()
    p0 = ee_position(ctx, start)

    # The manifold: the gripper TCP's z equals its home value.
    plane = Constraint(
        residual=ee_translation(ctx)[2] - float(p0[2]),
        q_sym=ctx.q,
        name="plane_with_obs",
    )

    # Project the pinned seed onto the manifold — guaranteed exactly
    # feasible, and produces the reliable geometry described above.
    goal = ctx.project(_GOAL_SEED, plane.residual)

    # Drop the sphere on the swept midpoint, nudged just below the
    # plane so its equator sits at the gripper's height.
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
        ),
        pointcloud=cloud,
        constraints=[plane],
    )
    run_demo(
        env,
        planner,
        start,
        goal,
        "plane + sphere: arm bends around the obstacle, gripper stays on the plane",
    )


if __name__ == "__main__":
    Fire(main)
