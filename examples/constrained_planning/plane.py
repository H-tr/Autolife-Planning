"""Plane constraint: the left gripper slides across a horizontal surface.

A single holonomic equation written as a CasADi expression — the
left gripper's world z-coordinate must equal its home value.  OMPL's
``ProjectedStateSpace`` projects every sampled state onto the plane
before validation, so the gripper trajectory is guaranteed flat
from start to goal.

A translucent blue plate is drawn at the target height to show the
manifold in 3D.

    pixi run python examples/constrained_planning/plane.py
"""

from _shared import SUBGROUP, ee_position, ee_translation, find_goal, run_demo, setup
from fire import Fire

from autolife_planning.planning import Constraint, create_planner
from autolife_planning.types import PlannerConfig


def main(time_limit: float = 5.0):
    env, ctx, start = setup()
    p0 = ee_position(ctx, start)

    # The manifold: the gripper TCP's z equals its home value.
    plane = Constraint(
        residual=ee_translation(ctx)[2] - float(p0[2]),
        q_sym=ctx.q,
        name="plane_z",
    )

    planner = create_planner(
        SUBGROUP, config=PlannerConfig(time_limit=time_limit), constraints=[plane]
    )
    goal = find_goal(
        ctx,
        plane.residual,
        start,
        planner,
        score=lambda xyz: abs(xyz[1] - p0[1]) + 0.3 * abs(xyz[0] - p0[0]),
    )

    # Draw the plane, centred between the start and goal TCP positions.
    # The plate sits a few cm below the TCP so the fingers don't clip
    # into it visually, and it's drawn wide enough that the gripper
    # never rides off the edge during the solve.
    p_goal = ee_position(ctx, goal)
    env.draw_plane(
        center=[
            float(0.5 * (p0[0] + p_goal[0])),
            float(0.5 * (p0[1] + p_goal[1])),
            float(p0[2]) - 0.05,
        ],
        half_sizes=(0.65, 0.65),
        color=(0.15, 0.55, 0.95, 0.22),
    )

    run_demo(env, planner, start, goal, "plane: gripper slides on a horizontal surface")


if __name__ == "__main__":
    Fire(main)
