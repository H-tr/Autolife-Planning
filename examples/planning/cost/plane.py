"""Plane cost: the left gripper is encouraged to stay at a fixed height.

Soft counterpart to ``constrained_planning/plane.py``.  The one-line
residual ``tcp_z - z0`` becomes ``(tcp_z - z0)^2`` — a single-term
quadratic penalty that RRT* trapezoidally integrates along every
motion.  The optimised path keeps the gripper very near the plane
without being locked to it; short off-plane excursions are allowed
whenever they're cheaper than the alternative.

A translucent blue plate is drawn at the target height.

    pixi run python examples/planning/cost/plane.py
"""

import casadi as ca
from _shared import SUBGROUP, ee_position, ee_translation, find_goal, run_demo, setup
from fire import Fire

from autolife_planning.planning import Cost, create_planner
from autolife_planning.types import PlannerConfig


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
        name="plane_z_cost",
        weight=weight,
    )

    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            # OMPL's default path simplifier uses geometric shortcuts
            # that ignore the custom cost — disabling it preserves
            # the plane-hugging shape RRT* produced.
            simplify=False,
        ),
        costs=[plane_cost],
    )
    goal = find_goal(
        ctx,
        residual,
        start,
        planner,
        score=lambda xyz: abs(xyz[1] - p0[1]) + 0.3 * abs(xyz[0] - p0[0]),
    )

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

    run_demo(
        env, planner, start, goal, "plane cost: gripper prefers a horizontal surface"
    )


if __name__ == "__main__":
    Fire(main)
