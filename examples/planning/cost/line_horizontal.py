"""Line cost: the left gripper is *encouraged* to ride a horizontal rail.

Soft counterpart to ``constrained_planning/line_horizontal.py``.
Same residual — TCP's ``y`` and ``z`` pinned, first two columns of
the gripper rotation pinned — but here it feeds a scalar cost
``||residual||^2`` that RRT* integrates along every motion instead
of a hard manifold.  The planner is free to drift off the rail; it
just pays a quadratic penalty for doing so, so the optimal path
hugs the rail except where bending slightly off it produces a
shorter solution.

A bold green cylinder marks the rail.

    pixi run python examples/planning/cost/line_horizontal.py
"""

import casadi as ca
from _shared import (
    EE_LINK,
    SUBGROUP,
    ee_position,
    ee_translation,
    find_goal,
    run_demo,
    setup,
)
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
    R0 = ctx.evaluate_link_pose(EE_LINK, start)[:3, :3]
    tcp = ee_translation(ctx)
    left_rot = ctx.link_rotation(EE_LINK)

    # Same residual the constrained version pins to zero; here we
    # feed its squared norm as a scalar penalty.
    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        tcp[1] - float(p0[1]),
        tcp[2] - float(p0[2]),
        left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    line_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="line_h_cost",
        weight=weight,
    )

    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            # Default path simplification uses geometric shortcuts that
            # ignore the custom cost — leaving it on would smooth the
            # RRT*-shaped path back to a near-straight line and wash
            # out the effect we're trying to demonstrate.
            simplify=False,
        ),
        costs=[line_cost],
    )
    # Goal landed via projection — sits exactly on the soft manifold.
    goal = find_goal(ctx, residual, start, planner, score=lambda xyz: xyz[0] - p0[0])

    p_goal = ee_position(ctx, goal)
    env.draw_rod(
        p1=[float(p0[0]) - 0.10, float(p0[1]), float(p0[2])],
        p2=[float(p_goal[0]) + 0.10, float(p0[1]), float(p0[2])],
        radius=0.008,
        color=(0.20, 0.90, 0.30, 1.0),
    )

    run_demo(
        env, planner, start, goal, "line_h cost: gripper prefers the horizontal rail"
    )


if __name__ == "__main__":
    Fire(main)
