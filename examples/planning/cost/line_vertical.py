"""Vertical-line cost: the left gripper is encouraged to move up and down.

Soft counterpart to ``constrained_planning/line_vertical.py``.  The
residual — TCP's ``x`` and ``y`` pinned, first two columns of the
gripper rotation pinned — is turned into a scalar quadratic penalty
``||residual||^2``.  RRT* integrates it along each motion, so the
optimised trajectory prefers a crisp vertical slide but is free to
bend off-axis when doing so shortens the path.

A bold yellow cylinder marks the rail.

    pixi run python examples/planning/cost/line_vertical.py
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

    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        tcp[0] - float(p0[0]),
        tcp[1] - float(p0[1]),
        left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    line_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="line_v_cost",
        weight=weight,
    )

    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            # Leave OMPL's default path simplifier off — it uses
            # geometric shortcuts that ignore the custom cost and
            # would undo the RRT*-shaped trajectory.
            simplify=False,
        ),
        costs=[line_cost],
    )
    goal = find_goal(ctx, residual, start, planner, score=lambda xyz: xyz[2] - p0[2])

    p_goal = ee_position(ctx, goal)
    env.draw_rod(
        p1=[float(p0[0]), float(p0[1]), float(p0[2]) - 0.10],
        p2=[float(p0[0]), float(p0[1]), float(p_goal[2]) + 0.10],
        radius=0.008,
        color=(0.95, 0.85, 0.10, 1.0),
    )

    run_demo(
        env, planner, start, goal, "line_v cost: gripper prefers the vertical rail"
    )


if __name__ == "__main__":
    Fire(main)
