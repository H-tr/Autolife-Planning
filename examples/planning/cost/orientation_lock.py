"""Orientation-lock cost: gripper rotation softly encouraged to stay put.

Soft counterpart to ``constrained_planning/orientation_lock.py``.
The six-element residual that pins the first two columns of the
gripper rotation matrix is squared and summed into a scalar cost.
The planner is free to wiggle the gripper orientation mid-path, but
any deviation pays a quadratic penalty integrated along the motion —
so the RRT*-optimised path keeps the rotation nearly fixed while
translation explores freely.

RGB coordinate frames are drawn at the start and goal gripper
positions — they look identical because the preferred rotation is
locked — and a faint magenta rod connects them.

    pixi run python examples/planning/cost/orientation_lock.py
"""

import casadi as ca
import numpy as np
from _shared import EE_LINK, SUBGROUP, ee_position, find_goal, run_demo, setup
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

    left_rot = ctx.link_rotation(EE_LINK)
    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    orient_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="orient_lock_cost",
        weight=weight,
    )

    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            # OMPL's default simplifier is distance-based and would
            # ignore the rotation-lock cost, undoing RRT*'s shaping.
            simplify=False,
        ),
        costs=[orient_cost],
    )
    goal = find_goal(
        ctx,
        residual,
        start,
        planner,
        score=lambda xyz: float(np.linalg.norm(xyz - p0)),
    )

    p_goal = ee_position(ctx, goal)
    env.draw_frame(p0, R0, size=0.12, radius=0.007)
    env.draw_frame(p_goal, R0, size=0.12, radius=0.007)
    env.draw_rod(p0, p_goal, radius=0.004, color=(0.85, 0.35, 0.95, 0.9))

    run_demo(
        env,
        planner,
        start,
        goal,
        "orient cost: translation free, rotation softly held (6-term quadratic penalty)",
    )


if __name__ == "__main__":
    Fire(main)
