"""Line constraint: the left gripper rides a horizontal rail.

Stacked residual — the gripper's ``y`` and ``z`` translation
components are pinned to their home values, and the first two
columns of its rotation matrix are pinned as well.  Translation
in ``x`` is the only free end-effector DOF; rotation is locked
so the gripper appears to slide cleanly along the rail without
flipping or rolling.  The residual has rank 5 on the 7-DOF left
arm, leaving a 2-D null space for the planner to reconfigure the
elbow and wrist while the gripper pose stays glued to the manifold.

A bold green cylinder marks the rail.

    pixi run python examples/constrained_planning/line_horizontal.py
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

from autolife_planning.planning import Constraint, create_planner
from autolife_planning.types import PlannerConfig


def main(time_limit: float = 5.0):
    env, ctx, start = setup()
    p0 = ee_position(ctx, start)
    R0 = ctx.evaluate_link_pose(EE_LINK, start)[:3, :3]
    tcp = ee_translation(ctx)
    left_rot = ctx.link_rotation(EE_LINK)

    # The manifold: TCP's y and z pinned, rotation locked; x is free.
    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        tcp[1] - float(p0[1]),
        tcp[2] - float(p0[2]),
        left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    line = Constraint(residual=residual, q_sym=ctx.q, name="line_h")

    planner = create_planner(
        SUBGROUP, config=PlannerConfig(time_limit=time_limit), constraints=[line]
    )
    # Push the TCP as far forward along the rail as the manifold allows.
    goal = find_goal(
        ctx, line.residual, start, planner, score=lambda xyz: xyz[0] - p0[0]
    )

    # Draw the rail — extend past both endpoints so it reads as "a line".
    p_goal = ee_position(ctx, goal)
    env.draw_rod(
        p1=[float(p0[0]) - 0.10, float(p0[1]), float(p0[2])],
        p2=[float(p_goal[0]) + 0.10, float(p0[1]), float(p0[2])],
        radius=0.008,
        color=(0.20, 0.90, 0.30, 1.0),
    )

    run_demo(env, planner, start, goal, "line_h: gripper rides a horizontal rail")


if __name__ == "__main__":
    Fire(main)
