"""Orientation-lock constraint: gripper rotation frozen, translation free.

Six holonomic equations — the first two columns of the gripper's
rotation matrix are pinned to their home values.  The third column
is fixed automatically by orthonormality of SO(3), so the entire
3x3 rotation is locked.  The residual is rank 3, which leaves a
4-DOF null space on the 7-DOF arm: translation in x, y, z plus one
redundant joint the planner can exploit.

RGB coordinate frames are drawn at the start and goal gripper
positions — they look identical because the rotation is locked —
and a faint magenta rod connects them to show the swept translation.

    pixi run python examples/constrained_planning/orientation_lock.py
"""

import casadi as ca
import numpy as np
from _shared import EE_LINK, SUBGROUP, ee_position, find_goal, run_demo, setup
from fire import Fire

from autolife_planning.planning import Constraint, create_planner
from autolife_planning.types import PlannerConfig


def main(time_limit: float = 5.0):
    env, ctx, start = setup()
    p0 = ee_position(ctx, start)
    R0 = ctx.evaluate_link_pose(EE_LINK, start)[:3, :3]

    # The manifold: first two columns of the rotation matrix pinned.
    # (Column 3 is determined by the first two via orthonormality.)
    left_rot = ctx.link_rotation(EE_LINK)
    residual: ca.SX = ca.vertcat(  # type: ignore[assignment]
        left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    orient = Constraint(residual=residual, q_sym=ctx.q, name="orient_lock")

    planner = create_planner(
        SUBGROUP, config=PlannerConfig(time_limit=time_limit), constraints=[orient]
    )
    goal = find_goal(
        ctx,
        orient.residual,
        start,
        planner,
        score=lambda xyz: float(np.linalg.norm(xyz - p0)),
    )

    # Start frame, goal frame, and a thin magenta rod between them.
    p_goal = ee_position(ctx, goal)
    env.draw_frame(p0, R0, size=0.12, radius=0.007)
    env.draw_frame(p_goal, R0, size=0.12, radius=0.007)
    env.draw_rod(p0, p_goal, radius=0.004, color=(0.85, 0.35, 0.95, 0.9))

    run_demo(
        env,
        planner,
        start,
        goal,
        "orient: translation free, rotation locked (6 eqns on SO(3))",
    )


if __name__ == "__main__":
    Fire(main)
