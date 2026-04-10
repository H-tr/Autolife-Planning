"""Constrained motion planning demo with PyBullet visualization.

Two hard manifold constraints, projected by OMPL's ProjectedStateSpace:

1. ``Knee = 2 * Ankle`` (LinearCoupling) on ``autolife_height``.
2. Left gripper pose locked in world frame (PoseLock) on ``autolife_body``
   — start and goal differ only in right-arm and neck joints, so the
   gripper pose is identical at both endpoints and the planner is
   forced to keep it locked while moving the rest of the body.

Both endpoints must already lie on the constraint manifold; the
planner does not run an IK pass on them.

    pixi run python examples/constrained_planning_example.py
"""

import numpy as np
import pinocchio as pin
from fire import Fire

from autolife_planning.config.robot_config import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import LinearCoupling, PoseLock, create_planner
from autolife_planning.types import PlannerConfig


def link_pose(link: str, full: np.ndarray) -> np.ndarray:
    """4x4 SE(3) pose of *link* at the full 24-DOF body config *full*."""
    model = pin.buildModelFromUrdf(
        autolife_robot_config.urdf_path, pin.JointModelPlanar()
    )
    data = model.createData()
    fid = model.getFrameId(link)
    q = np.empty(int(model.nq))  # type: ignore[arg-type]
    q[:2] = full[:2]
    q[2], q[3] = np.cos(full[2]), np.sin(full[2])
    q[4:] = full[3:]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, fid)
    pose = data.oMf[fid]  # type: ignore[index]
    M = np.eye(4)
    M[:3, :3] = pose.rotation
    M[:3, 3] = pose.translation
    return M


def main(planner_name: str = "rrtc"):
    env = PyBulletEnv(autolife_robot_config, visualize=True)
    config = PlannerConfig(planner_name=planner_name, time_limit=2.0)

    # ── Demo 1: Knee = 2 * Ankle ─────────────────────────────────────
    knee = LinearCoupling(master="Joint_Ankle", slave="Joint_Knee", multiplier=2.0)
    p1 = create_planner("autolife_height", config=config, constraints=[knee])

    r1 = p1.plan(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.6]))
    n1 = r1.path.shape[0] if r1.path is not None else 0
    print(f"[knee = 2 * ankle] {r1.status.value} — {n1} waypoints")
    if r1.success and r1.path is not None:
        env.animate_path(p1.embed_path(r1.path), fps=60)
    env.wait_key("n", "press 'n' for demo 2")

    # ── Demo 2: Left gripper locked in world frame ───────────────────
    target = link_pose("Link_Left_Gripper", HOME_JOINTS)
    lock = PoseLock(link="Link_Left_Gripper", target=target, frame="world")
    p2 = create_planner("autolife_body", config=config, constraints=[lock])

    # Start = HOME body (21 DOF).  Goal differs only in joints that
    # don't drive the left arm — the gripper pose at both endpoints
    # is identical, so the planner has to find a path that *keeps*
    # the gripper put while everything else moves.
    start = HOME_JOINTS[3:].copy()
    goal = start.copy()
    goal[11] = 0.30  # neck roll
    goal[12] = 0.40  # neck pitch
    goal[14] = -1.80  # right shoulder inner
    goal[16] = 0.80  # right upper arm
    goal[17] = -1.20  # right elbow

    r2 = p2.plan(start, goal)
    n2 = r2.path.shape[0] if r2.path is not None else 0
    print(f"[gripper lock] {r2.status.value} — {n2} waypoints")
    if r2.success and r2.path is not None:
        env.animate_path(p2.embed_path(r2.path), fps=60)
    env.wait_key("q", "press 'q' to quit")


if __name__ == "__main__":
    Fire(main)
