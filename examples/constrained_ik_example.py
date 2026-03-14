"""Constrained IK example — Pink solver with CoM stability and collision avoidance.

Shows the three main features of the Pink backend:
    1. Singularity-robust IK via Levenberg-Marquardt damping
    2. Center-of-mass stability cost (prevents tipping)
    3. Collision avoidance (self-collision + pointcloud obstacles)

All solvers are created through the same ``create_ik_solver`` factory.

Usage:
    python examples/constrained_ik_example.py
"""

import os

import numpy as np

from autolife_planning.config.robot_config import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
)
from autolife_planning.kinematics import create_ik_solver
from autolife_planning.kinematics.collision_model import (
    add_pointcloud_obstacles,
    build_collision_model,
)
from autolife_planning.kinematics.pink_ik_solver import PinkIKSolver
from autolife_planning.types import PinkIKConfig, SE3Pose

G = JOINT_GROUPS
HOME_WHOLE_BODY_LEFT = np.concatenate(
    [
        HOME_JOINTS[G["legs"]],
        HOME_JOINTS[G["waist"]],
        HOME_JOINTS[G["left_arm"]],
    ]
)


def basic_ik():
    """1. Basic Pink IK — same factory, same solve() interface as TRAC-IK."""
    print("=" * 60)
    print("1. Basic Pink IK (singularity-robust)")
    print("=" * 60)

    solver = create_ik_solver("whole_body", side="left", backend="pink")
    print(
        f"Chain: {solver.base_frame} -> {solver.ee_frame} "
        f"({solver.num_joints} joints)"
    )

    home_pose = solver.fk(HOME_WHOLE_BODY_LEFT)
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=home_pose.rotation,
    )

    # solve() returns IKResult — same interface as TracIKSolver
    result = solver.solve(target, seed=HOME_WHOLE_BODY_LEFT)
    print(f"Status: {result.status.value}")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")
    if result.success:
        print(f"  solution: {np.round(result.joint_positions, 4)}")
    print()


def com_and_camera_stability():
    """2. CoM + camera stability — keeps robot stable and chest camera steady."""
    print("=" * 60)
    print("2. CoM + camera stability")
    print("=" * 60)

    config = PinkIKConfig(
        lm_damping=1e-3,
        com_cost=0.1,  # keep CoM above support base
        camera_frame="Link_Waist_Yaw_to_Shoulder_Inner",  # chest camera
        camera_cost=0.1,  # penalize torso rotation to keep camera stable
        max_iterations=200,
    )

    solver = create_ik_solver(
        "whole_body",
        side="left",
        backend="pink",
        config=config,
    )
    assert isinstance(solver, PinkIKSolver)

    home_pose = solver.fk(HOME_WHOLE_BODY_LEFT)
    target = SE3Pose(
        position=home_pose.position + np.array([0.08, 0.05, -0.08]),
        rotation=home_pose.rotation,
    )

    # solve_constrained() gives the full trajectory
    result = solver.solve_constrained(target, seed=HOME_WHOLE_BODY_LEFT)
    print(f"Status: {result.status.value}  ({result.iterations} iterations)")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")
    if result.trajectory is not None:
        print(f"  trajectory shape:  {result.trajectory.shape}")
    if result.success:
        achieved = solver.fk(result.joint_positions)
        print(f"  achieved position: {np.round(achieved.position, 4)}")
    print()


def collision_avoidance():
    """3. Collision avoidance — self-collision + pointcloud obstacles."""
    print("=" * 60)
    print("3. Collision avoidance (self + obstacles)")
    print("=" * 60)

    # Use simplified collision geometry (35 geoms, 57 pairs) instead of the
    # full-mesh URDF (452 pairs).  Same kinematic model, much lighter QP.
    urdf_path = CHAIN_CONFIGS["whole_body_left"].urdf_path
    urdf_dir = os.path.dirname(urdf_path)
    collision_urdf = os.path.join(urdf_dir, "autolife_simple.urdf")
    srdf_path = os.path.join(urdf_dir, "autolife.srdf")
    collision_ctx = build_collision_model(collision_urdf, srdf_path=srdf_path)
    print(f"Self-collision pairs: {len(collision_ctx.collision_model.collisionPairs)}")

    # Obstacle cluster placed away from the home configuration
    obstacle_points = np.array(
        [
            [0.5, 0.3, 0.7],
            [0.52, 0.32, 0.72],
            [0.48, 0.28, 0.68],
        ]
    )
    n_obs = add_pointcloud_obstacles(collision_ctx, obstacle_points, radius=0.02)
    print(
        f"Added {n_obs} obstacle spheres "
        f"(total pairs: {len(collision_ctx.collision_model.collisionPairs)})"
    )

    config = PinkIKConfig(
        lm_damping=1e-3,
        com_cost=0.1,
        self_collision=True,
        collision_pairs=5,
        collision_d_min=0.005,
        solver="proxqp",
        max_iterations=300,
    )

    # Create solver and attach the collision context
    solver = create_ik_solver(
        "whole_body",
        side="left",
        backend="pink",
        config=config,
    )
    assert isinstance(solver, PinkIKSolver)
    solver.set_collision_context(collision_ctx)

    home_pose = solver.fk(HOME_WHOLE_BODY_LEFT)
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=home_pose.rotation,
    )

    result = solver.solve_constrained(target, seed=HOME_WHOLE_BODY_LEFT)
    print(f"Status: {result.status.value}  ({result.iterations} iterations)")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")
    if result.success and result.joint_positions is not None:
        achieved = solver.fk(result.joint_positions)
        print(f"  achieved position: {np.round(achieved.position, 4)}")
    print()


def main():
    basic_ik()
    com_and_camera_stability()
    collision_avoidance()


if __name__ == "__main__":
    main()
