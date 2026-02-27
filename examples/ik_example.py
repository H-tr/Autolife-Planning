"""Minimal IK example — no visualization, no PyBullet."""

import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

# Home configuration subsets for each chain
HOME_LEFT_ARM = np.array(HOME_JOINTS[4:11])
HOME_RIGHT_ARM = np.array(HOME_JOINTS[11:18])


def main():
    # Create IK solver for the left arm (7 DoF)
    solver = create_ik_solver(
        "left_arm", config=IKConfig(solve_type=SolveType.DISTANCE)
    )
    print(
        f"Chain: {solver.base_frame} -> {solver.ee_frame} ({solver.num_joints} joints)"
    )

    # Forward kinematics: get current end-effector pose at the home configuration
    home_pose = solver.fk(HOME_LEFT_ARM)
    print(f"Home EE position: {home_pose.position}")

    # Define a target pose: offset from home
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.05, 0.03]),
        rotation=home_pose.rotation,
    )

    # Solve IK
    result = solver.solve(target, seed=HOME_LEFT_ARM)
    print(f"IK status: {result.status.value}")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")

    if result.success:
        print(f"  solution: {np.round(result.joint_positions, 4)}")

        # Verify with FK
        achieved = solver.fk(result.joint_positions)
        print(f"  achieved position: {np.round(achieved.position, 4)}")
    else:
        print("  IK failed to find a valid solution.")


if __name__ == "__main__":
    main()
