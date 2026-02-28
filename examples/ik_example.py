"""Minimal IK example — no visualization, no PyBullet.

Available chains:
    "left_arm"              7 DOF   (shoulder to left wrist)
    "right_arm"             7 DOF   (shoulder to right wrist)
    "whole_body_left"      11 DOF   (ground vehicle to left wrist)
    "whole_body_right"     11 DOF   (ground vehicle to right wrist)
    "whole_body_base_left" 14 DOF   (zero point to left wrist, includes base)
    "whole_body_base_right"14 DOF   (zero point to right wrist, includes base)

    Shorthand: create_ik_solver("whole_body", side="left")

JOINT_GROUPS (indices into full 24-DOF config):
    base      [0:3]    Virtual_X, Virtual_Y, Virtual_Theta
    legs      [3:5]    Ankle, Knee
    waist     [5:7]    Waist Pitch, Yaw
    left_arm  [7:14]   Shoulder → Wrist (7 DOF)
    right_arm [14:21]  Shoulder → Wrist (7 DOF)
    neck      [21:24]  Roll, Pitch, Yaw
"""

import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

# Home configuration subsets for each chain (via JOINT_GROUPS)
G = JOINT_GROUPS
HOME_LEFT_ARM = HOME_JOINTS[G["left_arm"]]
HOME_RIGHT_ARM = HOME_JOINTS[G["right_arm"]]
HOME_WHOLE_BODY_LEFT = np.concatenate(
    [
        HOME_JOINTS[G["legs"]],
        HOME_JOINTS[G["waist"]],
        HOME_JOINTS[G["left_arm"]],
    ]
)
HOME_WHOLE_BODY_RIGHT = np.concatenate(
    [
        HOME_JOINTS[G["legs"]],
        HOME_JOINTS[G["waist"]],
        HOME_JOINTS[G["right_arm"]],
    ]
)


def main():
    # --- IKConfig (all fields shown with defaults) ---
    config = IKConfig(
        timeout=0.2,  # seconds per TRAC-IK attempt
        epsilon=1e-5,  # convergence tolerance
        solve_type=SolveType.SPEED,  # SPEED | DISTANCE | MANIP1 | MANIP2
        max_attempts=10,  # random restart attempts
        position_tolerance=1e-4,  # post-solve check (meters)
        orientation_tolerance=1e-4,  # post-solve check (radians)
    )

    # --- Create solver ---
    solver = create_ik_solver("left_arm", config=config)
    print(
        f"Chain: {solver.base_frame} -> {solver.ee_frame} ({solver.num_joints} joints)"
    )

    # Forward kinematics: get current end-effector pose at the home configuration
    home_pose = solver.fk(HOME_LEFT_ARM)
    print(f"Home EE position: {home_pose.position}")

    # Define a target pose: small offset, keep same orientation
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=home_pose.rotation,
    )

    # Solve IK (seed is optional; if None, uses random within joint limits)
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
