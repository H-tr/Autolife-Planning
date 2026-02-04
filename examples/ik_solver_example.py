"""Minimal IK solver example demonstrating core interfaces."""

import os

import numpy as np
from fire import Fire

from autolife_planning.config.ik_config import IKConfig
from autolife_planning.config.robot_config import HOME_JOINTS, autolife_robot_config
from autolife_planning.dataclass.ik_types import SE3Pose
from autolife_planning.kinematics.ik_solver import (
    compute_forward_kinematics,
    create_ik_context,
    solve_ik,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(
    PROJECT_ROOT, "third_party", "cricket", "resources", "autolife", "autolife.urdf"
)
LEFT_EE_FRAME = "Link_Left_Wrist_Lower_to_Gripper"


def main(visualize: bool = False) -> None:
    # 1. Create IK context
    context = create_ik_context(
        urdf_path=URDF_PATH,
        end_effector_frame=LEFT_EE_FRAME,
        joint_names=autolife_robot_config.joint_names,
    )

    # 2. Compute forward kinematics at home configuration
    home_config = np.array(HOME_JOINTS)
    current_pose = compute_forward_kinematics(context, home_config)

    # 3. Define target pose (15cm forward, 10cm left, 5cm up, rotated 30° around Z)
    target_position = current_pose.position + np.array([0.15, 0.10, 0.05])
    angle = np.deg2rad(30)
    rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    target_pose = SE3Pose(
        position=target_position, rotation=rot_z @ current_pose.rotation
    )

    # 4. Solve IK
    result = solve_ik(
        context=context,
        target_pose=target_pose,
        initial_config=home_config,
        config=IKConfig(
            max_iterations=200, position_tolerance=1e-4, orientation_tolerance=1e-4
        ),
    )

    print(
        f"IK {result.status.value}: pos_err={result.position_error:.4f}m, ori_err={result.orientation_error:.4f}rad"
    )

    if visualize and result.success and result.joint_positions is not None:
        visualize_result(home_config, result.joint_positions, target_pose)


def visualize_result(
    initial_config: np.ndarray, solution_config: np.ndarray, target_pose: SE3Pose
) -> None:
    try:
        from vamp import pybullet_interface as vpb
    except ImportError:
        print("Visualization requires vamp package")
        return

    urdf = os.path.join(
        PROJECT_ROOT,
        "third_party",
        "vamp",
        "resources",
        "autolife",
        "autolife_spherized.urdf",
    )
    sim = vpb.PyBulletSimulator(urdf, autolife_robot_config.joint_names, visualize=True)

    # Draw target: coordinate frame (RGB=XYZ) and sphere
    draw_frame(sim, target_pose.position, target_pose.rotation)
    sim.add_sphere(0.02, target_pose.position.tolist())

    # Show initial -> press 'n' -> show solution -> press 'q' to quit
    sim.set_joint_positions(initial_config.tolist())
    wait_key(sim, ord("n"), "Initial config. Press 'n' for solution.")

    sim.set_joint_positions(solution_config.tolist())
    wait_key(sim, ord("q"), "Solution. Press 'q' to quit.")


def draw_frame(sim, pos: np.ndarray, rot: np.ndarray, length: float = 0.08) -> None:
    origin = pos.tolist()
    sim.client.addUserDebugLine(
        origin, (pos + rot @ [length, 0, 0]).tolist(), [1, 0, 0], lineWidth=3
    )
    sim.client.addUserDebugLine(
        origin, (pos + rot @ [0, length, 0]).tolist(), [0, 1, 0], lineWidth=3
    )
    sim.client.addUserDebugLine(
        origin, (pos + rot @ [0, 0, length]).tolist(), [0, 0, 1], lineWidth=3
    )


def wait_key(sim, key: int, msg: str) -> None:
    print(msg)
    while key not in sim.client.getKeyboardEvents():
        pass


if __name__ == "__main__":
    Fire(main)
