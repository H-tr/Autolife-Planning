"""IK solver example using TRAC-IK with PyBullet visualization."""

import time

import numpy as np
import pybullet as pb

from autolife_planning.config.robot_config import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

# Home configuration subsets matching each chain's joint ordering (via JOINT_GROUPS)
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

# Mapping from chain solution indices to full 21-joint config indices (no base)
# These indices are relative to HOME_JOINTS[3:] (the 21 body joints)
CHAIN_TO_BODY = {
    "left_arm": list(range(4, 11)),
    "right_arm": list(range(11, 18)),
    "whole_body_left": list(range(0, 11)),
    "whole_body_right": list(range(0, 4)) + list(range(11, 18)),
}

CHAIN_SEEDS = {
    "left_arm": HOME_LEFT_ARM,
    "right_arm": HOME_RIGHT_ARM,
    "whole_body_left": HOME_WHOLE_BODY_LEFT,
    "whole_body_right": HOME_WHOLE_BODY_RIGHT,
}


def get_ee_link_index(env, link_name):
    """Find PyBullet link index by name."""
    client = env.sim.client
    for i in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    return -1


def draw_frame_at_link(env, link_index, length=0.08, width=3):
    """Draw RGB axes at a link's world pose. Returns debug line IDs."""
    client = env.sim.client
    state = client.getLinkState(env.sim.skel_id, link_index)
    pos = np.array(state[0])
    rot = np.array(client.getMatrixFromQuaternion(state[1])).reshape(3, 3)

    line_ids = []
    for axis_idx, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        axis = np.zeros(3)
        axis[axis_idx] = length
        end = (pos + rot @ axis).tolist()
        line_ids.append(
            client.addUserDebugLine(pos.tolist(), end, color, lineWidth=width)
        )
    return line_ids


def draw_frame_at_pose(env, pos, rot, length=0.08, width=3):
    """Draw RGB axes at a given world pose. Returns debug line IDs."""
    client = env.sim.client
    origin = pos.tolist()
    line_ids = []
    for axis_idx, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        axis = np.zeros(3)
        axis[axis_idx] = length
        end = (pos + rot @ axis).tolist()
        line_ids.append(client.addUserDebugLine(origin, end, color, lineWidth=width))
    return line_ids


def wait_key(env, key, msg):
    """Wait for a key press in the PyBullet GUI."""
    client = env.sim.client
    text_id = client.addUserDebugText(
        msg, [0, 0, 1.5], textColorRGB=[0, 0, 0], textSize=1.5
    )
    print(msg)
    while True:
        keys = client.getKeyboardEvents()
        if key in keys and keys[key] & pb.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.01)
    client.removeUserDebugItem(text_id)


def test_chain(env, chain_name):
    """Solve IK for one chain and visualize."""
    print(f"\n{'='*60}")
    print(f"Chain: {chain_name}")
    print(f"{'='*60}")

    # IKConfig controls solver behavior:
    #   timeout         - seconds for TRAC-IK dual-thread solve (default: 0.2)
    #   epsilon         - convergence tolerance (default: 1e-5)
    #   solve_type      - SPEED:    return first valid solution (fastest)
    #                     DISTANCE: minimize joint displacement from seed
    #                     MANIP1:   maximize manipulability (product of singular values)
    #                     MANIP2:   maximize isotropy (min/max singular value ratio)
    #   max_attempts    - random restart attempts (default: 10)
    #   position_tolerance    - post-solve validation in meters (default: 1e-4)
    #   orientation_tolerance - post-solve validation in radians (default: 1e-4)
    config = IKConfig(
        timeout=0.2,
        epsilon=1e-5,
        solve_type=SolveType.DISTANCE,
        max_attempts=10,
    )

    # Available chains: left_arm, right_arm, whole_body_left, whole_body_right,
    #                   whole_body_base_left, whole_body_base_right
    # Shorthand: create_ik_solver("whole_body", side="left")
    solver = create_ik_solver(chain_name, config=config)
    seed = CHAIN_SEEDS[chain_name]
    ee_link = CHAIN_CONFIGS[chain_name].ee_link
    ee_idx = get_ee_link_index(env, ee_link)

    print(f"  DOF: {solver.num_joints}")
    print(f"  base: {solver.base_frame}")
    print(f"  ee:   {solver.ee_frame}")

    # Show home config and draw current EE frame
    env.set_joint_states(HOME_JOINTS[3:])
    debug_lines = draw_frame_at_link(env, ee_idx, length=0.06, width=2)

    # FK to get current EE pose (in chain-local frame for IK target)
    current_pose = solver.fk(seed)

    # Define target pose
    if "whole_body" in chain_name:
        offset = np.array([0.10, 0.08, 0.05])
        angle = np.deg2rad(20)
        rot_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        target_pose = SE3Pose(
            position=current_pose.position + offset,
            rotation=rot_z @ current_pose.rotation,
        )
    else:
        # Arm-only: keep orientation, small position offset toward the front
        target_pose = SE3Pose(
            position=current_pose.position + np.array([0.05, 0.0, -0.05]),
            rotation=current_pose.rotation,
        )

    wait_key(env, ord("n"), f"[{chain_name}] Home config. Press 'n' to solve IK.")

    # Solve IK (seed is optional; if None, uses random within joint limits)
    # Joint limits can be overridden: solver.set_joint_limits(lower, upper)
    result = solver.solve(target_pose, seed=seed)
    print(
        f"  IK: {result.status.value}, "
        f"pos_err={result.position_error:.6f}m, "
        f"ori_err={result.orientation_error:.6f}rad"
    )

    if result.joint_positions is not None:
        # Apply solution: overlay IK result onto home body joints
        body_joints = HOME_JOINTS[3:].copy()
        for i, bi in enumerate(CHAIN_TO_BODY[chain_name]):
            body_joints[bi] = float(result.joint_positions[i])
        env.set_joint_states(body_joints)
        debug_lines += draw_frame_at_link(env, ee_idx, length=0.05, width=2)

    wait_key(env, ord("n"), f"[{chain_name}] Solution shown. Press 'n' for next.")

    # Clean up
    for lid in debug_lines:
        env.sim.client.removeUserDebugItem(lid)


def main():
    print("TRAC-IK Solver Example")
    print("=" * 60)

    env = PyBulletEnv(autolife_robot_config, visualize=True)

    for chain_name in ["left_arm", "right_arm", "whole_body_left", "whole_body_right"]:
        try:
            test_chain(env, chain_name)
        except Exception as e:
            print(f"  ERROR on {chain_name}: {e}")
            import traceback

            traceback.print_exc()

    wait_key(env, ord("q"), "All chains done. Press 'q' to quit.")
    print("\nDone.")


if __name__ == "__main__":
    main()
