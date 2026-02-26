"""IK solver example with PyBullet visualization using TRAC-IK."""

import time

import numpy as np
import pybullet as pb
import pybullet_data

from autolife_planning.config.robot_config import CHAIN_CONFIGS, HOME_JOINTS
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import SE3Pose

URDF_PATH = CHAIN_CONFIGS["left_arm"].urdf_path

# Home configuration subsets matching each chain's joint ordering
HOME_LEFT_ARM = np.array(HOME_JOINTS[4:11])
HOME_RIGHT_ARM = np.array(HOME_JOINTS[11:18])
HOME_WHOLE_BODY_LEFT = np.array(HOME_JOINTS[:11])
HOME_WHOLE_BODY_RIGHT = np.array(HOME_JOINTS[:4] + HOME_JOINTS[11:18])

# Mapping from chain solution indices to full 18-joint config indices
CHAIN_TO_FULL = {
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


class Visualizer:
    """PyBullet GUI wrapper for IK visualization."""

    def __init__(self):
        self.client = pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client
        )
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pb.loadURDF("plane.urdf", physicsClientId=self.client)

        self.robot_id = pb.loadURDF(
            URDF_PATH,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client,
        )

        # Build pybullet joint index map: joint_name -> pb_index
        self._joint_map: dict[str, int] = {}
        n = pb.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(n):
            info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            name = info[1].decode()
            if info[2] != pb.JOINT_FIXED:
                self._joint_map[name] = i

        # Ordered pybullet indices matching autolife_robot_config.joint_names
        from autolife_planning.config.robot_config import autolife_robot_config

        self._pb_indices = [
            self._joint_map[n] for n in autolife_robot_config.joint_names
        ]

        # Set home configuration
        self.set_full_config(HOME_JOINTS)

        # Camera
        pb.resetDebugVisualizerCamera(
            1.5, 45, -30, [0, 0, 0.5], physicsClientId=self.client
        )

        self._debug_lines: list[int] = []

    def set_full_config(self, full_joints: list[float] | np.ndarray) -> None:
        """Set all 18 controlled joints."""
        for idx, val in zip(self._pb_indices, full_joints):
            pb.resetJointState(self.robot_id, idx, val, physicsClientId=self.client)

    def set_chain_solution(self, chain_name: str, chain_joints: np.ndarray) -> None:
        """Overlay a chain IK solution onto the current full config."""
        full = list(HOME_JOINTS)
        for i, fi in enumerate(CHAIN_TO_FULL[chain_name]):
            full[fi] = float(chain_joints[i])
        self.set_full_config(full)

    def draw_frame(
        self, pos: np.ndarray, rot: np.ndarray, length: float = 0.08, width: float = 3
    ) -> None:
        """Draw RGB coordinate axes at a pose."""
        origin = pos.tolist()
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for axis_idx, color in enumerate(colors):
            axis = np.zeros(3)
            axis[axis_idx] = length
            end = (pos + rot @ axis).tolist()
            line_id = pb.addUserDebugLine(
                origin, end, color, lineWidth=width, physicsClientId=self.client
            )
            self._debug_lines.append(line_id)

    def draw_sphere(
        self, pos: np.ndarray, radius: float = 0.015, color: list | None = None
    ) -> int:
        """Draw a small sphere marker at a position."""
        if color is None:
            color = [1, 0.3, 0, 0.8]
        vis = pb.createVisualShape(
            pb.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=self.client
        )
        body = pb.createMultiBody(
            baseVisualShapeIndex=vis,
            basePosition=pos.tolist(),
            physicsClientId=self.client,
        )
        return body

    def clear_markers(self) -> None:
        for line_id in self._debug_lines:
            pb.removeUserDebugItem(line_id, physicsClientId=self.client)
        self._debug_lines.clear()

    def add_text(
        self, text: str, pos: list[float], color: list[float] | None = None
    ) -> int:
        if color is None:
            color = [0, 0, 0]
        return pb.addUserDebugText(
            text, pos, textColorRGB=color, textSize=1.5, physicsClientId=self.client
        )

    def wait_key(self, key: int, msg: str) -> None:
        text_id = self.add_text(msg, [0, 0, 1.5])
        print(msg)
        while True:
            keys = pb.getKeyboardEvents(physicsClientId=self.client)
            if key in keys and keys[key] & pb.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.01)
        pb.removeUserDebugItem(text_id, physicsClientId=self.client)

    def disconnect(self) -> None:
        pb.disconnect(self.client)


def test_chain_visualized(vis: Visualizer, chain_name: str) -> None:
    """Solve IK for one chain and visualize initial -> solution."""
    print(f"\n{'='*60}")
    print(f"Chain: {chain_name}")
    print(f"{'='*60}")

    solver = create_ik_solver(chain_name)
    seed = CHAIN_SEEDS[chain_name]

    print(f"  DOF: {solver.num_joints}")
    print(f"  base: {solver.base_frame}")
    print(f"  ee:   {solver.ee_frame}")

    # Show initial config
    vis.set_full_config(HOME_JOINTS)
    vis.clear_markers()

    # FK at seed to get current EE pose
    current_pose = solver.fk(seed)
    print(f"  Current EE pos: {np.round(current_pose.position, 4).tolist()}")

    # Draw current EE frame
    vis.draw_frame(current_pose.position, current_pose.rotation, length=0.06, width=2)

    # Target: offset from current pose
    target_position = current_pose.position + np.array([0.10, 0.08, 0.05])
    angle = np.deg2rad(20)
    rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    target_pose = SE3Pose(
        position=target_position,
        rotation=rot_z @ current_pose.rotation,
    )

    # Draw target
    vis.draw_frame(target_pose.position, target_pose.rotation)
    vis.draw_sphere(target_pose.position)
    print(f"  Target EE pos:  {np.round(target_position, 4).tolist()}")

    vis.wait_key(
        ord("n"), f"[{chain_name}] Initial config shown. Press 'n' to solve IK."
    )

    # Solve IK
    result = solver.solve(target_pose, seed=seed)
    print(
        f"  IK: {result.status.value}, pos_err={result.position_error:.6f}m, "
        f"ori_err={result.orientation_error:.6f}rad, attempts={result.iterations}"
    )

    if result.joint_positions is not None:
        print(f"  Solution: {np.round(result.joint_positions, 4).tolist()}")
        vis.set_chain_solution(chain_name, result.joint_positions)

        # Draw achieved EE frame
        achieved_pose = solver.fk(result.joint_positions)
        vis.draw_frame(
            achieved_pose.position, achieved_pose.rotation, length=0.05, width=2
        )

    vis.wait_key(ord("n"), f"[{chain_name}] Solution shown. Press 'n' for next chain.")


def main() -> None:
    print("TRAC-IK Solver — PyBullet Visualization")
    print("=" * 60)
    print("Controls: 'n' = next step, 'q' = quit")

    vis = Visualizer()

    chains = ["left_arm", "right_arm", "whole_body_left", "whole_body_right"]

    for chain_name in chains:
        try:
            test_chain_visualized(vis, chain_name)
        except Exception as e:
            print(f"  ERROR on {chain_name}: {e}")
            import traceback

            traceback.print_exc()

    vis.wait_key(ord("q"), "All chains done. Press 'q' to quit.")
    vis.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
