"""TRAC-IK based inverse kinematics solver.

Wraps the vendored TRAC-IK C++ library (dual KDL + NLopt threading)
via pybind11 bindings. Pinocchio is still used for FK/Jacobian in
motion planning; this module handles IK only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

from autolife_planning.config.robot_config import CHAIN_CONFIGS
from autolife_planning.types import (
    ChainConfig,
    IKConfig,
    IKResult,
    IKStatus,
    SE3Pose,
    SolveType,
)


@runtime_checkable
class IKSolverBase(Protocol):
    """Protocol for IK solver backends."""

    @property
    def base_frame(self) -> str:
        ...

    @property
    def ee_frame(self) -> str:
        ...

    @property
    def num_joints(self) -> int:
        ...

    def solve(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: IKConfig | None = None,
    ) -> IKResult:
        ...


# Map our SolveType enum to pytracik C++ enum values
_SOLVE_TYPE_MAP = {
    SolveType.SPEED: "Speed",
    SolveType.DISTANCE: "Distance",
    SolveType.MANIP1: "Manip1",
    SolveType.MANIP2: "Manip2",
}


class TracIKSolver:
    """IK solver wrapping the TRAC-IK C++ library."""

    def __init__(
        self,
        chain_config: ChainConfig,
        config: IKConfig | None = None,
    ) -> None:
        if config is None:
            config = IKConfig()

        import pytracik

        # Read URDF string for TRAC-IK constructor
        with open(chain_config.urdf_path) as f:
            urdf_string = f.read()

        # Map SolveType enum
        cpp_solve_type = getattr(pytracik.SolveType, _SOLVE_TYPE_MAP[config.solve_type])

        self._trac_ik = pytracik.TRAC_IK(
            chain_config.base_link,
            chain_config.ee_link,
            urdf_string,
            config.timeout,
            config.epsilon,
            cpp_solve_type,
        )
        self._pytracik = pytracik
        self._chain_config = chain_config
        self._config = config

        # Cache joint limits
        self._lower_bounds = np.array(pytracik.get_joint_lower_bounds(self._trac_ik))
        self._upper_bounds = np.array(pytracik.get_joint_upper_bounds(self._trac_ik))
        self._n_joints = pytracik.get_num_joints(self._trac_ik)

    @property
    def base_frame(self) -> str:
        return self._chain_config.base_link

    @property
    def ee_frame(self) -> str:
        return self._chain_config.ee_link

    @property
    def num_joints(self) -> int:
        return self._n_joints

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits as (lower_bounds, upper_bounds) arrays."""
        return self._lower_bounds.copy(), self._upper_bounds.copy()

    def set_joint_limits(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Override joint limits for the solver.

        Input:
            lower: Lower bounds array of length num_joints
            upper: Upper bounds array of length num_joints
        """
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if len(lower) != self._n_joints or len(upper) != self._n_joints:
            raise ValueError(
                f"Expected {self._n_joints} limits, "
                f"got lower={len(lower)}, upper={len(upper)}"
            )
        self._pytracik.set_joint_limits(self._trac_ik, lower, upper)
        self._lower_bounds = lower.copy()
        self._upper_bounds = upper.copy()

    def fk(self, joint_positions: np.ndarray) -> SE3Pose:
        """Compute forward kinematics via TRAC-IK's built-in FK.

        Input:
            joint_positions: Joint angles array of length num_joints
        Output:
            SE3Pose of the end effector
        """
        q = np.asarray(joint_positions, dtype=np.float64)
        T = self._pytracik.fk(self._trac_ik, q)
        if T.size == 0:
            raise RuntimeError("FK failed — invalid joint configuration")
        return SE3Pose.from_matrix(T)

    def solve(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: IKConfig | None = None,
    ) -> IKResult:
        """Solve IK with random restart loop.

        Input:
            target_pose: Desired end-effector pose
            seed: Initial joint configuration (random if None)
            config: Override default IK config for this solve
        Output:
            IKResult with solution status and joint positions
        """
        cfg = config if config is not None else self._config

        # Extract target position and quaternion (x, y, z, w) for pytracik
        x, y, z = target_pose.position
        quat_xyzw = Rotation.from_matrix(target_pose.rotation).as_quat()  # [x,y,z,w]
        qx, qy, qz, qw = quat_xyzw

        best_result: IKResult | None = None

        for attempt in range(cfg.max_attempts):
            if seed is not None and attempt == 0:
                q_seed = np.asarray(seed, dtype=np.float64)
            else:
                # Random seed within joint limits
                q_seed = np.random.uniform(self._lower_bounds, self._upper_bounds)

            result_arr = self._pytracik.ik(
                self._trac_ik, q_seed, x, y, z, qx, qy, qz, qw
            )

            ret_code = int(result_arr[0])
            if ret_code >= 0:
                q_solution = result_arr[1:]

                # Validate solution via FK
                achieved_pose = self.fk(q_solution)
                pos_err = float(
                    np.linalg.norm(achieved_pose.position - target_pose.position)
                )
                # Orientation error: angle of rotation difference
                R_err = achieved_pose.rotation.T @ target_pose.rotation
                ori_err = float(np.linalg.norm(Rotation.from_matrix(R_err).as_rotvec()))

                result = IKResult(
                    status=IKStatus.SUCCESS,
                    joint_positions=q_solution,
                    final_error=pos_err + ori_err,
                    iterations=attempt + 1,
                    position_error=pos_err,
                    orientation_error=ori_err,
                )

                # Check if within post-solve tolerances
                if (
                    pos_err < cfg.position_tolerance
                    and ori_err < cfg.orientation_tolerance
                ):
                    return result

                # Keep best result so far
                if best_result is None or result.final_error < best_result.final_error:
                    best_result = result

        # Return best result found, or failure
        if best_result is not None:
            return best_result

        return IKResult(
            status=IKStatus.FAILED,
            joint_positions=None,
            final_error=float("inf"),
            iterations=cfg.max_attempts,
            position_error=float("inf"),
            orientation_error=float("inf"),
        )


def create_ik_solver(
    chain_name: str,
    config: IKConfig | None = None,
    side: str | None = None,
) -> TracIKSolver:
    """Factory function to create a TRAC-IK solver for a named chain.

    Input:
        chain_name: Name of the chain (e.g. "left_arm", "whole_body")
        config: IK configuration (uses defaults if None)
        side: Optional "left" or "right" suffix for compound names
    Output:
        TracIKSolver instance

    Examples:
        create_ik_solver("left_arm")
        create_ik_solver("whole_body", side="left")  # resolves to "whole_body_left"
    """
    if side is not None:
        chain_name = f"{chain_name}_{side}"

    if chain_name not in CHAIN_CONFIGS:
        available = ", ".join(sorted(CHAIN_CONFIGS.keys()))
        raise ValueError(f"Unknown chain '{chain_name}'. Available chains: {available}")

    return TracIKSolver(CHAIN_CONFIGS[chain_name], config)
