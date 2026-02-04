"""
Inverse Kinematics solver using Pinocchio.

This module provides functional interfaces for solving IK problems
using damped least squares (Levenberg-Marquardt) method.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from autolife_planning.config.ik_config import IKConfig
from autolife_planning.dataclass.ik_types import IKResult, IKStatus, SE3Pose


@dataclass
class IKContext:
    """Context holding Pinocchio model and data for IK computations."""

    model: pin.Model
    data: pin.Data
    end_effector_frame_id: int
    joint_names: list[str]
    joint_ids: list[int]


def create_ik_context(
    urdf_path: str,
    end_effector_frame: str,
    joint_names: list[str] | None = None,
) -> IKContext:
    """
    Create an IK context from URDF file.

    Input:
        urdf_path: Path to the URDF file
        end_effector_frame: Name of the end effector frame/link
        joint_names: Optional list of joint names to control (if None, uses all joints)
    Output:
        IKContext containing model and data for IK computations
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # Get end effector frame ID
    if not model.existFrame(end_effector_frame):
        available_frames = [
            model.frames[i].name for i in range(int(model.nframes))  # type: ignore[arg-type]
        ]
        raise ValueError(
            f"Frame '{end_effector_frame}' not found. Available frames: {available_frames}"
        )
    ee_frame_id = model.getFrameId(end_effector_frame)

    # Get joint IDs for the specified joint names
    actual_joint_names: list[str]
    if joint_names is None:
        # Use all actuated joints (exclude universe joint)
        joint_ids = list(range(1, int(model.njoints)))  # type: ignore[arg-type]
        actual_joint_names = [str(model.names[i]) for i in joint_ids]  # type: ignore[index]
    else:
        joint_ids = []
        model_names_list = list(model.names)  # type: ignore[arg-type]
        for name in joint_names:
            if name not in model_names_list:
                raise ValueError(f"Joint '{name}' not found in model")
            joint_ids.append(model.getJointId(name))
        actual_joint_names = joint_names

    return IKContext(
        model=model,
        data=data,
        end_effector_frame_id=ee_frame_id,
        joint_names=actual_joint_names,
        joint_ids=joint_ids,
    )


def compute_forward_kinematics(
    context: IKContext,
    joint_positions: np.ndarray,
) -> SE3Pose:
    """
    Compute forward kinematics for the end effector.

    Input:
        context: IK context with model and data
        joint_positions: Joint positions array
    Output:
        SE3Pose of the end effector
    """
    q = _to_pinocchio_config(context, joint_positions)
    pin.forwardKinematics(context.model, context.data, q)
    pin.updateFramePlacements(context.model, context.data)

    oMf = context.data.oMf[context.end_effector_frame_id]

    return SE3Pose(
        position=np.array(oMf.translation),
        rotation=np.array(oMf.rotation),
    )


def compute_jacobian(
    context: IKContext,
    joint_positions: np.ndarray,
    local_frame: bool = False,
) -> np.ndarray:
    """
    Compute the Jacobian matrix at the end effector.

    Input:
        context: IK context with model and data
        joint_positions: Joint positions array
        local_frame: If True, compute Jacobian in local frame; else world frame
    Output:
        Jacobian matrix, shape (6, n_joints)
    """
    q = _to_pinocchio_config(context, joint_positions)
    pin.computeJointJacobians(context.model, context.data, q)
    pin.updateFramePlacements(context.model, context.data)

    reference_frame = pin.LOCAL if local_frame else pin.LOCAL_WORLD_ALIGNED

    J_full = pin.getFrameJacobian(
        context.model,
        context.data,
        context.end_effector_frame_id,
        reference_frame,
    )

    # Extract columns for controlled joints
    J = _extract_controlled_jacobian(context, J_full)

    return J


def solve_ik(
    context: IKContext,
    target_pose: SE3Pose,
    initial_config: np.ndarray,
    config: IKConfig | None = None,
) -> IKResult:
    """
    Solve inverse kinematics using damped least squares method.

    Input:
        context: IK context with model and data
        target_pose: Target SE3 pose for the end effector
        initial_config: Initial joint configuration for iteration
        config: IK solver configuration (uses defaults if None)
    Output:
        IKResult containing solution status and joint positions
    """
    if config is None:
        config = IKConfig()

    # Convert target to Pinocchio SE3
    target_se3 = pin.SE3(target_pose.rotation, target_pose.position)

    # Initialize with current configuration
    q = _to_pinocchio_config(context, initial_config)
    q_controlled = initial_config.astype(np.float64).copy()

    # Get joint limits
    q_min, q_max = _get_joint_limits(context)

    # Initialize error tracking
    error = np.zeros(6)
    position_error = float("inf")
    orientation_error = float("inf")

    for iteration in range(config.max_iterations):
        # Compute current pose
        pin.forwardKinematics(context.model, context.data, q)
        pin.updateFramePlacements(context.model, context.data)

        current_se3 = context.data.oMf[context.end_effector_frame_id]

        # Compute pose error in local frame
        error_se3 = current_se3.actInv(target_se3)
        error = np.array(pin.log6(error_se3).vector)

        # Compute position and orientation errors
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))

        # Check convergence
        if (
            position_error < config.position_tolerance
            and orientation_error < config.orientation_tolerance
        ):
            return IKResult(
                status=IKStatus.SUCCESS,
                joint_positions=q_controlled.copy(),
                final_error=float(np.linalg.norm(error)),
                iterations=iteration + 1,
                position_error=position_error,
                orientation_error=orientation_error,
            )

        # Compute Jacobian in local frame
        pin.computeJointJacobians(context.model, context.data, q)
        J_full = pin.getFrameJacobian(
            context.model,
            context.data,
            context.end_effector_frame_id,
            pin.LOCAL,
        )

        # Extract Jacobian for controlled joints
        J = _extract_controlled_jacobian(context, J_full)

        # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
        JJT = J @ J.T
        damping_matrix = config.damping**2 * np.eye(6)

        try:
            dq = J.T @ np.linalg.solve(JJT + damping_matrix, error)
        except np.linalg.LinAlgError:
            return IKResult(
                status=IKStatus.SINGULAR,
                joint_positions=None,
                final_error=float(np.linalg.norm(error)),
                iterations=iteration + 1,
                position_error=position_error,
                orientation_error=orientation_error,
            )

        # Apply step
        q_controlled = q_controlled + config.step_size * dq

        # Apply joint limits if enabled
        if config.use_limits:
            q_controlled = np.clip(q_controlled, q_min, q_max)

        # Update full configuration
        q = _to_pinocchio_config(context, q_controlled)

    # Did not converge
    return IKResult(
        status=IKStatus.MAX_ITERATIONS,
        joint_positions=q_controlled.copy(),
        final_error=float(np.linalg.norm(error)),
        iterations=config.max_iterations,
        position_error=position_error,
        orientation_error=orientation_error,
    )


def solve_ik_position_only(
    context: IKContext,
    target_position: np.ndarray,
    initial_config: np.ndarray,
    config: IKConfig | None = None,
) -> IKResult:
    """
    Solve inverse kinematics for position only (ignores orientation).

    Input:
        context: IK context with model and data
        target_position: Target position (3,) for the end effector
        initial_config: Initial joint configuration for iteration
        config: IK solver configuration (uses defaults if None)
    Output:
        IKResult containing solution status and joint positions
    """
    if config is None:
        config = IKConfig()

    target_position = np.asarray(target_position, dtype=np.float64)

    # Initialize with current configuration
    q = _to_pinocchio_config(context, initial_config)
    q_controlled = initial_config.astype(np.float64).copy()

    # Get joint limits
    q_min, q_max = _get_joint_limits(context)

    # Initialize error tracking
    error = np.zeros(3)
    position_error = float("inf")

    for iteration in range(config.max_iterations):
        # Compute current pose
        pin.forwardKinematics(context.model, context.data, q)
        pin.updateFramePlacements(context.model, context.data)

        current_position = np.array(
            context.data.oMf[context.end_effector_frame_id].translation
        )

        # Compute position error
        error = target_position - current_position
        position_error = float(np.linalg.norm(error))

        # Check convergence
        if position_error < config.position_tolerance:
            return IKResult(
                status=IKStatus.SUCCESS,
                joint_positions=q_controlled.copy(),
                final_error=position_error,
                iterations=iteration + 1,
                position_error=position_error,
                orientation_error=0.0,
            )

        # Compute Jacobian (only position part)
        pin.computeJointJacobians(context.model, context.data, q)
        J_full = pin.getFrameJacobian(
            context.model,
            context.data,
            context.end_effector_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )

        # Extract Jacobian for controlled joints (position rows only)
        J = _extract_controlled_jacobian(context, J_full)[:3, :]

        # Damped least squares
        JJT = J @ J.T
        damping_matrix = config.damping**2 * np.eye(3)

        try:
            dq = J.T @ np.linalg.solve(JJT + damping_matrix, error)
        except np.linalg.LinAlgError:
            return IKResult(
                status=IKStatus.SINGULAR,
                joint_positions=None,
                final_error=position_error,
                iterations=iteration + 1,
                position_error=position_error,
                orientation_error=0.0,
            )

        # Apply step
        q_controlled = q_controlled + config.step_size * dq

        # Apply joint limits
        if config.use_limits:
            q_controlled = np.clip(q_controlled, q_min, q_max)

        # Update full configuration
        q = _to_pinocchio_config(context, q_controlled)

    # Did not converge
    return IKResult(
        status=IKStatus.MAX_ITERATIONS,
        joint_positions=q_controlled.copy(),
        final_error=position_error,
        iterations=config.max_iterations,
        position_error=position_error,
        orientation_error=0.0,
    )


# --- Internal helper functions ---


def _to_pinocchio_config(context: IKContext, joint_positions: np.ndarray) -> np.ndarray:
    """Convert controlled joint positions to full Pinocchio configuration."""
    q = pin.neutral(context.model)
    for i, jid in enumerate(context.joint_ids):
        idx = context.model.joints[jid].idx_q
        q[idx] = joint_positions[i]
    return q


def _extract_controlled_jacobian(context: IKContext, J_full: np.ndarray) -> np.ndarray:
    """Extract Jacobian columns for controlled joints only."""
    cols = []
    for jid in context.joint_ids:
        idx_v = context.model.joints[jid].idx_v
        cols.append(J_full[:, idx_v])
    return np.column_stack(cols)


def _get_joint_limits(context: IKContext) -> tuple[np.ndarray, np.ndarray]:
    """Get joint limits for controlled joints."""
    q_min = np.array(
        [
            context.model.lowerPositionLimit[context.model.joints[jid].idx_q]
            for jid in context.joint_ids
        ]
    )
    q_max = np.array(
        [
            context.model.upperPositionLimit[context.model.joints[jid].idx_q]
            for jid in context.joint_ids
        ]
    )
    return q_min, q_max
