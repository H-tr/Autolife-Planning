"""
Pinocchio-based forward kinematics and Jacobian computation.

Used by motion planning. IK solving is handled by trac_ik_solver.py.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from autolife_planning.types import SE3Pose

pin = importlib.import_module("pinocchio")


@dataclass
class PinocchioContext:
    """Context holding Pinocchio model and data for FK/Jacobian computations."""

    model: Any
    data: Any
    end_effector_frame_id: int
    joint_names: list[str]
    joint_ids: list[int]


def create_pinocchio_context(
    urdf_path: str,
    end_effector_frame: str,
    joint_names: list[str] | None = None,
) -> PinocchioContext:
    """
    Create a Pinocchio context from URDF file.

    Input:
        urdf_path: Path to the URDF file
        end_effector_frame: Name of the end effector frame/link
        joint_names: Optional list of joint names to control (if None, uses all joints)
    Output:
        PinocchioContext containing model and data for FK/Jacobian computations
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

    return PinocchioContext(
        model=model,
        data=data,
        end_effector_frame_id=ee_frame_id,
        joint_names=actual_joint_names,
        joint_ids=joint_ids,
    )


def compute_forward_kinematics(
    context: PinocchioContext,
    joint_positions: np.ndarray,
) -> SE3Pose:
    """
    Compute forward kinematics for the end effector.

    Input:
        context: Pinocchio context with model and data
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
    context: PinocchioContext,
    joint_positions: np.ndarray,
    local_frame: bool = False,
) -> np.ndarray:
    """
    Compute the Jacobian matrix at the end effector.

    Input:
        context: Pinocchio context with model and data
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


# --- Internal helper functions ---


def _to_pinocchio_config(
    context: PinocchioContext, joint_positions: np.ndarray
) -> np.ndarray:
    """Convert controlled joint positions to full Pinocchio configuration."""
    q = pin.neutral(context.model)
    for i, jid in enumerate(context.joint_ids):
        idx = context.model.joints[jid].idx_q
        q[idx] = joint_positions[i]
    return q


def _extract_controlled_jacobian(
    context: PinocchioContext, J_full: np.ndarray
) -> np.ndarray:
    """Extract Jacobian columns for controlled joints only."""
    cols = []
    for jid in context.joint_ids:
        idx_v = context.model.joints[jid].idx_v
        cols.append(J_full[:, idx_v])
    return np.column_stack(cols)
