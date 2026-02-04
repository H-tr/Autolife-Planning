"""Rotation utility functions for conversions between representations."""

import numpy as np


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Input:
        quat: Quaternion in (w, x, y, z) format, shape (4,)
    Output:
        rotation matrix, shape (3, 3)
    """
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat
    # Normalize
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion.

    Input:
        rot: Rotation matrix, shape (3, 3)
    Output:
        quaternion in (w, x, y, z) format, shape (4,)
    """
    rot = np.asarray(rot, dtype=np.float64)
    trace = np.trace(rot)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll-pitch-yaw (XYZ Euler angles) to rotation matrix.

    Input:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)
    Output:
        rotation matrix, shape (3, 3)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """
    Convert rotation matrix to roll-pitch-yaw (XYZ Euler angles).

    Input:
        rot: Rotation matrix, shape (3, 3)
    Output:
        (roll, pitch, yaw) in radians
    """
    rot = np.asarray(rot, dtype=np.float64)

    if abs(rot[2, 0]) < 1.0 - 1e-10:
        pitch = -np.arcsin(rot[2, 0])
        roll = np.arctan2(rot[2, 1] / np.cos(pitch), rot[2, 2] / np.cos(pitch))
        yaw = np.arctan2(rot[1, 0] / np.cos(pitch), rot[0, 0] / np.cos(pitch))
    else:
        # Gimbal lock
        yaw = 0.0
        if rot[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(rot[0, 1], rot[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-rot[0, 1], -rot[0, 2])

    return roll, pitch, yaw


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix (Rodrigues formula).

    Input:
        axis: Unit vector representing rotation axis, shape (3,)
        angle: Rotation angle in radians
    Output:
        rotation matrix, shape (3, 3)
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def matrix_to_axis_angle(rot: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convert rotation matrix to axis-angle representation.

    Input:
        rot: Rotation matrix, shape (3, 3)
    Output:
        (axis, angle) where axis is unit vector shape (3,), angle in radians
    """
    rot = np.asarray(rot, dtype=np.float64)

    angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))

    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0

    if abs(angle - np.pi) < 1e-10:
        # Find column of (R + I) with largest norm
        B = rot + np.eye(3)
        col_norms = np.linalg.norm(B, axis=0)
        idx = np.argmax(col_norms)
        axis = B[:, idx] / col_norms[idx]
        return axis, angle

    axis = np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
    )
    axis = axis / (2 * np.sin(angle))

    return axis, angle
