"""SE3Pose construction and round-trip conversions."""

from __future__ import annotations

import numpy as np
import pytest

from autolife_planning.types import SE3Pose


def test_construction_validates_shapes():
    SE3Pose(position=np.zeros(3), rotation=np.eye(3))

    with pytest.raises(ValueError):
        SE3Pose(position=np.zeros(2), rotation=np.eye(3))
    with pytest.raises(ValueError):
        SE3Pose(position=np.zeros(3), rotation=np.eye(4))


def test_matrix_round_trip():
    matrix = np.array(
        [
            [0.0, -1.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 1.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose = SE3Pose.from_matrix(matrix)
    np.testing.assert_allclose(pose.to_matrix(), matrix)


def test_from_matrix_rejects_wrong_shape():
    with pytest.raises(ValueError):
        SE3Pose.from_matrix(np.eye(3))


def test_quaternion_round_trip():
    pose = SE3Pose.from_position_quat(
        position=np.array([1.0, 2.0, 3.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # identity
    )
    quat = pose.to_quaternion()
    # Identity quaternion: scalar part 1, vector part 0 (sign-agnostic).
    assert abs(abs(quat[0]) - 1.0) < 1e-9
    np.testing.assert_allclose(quat[1:], 0.0, atol=1e-9)


def test_rpy_round_trip():
    rpy = (0.1, -0.3, 1.2)
    pose = SE3Pose.from_position_rpy(
        position=np.zeros(3),
        roll=rpy[0],
        pitch=rpy[1],
        yaw=rpy[2],
    )
    out = pose.to_rpy()
    np.testing.assert_allclose(out, rpy, atol=1e-9)
