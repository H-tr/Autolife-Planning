from dataclasses import dataclass
from enum import Enum

import numpy as np

from autolife_planning.utils.rot_utils import matrix_to_quaternion, quaternion_to_matrix


class IKStatus(Enum):
    """Status of IK solution attempt."""

    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    SINGULAR = "singular"
    FAILED = "failed"


@dataclass
class SE3Pose:
    """
    Represents a 6D pose in SE(3) - position and orientation.
    Orientation is stored as a 3x3 rotation matrix.
    """

    position: np.ndarray  # (3,) xyz
    rotation: np.ndarray  # (3, 3) rotation matrix

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,), got {self.position.shape}")
        if self.rotation.shape != (3, 3):
            raise ValueError(
                f"Rotation must be shape (3, 3), got {self.rotation.shape}"
            )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "SE3Pose":
        """Create SE3Pose from 4x4 homogeneous transformation matrix."""
        matrix = np.asarray(matrix)
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be shape (4, 4), got {matrix.shape}")
        return cls(position=matrix[:3, 3], rotation=matrix[:3, :3])

    @classmethod
    def from_position_quat(
        cls, position: np.ndarray, quaternion: np.ndarray
    ) -> "SE3Pose":
        """Create SE3Pose from position and quaternion (w, x, y, z)."""
        position = np.asarray(position)
        quaternion = np.asarray(quaternion)
        rotation = quaternion_to_matrix(quaternion)
        return cls(position=position, rotation=rotation)

    @classmethod
    def from_position_rpy(
        cls, position: np.ndarray, roll: float, pitch: float, yaw: float
    ) -> "SE3Pose":
        """Create SE3Pose from position and roll-pitch-yaw angles."""
        from autolife_planning.utils.rot_utils import rpy_to_matrix

        position = np.asarray(position)
        rotation = rpy_to_matrix(roll, pitch, yaw)
        return cls(position=position, rotation=rotation)

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.position
        return matrix

    def to_quaternion(self) -> np.ndarray:
        """Convert rotation to quaternion (w, x, y, z)."""
        return matrix_to_quaternion(self.rotation)

    def to_rpy(self) -> tuple[float, float, float]:
        """Convert rotation to roll-pitch-yaw angles."""
        from autolife_planning.utils.rot_utils import matrix_to_rpy

        return matrix_to_rpy(self.rotation)


@dataclass
class IKResult:
    """Result of an IK solution attempt."""

    status: IKStatus
    joint_positions: np.ndarray | None  # Solution if successful
    final_error: float  # Final pose error
    iterations: int  # Number of iterations performed
    position_error: float  # Final position error (meters)
    orientation_error: float  # Final orientation error (radians)

    @property
    def success(self) -> bool:
        return self.status == IKStatus.SUCCESS
