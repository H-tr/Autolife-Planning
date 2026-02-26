from dataclasses import dataclass
from enum import Enum

import numpy as np


class IKStatus(Enum):
    """Status of IK solution attempt."""

    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    SINGULAR = "singular"
    FAILED = "failed"


class SolveType(Enum):
    """TRAC-IK solve type.

    SPEED:    Return first valid solution (fastest).
    DISTANCE: Minimize joint displacement from seed configuration.
    MANIP1:   Maximize manipulability (product of Jacobian singular values).
    MANIP2:   Maximize isotropy (min/max Jacobian singular value ratio).
    """

    SPEED = "Speed"
    DISTANCE = "Distance"
    MANIP1 = "Manip1"
    MANIP2 = "Manip2"


@dataclass
class IKConfig:
    """Configuration parameters for TRAC-IK solver."""

    timeout: float = 0.2  # Seconds for TRAC-IK dual-thread solve
    epsilon: float = 1e-5  # TRAC-IK convergence tolerance
    solve_type: SolveType = SolveType.SPEED  # Solve strategy
    max_attempts: int = 10  # Python-level random restart attempts
    position_tolerance: float = 1e-4  # Post-solve position validation (meters)
    orientation_tolerance: float = 1e-4  # Post-solve orientation validation (radians)

    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.position_tolerance <= 0:
            raise ValueError("position_tolerance must be > 0")
        if self.orientation_tolerance <= 0:
            raise ValueError("orientation_tolerance must be > 0")


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
