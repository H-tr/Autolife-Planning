from dataclasses import dataclass


@dataclass
class IKConfig:
    """Configuration parameters for IK solver."""

    max_iterations: int = 200
    position_tolerance: float = 1e-4  # meters
    orientation_tolerance: float = 1e-4  # radians
    damping: float = 1e-6  # Damped least squares regularization
    step_size: float = 1.0  # Step size for iteration
    use_limits: bool = True  # Respect joint limits

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.position_tolerance <= 0:
            raise ValueError("position_tolerance must be > 0")
        if self.orientation_tolerance <= 0:
            raise ValueError("orientation_tolerance must be > 0")
