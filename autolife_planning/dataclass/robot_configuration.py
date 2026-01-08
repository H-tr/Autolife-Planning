from dataclasses import dataclass
import numpy as np

@dataclass
class BaseConfiguration:
    """
    Represents the 3 DoF base configuration (x, y, theta).
    """
    x: float
    y: float
    theta: float

    @classmethod
    def from_array(cls, array: np.ndarray | list[float]) -> "BaseConfiguration":
        if len(array) != 3:
            raise ValueError(f"Expected 3 elements for BaseConfiguration, got {len(array)}")
        return cls(x=array[0], y=array[1], theta=array[2])
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

@dataclass
class RobotConfiguration:
    """
    Represents the 18 DoF robot configuration (joints).
    """
    values: np.ndarray

    def __post_init__(self):
        if len(self.values) != 18:
            raise ValueError(f"Expected 18 elements for RobotConfiguration, got {len(self.values)}")
        self.values = np.array(self.values)

    @classmethod
    def from_array(cls, array: np.ndarray | list[float]) -> "RobotConfiguration":
        return cls(values=np.array(array))

    def to_array(self) -> np.ndarray:
        return self.values

@dataclass
class ArmConfiguration:
    """
    Represents the arm configuration for a specific side (left or right).
    """
    values: np.ndarray
    side: str

    def __post_init__(self):
        if self.side not in ["left", "right"]:
            raise ValueError(f"Side must be 'left' or 'right', got '{self.side}'")
        self.values = np.array(self.values)

    @classmethod
    def from_array(cls, array: np.ndarray | list[float], side: str) -> "ArmConfiguration":
        return cls(values=np.array(array), side=side)

    def to_array(self) -> np.ndarray:
        return self.values

@dataclass
class WholeBodyConfiguration:
    """
    Represents the 21 DoF whole body configuration (18 DoF Robot + 3 DoF Base).
    """
    robot_config: RobotConfiguration
    base_config: BaseConfiguration

    @classmethod
    def from_array(cls, array: np.ndarray | list[float]) -> "WholeBodyConfiguration":
        if len(array) != 21:
            raise ValueError(f"Expected 21 elements for WholeBodyConfiguration, got {len(array)}")
        
        # Assuming the order is RobotConfiguration (18) + BaseConfiguration (3)
        # based on "configuration + base" description.
        robot_values = array[:18]
        base_values = array[18:]
        
        return cls(
            robot_config=RobotConfiguration.from_array(robot_values),
            base_config=BaseConfiguration.from_array(base_values)
        )

    def to_array(self) -> np.ndarray:
        return np.concatenate((self.robot_config.to_array(), self.base_config.to_array()))
