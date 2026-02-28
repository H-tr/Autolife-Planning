from dataclasses import dataclass


@dataclass
class CameraConfig:
    link_name: str
    width: int
    height: int
    fov: float
    near: float
    far: float


@dataclass
class RobotConfig:
    urdf_path: str
    joint_names: list[str]
    camera: CameraConfig


@dataclass(frozen=True)
class ChainConfig:
    """Configuration for a kinematic chain used by TRAC-IK."""

    base_link: str
    ee_link: str
    num_joints: int
    urdf_path: str
