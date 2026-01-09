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
