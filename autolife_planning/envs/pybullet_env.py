import numpy as np
from typing import Any
from autolife_planning.envs.base_env import BaseEnv
from autolife_planning.dataclass.robot_configuration import RobotConfiguration, BaseConfiguration
from vamp import pybullet_interface as vpb

class PyBulletEnv(BaseEnv):
    def __init__(self, urdf_path: str, joint_names: list[str], visualize: bool = True):
        self.sim = vpb.PyBulletSimulator(urdf_path, joint_names, visualize=visualize)
        self.joint_names = joint_names

    def get_joint_states(self) -> RobotConfiguration:
        states = []
        # PyBulletSimulator stores self.joints as indices
        for joint_idx in self.sim.joints:
            state = self.sim.client.getJointState(self.sim.skel_id, joint_idx)
            states.append(state[0])
        return RobotConfiguration.from_array(states)

    def set_joint_states(self, config: RobotConfiguration):
        self.sim.set_joint_positions(config.to_array())

    def get_localization(self) -> BaseConfiguration:
        pos, orn = self.sim.client.getBasePositionAndOrientation(self.sim.skel_id)
        euler = self.sim.client.getEulerFromQuaternion(orn)
        return BaseConfiguration(x=pos[0], y=pos[1], theta=euler[2])

    def get_obs(self) -> Any:
        return {
            "joint_states": self.get_joint_states(),
            "localization": self.get_localization()
        }

    def step(self):
        self.sim.client.stepSimulation()

    def add_pointcloud(self, points: np.ndarray, lifetime: float = 0., pointsize: int = 3):
        self.sim.draw_pointcloud(points, lifetime, pointsize)
