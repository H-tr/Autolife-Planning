from typing import Any

import numpy as np
from vamp import pybullet_interface as vpb

from autolife_planning.dataclass.robot_configuration import (
    BaseConfiguration,
    RobotConfiguration,
)
from autolife_planning.envs.base_env import BaseEnv


class PyBulletEnv(BaseEnv):
    def __init__(self, urdf_path: str, joint_names: list[str], visualize: bool = True):
        self.sim = vpb.PyBulletSimulator(urdf_path, joint_names, visualize=visualize)
        self.joint_names = joint_names

        # Find camera link index
        self.camera_link_idx = -1
        # Use skel_id (body id) to query joints
        num_joints = self.sim.client.getNumJoints(self.sim.skel_id)
        for i in range(num_joints):
            info = self.sim.client.getJointInfo(self.sim.skel_id, i)
            # info[12] is the child link name (bytes)
            if info[12].decode("utf-8") == "Link_Camera_Chest":
                self.camera_link_idx = i
                break

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

    def get_rgbd(self):
        if self.camera_link_idx == -1:
            return None

        # Get link state
        ls = self.sim.client.getLinkState(self.sim.skel_id, self.camera_link_idx)
        pos = ls[0]
        orn = ls[1]

        # Calculate view matrix
        rot_mat = np.array(self.sim.client.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # Assume camera link X is forward, Z is up (common sensor frame)
        forward = rot_mat[:, 0]
        up = rot_mat[:, 2]

        view_matrix = self.sim.client.computeViewMatrix(
            cameraEyePosition=pos, cameraTargetPosition=pos + forward, cameraUpVector=up
        )

        width = 640
        height = 480
        fov = 60
        aspect = width / height
        near = 0.1
        far = 10.0

        proj_matrix = self.sim.client.computeProjectionMatrixFOV(fov, aspect, near, far)

        img = self.sim.client.getCameraImage(width, height, view_matrix, proj_matrix)

        # img[2] is RGB, img[3] is Depth
        rgb = np.array(img[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        depth = np.array(img[3], dtype=np.float32).reshape(height, width)

        return {"rgb": rgb, "depth": depth}

    def get_obs(self) -> Any:
        return {
            "joint_states": self.get_joint_states(),
            "localization": self.get_localization(),
            "camera_chest": self.get_rgbd(),
        }

    def step(self):
        self.sim.client.stepSimulation()

    def add_pointcloud(
        self, points: np.ndarray, lifetime: float = 0.0, pointsize: int = 3
    ):
        self.sim.draw_pointcloud(points, lifetime, pointsize)
