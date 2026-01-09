import os
from typing import Any

import numpy as np
import pybullet as pb
from vamp import pybullet_interface as vpb

from autolife_planning.dataclass.robot_configuration import (
    BaseConfiguration,
    RobotConfiguration,
)
from autolife_planning.dataclass.robot_description import RobotConfig
from autolife_planning.envs.base_env import BaseEnv


class PyBulletEnv(BaseEnv):
    def __init__(self, config: RobotConfig, visualize: bool = True):
        self.config = config
        self.sim = vpb.PyBulletSimulator(
            config.urdf_path, config.joint_names, visualize=visualize
        )
        self.joint_names = config.joint_names

        # Find camera link index
        self.camera_link_idx = -1
        # Use skel_id (body id) to query joints
        num_joints = self.sim.client.getNumJoints(self.sim.skel_id)
        for i in range(num_joints):
            info = self.sim.client.getJointInfo(self.sim.skel_id, i)
            # info[12] is the child link name (bytes)
            if info[12].decode("utf-8") == config.camera.link_name:
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

        # Camera link Z is forward, Y is up (URDF frame)
        forward = rot_mat[:, 2]
        up = rot_mat[:, 1]

        view_matrix = self.sim.client.computeViewMatrix(
            cameraEyePosition=pos, cameraTargetPosition=pos + forward, cameraUpVector=up
        )

        width = self.config.camera.width
        height = self.config.camera.height
        fov = self.config.camera.fov
        aspect = width / height
        near = self.config.camera.near
        far = self.config.camera.far

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

    def add_mesh(
        self,
        mesh_file: str,
        position: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        scale: np.ndarray = np.ones(3),
        mass: float = 0.0,
        name: str = None,
    ):
        """
        Add a mesh to the simulation environment directly using the raw PyBullet client,
        bypassing the wrapper to avoid modifying third_party code.
        """
        # Ensure the simulator isn't rendering while we load to speed it up
        # We can't easily use the DisableRendering context manager from vamp here
        # without importing it, but we can access the client directly.
        self.sim.client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        vis_shape_id = self.sim.client.createVisualShape(
            shapeType=pb.GEOM_MESH,
            fileName=mesh_file,
            meshScale=scale.tolist(),
        )
        col_shape_id = self.sim.client.createCollisionShape(
            shapeType=pb.GEOM_MESH,
            fileName=mesh_file,
            meshScale=scale.tolist(),
        )

        multibody_id = self.sim.client.createMultiBody(
            baseVisualShapeIndex=vis_shape_id,
            baseCollisionShapeIndex=col_shape_id,
            basePosition=position.tolist(),
            baseOrientation=orientation.tolist(),
            baseMass=mass,
        )

        if name:
            # Add debug text
            self.sim.client.addUserDebugText(
                text=name,
                textPosition=position.tolist(),
                textColorRGB=[0.0, 0.0, 0.0],
            )

        # Try to load texture if it exists
        base_path = os.path.splitext(mesh_file)[0]
        for ext in [".png", ".jpg", ".jpeg", ".tga"]:
            tex_path = base_path + ext
            if os.path.exists(tex_path):
                tex_id = self.sim.client.loadTexture(tex_path)
                self.sim.client.changeVisualShape(
                    multibody_id, -1, textureUniqueId=tex_id
                )
                break

        self.sim.client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        return multibody_id
