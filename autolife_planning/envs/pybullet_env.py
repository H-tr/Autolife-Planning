import os
import re
import tempfile
import time
from typing import Any

import numpy as np
import pybullet as pb

from autolife_planning.envs.base_env import BaseEnv
from autolife_planning.types import RobotConfig
from autolife_planning.utils import pybullet_interface as vpb


def _resolve_package_paths(urdf_path: str) -> str:
    """Resolve ``package://`` URIs in a URDF to absolute paths.

    PyBullet resolves ``package://`` relative to the URDF file's directory,
    which breaks for ROS-style ``package://pkg_name/...`` references.  This
    helper rewrites them to absolute paths in a temporary file.
    """
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    with open(urdf_path, "r") as f:
        content = f.read()

    def _replace(m: re.Match) -> str:
        pkg_name = m.group(1)
        rest = m.group(2)
        # Walk up from the URDF directory to find the package directory
        search = urdf_dir
        for _ in range(10):
            candidate = os.path.join(search, pkg_name)
            if os.path.isdir(candidate):
                return os.path.join(candidate, rest)
            parent = os.path.dirname(search)
            if parent == search:
                break
            search = parent
        return m.group(0)  # leave unchanged if not found

    resolved = re.sub(r"package://([^/]+)/([\w/.\-]+)", _replace, content)

    if resolved == content:
        return urdf_path  # nothing changed

    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", prefix="viz_", delete=False, mode="w"
    )
    tmp.write(resolved)
    tmp.close()
    return tmp.name


class PyBulletEnv(BaseEnv):
    def __init__(
        self,
        config: RobotConfig,
        visualize: bool = True,
        viz_urdf_path: str | None = None,
    ):
        self.config = config
        urdf = viz_urdf_path if viz_urdf_path else config.urdf_path
        urdf = _resolve_package_paths(urdf)
        self.sim = vpb.PyBulletSimulator(urdf, config.joint_names, visualize=visualize)
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

        # Set initial pose
        from autolife_planning.config.robot_config import HOME_JOINTS

        self.set_joint_states(HOME_JOINTS[3:])

    def get_joint_states(self) -> np.ndarray:
        states = []
        for joint_idx in self.sim.joints:
            state = self.sim.client.getJointState(self.sim.skel_id, joint_idx)
            states.append(state[0])
        return np.array(states)

    def set_joint_states(self, config: np.ndarray):
        self.sim.set_joint_positions(np.asarray(config))

    def set_base_position(self, x: float, y: float, theta: float):
        """Move the robot base to (x, y) with yaw=theta in the world frame."""
        quat = self.sim.client.getQuaternionFromEuler([0, 0, theta])
        self.sim.client.resetBasePositionAndOrientation(
            self.sim.skel_id, [x, y, 0], quat
        )

    def set_configuration(self, config: np.ndarray):
        """Apply a full 24-DOF config (3 base + 21 joints) to the visualization."""
        self.set_base_position(config[0], config[1], config[2])
        self.set_joint_states(config[3:])

    def get_localization(self) -> np.ndarray:
        pos, orn = self.sim.client.getBasePositionAndOrientation(self.sim.skel_id)
        euler = self.sim.client.getEulerFromQuaternion(orn)
        return np.array([pos[0], pos[1], euler[2]])

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

    def wait_key(self, key: str | int, message: str = "") -> None:
        """Block until *key* is pressed in the PyBullet GUI.

        Draws *message* as a temporary on-screen label and prints it
        to stdout, then polls keyboard events until the requested key
        is triggered.  Accepts either a single character (``"n"``) or
        a raw PyBullet key code (e.g. ``pb.B3G_RIGHT_ARROW``).
        """
        client = self.sim.client
        key_code = ord(key) if isinstance(key, str) else int(key)
        text_id = (
            client.addUserDebugText(
                message, [0, 0, 1.5], textColorRGB=[0, 0, 0], textSize=1.5
            )
            if message
            else None
        )
        if message:
            print(message)
        while True:
            keys = client.getKeyboardEvents()
            if key_code in keys and keys[key_code] & pb.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.01)
        if text_id is not None:
            client.removeUserDebugItem(text_id)

    def animate_path(self, path: np.ndarray, fps: float = 120.0) -> None:
        """Play back a full 24-DOF path as a smooth animation.

        Each row of *path* is a configuration accepted by
        :meth:`set_configuration`; consecutive rows are displayed at
        the requested frame rate.
        """
        if path is None or len(path) == 0:
            return
        dt = 1.0 / fps
        for config in path:
            self.set_configuration(config)
            time.sleep(dt)

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
