import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pybullet as pb

from autolife_planning.envs.base_env import BaseEnv
from autolife_planning.types import RobotConfig
from autolife_planning.utils import pybullet_interface as vpb


def _prepare_urdf_for_pybullet(urdf_path: str) -> str:
    """Resolve ``package://`` URIs and redirect visual meshes to ``viz_meshes/``.

    PyBullet resolves ``package://`` relative to the URDF file's directory,
    which breaks for ROS-style ``package://pkg_name/...`` references — so
    this helper rewrites every ``<mesh filename="...">`` in the URDF to an
    absolute path, then writes the result to a temporary URDF file.

    Additionally, if a sibling ``viz_meshes/`` directory exists next to the
    URDF, visual ``<mesh>`` elements whose filename lives under
    ``package://meshes/`` are redirected to the matching file in
    ``viz_meshes/``.  This lets the collision pipeline use a repaired
    (sometimes convex-hulled) copy of each mesh for sphere-tree validation,
    while the PyBullet viewer still renders the pristine high-poly original.
    """
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    viz_meshes_dir = os.path.join(urdf_dir, "viz_meshes")
    use_viz = os.path.isdir(viz_meshes_dir)

    def _resolve_package(pkg_name: str) -> str | None:
        """Walk up from the URDF directory to find a ``pkg_name`` folder."""
        search = urdf_dir
        for _ in range(10):
            candidate = os.path.join(search, pkg_name)
            if os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(search)
            if parent == search:
                return None
            search = parent
        return None

    pkg_re = re.compile(r"package://([^/]+)/(.+)")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    changed = False
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for node in link.findall(tag):
                geom = node.find("geometry")
                if geom is None:
                    continue
                mesh = geom.find("mesh")
                if mesh is None:
                    continue
                fn = mesh.get("filename", "")
                new_fn = fn
                m = pkg_re.match(fn)
                if m:
                    pkg_name, rest = m.group(1), m.group(2)
                    pkg_dir = _resolve_package(pkg_name)
                    if pkg_dir is not None:
                        new_fn = os.path.join(pkg_dir, rest)
                    # Visual meshes under package://meshes/ get redirected to
                    # the untouched originals in viz_meshes/ when present.
                    if tag == "visual" and use_viz and pkg_name == "meshes":
                        viz_candidate = os.path.join(viz_meshes_dir, rest)
                        if os.path.isfile(viz_candidate):
                            new_fn = viz_candidate
                if new_fn != fn:
                    mesh.set("filename", new_fn)
                    changed = True

    if not changed:
        return urdf_path

    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", prefix="viz_", delete=False, mode="wb"
    )
    tree.write(tmp.name, encoding="utf-8", xml_declaration=True)
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
        urdf = _prepare_urdf_for_pybullet(urdf)
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

        Prints *message* to stdout, then polls keyboard events until the
        requested key is triggered.  Accepts either a single character
        (``"n"``) or a raw PyBullet key code (e.g. ``pb.B3G_RIGHT_ARROW``).
        Returns early if the GUI window is closed.
        """
        client = self.sim.client
        key_code = ord(key) if isinstance(key, str) else int(key)
        if message:
            print(message)
        try:
            while client.isConnected():
                keys = client.getKeyboardEvents()
                if key_code in keys and keys[key_code] & pb.KEY_WAS_TRIGGERED:
                    break
                time.sleep(0.01)
        except pb.error:
            pass

    def wait_for_close(self) -> None:
        """Block until the user closes the PyBullet GUI window."""
        client = self.sim.client
        try:
            while client.isConnected():
                client.getKeyboardEvents()  # keeps the event queue alive
                time.sleep(0.05)
        except pb.error:
            pass

    def animate_path(
        self,
        path: np.ndarray,
        fps: float = 60.0,
        next_key: str | int | None = None,
    ) -> bool:
        """Interactively play back a full-DOF path, VAMP-style.

        Each row of *path* is handed to :meth:`set_configuration`.  The
        method blocks on the PyBullet GUI with these controls:

        * ``SPACE``             — toggle auto-play (loops end → start)
        * ``← / →``             — step one waypoint back / forward (while paused)
        * ``next_key`` (opt.)   — exit and return ``True`` — useful when
          the caller wants to drive a sequence of demos

        Exits when the user closes the GUI window (returns ``False``) or
        presses *next_key* if one is provided (returns ``True``).  The
        animation starts paused so the caller can read the banner before
        anything moves.

        Args:
          path: ``(N, dof)`` array of full-DOF configurations.
          fps:  playback frame rate while auto-playing.
          next_key: optional single-character string or raw PyBullet key
            code; pressing it exits the viewer.

        Returns:
          ``True`` if *next_key* was pressed, ``False`` otherwise.
        """
        if path is None or len(path) == 0:
            return False

        client = self.sim.client
        dt = 1.0 / fps
        n = int(len(path))
        idx = 0
        playing = False

        left = pb.B3G_LEFT_ARROW
        right = pb.B3G_RIGHT_ARROW
        space = ord(" ")
        next_code: int | None = None
        if next_key is not None:
            next_code = ord(next_key) if isinstance(next_key, str) else int(next_key)

        parts = ["SPACE play/pause", "←/→ step"]
        if next_code is not None:
            parts.append(f"'{next_key}' next")
        parts.append("close window to exit")
        print("  " + "  |  ".join(parts))

        advanced_to_next = False
        try:
            while client.isConnected():
                self.set_configuration(path[idx])

                keys = client.getKeyboardEvents()
                if space in keys and keys[space] & pb.KEY_WAS_TRIGGERED:
                    playing = not playing
                elif (
                    next_code is not None
                    and next_code in keys
                    and keys[next_code] & pb.KEY_WAS_TRIGGERED
                ):
                    advanced_to_next = True
                    break
                elif not playing and left in keys and keys[left] & pb.KEY_WAS_TRIGGERED:
                    idx = (idx - 1) % n
                elif (
                    not playing and right in keys and keys[right] & pb.KEY_WAS_TRIGGERED
                ):
                    idx = (idx + 1) % n
                elif playing:
                    idx = (idx + 1) % n

                time.sleep(dt)
        except pb.error:
            pass

        return advanced_to_next

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
        name: str | None = None,
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

    # ── scene-setup primitives ────────────────────────────────────────

    def draw_plane(
        self,
        center,
        half_sizes: tuple[float, float] = (0.35, 0.35),
        normal=(0.0, 0.0, 1.0),
        color: tuple[float, float, float, float] = (0.15, 0.55, 0.95, 0.35),
    ) -> int:
        """Visual-only translucent plate at *center*.

        The plate is a thin ``GEOM_BOX`` (1 mm thick along its normal),
        oriented so its thin axis lines up with *normal*.  Useful for
        showing a plane manifold — e.g. ``z = z0`` — in the scene.
        """
        client = self.sim.client
        vid = client.createVisualShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=[float(half_sizes[0]), float(half_sizes[1]), 0.001],
            rgbaColor=list(color),
        )
        return client.createMultiBody(
            baseVisualShapeIndex=vid,
            basePosition=list(np.asarray(center, dtype=float)),
            baseOrientation=_quat_from_z_axis(normal),
        )

    def draw_rod(
        self,
        p1,
        p2,
        radius: float = 0.008,
        color: tuple[float, float, float, float] = (0.20, 0.90, 0.30, 1.0),
    ) -> int | None:
        """Visual-only solid rod (``GEOM_CYLINDER``) from *p1* to *p2*.

        Uses a real 3D cylinder rather than ``addUserDebugLine`` so the
        segment has a proper thickness, catches lighting, and shows up
        correctly in screenshots.  Returns ``None`` if *p1* and *p2*
        coincide.
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        delta = p2 - p1
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            return None
        client = self.sim.client
        vid = client.createVisualShape(
            shapeType=pb.GEOM_CYLINDER,
            radius=float(radius),
            length=length,
            rgbaColor=list(color),
        )
        return client.createMultiBody(
            baseVisualShapeIndex=vid,
            basePosition=(0.5 * (p1 + p2)).tolist(),
            baseOrientation=_quat_from_z_axis(delta / length),
        )

    def draw_sphere(
        self,
        center,
        radius: float,
        color: tuple[float, float, float, float] = (0.95, 0.30, 0.30, 0.55),
    ) -> int:
        """Visual-only sphere — handy for marking obstacles or targets."""
        client = self.sim.client
        vid = client.createVisualShape(
            shapeType=pb.GEOM_SPHERE,
            radius=float(radius),
            rgbaColor=list(color),
        )
        return client.createMultiBody(
            baseVisualShapeIndex=vid,
            basePosition=list(np.asarray(center, dtype=float)),
        )

    def draw_frame(
        self,
        position,
        rotation,
        size: float = 0.12,
        radius: float = 0.006,
    ) -> list[int]:
        """RGB coordinate axes at *position* with a 3x3 *rotation* matrix.

        Red / green / blue rods along the rotation's first / second /
        third columns — useful for visualising an orientation-lock
        constraint (the frame at the start and goal look identical).
        """
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        colors = [
            (1.0, 0.15, 0.15, 1.0),
            (0.15, 1.0, 0.15, 1.0),
            (0.15, 0.40, 1.0, 1.0),
        ]
        ids: list[int] = []
        for i, color in enumerate(colors):
            tip = position + rotation[:, i] * size
            body_id = self.draw_rod(position, tip, radius=radius, color=color)
            if body_id is not None:
                ids.append(body_id)
        return ids


def _quat_from_z_axis(direction) -> list[float]:
    """Quaternion ``[x, y, z, w]`` that rotates ``+Z`` onto *direction*.

    Used to orient thin ``GEOM_BOX`` plates and ``GEOM_CYLINDER`` rods,
    both of which default to the +Z axis in PyBullet.
    """
    d = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, d))
    if dot > 0.99999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.99999:
        # 180° around X
        return [1.0, 0.0, 0.0, 0.0]
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    angle = float(np.arccos(dot))
    s = float(np.sin(angle / 2.0))
    return [
        float(axis[0] * s),
        float(axis[1] * s),
        float(axis[2] * s),
        float(np.cos(angle / 2.0)),
    ]
