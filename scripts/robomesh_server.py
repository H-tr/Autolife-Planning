#!/usr/bin/env python
import glob
import logging
import os
import threading
import time

import numpy as np

# Import project interfaces
from autolife_planning.interfaces.robomesh import app
from autolife_planning.utils.stream_utils import VideoStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobomeshServer")


def load_ply(filename):
    """
    Simple PLY loader for binary_little_endian 1.0 files from Open3D.
    Assumes properties: double x, y, z, nx, ny, nz, uchar red, green, blue.
    """
    with open(filename, "rb") as f:
        # Read header
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break

        # Define dtype based on observed Open3D format
        dt = np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("nx", "<f8"),
                ("ny", "<f8"),
                ("nz", "<f8"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )

        data = np.frombuffer(f.read(), dtype=dt)
        points = np.stack([data["x"], data["y"], data["z"]], axis=-1)
        return points


class RobomeshServer:
    def __init__(self, host="0.0.0.0", port=11111):
        self.host = host
        self.port = port
        self.app = app

        # Image streaming configuration
        rtp_ip = os.getenv("RTP_VIDEO_IP", "127.0.0.1")
        rtp_port = os.getenv("RTP_VIDEO_PORT", "5004")
        self.fps = 15

        # Construct destination URL with packet size
        destination = f"rtp://{rtp_ip}:{rtp_port}?pkt_size=1500"

        # Initialize VideoStreamer with libvpx (WebRTC friendly)
        self.streamer = VideoStreamer(
            destination=destination,
            width=640,  # Default, will be updated on first frame if needed logic was here?
            height=480,  # Note: VideoStreamer expects fixed size or re-init.
            fps=self.fps,
            codec="libvpx",
        )
        self.streamer_initialized = False

        logger.info("RobomeshServer initialized")

    def stream_image(self, image_rgb: np.ndarray):
        """
        Stream an image frame via RTP.
        Args:
            image_rgb: numpy array of shape (H, W, 3) in RGB format.
        """
        if image_rgb is None:
            return

        h, w, c = image_rgb.shape
        if c != 3:
            logger.error(f"Image must have 3 channels (RGB), got {c}")
            return

        # Handle resolution change or first initialization
        if (
            not self.streamer_initialized
            or self.streamer.width != w
            or self.streamer.height != h
        ):
            # Re-initialize streamer with correct dimensions
            if self.streamer:
                self.streamer.close()

            rtp_ip = os.getenv("RTP_VIDEO_IP", "127.0.0.1")
            rtp_port = os.getenv("RTP_VIDEO_PORT", "5004")
            destination = f"rtp://{rtp_ip}:{rtp_port}?pkt_size=1500"

            self.streamer = VideoStreamer(
                destination=destination, width=w, height=h, fps=self.fps, codec="libvpx"
            )
            self.streamer_initialized = True

        self.streamer.push_frame(image_rgb)

    def stream_rgbd(self, rgb: np.ndarray, depth: np.ndarray):
        # Normalize depth to 0-255 uint8 for visualization
        depth_visual = (np.clip(depth, 0, 5.0) / 5.0 * 255).astype(np.uint8)
        depth_visual = np.stack([depth_visual] * 3, axis=-1)

        # Concatenate side-by-side
        combined = np.hstack((rgb, depth_visual))

        self.stream_image(combined)

    def run(self):
        """Run the Flask app"""
        logger.info(f"Starting RobomeshServer Flask app on port {self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def run_threaded(self):
        """Run the Flask app in a separate thread"""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t


if __name__ == "__main__":
    from autolife_planning.envs.pybullet_env import PyBulletEnv

    urdf_path = (
        "/media/run/Extend/Autolife-Planning/assets/robot/autolife/autolife.urdf"
    )
    joint_names = [
        "Joint_Ankle",
        "Joint_Knee",
        "Joint_Waist_Pitch",
        "Joint_Waist_Yaw",
        "Joint_Left_Shoulder_Inner",
        "Joint_Left_Shoulder_Outer",
        "Joint_Left_UpperArm",
        "Joint_Left_Elbow",
        "Joint_Left_Forearm",
        "Joint_Left_Wrist_Upper",
        "Joint_Left_Wrist_Lower",
        "Joint_Right_Shoulder_Inner",
        "Joint_Right_Shoulder_Outer",
        "Joint_Right_UpperArm",
        "Joint_Right_Elbow",
        "Joint_Right_Forearm",
        "Joint_Right_Wrist_Upper",
        "Joint_Right_Wrist_Lower",
    ]
    env = PyBulletEnv(urdf_path, joint_names, visualize=True)

    # Load environment pointclouds
    pcd_dir = "/media/run/Extend/Autolife-Planning/assets/envs/rls_env/pcd"

    if os.path.exists(pcd_dir):
        ply_files = glob.glob(os.path.join(pcd_dir, "*.ply"))
        logger.info(f"Found {len(ply_files)} ply files in {pcd_dir}")

        for ply_file in ply_files:
            points = load_ply(ply_file)
            env.add_pointcloud(points)
            logger.info(f"Loaded pointcloud: {ply_file} with {len(points)} points")
    else:
        logger.warning(
            f"Pointcloud directory {pcd_dir} not found. Please run 'bash scripts/download_assets.sh'"
        )

    server = RobomeshServer()
    server.run_threaded()

    # Strict loop, assumes valid observations
    while True:
        env.step()
        obs = env.get_obs()
        cam = obs["camera_chest"]
        server.stream_rgbd(cam["rgb"], cam["depth"])
        time.sleep(1.0 / 30.0)
