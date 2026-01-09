#!/usr/bin/env python
import glob
import logging
import os
import threading
import time

import numpy as np
from flask import Flask, jsonify, request

# Import project utilities
from autolife_planning.utils.stream_utils import VideoStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobomeshServer")


class RobomeshServer:
    def __init__(self, orchestrator=None, host="0.0.0.0", port=11111):
        self.host = host
        self.port = port
        self.orchestrator = orchestrator

        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()

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

    def setup_routes(self):
        """Setup Flask routes for HTTP API"""

        @self.app.route("/chat", methods=["POST"])
        def chat():
            try:
                # Validate and parse input using dataclass
                from autolife_planning.dataclass.commands import TextCommand

                message = TextCommand.from_dict(request.json)

                logger.info(f"Received user input: {message.text}")

                # Hand over instruction to orchestrator
                if self.orchestrator:
                    self.orchestrator.submit_task(context=message, point=None)

                # Return acknowledgment
                return jsonify(
                    {"status": "received", "message": "Processing request..."}
                )

            except Exception as e:
                logger.error(f"Error in chat endpoint: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/point", methods=["POST"])
        def point():
            try:
                # Validate and parse input using dataclass
                from autolife_planning.dataclass.commands import PointCommand

                message = PointCommand.from_dict(request.json)

                logger.info(f"Received point coordinates: ({message.x}, {message.y})")

                # Hand over point to orchestrator
                if self.orchestrator:
                    self.orchestrator.submit_task(context=None, point=message)

                return jsonify(
                    {
                        "status": "received",
                        "point": [message.x, message.y],
                        "message": "Point received successfully",
                    }
                )

            except Exception as e:
                logger.error(f"Error in point endpoint: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify("OK"), 200

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
    from autolife_planning.config.robot_config import autolife_robot_config
    from autolife_planning.core.orchestrator import Orchestrator
    from autolife_planning.envs.pybullet_env import PyBulletEnv

    env = PyBulletEnv(autolife_robot_config, visualize=True)
    orchestrator = Orchestrator(env)

    # Load environment meshes
    mesh_dir = (
        "/media/run/Extend/Autolife-Planning/assets/envs/rls_env/simplified_meshes"
    )

    if os.path.exists(mesh_dir):
        # Recursively find all .obj files
        obj_files = glob.glob(os.path.join(mesh_dir, "**/*.obj"), recursive=True)
        logger.info(f"Found {len(obj_files)} obj files in {mesh_dir}")

        allowed_meshes = {
            "open_kitchen",
            "rls_2",
            "table",
            "wall",
            "workstation",
            "sofa",
            "tea_table",
        }

        for obj_file in obj_files:
            # We can use the filename stem as the object name
            obj_name = os.path.splitext(os.path.basename(obj_file))[0]

            if obj_name in allowed_meshes:
                body_id = env.add_mesh(obj_file, name=obj_name)
                aabb_min, aabb_max = env.sim.client.getAABB(body_id)
                logger.info(
                    f"Loaded mesh: {obj_name} | ID: {body_id} | Bounds: {aabb_min} to {aabb_max}"
                )
            else:
                logger.debug(f"Skipping mesh: {obj_name}")
    else:
        logger.warning(
            f"Mesh directory {mesh_dir} not found. Please run 'bash scripts/download_assets.sh'"
        )

    server = RobomeshServer(orchestrator=orchestrator)
    server.run_threaded()

    # Strict loop, assumes valid observations
    while True:
        orchestrator.update()  # Orchestrator wraps step and behavior execution
        obs = orchestrator.env.get_obs()
        cam = obs["camera_chest"]
        server.stream_rgbd(cam["rgb"], cam["depth"])
        time.sleep(1.0 / 60.0)
