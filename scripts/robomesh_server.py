#!/usr/bin/env python
import logging
import os
import threading

import numpy as np

# Import project interfaces
from autolife_planning.interfaces.robomesh import app
from autolife_planning.utils.stream_utils import VideoStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobomeshServer")


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
    server = RobomeshServer()
    server.run()
