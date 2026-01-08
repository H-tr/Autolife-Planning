import subprocess
import numpy as np
from typing import Callable, Optional

class VideoStreamer:
    def __init__(self, destination: str, width: int, height: int, fps: int = 30, codec: str = 'libx264'):
        self.destination = destination
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.process = None

    def start(self):
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', self.codec,
        ]

        if self.codec == 'libx264':
            command.extend(['-preset', 'ultrafast', '-tune', 'zerolatency'])
        elif self.codec == 'libvpx':
            command.extend(['-deadline', 'realtime', '-cpu-used', '5'])

        command.extend(['-f', 'rtp', self.destination])
        
        # Determine pkt_size if needed, usually appended to destination or added as arg
        # The original stream_utils didn't have pkt_size, but the reference did. Keeping original behavior unless generic.
        
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def push_frame(self, image: np.ndarray):
        if self.process is None:
            self.start()
        
        if image is not None:
             self.process.stdin.write(image.tobytes())

    def close(self):
        if self.process and self.process.stdin:
            self.process.stdin.close()
        if self.process:
            self.process.wait()
            self.process = None
