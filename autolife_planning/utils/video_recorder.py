"""Offscreen MP4 recorder for :class:`PyBulletEnv`.

Records high-quality MP4 clips from a headless PyBullet session by
rendering frames via ``getCameraImage`` and piping them into ``ffmpeg``.
Works with a ``DIRECT`` connection (no GUI window required), which makes
it safe to use inside build pipelines, CI, or a doc-rendering driver.

Typical usage::

    from autolife_planning.utils.video_recorder import VideoRecorder

    with VideoRecorder(env, "output.mp4", camera=CameraView.front_left()) as rec:
        rec.hold(seconds=0.5)            # pause on the start pose
        rec.play_path(path, duration=6)  # interpolate smoothly through waypoints
        rec.hold(seconds=1.0)            # pause on the final pose

The recorder assumes ``ffmpeg`` is on ``PATH``.  Frames are rendered at
the requested resolution using the CPU ``TinyRenderer``; quality is
solid at 720p and the throughput (~20-60 fps on a modern laptop) is
fast enough to render every example clip in under a minute.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pybullet as pb

if TYPE_CHECKING:
    from autolife_planning.envs.pybullet_env import PyBulletEnv


@dataclass
class CameraView:
    """Static orbit-camera parameters passed to ``computeViewMatrixFromYawPitchRoll``.

    ``target`` points at a world-space location (typically the centre
    of the robot torso), ``distance`` is the orbit radius, and
    ``yaw``/``pitch`` orbit around ``target`` in degrees.  ``fov``
    controls the vertical field of view of the projection matrix.

    The view is intentionally static: the camera does not move during
    a clip — every frame is rendered from exactly the same pose so the
    only motion the viewer sees is the robot itself.  The defaults
    below are tuned as a single "house view" that frames the whole
    robot while still showing enough of the end effector for the
    constraint demos.
    """

    target: tuple[float, float, float] = (0.10, 0.00, 0.85)
    distance: float = 2.00
    yaw: float = 55.0
    pitch: float = -10.0
    roll: float = 0.0
    fov: float = 50.0
    near: float = 0.05
    far: float = 20.0


@dataclass
class VideoRecorder:
    """Render frames from a :class:`PyBulletEnv` into an H.264 MP4.

    The recorder owns a running ``ffmpeg`` subprocess as long as the
    context is open; frames are written directly to its stdin in
    ``rgba`` format, avoiding any intermediate PNG files.

    If ``ground`` is true, a large soft-grey plate is added at ``z = 0``
    when the recording opens (and removed when it closes) so the robot
    has something to stand on — without it the robot renders as if it
    were floating in empty space, which reads badly for motion demos.
    The camera is static for the full duration of the clip: every
    frame is rendered from exactly the same view so the only motion
    the viewer perceives is the robot.
    """

    env: "PyBulletEnv"
    output_path: Path | str
    fps: int = 30
    resolution: tuple[int, int] = (1280, 720)
    camera: CameraView = field(default_factory=CameraView)
    shadow: bool = True
    light_direction: tuple[float, float, float] = (0.55, 0.40, 1.30)
    crf: int = 20
    ground: bool = True

    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _view: list[float] = field(default_factory=list, init=False, repr=False)
    _proj: list[float] = field(default_factory=list, init=False, repr=False)
    _n_written: int = field(default=0, init=False, repr=False)
    _ground_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        w, h = self.resolution
        if w % 2 or h % 2:
            raise ValueError(f"resolution must be even: got {self.resolution}")

        if which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH — install ffmpeg to record videos"
            )

    # ── context management ──────────────────────────────────────────

    def __enter__(self) -> "VideoRecorder":
        self._rebuild_camera_matrices()
        self._configure_background()
        if self.ground:
            self._add_ground_plate()
        self._spawn_ffmpeg()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        self._remove_ground_plate()

    def close(self) -> None:
        """Finalize the MP4 by closing ffmpeg's stdin and waiting."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc = None

    # ── frame sources ───────────────────────────────────────────────

    def capture(self) -> None:
        """Render the current env state as a single frame into the clip."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError(
                "VideoRecorder is not open — use it as a context manager"
            )

        w, h = self.resolution
        client = self.env.sim.client
        img = client.getCameraImage(
            width=w,
            height=h,
            viewMatrix=self._view,
            projectionMatrix=self._proj,
            shadow=1 if self.shadow else 0,
            lightDirection=list(self.light_direction),
            lightColor=[1.0, 0.98, 0.94],
            lightDistance=3.5,
            lightAmbientCoeff=0.42,
            lightDiffuseCoeff=0.68,
            lightSpecularCoeff=0.22,
            renderer=pb.ER_TINY_RENDERER,
        )
        rgba = np.asarray(img[2], dtype=np.uint8).reshape(h, w, 4)
        self._proc.stdin.write(rgba.tobytes())
        self._n_written += 1

    def hold(self, seconds: float = 0.5, frames: int | None = None) -> None:
        """Freeze on the current configuration for *seconds* (or *frames*)."""
        n = frames if frames is not None else max(1, int(round(seconds * self.fps)))
        for _ in range(n):
            self.capture()

    def play_path(
        self,
        path: np.ndarray,
        *,
        duration: float | None = None,
        frames: int | None = None,
        on_frame: "Callable[[np.ndarray], None] | None" = None,
    ) -> None:
        """Smoothly interpolate through *path* and capture each frame.

        The path is re-sampled to a uniform time grid with
        ``frames = int(duration * fps)`` samples (linearly interpolating
        between consecutive waypoints), so the playback duration is the
        same regardless of how many waypoints the planner produced.

        ``on_frame`` is an optional callback invoked with the current
        interpolated configuration right after ``set_configuration``
        and right before ``capture`` — useful for updating visual
        markers (constraint points, EE frames, …) in lockstep with
        the robot.
        """
        path = np.asarray(path)
        if path.ndim != 2 or len(path) == 0:
            return

        if frames is None:
            if duration is None:
                duration = max(2.0, min(10.0, len(path) / 30.0))
            frames = max(2, int(round(duration * self.fps)))

        n_wp = len(path)
        if n_wp == 1:
            for _ in range(frames):
                self.env.set_configuration(path[0])
                if on_frame is not None:
                    on_frame(path[0])
                self.capture()
            return

        # Resample: for each output frame, find the fractional waypoint
        # index and linearly blend between the two nearest waypoints.
        ts = np.linspace(0.0, n_wp - 1, frames)
        for t in ts:
            lo = int(np.floor(t))
            hi = min(lo + 1, n_wp - 1)
            alpha = float(t - lo)
            cfg = (1.0 - alpha) * path[lo] + alpha * path[hi]
            self.env.set_configuration(cfg)
            if on_frame is not None:
                on_frame(cfg)
            self.capture()

    def play_sequence(
        self,
        segments: Sequence[np.ndarray],
        *,
        segment_duration: float = 3.0,
        hold_between: float = 0.3,
    ) -> None:
        """Render several paths back-to-back with a short hold in between.

        Useful for IK-style demos where each "segment" is a one-frame
        pose (home → solved → home → solved …).
        """
        for i, seg in enumerate(segments):
            seg = np.asarray(seg)
            if seg.ndim == 1:
                seg = seg[None, :]
            self.play_path(seg, duration=segment_duration)
            if i != len(segments) - 1:
                self.hold(seconds=hold_between)

    # ── internals ───────────────────────────────────────────────────

    def _rebuild_camera_matrices(self) -> None:
        client = self.env.sim.client
        self._view = client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(self.camera.target),
            distance=self.camera.distance,
            yaw=self.camera.yaw,
            pitch=self.camera.pitch,
            roll=self.camera.roll,
            upAxisIndex=2,
        )
        w, h = self.resolution
        self._proj = client.computeProjectionMatrixFOV(
            fov=self.camera.fov,
            aspect=w / h,
            nearVal=self.camera.near,
            farVal=self.camera.far,
        )

    def _configure_background(self) -> None:
        """Set a clean rgb background via PyBullet's debug visualizer."""
        client = self.env.sim.client
        try:
            client.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            client.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        except pb.error:
            pass  # DIRECT mode: debug visualizer calls are no-ops

    def _add_ground_plate(self) -> None:
        """Visual-only flat plate at ``z = 0`` — gives the robot something
        to stand on so renders don't look like the robot is floating.
        """
        client = self.env.sim.client
        vid = client.createVisualShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=[4.0, 4.0, 0.005],
            rgbaColor=[0.88, 0.90, 0.94, 1.0],
            specularColor=[0.10, 0.10, 0.12],
        )
        self._ground_id = client.createMultiBody(
            baseVisualShapeIndex=vid,
            basePosition=[0.0, 0.0, -0.005],
        )

    def _remove_ground_plate(self) -> None:
        if self._ground_id is None:
            return
        try:
            self.env.sim.client.removeBody(self._ground_id)
        except pb.error:
            pass
        self._ground_id = None

    def _spawn_ffmpeg(self) -> None:
        w, h = self.resolution
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{w}x{h}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
