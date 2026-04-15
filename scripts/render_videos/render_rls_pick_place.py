"""Record the ``rls_pick_place`` demo as a narrated multi-pane MP4.

The script drives the demo's planning pipeline in headless mode
via monkey-patches (no changes to ``examples/demos/rls_pick_place.py``),
captures the produced segments + per-stage planning timings, then
replays them through three simultaneous camera views composited
side-by-side into one MP4:

  * left pane   — follow-cam that keeps the robot large and centred
                  while it drives through the room
  * middle pane — top-down nav view that shows the whole apartment
                  (table / sofa / kitchen / coffee / beam) at a glance
  * right pane  — close-up on whichever hand currently carries the
                  active object (apple/bowl/bottle) so the constraint
                  satisfaction — horizontal bowl, squat-under-beam —
                  is unambiguous

The scene meshes (table / kitchen / wall / workstation / sofa /
coffee table) are loaded explicitly for the renderer — the demo
itself only adds them to the planner as a pointcloud, which
TinyRenderer cannot display.  A ground plate, softer lighting, and
a highlight halo under the currently-held object round out the scene.

Per-segment captions, a progress bar, and a running "stage X of N"
chip are baked in by passing the captured segment schedule to a
single ffmpeg ``filter_complex`` — no per-frame Python overlay in
the render hot path.

Usage::

    pixi run python scripts/render_videos/render_rls_pick_place.py
    pixi run python scripts/render_videos/render_rls_pick_place.py \\
        --out docs/assets/rls_pick_place.mp4 --pane_width 720 \\
        --pane_height 720 --fps 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which

import numpy as np
import pybullet as pb

# ── Make the demo importable without touching sys.path inside it ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples" / "demos"))

from autolife_planning.envs.pybullet_env import PyBulletEnv  # noqa: E402

# ── Force headless planning ───────────────────────────────────────
_orig_env_init = PyBulletEnv.__init__


def _headless_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs["visualize"] = False
    _orig_env_init(self, *args, **kwargs)


PyBulletEnv.__init__ = _headless_init  # type: ignore[assignment]

import rls_pick_place as demo  # noqa: E402

# ── Capture hooks on the demo ─────────────────────────────────────

_captured: dict = {}
_timings: list[tuple[str, float]] = []


def _capture_segs(env, segs, fps: float = 60.0) -> None:
    _captured["env"] = env
    _captured["segs"] = segs


demo.play_segments = _capture_segs

_orig_plan = demo._plan


def _plan_timed(planner, start, goal, label, *args, **kwargs):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    path = _orig_plan(planner, start, goal, label, *args, **kwargs)
    _timings.append((label, (time.perf_counter() - t0) * 1000.0))
    return path


demo._plan = _plan_timed

_orig_direct = demo.plan_body_locked_line_direct


def _direct_timed(*args, **kwargs):  # type: ignore[no-untyped-def]
    label = args[4] if len(args) > 4 else kwargs.get("label", "direct")
    t0 = time.perf_counter()
    path = _orig_direct(*args, **kwargs)
    _timings.append((label, (time.perf_counter() - t0) * 1000.0))
    return path


demo.plan_body_locked_line_direct = _direct_timed


# ── Narration / stage metadata ────────────────────────────────────

# Human-readable story beat for each raw banner coming out of the demo.
# Keyed by the exact banner string the demo sets on Segment.banner.
_STORY_BEATS: dict[str, tuple[str, str, str]] = {
    # banner : (chapter, title, subtitle)
    "home pose": (
        "intro",
        "Autolife pick-and-place demo",
        "10 stages · 4 planners · 1 room",
    ),
    "s1a squat": (
        "Stage 1a",
        "Squat under the table",
        "subgroup · autolife_body · knee=2·ankle",
    ),
    "s1b approach": ("Stage 1b", "Reach for the apple", "free motion · 21 DOF"),
    "s1b approach line": (
        "Stage 1b",
        "Straight-line pregrasp",
        "line constraint · 2 eq",
    ),
    "s1b lift": ("Stage 1b", "Lift the apple", "line constraint · leg pin"),
    "s1c stand": ("Stage 1c", "Stand back up", "apple attached · body subgroup"),
    "nav→place pos": ("Stage 2", "Shuffle to table edge", "base subgroup · 3 DOF"),
    "s2 carry": ("Stage 2", "Carry apple to table", "torso + left arm · 9 DOF"),
    "s2 lower": ("Stage 2", "Lower onto table", "straight-line place"),
    "s2 retreat": ("Stage 2", "Retreat from table", "arm free · apple placed"),
    "retract arm": ("transit", "Retract arm to home", "getting ready to drive"),
    "s3 nav→kitchen": ("Stage 3", "Drive to kitchen counter", "base subgroup · 3 DOF"),
    "s4 approach": ("Stage 4", "Reach for the bowl", "right arm · free motion"),
    "s4 approach line": ("Stage 4", "Straight-line pregrasp", "line constraint · 2 eq"),
    "s4 lift": ("Stage 4", "Lift bowl · lock rotation", "3-eq SO(3) lock + leg pin"),
    "s5 drive→coffee (auto-squat)": (
        "Stage 5",
        "Drive + auto-squat under beam",
        "6-DOF · horizontal bowl + knee=2·ankle · RRT-Connect",
    ),
    "s6 carry": (
        "Stage 6",
        "Carry bowl to coffee table",
        "body subgroup · rotation-locked",
    ),
    "s6 lower": (
        "Stage 6",
        "Lower bowl onto table",
        "line + SO(3) lock · projected IK",
    ),
    "s6 retreat": (
        "Stage 6",
        "Retreat from coffee table",
        "bowl released · line constraint",
    ),
    "s7 nav→sofa": ("Stage 7", "Drive to sofa", "base subgroup · 3 DOF"),
    "s8 approach": ("Stage 8", "Reach for the bottle", "left arm · free motion"),
    "s8 approach line": ("Stage 8", "Straight-line pregrasp", "line constraint · 2 eq"),
    "s8 lift": ("Stage 8", "Lift the bottle", "line constraint · arm subgroup"),
    "s9 nav→coffee": ("Stage 9", "Drive to coffee table", "base subgroup · 3 DOF"),
    "s10 carry": (
        "Stage 10",
        "Carry bottle to coffee table",
        "torso + left arm · 9 DOF",
    ),
    "s10 lower": ("Stage 10", "Lower bottle onto table", "straight-line place"),
    "s10 retreat": (
        "Stage 10",
        "Retreat · task complete",
        "bottle placed · home stretch",
    ),
}


def _beat_for(banner: str) -> tuple[str, str, str]:
    if banner in _STORY_BEATS:
        return _STORY_BEATS[banner]
    return ("stage", banner, "")


# "What to watch for" — surfaced in the right edge of the caption bar.
_LEARN_GOAL: dict[str, str] = {
    "s1a squat": "Watch the legs fold and the torso stay over the foot",
    "s1b lift": "Watch the gripper rise in a straight line",
    "s4 lift": "Watch the bowl stay perfectly level (rotation locked)",
    "s5 drive→coffee (auto-squat)": "Watch the body dip only as much as the beam demands",
    "s6 lower": "Watch a straight-line place with rotation still locked",
    "s2 lower": "Watch the apple settle on the table along a line",
    "s10 lower": "Watch the bottle come down on the same line",
}


# Which "focus object" (mesh id) is in play during each segment, so the
# close-up pane can follow the action even when nothing is attached yet
# (e.g. during the pregrasp approaches).  Resolved to body ids in
# ``_render``.
_FOCUS_OBJECT: dict[str, str] = {
    "home pose": "apple",
    "s1a squat": "apple",
    "s1b approach": "apple",
    "s1b approach line": "apple",
    "s1b lift": "apple",
    "s1c stand": "apple",
    "nav→place pos": "apple",
    "s2 carry": "apple",
    "s2 lower": "apple",
    "s2 retreat": "apple",
    "retract arm": "robot",
    "s3 nav→kitchen": "bowl",
    "s4 approach": "bowl",
    "s4 approach line": "bowl",
    "s4 lift": "bowl",
    "s5 drive→coffee (auto-squat)": "bowl",
    "s6 carry": "bowl",
    "s6 lower": "bowl",
    "s6 retreat": "bowl",
    "s7 nav→sofa": "bottle",
    "s8 approach": "bottle",
    "s8 approach line": "bottle",
    "s8 lift": "bottle",
    "s9 nav→coffee": "bottle",
    "s10 carry": "bottle",
    "s10 lower": "bottle",
    "s10 retreat": "bottle",
}


# Which gripper link holds the currently-active object (for the follow-cam).
# During pure nav/transit segments the close-up pane is better off
# showing a waist-up tracking shot of the robot rather than an empty
# piece of furniture.  Key off the banner.
_NAV_WAIST_UP_SEGMENTS = {
    "s3 nav→kitchen",
    "s7 nav→sofa",
    "s9 nav→coffee",
    "nav→place pos",
    "retract arm",
    "retract arm→home",
}

# Destination pin for each nav segment — drawn on the iso map so the
# viewer sees WHERE the robot is going, not just that it's moving.
_NAV_DESTINATION: dict[str, np.ndarray] = {
    "nav→place pos": np.array([-1.30, 1.67, 0.02]),
    "s3 nav→kitchen": np.array([-3.80, -2.40, 0.02]),
    "s7 nav→sofa": np.array([2.00, 1.30, 0.02]),
    "s9 nav→coffee": np.array([3.60, 1.30, 0.02]),
}

# Local-frame target for the waist-up shot (robot forward x, lateral y, up z).
_NAV_TARGET_LOCAL = np.array([0.18, 0.00, 1.05])


def _robot_local_target(
    base: np.ndarray, theta: float, local: np.ndarray
) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    dx, dy, dz = local
    return np.array([base[0] + c * dx - s * dy, base[1] + s * dx + c * dy, dz])


_FOCUS_GRIPPER: dict[str, str] = {
    # Left-gripper chapters: apple + bottle
    **{
        k: demo.GRIPPER_LINK
        for k in [
            "home pose",
            "s1a squat",
            "s1b approach",
            "s1b approach line",
            "s1b lift",
            "s1c stand",
            "nav→place pos",
            "s2 carry",
            "s2 lower",
            "s2 retreat",
            "s8 approach",
            "s8 approach line",
            "s8 lift",
            "s10 carry",
            "s10 lower",
            "s10 retreat",
        ]
    },
    # Right-gripper chapters: bowl
    **{
        k: demo.RIGHT_GRIPPER_LINK
        for k in [
            "s4 approach",
            "s4 approach line",
            "s4 lift",
            "s5 drive→coffee (auto-squat)",
            "s6 carry",
            "s6 lower",
            "s6 retreat",
        ]
    },
}


# ── Scene dressing ─────────────────────────────────────────────────


_PC_SCENE_OBJ = _REPO_ROOT / "tmp/rls_render/pc_scene/scene.obj"


# Walls are now capped at chest height by prepare_pointcloud_scene.py,
# so we can let them be reasonably opaque (lower walls don't occlude
# the robot from above).  Furniture stays solid; the floor/outer shell
# is just translucent enough to read as a backdrop.
_PC_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "rls_2": (0.78, 0.80, 0.84, 0.45),
    "wall": (0.78, 0.80, 0.84, 0.45),
    "open_kitchen": (0.88, 0.78, 0.67, 0.85),
    "workstation": (0.47, 0.55, 0.67, 0.95),
    "table": (0.67, 0.51, 0.35, 0.95),
    "sofa": (0.84, 0.76, 0.63, 0.95),
    "tea_table": (0.37, 0.28, 0.22, 0.95),
}


def _spawn_scene_meshes(env) -> list[int]:
    """Load the voxelised pointcloud scene as one body per source tag
    (walls, kitchen, workstation, tables, sofa) and colour each via
    ``changeVisualShape``.  The voxel diorama is generated by
    ``prepare_pointcloud_scene.py`` — run that once before rendering.
    """
    pc_dir = _PC_SCENE_OBJ.parent
    split_objs = sorted(pc_dir.glob("scene_*.obj"))
    if not split_objs:
        raise RuntimeError(
            f"no voxel scene under {pc_dir}; run "
            "`pixi run python scripts/render_videos/prepare_pointcloud_scene.py`"
        )
    ids: list[int] = []
    for obj in split_objs:
        name = obj.stem.removeprefix("scene_")
        bid = env.add_mesh(str(obj), position=np.zeros(3))
        rgba = _PC_COLOURS.get(name, (0.80, 0.80, 0.82, 1.0))
        env.sim.client.changeVisualShape(bid, -1, rgbaColor=list(rgba))
        ids.append(bid)
    return ids


def _spawn_ground_plate(client) -> int:
    """Light-grey plate at z=0 so the room doesn't float in a void."""
    vid = client.createVisualShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=[8.0, 8.0, 0.005],
        rgbaColor=[0.90, 0.91, 0.93, 1.0],
        specularColor=[0.05, 0.05, 0.06],
    )
    return client.createMultiBody(baseVisualShapeIndex=vid, basePosition=[0, 0, -0.01])


def _spawn_beam(client, hidden: bool = True) -> int:
    """Low-hanging red beam, hidden underground until its segment fires."""
    vid = client.createVisualShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=demo.BEAM_HALF_EXTENTS.tolist(),
        rgbaColor=[0.78, 0.30, 0.28, 0.55],
        specularColor=[0.04, 0.02, 0.02],
    )
    return client.createMultiBody(
        baseVisualShapeIndex=vid,
        basePosition=[0, 0, -50.0] if hidden else demo.BEAM_CENTER.tolist(),
    )


def _show_beam(client, beam_id: int) -> None:
    client.resetBasePositionAndOrientation(
        beam_id, demo.BEAM_CENTER.tolist(), [0, 0, 0, 1]
    )


def _paint_graspables(client, apple_id: int, bowl_id: int, bottle_id: int) -> None:
    """Override graspable colours so they read against the neutral scene."""
    # (id, rgba) — hi-saturation primaries that pop in thumbnails
    for body_id, rgba in (
        (apple_id, [0.90, 0.12, 0.14, 1.0]),  # crimson apple
        (bowl_id, [0.20, 0.35, 0.95, 1.0]),  # cobalt blue bowl
        (bottle_id, [0.15, 0.70, 0.30, 1.0]),  # emerald green bottle
    ):
        # The mesh may have several sub-shapes; colour every visual slot.
        for info in client.getVisualShapeData(body_id):
            link_idx = info[1]
            client.changeVisualShape(body_id, link_idx, rgbaColor=rgba)


def _spawn_halo(client, radius: float = 0.16) -> int:
    """Flat glowing disc (ring marker) we slide under the active object."""
    vid = client.createVisualShape(
        shapeType=pb.GEOM_CYLINDER,
        radius=radius,
        length=0.003,
        rgbaColor=[1.0, 0.82, 0.12, 0.48],
        specularColor=[0.12, 0.10, 0.02],
    )
    body = client.createMultiBody(
        baseVisualShapeIndex=vid,
        basePosition=[0, 0, -100.0],  # parked offscreen until used
    )
    return body


def _spawn_destination_pin(client) -> int:
    """Cyan floor marker dropped at each nav segment's goal."""
    vid = client.createVisualShape(
        shapeType=pb.GEOM_CYLINDER,
        radius=0.20,
        length=0.006,
        rgbaColor=[0.16, 0.77, 0.96, 0.62],
        specularColor=[0.05, 0.10, 0.12],
    )
    return client.createMultiBody(
        baseVisualShapeIndex=vid,
        basePosition=[0, 0, -100.0],
    )


def _hide(client, body_id: int) -> None:
    client.resetBasePositionAndOrientation(body_id, [0, 0, -100.0], [0, 0, 0, 1])


def _move_to(client, body_id: int, xyz: np.ndarray) -> None:
    client.resetBasePositionAndOrientation(body_id, xyz.tolist(), [0, 0, 0, 1])


# ── Multi-pane camera config ──────────────────────────────────────


@dataclass
class Pane:
    name: str
    kind: str  # "follow_robot" | "topdown" | "follow_action"
    distance: float
    yaw: float
    pitch: float
    fov: float = 50.0
    target: tuple[float, float, float] = (0.0, 0.0, 0.7)
    smoothing: float = 0.18  # low-pass alpha (0 = stick, 1 = teleport)


# The three panes are tuned for a 720×720 cell.  The outer two track
# motion; the middle pane is configurable via ``_build_panes``:
#   * "chase"    — over-the-shoulder chase cam behind the robot
#   * "overhead" — top-down follow cam riding above the robot
#   * "map"      — fixed high-angle iso covering the whole apartment
def _build_panes(middle_view: str = "chase") -> list[Pane]:
    hero = Pane(
        "hero",
        kind="follow_robot",
        distance=4.4,
        yaw=42.0,
        pitch=-24.0,
        fov=60.0,
        smoothing=0.12,
    )
    closeup = Pane(
        "closeup",
        kind="follow_action",
        distance=1.20,
        yaw=55.0,
        pitch=-18.0,
        fov=55.0,
        smoothing=0.22,
    )
    if middle_view == "chase":
        # pane.yaw=270 places the camera behind a robot facing +X; the
        # ``yaw_override = theta + pane.yaw`` formula then keeps it locked
        # behind the robot as it turns.  Pitch tilts the shot slightly
        # downward so the floor ahead is visible.
        middle = Pane(
            "chase",
            kind="chase_robot",
            distance=3.0,
            yaw=270.0,
            pitch=-18.0,
            fov=58.0,
            smoothing=0.18,
        )
    elif middle_view == "overhead":
        # World-fixed yaw keeps "up" in the image stable as the robot
        # drives around — easier to read than a heading-locked overhead
        # which would spin the whole world every time the robot turns.
        middle = Pane(
            "overhead",
            kind="overhead_robot",
            distance=3.8,
            yaw=30.0,
            pitch=-72.0,
            fov=62.0,
            smoothing=0.18,
        )
    elif middle_view == "map":
        # High-angle iso (not straight top-down) so walls and furniture read
        # in context — a true pitch=-89 sees only the roofs.  Target + distance
        # picked to fit the whole apartment (x∈[-5.5, 4.5], y∈[-3.5, 2.5]).
        middle = Pane(
            "map",
            kind="iso",
            distance=14.2,
            yaw=30.0,
            pitch=-55.0,
            fov=52.0,
            target=(-0.5, -0.5, 0.8),
        )
    else:
        raise ValueError(
            f"unknown middle_view {middle_view!r} "
            "(expected 'chase', 'overhead', or 'map')"
        )
    return [hero, middle, closeup]


PANES: list[Pane] = _build_panes()


def _compute_view(
    client,
    pane: Pane,
    target: np.ndarray,
    yaw_override: float | None = None,
    dist_override: float | None = None,
    pitch_override: float | None = None,
):
    return client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(),
        distance=pane.distance if dist_override is None else dist_override,
        yaw=(pane.yaw if yaw_override is None else yaw_override),
        pitch=pane.pitch if pitch_override is None else pitch_override,
        roll=0.0,
        upAxisIndex=2,
    )


def _robot_root_xy(client, env) -> np.ndarray:
    """World position of the robot's floating base."""
    pos, _ = client.getBasePositionAndOrientation(env.sim.skel_id)
    return np.asarray(pos, dtype=np.float64)


def _robot_yaw(client, env) -> float:
    """World-frame yaw of the robot (radians)."""
    _, orn = client.getBasePositionAndOrientation(env.sim.skel_id)
    eul = client.getEulerFromQuaternion(orn)
    return float(eul[2])


# ── Caption timeline → ffmpeg filter_complex ──────────────────────


def _escape(s: str) -> str:
    # Only the subset of chars ffmpeg's drawtext actually misbehaves on.
    return (
        s.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def _build_drawtext(segs_timing: list[dict], frame_w: int, frame_h: int) -> str:
    """Build a filter_complex string that paints a caption for every segment.

    ``segs_timing`` is a list of {start_s, end_s, chapter, title, subtitle,
    idx, total}; each entry becomes a drawtext block enabled only during
    its time window.  A persistent top-left chip shows the demo title,
    a persistent bottom-left shows "stage k / N", and a full-width bottom
    bar draws the per-segment title/subtitle.
    """
    chain: list[str] = []

    # Persistent top bar — static title.
    chain.append("drawbox=x=0:y=0:w=iw:h=60:color=black@0.55:t=fill")
    chain.append(
        "drawtext=text='rls_pick_place · 10-stage planner showcase':"
        "fontcolor=white:fontsize=24:"
        "x=24:y=18:"
        "box=0"
    )

    # Per-segment bottom overlay.
    bar_h = 110
    bar_y = frame_h - bar_h
    chain.append(f"drawbox=x=0:y={bar_y}:w=iw:h={bar_h}:color=black@0.62:t=fill")

    for s in segs_timing:
        start = f"{s['start_s']:.3f}"
        end = f"{s['end_s']:.3f}"
        # Exclusive upper bound prevents two adjacent captions from
        # double-drawing for one frame at the boundary.
        enable = f"gte(t\\,{start})*lt(t\\,{end})"
        chapter = _escape(s["chapter"])
        title = _escape(s["title"])
        subtitle = _escape(s["subtitle"])
        stage_chip = _escape(f"{s['idx'] + 1} / {s['total']}")
        goal = _escape(_LEARN_GOAL.get(s["banner"], ""))

        # Chapter badge (left edge, yellow)
        chain.append(
            f"drawtext=text='{chapter}':enable='{enable}':"
            f"fontcolor=0xFFD166:fontsize=22:"
            f"x=28:y={bar_y + 14}"
        )
        # Title (big, white)
        chain.append(
            f"drawtext=text='{title}':enable='{enable}':"
            f"fontcolor=white:fontsize=30:"
            f"x=28:y={bar_y + 40}"
        )
        # Subtitle (slate, small)
        chain.append(
            f"drawtext=text='{subtitle}':enable='{enable}':"
            f"fontcolor=0xB8C5D6:fontsize=18:"
            f"x=28:y={bar_y + 78}"
        )
        # Stage counter chip (bottom-right corner, closer to the edge).
        chain.append(
            f"drawtext=text='{stage_chip}':enable='{enable}':"
            f"fontcolor=white:fontsize=22:"
            f"box=1:boxcolor=0x2D5DA1@0.85:boxborderw=10:"
            f"x=W-text_w-28:y={bar_y + 78}"
        )
        # Mint-coloured "what to watch for" hint — right-aligned on the
        # middle row so it doesn't crowd the title.
        if goal:
            chain.append(
                f"drawtext=text='{goal}':enable='{enable}':"
                f"fontcolor=0x9FE3A8:fontsize=19:"
                f"x=W-text_w-28:y={bar_y + 46}"
            )

    # One-shot flash when the beam is revealed (start of stage 5).
    for s in segs_timing:
        rs = s.get("reveal_s")
        if rs is None:
            continue
        rs_s, rs_e = f"{rs:.3f}", f"{rs + 1.3:.3f}"
        chain.append(
            f"drawtext=text='Beam obstacle installed':"
            f"enable='gte(t\\,{rs_s})*lt(t\\,{rs_e})':"
            f"fontcolor=0xFF6B6B:fontsize=24:"
            f"x=(W-text_w)/2:y=72:"
            f"box=1:boxcolor=black@0.62:boxborderw=10"
        )

    # Chapter-title fade-in at the start of every new "Stage N" — gives
    # the viewer a clean demarcation between the 10 planning stages.
    last_chapter: str | None = None
    for s in segs_timing:
        chapter = s["chapter"]
        if chapter == last_chapter or not chapter.startswith("Stage"):
            last_chapter = chapter
            continue
        last_chapter = chapter
        t0 = s["start_s"]
        t1 = t0 + 1.6  # visible window
        alpha_expr = (
            f"if(lt(t\\,{t0 + 0.35:.3f})\\,(t-{t0:.3f})/0.35\\,"
            f"if(gt(t\\,{t1 - 0.35:.3f})\\,({t1:.3f}-t)/0.35\\,1))"
        )
        title = _escape(f"{chapter} · {s['title']}")
        chain.append(
            f"drawtext=text='{title}':"
            f"enable='gte(t\\,{t0:.3f})*lt(t\\,{t1:.3f})':"
            f"fontcolor=white:fontsize=46:"
            f"alpha='{alpha_expr}':"
            f"x=(W-text_w)/2:y=(H-180)/2:"
            f"box=1:boxcolor=0x10141A@0.72:boxborderw=18"
        )

    # Opening intro card: show briefly over the "home pose" segment.
    if segs_timing and segs_timing[0]["banner"] == "home pose":
        t_end = min(segs_timing[0]["end_s"], 2.0)
        alpha_expr = (
            f"if(lt(t\\,0.35)\\,t/0.35\\,"
            f"if(gt(t\\,{t_end - 0.35:.3f})\\,({t_end:.3f}-t)/0.35\\,1))"
        )
        chain.append(
            f"drawtext=text='Autolife · rls_pick_place':"
            f"enable='lt(t\\,{t_end:.3f})':"
            f"fontcolor=white:fontsize=56:alpha='{alpha_expr}':"
            f"x=(W-text_w)/2:y=(H-180)/2:"
            f"box=1:boxcolor=0x10141A@0.80:boxborderw=22"
        )
        chain.append(
            f"drawtext=text='10-stage long-horizon planning showcase':"
            f"enable='lt(t\\,{t_end:.3f})':"
            f"fontcolor=0xB8C5D6:fontsize=26:alpha='{alpha_expr}':"
            f"x=(W-text_w)/2:y=(H-180)/2+80"
        )

    # Closing bow: after the last segment, flash a "complete" card.
    if segs_timing:
        last = segs_timing[-1]
        t0 = last["end_s"] - 1.8
        t1 = last["end_s"]
        if t0 < t1 - 0.4:
            alpha_expr = (
                f"if(lt(t\\,{t0 + 0.35:.3f})\\,(t-{t0:.3f})/0.35\\,"
                f"if(gt(t\\,{t1 - 0.35:.3f})\\,({t1:.3f}-t)/0.35\\,1))"
            )
            chain.append(
                f"drawtext=text='Task complete · 10 stages · 4 planners':"
                f"enable='gte(t\\,{t0:.3f})*lt(t\\,{t1:.3f})':"
                f"fontcolor=white:fontsize=40:alpha='{alpha_expr}':"
                f"x=(W-text_w)/2:y=(H-180)/2:"
                f"box=1:boxcolor=0x10141A@0.78:boxborderw=18"
            )

    # Progress bar — a thin rule across the very bottom.
    total = segs_timing[-1]["end_s"] if segs_timing else 1.0
    chain.append(
        f"drawbox=x=0:y=ih-4:w=iw*(t/{total:.3f}):h=4:" f"color=0x29C5F6@0.95:t=fill"
    )

    return ",".join(chain)


# ── Main render loop ──────────────────────────────────────────────


def _spawn_ffmpeg(out: Path, frame_w: int, frame_h: int, fps: int, filter_chain: str):
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg to record videos")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Persist the filter graph for postmortems — the full string can be
    # several kilobytes and is the #1 cause of a "broken pipe" on startup.
    (out.parent / (out.stem + ".ffmpeg-filter.txt")).write_text(filter_chain)
    stderr_log = out.parent / (out.stem + ".ffmpeg-stderr.log")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{frame_w}x{frame_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vf",
        filter_chain,
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    return (
        subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=open(stderr_log, "wb")),
        stderr_log,
    )


def _stride_for(seg, base: int) -> int:
    """Stride = how many waypoints we skip per rendered frame.  Larger = faster.
    Climax moments (drive-squat + line constraints) use ~half the base stride
    so the viewer can see the dip + locked rotation happen.
    """
    b = seg.banner
    if "drive" in b and "squat" in b:
        return max(1, base // 2)
    if b.endswith("lift") or b.endswith("lower"):
        return max(1, base // 2)
    return base


def _render(
    out: Path,
    pane_w: int,
    pane_h: int,
    fps: int,
    playback_speed: float,
    hold_frames: int,
    shadows: int = 1,
) -> None:
    env = _captured["env"]
    segs = _captured["segs"]
    client = env.sim.client

    # Scene dressing.
    _spawn_ground_plate(client)
    _spawn_scene_meshes(env)
    beam_id = _spawn_beam(client, hidden=True)
    halo_id = _spawn_halo(client, radius=0.16)
    dest_pin_id = _spawn_destination_pin(client)

    # Colour the graspables.  place_graspable in the demo uses the
    # mesh's default material; recolour for legibility.  The body ids
    # were deterministic in demo.main — apple/bowl/bottle are the 3
    # last meshes added in that exact order.
    n_bodies = client.getNumBodies()
    # Anchor: the 3 most-recently-added bodies before we spawned scene/beam
    # are the graspables; but we spawned new bodies since, so fish by name.
    # Easiest: iterate bodies, match visual shape filename.
    apple_id = bowl_id = bottle_id = None
    for i in range(n_bodies):
        uid = client.getBodyUniqueId(i)
        for info in client.getVisualShapeData(uid):
            mesh_asset = info[4]
            if isinstance(mesh_asset, bytes):
                mesh_asset = mesh_asset.decode("utf-8", errors="ignore")
            name = os.path.basename(mesh_asset) if mesh_asset else ""
            if name == "apple.obj":
                apple_id = uid
            elif name == "bowl.obj":
                bowl_id = uid
            elif name == "bottle.obj":
                bottle_id = uid
    assert apple_id and bowl_id and bottle_id, "graspable mesh ids not found"
    _paint_graspables(client, apple_id, bowl_id, bottle_id)

    obj_name_to_id = {"apple": apple_id, "bowl": bowl_id, "bottle": bottle_id}

    # Link indices for the follow-cam.
    link_ix = {
        demo.GRIPPER_LINK: demo.find_link_index(env, demo.GRIPPER_LINK),
        demo.RIGHT_GRIPPER_LINK: demo.find_link_index(env, demo.RIGHT_GRIPPER_LINK),
    }

    # Compute per-segment timing (seconds in the final video).
    # playback_speed is the stride (frames to skip per rendered frame); larger
    # means a shorter clip.  Old formula was inverted and collapsed to stride=1
    # for all speeds > 1.5 — hence iter02 rendered every waypoint.
    strides = [_stride_for(s, max(1, int(round(playback_speed)))) for s in segs]
    frame_counts = [
        max(1, (s.path.shape[0] + strides[i] - 1) // strides[i])
        for i, s in enumerate(segs)
    ]

    # Optional per-segment tail hold (freeze on last frame) for viewers
    # to absorb each beat.  Skip for "home pose" since it's already a hold.
    # The auto-squat climax gets a longer beat so the viewer can see the
    # robot fully standing back up.
    def _hold_for(seg):
        if seg.banner == "home pose":
            return 0
        if "drive" in seg.banner and "squat" in seg.banner:
            return max(hold_frames, 18)
        return hold_frames

    tail_holds = [_hold_for(s) for s in segs]
    cumulative = 0
    timings_table = []
    for i, s in enumerate(segs):
        fcount = frame_counts[i] + tail_holds[i]
        start_s = cumulative / fps
        end_s = (cumulative + fcount) / fps
        chapter, title, subtitle = _beat_for(s.banner)
        # Disambiguate the two transit "retract arm" moments.
        if s.banner == "retract arm":
            subtitle = (
                f"getting ready to drive · {'left' if i < 15 else 'right'} arm → home"
            )
        reveal_s = start_s if s.reveal_points is not None else None
        timings_table.append(
            dict(
                idx=i,
                total=len(segs),
                start_s=start_s,
                end_s=end_s,
                chapter=chapter,
                title=title,
                subtitle=subtitle,
                banner=s.banner,
                reveal_s=reveal_s,
            )
        )
        cumulative += fcount

    frame_w = pane_w * len(PANES)
    frame_h = pane_h + 0  # captions sit inside the pane height
    filter_chain = _build_drawtext(timings_table, frame_w, frame_h)

    proc, stderr_log = _spawn_ffmpeg(out, frame_w, frame_h, fps, filter_chain)
    assert proc.stdin is not None

    # Render every pane at full resolution; the iso pane is noisy with
    # nearest-neighbour upsampling (visible block structure on walls).
    pane_render_sizes = [(pane_w, pane_h) for _ in PANES]

    # Projection matrices — static per pane.
    projs = [
        client.computeProjectionMatrixFOV(
            fov=p.fov, aspect=pane_w / pane_h, nearVal=0.05, farVal=40.0
        )
        for p in PANES
    ]

    # Track per-pane smoothed target.
    cam_targets: list[np.ndarray] = []
    for pane in PANES:
        cam_targets.append(np.asarray(pane.target, dtype=np.float64))

    light_dir = [0.55, 0.25, 1.05]
    light_color = [1.0, 0.98, 0.95]
    # Slightly warmer / brighter than iter02 — iter02's iso pane read
    # a little gray.  Still darker than the original "washed out" preset.
    light_ambient = 0.42
    light_diffuse = 0.60
    light_specular = 0.10

    total_frames = 0
    t_start = time.perf_counter()

    for si, seg in enumerate(segs):
        chapter, title, _ = _beat_for(seg.banner)
        beat_ms = dict((lbl, ms) for lbl, ms in _timings)
        ms = beat_ms.get(seg.banner.strip(), None)
        if ms is None:
            # Some demo labels have trailing "line" etc. — try prefix match.
            for k, v in beat_ms.items():
                if seg.banner.strip().startswith(k):
                    ms = v
                    break
        extra = f" · solved in {ms:.0f} ms" if ms is not None else ""
        print(f"  [{si:02d}] {chapter}: {title} ({seg.path.shape[0]} wp){extra}")

        if seg.reveal_points is not None:
            _show_beam(client, beam_id)

        # Destination marker: drop it for nav, park offscreen otherwise.
        if seg.banner in _NAV_DESTINATION:
            _move_to(client, dest_pin_id, _NAV_DESTINATION[seg.banner])
        else:
            _hide(client, dest_pin_id)

        focus_obj_name = _FOCUS_OBJECT.get(seg.banner, "robot")
        focus_grip_link = _FOCUS_GRIPPER.get(seg.banner, demo.RIGHT_GRIPPER_LINK)
        focus_grip_idx = link_ix[focus_grip_link]

        stride = strides[si]
        path_indices = list(range(0, seg.path.shape[0], stride))
        if path_indices[-1] != seg.path.shape[0] - 1:
            path_indices.append(seg.path.shape[0] - 1)
        tail = tail_holds[si]
        path_indices.extend([seg.path.shape[0] - 1] * tail)

        for fidx in path_indices:
            cfg = seg.path[fidx]
            env.set_configuration(cfg)
            if (
                seg.attach_body_id is not None
                and seg.attach_local_tf is not None
                and seg.attach_link_idx is not None
            ):
                demo.apply_attachment(
                    env, seg.attach_link_idx, seg.attach_body_id, seg.attach_local_tf
                )

            # Halo under the focus object when it's a known graspable.
            # Pin to floor whenever the object is held (z becomes stale as
            # the gripper moves around) — a floor disc reads as a map-pin.
            if focus_obj_name in obj_name_to_id:
                fx = obj_name_to_id[focus_obj_name]
                pos, _ = client.getBasePositionAndOrientation(fx)
                if seg.attach_body_id == fx:
                    halo_pos = np.array([pos[0], pos[1], 0.012])
                else:
                    halo_pos = np.array([pos[0], pos[1], max(0.012, pos[2] - 0.02)])
                _move_to(client, halo_id, halo_pos)
            else:
                _hide(client, halo_id)

            # Per-pane camera targets.
            robot_xy = _robot_root_xy(client, env)
            robot_theta = _robot_yaw(client, env)
            grip_pos = np.asarray(
                client.getLinkState(env.sim.skel_id, focus_grip_idx)[0]
            )
            focus_world = (
                np.asarray(
                    client.getBasePositionAndOrientation(
                        obj_name_to_id[focus_obj_name]
                    )[0]
                )
                if focus_obj_name in obj_name_to_id
                else grip_pos
            )

            panes_rgba = []
            is_nav = seg.banner in _NAV_WAIST_UP_SEGMENTS and seg.attach_body_id is None
            for pi, (pane, proj) in enumerate(zip(PANES, projs)):
                yaw_override = None
                dist_override = None
                pitch_override = None
                if pane.kind == "follow_robot":
                    tgt_raw = np.array([robot_xy[0], robot_xy[1], 1.10])
                    # Yaw the camera to the robot's front-right so we see the
                    # face/chest rather than the back.  ``pane.yaw`` is the
                    # offset (30-45°) from the robot's facing direction.
                    yaw_override = float(np.degrees(robot_theta)) + pane.yaw
                elif pane.kind == "chase_robot":
                    tgt_raw = np.array([robot_xy[0], robot_xy[1], 1.20])
                    yaw_override = float(np.degrees(robot_theta)) + pane.yaw
                elif pane.kind == "overhead_robot":
                    # Static world-frame yaw (pane.yaw) — don't override —
                    # so the image "up" direction stays stable while the
                    # target tracks the robot on the floor plane.
                    tgt_raw = np.array([robot_xy[0], robot_xy[1], 0.80])
                elif pane.kind in ("topdown", "iso"):
                    tgt_raw = np.array(pane.target)
                elif is_nav:
                    # Waist-up follow shot replaces the close-up during transit.
                    tgt_raw = _robot_local_target(
                        robot_xy, robot_theta, _NAV_TARGET_LOCAL
                    )
                    yaw_override = float(np.degrees(robot_theta)) + 32.0
                    pitch_override = -12.0
                    dist_override = 2.25
                else:  # follow_action
                    tgt_raw = 0.55 * focus_world + 0.45 * grip_pos
                    yaw_override = float(np.degrees(robot_theta)) + pane.yaw

                # Low-pass smoothing so the cameras don't jitter.
                alpha = pane.smoothing
                cam_targets[pi] = (1 - alpha) * cam_targets[pi] + alpha * tgt_raw
                view = _compute_view(
                    client,
                    pane,
                    cam_targets[pi],
                    yaw_override,
                    dist_override=dist_override,
                    pitch_override=pitch_override,
                )

                rw, rh = pane_render_sizes[pi]
                img = client.getCameraImage(
                    width=rw,
                    height=rh,
                    viewMatrix=view,
                    projectionMatrix=proj,
                    shadow=shadows,
                    lightDirection=light_dir,
                    lightColor=light_color,
                    lightDistance=5.5,
                    lightAmbientCoeff=light_ambient,
                    lightDiffuseCoeff=light_diffuse,
                    lightSpecularCoeff=light_specular,
                    renderer=pb.ER_TINY_RENDERER,
                )
                rgba = np.asarray(img[2], dtype=np.uint8).reshape(rh, rw, 4)
                if (rw, rh) != (pane_w, pane_h):
                    # Nearest-neighbour 2× upscale — keeps silhouettes crisp
                    # and avoids another pass over the bitmap.
                    rgba = np.repeat(np.repeat(rgba, 2, axis=0), 2, axis=1)
                    rgba = rgba[:pane_h, :pane_w]
                panes_rgba.append(rgba)

            frame = np.concatenate(panes_rgba, axis=1)
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                proc.wait(timeout=10)
                err = stderr_log.read_text(errors="ignore")
                sys.stderr.write(err)
                raise
            total_frames += 1

    proc.stdin.close()
    rc = proc.wait(timeout=180)
    if rc != 0:
        sys.stderr.write(stderr_log.read_text(errors="ignore"))
        raise RuntimeError(f"ffmpeg exited {rc}")

    wall = time.perf_counter() - t_start
    size_kb = out.stat().st_size / 1024 if out.exists() else 0.0
    print(
        f"\n  rendered {total_frames} frames in {wall:.1f} s → "
        f"{out} ({size_kb:.0f} KB, {total_frames / fps:.1f} s clip)"
    )

    # Dump a JSON sidecar that the review loop can consume.
    meta = {
        "video": str(out),
        "fps": fps,
        "pane_w": pane_w,
        "pane_h": pane_h,
        "total_frames": total_frames,
        "duration_s": total_frames / fps,
        "segments": timings_table,
    }
    side = out.with_suffix(".meta.json")
    side.write_text(json.dumps(meta, indent=2))
    print(f"  metadata → {side}")


def _preview_frames(out_dir: Path, pane_w: int, pane_h: int) -> None:
    """Cheap dry-run: render a single representative frame per segment as PNG.

    No ffmpeg, no drawtext — purely a visual sanity check that the
    scene meshes are in place, the cameras are framed sensibly, and
    the graspable colours / halo / beam are showing up.  Takes ~10 s.
    """
    # Serialize each RGBA frame to PNG through ffmpeg — avoids adding a
    # hard dep on Pillow/imageio.
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_png(path: Path, rgba: np.ndarray) -> None:
        h, w, _ = rgba.shape
        p = subprocess.Popen(
            [
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
                "-i",
                "-",
                "-frames:v",
                "1",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        assert p.stdin is not None
        p.stdin.write(rgba.tobytes())
        p.stdin.close()
        p.wait(timeout=30)

    env = _captured["env"]
    segs = _captured["segs"]
    client = env.sim.client

    _spawn_ground_plate(client)
    _spawn_scene_meshes(env)
    beam_id = _spawn_beam(client, hidden=True)
    halo_id = _spawn_halo(client, radius=0.16)
    dest_pin_id = _spawn_destination_pin(client)

    # Discover graspable body IDs by mesh asset name.
    apple_id = bowl_id = bottle_id = None
    for i in range(client.getNumBodies()):
        uid = client.getBodyUniqueId(i)
        for info in client.getVisualShapeData(uid):
            mesh_asset = info[4]
            if isinstance(mesh_asset, bytes):
                mesh_asset = mesh_asset.decode("utf-8", errors="ignore")
            name = os.path.basename(mesh_asset) if mesh_asset else ""
            if name == "apple.obj":
                apple_id = uid
            elif name == "bowl.obj":
                bowl_id = uid
            elif name == "bottle.obj":
                bottle_id = uid
    assert apple_id and bowl_id and bottle_id, "graspable mesh ids not found"
    _paint_graspables(client, apple_id, bowl_id, bottle_id)
    obj_name_to_id = {"apple": apple_id, "bowl": bowl_id, "bottle": bottle_id}

    link_ix = {
        demo.GRIPPER_LINK: demo.find_link_index(env, demo.GRIPPER_LINK),
        demo.RIGHT_GRIPPER_LINK: demo.find_link_index(env, demo.RIGHT_GRIPPER_LINK),
    }

    projs = [
        client.computeProjectionMatrixFOV(
            fov=p.fov, aspect=pane_w / pane_h, nearVal=0.05, farVal=40.0
        )
        for p in PANES
    ]
    cam_targets = [np.asarray(p.target, dtype=np.float64) for p in PANES]

    for si, seg in enumerate(segs):
        if seg.reveal_points is not None:
            _show_beam(client, beam_id)

        if seg.banner in _NAV_DESTINATION:
            _move_to(client, dest_pin_id, _NAV_DESTINATION[seg.banner])
        else:
            _hide(client, dest_pin_id)

        # Sample the middle of each segment — it's the most informative pose.
        fidx = seg.path.shape[0] // 2
        cfg = seg.path[fidx]
        env.set_configuration(cfg)
        if (
            seg.attach_body_id is not None
            and seg.attach_local_tf is not None
            and seg.attach_link_idx is not None
        ):
            demo.apply_attachment(
                env, seg.attach_link_idx, seg.attach_body_id, seg.attach_local_tf
            )

        focus_obj_name = _FOCUS_OBJECT.get(seg.banner, "robot")
        focus_grip_link = _FOCUS_GRIPPER.get(seg.banner, demo.RIGHT_GRIPPER_LINK)
        focus_grip_idx = link_ix[focus_grip_link]

        if focus_obj_name in obj_name_to_id:
            fx = obj_name_to_id[focus_obj_name]
            pos, _ = client.getBasePositionAndOrientation(fx)
            if seg.attach_body_id == fx:
                halo_pos = np.array([pos[0], pos[1], 0.012])
            else:
                halo_pos = np.array([pos[0], pos[1], max(0.012, pos[2] - 0.02)])
            _move_to(client, halo_id, halo_pos)
        else:
            _hide(client, halo_id)

        robot_xy = _robot_root_xy(client, env)
        robot_theta = _robot_yaw(client, env)
        grip_pos = np.asarray(client.getLinkState(env.sim.skel_id, focus_grip_idx)[0])
        focus_world = (
            np.asarray(
                client.getBasePositionAndOrientation(obj_name_to_id[focus_obj_name])[0]
            )
            if focus_obj_name in obj_name_to_id
            else grip_pos
        )

        panes_rgba = []
        is_nav = seg.banner in _NAV_WAIST_UP_SEGMENTS and seg.attach_body_id is None
        for pi, (pane, proj) in enumerate(zip(PANES, projs)):
            yaw_override = None
            dist_override = None
            pitch_override = None
            if pane.kind == "follow_robot":
                tgt = np.array([robot_xy[0], robot_xy[1], 1.10])
                yaw_override = float(np.degrees(robot_theta)) + pane.yaw
            elif pane.kind == "chase_robot":
                tgt = np.array([robot_xy[0], robot_xy[1], 1.20])
                yaw_override = float(np.degrees(robot_theta)) + pane.yaw
            elif pane.kind == "overhead_robot":
                tgt = np.array([robot_xy[0], robot_xy[1], 0.80])
            elif pane.kind in ("topdown", "iso"):
                tgt = np.array(pane.target)
            elif is_nav:
                tgt = _robot_local_target(robot_xy, robot_theta, _NAV_TARGET_LOCAL)
                yaw_override = float(np.degrees(robot_theta)) + 32.0
                pitch_override = -12.0
                dist_override = 2.25
            else:
                tgt = 0.55 * focus_world + 0.45 * grip_pos
                yaw_override = float(np.degrees(robot_theta)) + pane.yaw
            cam_targets[pi] = tgt  # no smoothing in preview
            view = _compute_view(
                client,
                pane,
                cam_targets[pi],
                yaw_override,
                dist_override=dist_override,
                pitch_override=pitch_override,
            )
            img = client.getCameraImage(
                width=pane_w,
                height=pane_h,
                viewMatrix=view,
                projectionMatrix=proj,
                shadow=1,
                lightDirection=[0.55, 0.20, 1.00],
                lightColor=[1.0, 0.97, 0.94],
                lightDistance=5.5,
                lightAmbientCoeff=0.36,
                lightDiffuseCoeff=0.55,
                lightSpecularCoeff=0.10,
                renderer=pb.ER_TINY_RENDERER,
            )
            panes_rgba.append(
                np.asarray(img[2], dtype=np.uint8).reshape(pane_h, pane_w, 4)
            )
        frame = np.concatenate(panes_rgba, axis=1)
        chapter, title, _ = _beat_for(seg.banner)
        safe = title.replace(" ", "_").replace("/", "").replace(",", "")
        out_path = out_dir / f"{si:02d}_{chapter.replace(' ', '_')}_{safe[:40]}.png"
        _write_png(out_path, frame)
        print(f"  [{si:02d}] {chapter} · {title} → {out_path.name}")


# ── CLI ───────────────────────────────────────────────────────────


_CACHE_VERSION = 2


def _segments_to_cache(segs, out: Path) -> None:
    """Persist planned segments so iterations can skip the 30+s planner."""
    import pickle

    data = {
        "version": _CACHE_VERSION,
        "segments": [
            {
                "path": s.path,
                "banner": s.banner,
                "attach_body_id": s.attach_body_id,
                "attach_local_tf": s.attach_local_tf,
                "attach_link_idx": s.attach_link_idx,
                "reveal_points": s.reveal_points,
            }
            for s in segs
        ],
        "timings": list(_timings),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def _segments_from_cache(path: Path):
    import pickle

    from rls_pick_place import Segment  # type: ignore

    data = pickle.loads(Path(path).read_bytes())
    if data.get("version") != _CACHE_VERSION:
        raise RuntimeError(
            f"stale segment cache at {path} (v{data.get('version')} ≠ v{_CACHE_VERSION})"
        )
    segs = [Segment(**d) for d in data["segments"]]
    _timings.clear()
    _timings.extend([tuple(t) for t in data["timings"]])
    return segs


def _rebuild_env_for_render():
    """Re-create the PyBullet env + graspables in the same order the demo
    does, so cached ``attach_body_id`` values still identify the apple /
    bowl / bottle correctly.  No pointcloud — it's invisible to
    TinyRenderer and slows nothing but the env load."""
    env = PyBulletEnv(demo.autolife_robot_config, visualize=False)
    current = demo.HOME_JOINTS.copy()
    current[:3] = demo.BASE_SQUAT_EAST
    current[6] = 0.0
    env.set_configuration(current)
    demo.place_graspable(env, "apple", demo.APPLE_MESH_INIT)
    demo.place_graspable(env, "bowl", demo.BOWL_MESH_INIT)
    demo.place_graspable(env, "bottle", demo.BOTTLE_MESH_INIT)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/assets/rls_pick_place.mp4")
    parser.add_argument("--pane_width", type=int, default=720)
    parser.add_argument("--pane_height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument(
        "--hold_frames",
        type=int,
        default=6,
        help="Extra frames to freeze on each segment's last config (absorb beats).",
    )
    parser.add_argument(
        "--segment_cache",
        default="tmp/rls_render/segments.cache.pkl",
        help=(
            "Path where planned segments are persisted.  If the cache "
            "exists it's used and planning is skipped; otherwise the "
            "demo is run and the cache is written."
        ),
    )
    parser.add_argument(
        "--force_replan",
        action="store_true",
        help="Ignore the cache and re-run the demo's planning pipeline.",
    )
    parser.add_argument(
        "--preview_frames_dir",
        default=None,
        help=(
            "If set, render only the first frame of every segment as PNGs "
            "into this directory and exit — fast dry-run for visual spot-checks."
        ),
    )
    parser.add_argument(
        "--shadows",
        type=int,
        default=1,
        help="Enable PyBullet shadows (1) or not (0).  Off is ~2× faster.",
    )
    parser.add_argument(
        "--middle_view",
        choices=("chase", "overhead", "map"),
        default="chase",
        help=(
            "Camera for the middle pane: 'chase' over-the-shoulder behind "
            "the robot (default), 'overhead' top-down following the robot, "
            "'map' fixed high-angle iso of the whole apartment."
        ),
    )
    args = parser.parse_args()

    global PANES
    PANES = _build_panes(args.middle_view)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path

    cache_path = Path(args.segment_cache)
    if not cache_path.is_absolute():
        cache_path = _REPO_ROOT / cache_path

    if args.pane_width % 2 or args.pane_height % 2:
        raise ValueError("pane_width / pane_height must both be even")

    if cache_path.exists() and not args.force_replan:
        print(f"── loading planned segments from {cache_path} ──")
        segs = _segments_from_cache(cache_path)
        env = _rebuild_env_for_render()
        _captured["env"] = env
        _captured["segs"] = segs
    else:
        print("── planning (headless) ──")
        demo.main(visualize=True)
        _segments_to_cache(_captured["segs"], cache_path)
        print(f"  cached → {cache_path}")

    total_ms = sum(ms for _, ms in _timings)
    print("\n── planning timings ──")
    for label, ms in _timings:
        print(f"  {label:40s}  {ms:7.1f} ms")
    print(f"  {'TOTAL':40s}  {total_ms:7.1f} ms   ({len(_timings)} plans)")

    if args.preview_frames_dir:
        preview_dir = Path(args.preview_frames_dir)
        if not preview_dir.is_absolute():
            preview_dir = _REPO_ROOT / preview_dir
        print(f"\n── PREVIEW: writing first frame of every segment to {preview_dir} ──")
        _preview_frames(preview_dir, args.pane_width, args.pane_height)
        return

    print(f"\n── rendering → {out_path} ──")
    _render(
        out_path,
        args.pane_width,
        args.pane_height,
        args.fps,
        args.playback_speed,
        args.hold_frames,
        shadows=args.shadows,
    )


if __name__ == "__main__":
    main()
