"""Long-horizon pick-and-place inside the full RLS-env room.

A single end-to-end showcase of the project's planning stack:

* **Four subgroup planners** combined in one sequence:
  ``autolife_height`` (squat/stand), ``autolife_body`` (whole upper
  body with leg-pin constraint), ``autolife_torso_left_arm``
  (9-DOF arm), and ``autolife_base`` (navigation).
* **CasADi constraints** — the under-table pick pins the ankle +
  knee via a compiled holonomic constraint while the arm plans on
  the remaining 19 DOF; every pregrasp→grasp sweep uses a
  2-equation straight-line manifold.
* **Collision avoidance** — every planner gets the full 151 k-point
  cloud built from all seven ``pcd/*.ply`` scans.
* **Hardcoded grasp configs** — each grasp/pregrasp/place pose is a
  pre-solved 24-DOF configuration so the demo is 100 % deterministic.

Storyline:

    1a. squat down  (autolife_height, 3 DOF)
    1b. pick apple from under the table  (autolife_body, 21 DOF + leg pin)
    1c. stand up  (autolife_height)
    2.  place apple on table top  (autolife_torso_left_arm, 9 DOF)
    3.  navigate to sofa  (autolife_base, 3 DOF)
    4.  pick bottle from sofa  (autolife_torso_left_arm)
    5.  navigate to coffee table  (autolife_base)
    6.  place bottle on coffee table  (autolife_torso_left_arm)

Usage::

    pixi run python examples/rls_pick_place_demo.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pybullet as pb
import trimesh
from fire import Fire

from autolife_planning.config.robot_config import (
    HOME_JOINTS,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import Constraint, SymbolicContext, create_planner
from autolife_planning.types import PlannerConfig

# ── Subgroups ──────────────────────────────────────────────────────
BASE_SUBGROUP = "autolife_base"  # 3 DOF: virtual x/y/yaw
HEIGHT_SUBGROUP = "autolife_height"  # 3 DOF: ankle, knee, waist_pitch
ARM_SUBGROUP = "autolife_torso_left_arm"  # 9 DOF: waist + left arm
BODY_SUBGROUP = "autolife_body"  # 21 DOF: legs + waist + arms + neck

TORSO_ARM_IDX = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13])
HEIGHT_IDX = np.array([3, 4, 5])
BODY_IDX = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
)
BODY_ANKLE_IDX = 0  # inside the 21-element active vector
BODY_KNEE_IDX = 1

GRIPPER_LINK = "Link_Left_Gripper"

# ── Asset paths ────────────────────────────────────────────────────
RLS_ROOT = "assets/envs/rls_env"
MESH_DIR = f"{RLS_ROOT}/meshes"
PCD_DIR = f"{RLS_ROOT}/pcd"
SCENE_PROPS = [
    ("rls_2", "rls_2"),
    ("open_kitchen", "open_kitchen"),
    ("wall", "wall"),
    ("workstation", "workstation"),
    ("table", "table"),
    ("sofa", "sofa"),
    ("tea_table", "coffee_table"),
]

# ── Robot base poses ──────────────────────────────────────────────
BASE_NEAR_TABLE_EAST = np.array([-1.30, 1.67, np.pi])
BASE_NEAR_SOFA = np.array([2.00, 1.30, -np.pi / 2])
BASE_NEAR_COFFEE = np.array([3.60, 1.30, -np.pi / 2])
STANDING_STANCE = np.array([0.0, 0.0, 0.0])
SQUAT_STANCE = np.array([1.0, 2.0, 0.0])

# ── Graspable positions + approach vectors ────────────────────────
APPLE_MESH_INIT = np.array([-2.05, 1.67, 0.32])
APPLE_MESH_PLACE = np.array([-2.05, 1.67, 0.75])
APPLE_APPROACH = np.array([-1.0, 0.0, 0.0])
APPLE_GRASP_Z = 0.13

BOTTLE_MESH_INIT = np.array([1.95, 0.30, 0.69])
BOTTLE_MESH_PLACE = np.array([3.60, 0.30, 0.58])
BOTTLE_APPROACH = np.array([0.0, -1.0, 0.0])
BOTTLE_GRASP_Z = 0.14

PREGRASP_OFFSET = 0.12
PREGRASP_OFFSET_UNDER_TABLE = 0.05

# ── Hardcoded 24-DOF grasp/pregrasp configs ───────────────────────
# Pre-solved via TRAC-IK on the whole_body_base_left chain and
# validated against the scene pointcloud.  Each config is the FULL
# 24-DOF body state (base + legs + waist + arms + neck).
# fmt: off
_NH = [0.0, 0.0, 0.0, -0.7, 0.14, -0.09, -2.31, -0.04, -0.4, 0.0]

APPLE_GRASP_FULL = np.array(
    [-1.3, 1.67, 3.14159, 1.0, 2.0, 1.69081, -1.29081,
     0.26294, -0.89298, -0.42125, 1.79897, -0.44705,
     -1.16186, -0.14746] + _NH)
APPLE_PREGRASP_FULL = np.array(
    [-1.3, 1.67, 3.14159, 1.0, 2.0, 1.66137, -1.28397,
     0.30579, -0.77443, -0.34967, 1.86676, -0.40673,
     -1.17836, -0.1811] + _NH)
APPLE_PLACE_FULL = np.array(
    [-1.3, 1.67, 3.14159, 0.0, 0.0, 0.54695, -0.94467,
     -0.39819, -0.36978, -0.32065, 1.66405, 0.13771,
     -0.5133, 0.13768] + _NH)
APPLE_PREPLACE_FULL = np.array(
    [-1.3, 1.67, 3.14159, 0.0, 0.0, 0.50202, -0.84614,
     -0.21316, -0.11009, -0.19886, 1.7565, 0.22301,
     -0.59207, 0.00636] + _NH)
BOTTLE_GRASP_FULL = np.array(
    [2.0, 1.3, -1.5708, 0.0, 0.0, 0.7048, -1.4169,
     -0.11159, -1.38957, -0.47527, 1.08013, -0.72665,
     -0.71251, -0.15198] + _NH)
BOTTLE_PREGRASP_FULL = np.array(
    [2.0, 1.3, -1.5708, 0.0, 0.0, 0.63199, -1.31205,
     -0.30518, -0.83376, -0.62307, 1.42075, -0.09389,
     -0.54424, 0.07077] + _NH)
BOTTLE_PLACE_FULL = np.array(
    [3.6, 1.3, -1.5708, 0.0, 0.0, 0.926, -1.38697,
     -0.32474, -1.45883, -0.5743, 1.09816, -0.4848,
     -0.48822, -0.1056] + _NH)
BOTTLE_PREPLACE_FULL = np.array(
    [3.6, 1.3, -1.5708, 0.0, 0.0, 0.90982, -1.23524,
     -0.56996, -0.89735, -0.65942, 1.53732, 0.04446,
     -0.26316, 0.08853] + _NH)
# fmt: on

ARM_FREE_TIME = 4.0
ARM_LINE_TIME = 8.0
BASE_NAV_TIME = 6.0


# ── Scene setup ────────────────────────────────────────────────────


def load_room_meshes(env: PyBulletEnv) -> None:
    for mesh_name, _ in SCENE_PROPS:
        env.add_mesh(
            os.path.abspath(f"{MESH_DIR}/{mesh_name}/{mesh_name}.obj"),
            position=np.zeros(3),
        )


def load_room_pointcloud(stride: int = 1) -> np.ndarray:
    chunks = []
    for _, pcd_name in SCENE_PROPS:
        pc = trimesh.load(os.path.abspath(f"{PCD_DIR}/{pcd_name}.ply"))
        v = np.asarray(pc.vertices, dtype=np.float32)  # type: ignore[union-attr]
        if stride > 1:
            v = v[::stride]
        chunks.append(v)
    return np.concatenate(chunks, axis=0)


def place_graspable(env: PyBulletEnv, name: str, xyz: np.ndarray) -> int:
    return env.add_mesh(
        os.path.abspath(f"{MESH_DIR}/{name}/{name}.obj"),
        position=np.asarray(xyz, dtype=float),
    )


# ── Constraint helpers ─────────────────────────────────────────────


def _sanitize(s: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _project_and_clamp(ctx, residual, q, lo, hi, iters=5):
    for _ in range(iters):
        q = ctx.project(q, residual)
        q = np.clip(q, lo + 1e-5, hi - 1e-5)
    return q


def make_line_constraint(ctx, p_from, p_to, name):
    p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
    d = p_to - p_from
    d /= float(np.linalg.norm(d))
    seed = np.array([1, 0, 0.0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0.0])
    u = np.cross(d, seed)
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)
    gripper = ctx.link_translation(GRIPPER_LINK)
    diff = gripper - ca.DM(p_from.tolist())
    res = ca.vertcat(ca.dot(diff, ca.DM(u.tolist())), ca.dot(diff, ca.DM(v.tolist())))
    return Constraint(residual=res, q_sym=ctx.q, name=name)


def make_leg_pin(ctx, ankle, knee, name):
    res = ca.vertcat(
        ctx.q[BODY_ANKLE_IDX] - ca.DM(float(ankle)),
        ctx.q[BODY_KNEE_IDX] - ca.DM(float(knee)),
    )
    return Constraint(residual=res, q_sym=ctx.q, name=name)


def grasp_rotation_from_approach(approach):
    a = np.asarray(approach, float)
    a /= float(np.linalg.norm(a))
    y = -a
    z = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1, 0, 0.0])
    z = z - float(np.dot(z, y)) * y
    z /= float(np.linalg.norm(z))
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


FINGER_MIDPOINT_IN_GRIPPER = np.array([-0.031, -0.064, 0.0])


def grasp_line_endpoints(finger_xyz, approach, pregrasp_offset=PREGRASP_OFFSET):
    R = grasp_rotation_from_approach(approach)
    a = np.asarray(approach, float)
    a /= float(np.linalg.norm(a))
    grasp_pos = np.asarray(finger_xyz, float) - R @ FINGER_MIDPOINT_IN_GRIPPER
    pregrasp_pos = grasp_pos - pregrasp_offset * a
    return grasp_pos, pregrasp_pos


# ── Planning wrappers ──────────────────────────────────────────────


def _report(label, result):
    n = result.path.shape[0] if result.path is not None else 0
    ms = result.planning_time_ns / 1e6
    print(
        f"  [{label}] {result.status.value}: {n} wp in {ms:.0f} ms"
        + (f", cost {result.path_cost:.2f}" if result.success else "")
    )


def plan_arm_free(current_full, goal_full, cloud, label):
    p = create_planner(
        ARM_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_FREE_TIME),
        pointcloud=cloud,
        base_config=current_full,
    )
    r = p.plan(current_full[TORSO_ARM_IDX], goal_full[TORSO_ARM_IDX])
    _report(label, r)
    assert r.success and r.path is not None, f"{label}: {r.status.value}"
    return p.embed_path(r.path)


def plan_arm_line(current_full, goal_full, p_from, p_to, cloud, label):
    ctx = SymbolicContext(ARM_SUBGROUP, base_config=current_full)
    c = make_line_constraint(ctx, p_from, p_to, f"line_{_sanitize(label)}")
    p = create_planner(
        ARM_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_LINE_TIME),
        pointcloud=cloud,
        base_config=current_full,
        constraints=[c],
    )
    lo, hi = np.array(p._planner.lower_bounds()), np.array(p._planner.upper_bounds())
    start = _project_and_clamp(
        ctx, c.residual, current_full[TORSO_ARM_IDX].copy(), lo, hi
    )
    goal = _project_and_clamp(ctx, c.residual, goal_full[TORSO_ARM_IDX].copy(), lo, hi)
    r = p.plan(start, goal)
    _report(label, r)
    assert r.success and r.path is not None, f"{label}: {r.status.value}"
    return p.embed_path(r.path)


def plan_body_free(current_full, goal_full, cloud, label):
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    lp = make_leg_pin(
        ctx, current_full[3], current_full[4], f"legpin_{_sanitize(label)}"
    )
    p = create_planner(
        BODY_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_FREE_TIME),
        pointcloud=cloud,
        base_config=current_full,
        constraints=[lp],
    )
    lo, hi = np.array(p._planner.lower_bounds()), np.array(p._planner.upper_bounds())
    start = _project_and_clamp(ctx, lp.residual, current_full[BODY_IDX].copy(), lo, hi)
    goal = _project_and_clamp(ctx, lp.residual, goal_full[BODY_IDX].copy(), lo, hi)
    r = p.plan(start, goal)
    _report(label, r)
    assert r.success and r.path is not None, f"{label}: {r.status.value}"
    return p.embed_path(r.path)


def plan_body_line(current_full, goal_full, p_from, p_to, cloud, label):
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    leg_res = ca.vertcat(
        ctx.q[BODY_ANKLE_IDX] - ca.DM(float(current_full[3])),
        ctx.q[BODY_KNEE_IDX] - ca.DM(float(current_full[4])),
    )
    pf, pt = np.asarray(p_from, float), np.asarray(p_to, float)
    d = pt - pf
    d /= float(np.linalg.norm(d))
    seed = np.array([1, 0, 0.0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0.0])
    u = np.cross(d, seed)
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)
    gripper = ctx.link_translation(GRIPPER_LINK)
    diff = gripper - ca.DM(pf.tolist())
    line_res = ca.vertcat(
        ca.dot(diff, ca.DM(u.tolist())), ca.dot(diff, ca.DM(v.tolist()))
    )
    combined = ca.vertcat(leg_res, line_res)
    c = Constraint(residual=combined, q_sym=ctx.q, name=f"bodyline_{_sanitize(label)}")
    p = create_planner(
        BODY_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_LINE_TIME),
        pointcloud=cloud,
        base_config=current_full,
        constraints=[c],
    )
    lo, hi = np.array(p._planner.lower_bounds()), np.array(p._planner.upper_bounds())
    start = _project_and_clamp(ctx, c.residual, current_full[BODY_IDX].copy(), lo, hi)
    goal = _project_and_clamp(ctx, c.residual, goal_full[BODY_IDX].copy(), lo, hi)
    r = p.plan(start, goal)
    _report(label, r)
    assert r.success and r.path is not None, f"{label}: {r.status.value}"
    return p.embed_path(r.path)


def plan_base(current_full, goal_base, cloud, label):
    p = create_planner(
        BASE_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=BASE_NAV_TIME),
        pointcloud=cloud,
        base_config=current_full,
    )
    r = p.plan(current_full[:3].copy(), np.asarray(goal_base, dtype=np.float64))
    _report(label, r)
    assert r.success and r.path is not None, f"{label}: {r.status.value}"
    return p.embed_path(r.path)


def plan_height(current_full, target, cloud, label):
    p = create_planner(
        HEIGHT_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_FREE_TIME),
        pointcloud=cloud,
        base_config=current_full,
    )
    r = p.plan(current_full[HEIGHT_IDX].copy(), np.asarray(target, dtype=np.float64))
    _report(label, r)
    assert r.success and r.path is not None, f"{label}: {r.status.value}"
    return p.embed_path(r.path)


# ── Playback ───────────────────────────────────────────────────────


@dataclass
class Segment:
    path: np.ndarray
    attach_body_id: int | None
    attach_local_tf: np.ndarray | None
    banner: str


def find_link_index(env, name):
    for i in range(env.sim.client.getNumJoints(env.sim.skel_id)):
        if env.sim.client.getJointInfo(env.sim.skel_id, i)[12].decode("utf-8") == name:
            return i
    raise RuntimeError(f"link {name!r} not found")


def capture_local_transform(env, link_idx, body_id):
    c = env.sim.client
    lp, lq = (
        c.getLinkState(env.sim.skel_id, link_idx)[0],
        c.getLinkState(env.sim.skel_id, link_idx)[1],
    )
    lr = np.asarray(c.getMatrixFromQuaternion(lq)).reshape(3, 3)
    op, oq = c.getBasePositionAndOrientation(body_id)
    orr = np.asarray(c.getMatrixFromQuaternion(oq)).reshape(3, 3)
    tf = np.eye(4)
    tf[:3, :3] = lr.T @ orr
    tf[:3, 3] = lr.T @ (np.asarray(op) - np.asarray(lp))
    return tf


def apply_attachment(env, link_idx, body_id, local_tf):
    c = env.sim.client
    lp, lq = (
        c.getLinkState(env.sim.skel_id, link_idx)[0],
        c.getLinkState(env.sim.skel_id, link_idx)[1],
    )
    lr = np.asarray(c.getMatrixFromQuaternion(lq)).reshape(3, 3)
    wr = lr @ local_tf[:3, :3]
    wp = lr @ local_tf[:3, 3] + np.asarray(lp)
    m = wr
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0:
        s = float(np.sqrt(t + 1) * 2)
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = float(np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2)
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2)
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = float(np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2)
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    c.resetBasePositionAndOrientation(body_id, wp.tolist(), [x, y, z, w])


def play_segments(env, segments, gripper_link_idx, fps=60.0):
    c = env.sim.client
    dt = 1.0 / fps
    frames = [
        (si, ri) for si, seg in enumerate(segments) for ri in range(seg.path.shape[0])
    ]
    idx, n, playing, last_si = 0, len(frames), False, -1
    print("\nControls: SPACE play/pause   ←/→ step   close window to exit\n")
    for s in segments:
        print(f"  → {s.banner} ({s.path.shape[0]} wp)")
    try:
        while c.isConnected():
            si, ri = frames[idx]
            seg = segments[si]
            env.set_configuration(seg.path[ri])
            if seg.attach_body_id is not None and seg.attach_local_tf is not None:
                apply_attachment(
                    env, gripper_link_idx, seg.attach_body_id, seg.attach_local_tf
                )
            if si != last_si:
                print(f"[{si}] {seg.banner}")
                last_si = si
            keys = c.getKeyboardEvents()
            if ord(" ") in keys and keys[ord(" ")] & pb.KEY_WAS_TRIGGERED:
                playing = not playing
            elif (
                not playing
                and pb.B3G_LEFT_ARROW in keys
                and keys[pb.B3G_LEFT_ARROW] & pb.KEY_WAS_TRIGGERED
            ):
                idx = (idx - 1) % n
            elif (
                not playing
                and pb.B3G_RIGHT_ARROW in keys
                and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_WAS_TRIGGERED
            ):
                idx = (idx + 1) % n
            elif playing:
                idx = (idx + 1) % n
            time.sleep(dt)
    except pb.error:
        pass


# ── Main ───────────────────────────────────────────────────────────


def main(pcd_stride: int = 1, visualize: bool = True) -> None:
    env = PyBulletEnv(autolife_robot_config, visualize=visualize)
    print("── scene setup ──")
    load_room_meshes(env)
    cloud = load_room_pointcloud(stride=pcd_stride)
    print(f"  collision cloud: {len(cloud)} points (stride={pcd_stride})")
    env.add_pointcloud(cloud[::4], pointsize=2)
    if visualize:
        env.sim.client.resetDebugVisualizerCamera(
            cameraDistance=4.5,
            cameraYaw=-90.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[-0.5, -0.3, 0.7],
        )

    current = HOME_JOINTS.copy()
    current[:3] = BASE_NEAR_TABLE_EAST
    env.set_configuration(current)

    apple_id = place_graspable(env, "apple", APPLE_MESH_INIT)
    bottle_id = place_graspable(env, "bottle", BOTTLE_MESH_INIT)
    gripper_link = find_link_index(env, GRIPPER_LINK)
    client = env.sim.client
    segs: list[Segment] = []

    def add(path, label, attach_id=None, attach_tf=None):
        segs.append(
            Segment(
                path=path,
                attach_body_id=attach_id,
                attach_local_tf=attach_tf,
                banner=label,
            )
        )

    # ── Stage 1a: squat (height subgroup) ──
    print("\n── stage 1a: squat ──")
    path = plan_height(current, SQUAT_STANCE, cloud, "s1a squat")
    add(path, "s1a squat")
    current = path[-1]

    # ── Stage 1b: pick apple under table (body + leg-pin) ──
    print("\n── stage 1b: pick apple under table (body + leg-pin constraint) ──")
    gp, pp = grasp_line_endpoints(
        APPLE_MESH_INIT + [0, 0, APPLE_GRASP_Z],
        APPLE_APPROACH,
        PREGRASP_OFFSET_UNDER_TABLE,
    )

    path = plan_body_free(current, APPLE_PREGRASP_FULL, cloud, "s1b free→pregrasp")
    add(path, "s1b approach")
    current = path[-1]

    path = plan_body_line(current, APPLE_GRASP_FULL, pp, gp, cloud, "s1b approach")
    add(path, "s1b approach line")
    current = path[-1]

    # Capture attachment
    env.set_configuration(APPLE_GRASP_FULL)
    client.resetBasePositionAndOrientation(
        apple_id, APPLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    apple_tf = capture_local_transform(env, gripper_link, apple_id)

    path = plan_body_line(current, APPLE_PREGRASP_FULL, gp, pp, cloud, "s1b lift")
    add(path, "s1b lift", apple_id, apple_tf)
    current = path[-1]

    # ── Stage 1c: stand up (height, apple attached) ──
    print("\n── stage 1c: stand up ──")
    path = plan_height(current, STANDING_STANCE, cloud, "s1c stand")
    add(path, "s1c stand", apple_id, apple_tf)
    current = path[-1]

    # ── Stage 2: place apple on table (torso+arm) ──
    print("\n── stage 2: place apple on table ──")
    gp2, pp2 = grasp_line_endpoints(
        APPLE_MESH_PLACE + [0, 0, APPLE_GRASP_Z], APPLE_APPROACH
    )

    path = plan_arm_free(current, APPLE_PREPLACE_FULL, cloud, "s2 free→preplace")
    add(path, "s2 carry", apple_id, apple_tf)
    current = path[-1]

    path = plan_arm_line(current, APPLE_PLACE_FULL, pp2, gp2, cloud, "s2 lower")
    add(path, "s2 lower", apple_id, apple_tf)
    current = path[-1]

    path = plan_arm_line(current, APPLE_PREPLACE_FULL, gp2, pp2, cloud, "s2 retreat")
    add(path, "s2 retreat")
    current = path[-1]

    # ── Stage 3: nav to sofa ──
    print("\n── stage 3: nav → sofa ──")
    path = plan_base(current, BASE_NEAR_SOFA, cloud, "s3 nav→sofa")
    add(path, "s3 nav→sofa")
    current = path[-1]

    # ── Stage 4: pick bottle from sofa ──
    print("\n── stage 4: pick bottle from sofa ──")
    gp3, pp3 = grasp_line_endpoints(
        BOTTLE_MESH_INIT + [0, 0, BOTTLE_GRASP_Z], BOTTLE_APPROACH
    )

    path = plan_arm_free(current, BOTTLE_PREGRASP_FULL, cloud, "s4 free→pregrasp")
    add(path, "s4 approach")
    current = path[-1]

    path = plan_arm_line(current, BOTTLE_GRASP_FULL, pp3, gp3, cloud, "s4 approach")
    add(path, "s4 approach line")
    current = path[-1]

    env.set_configuration(BOTTLE_GRASP_FULL)
    client.resetBasePositionAndOrientation(
        bottle_id, BOTTLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    bottle_tf = capture_local_transform(env, gripper_link, bottle_id)

    path = plan_arm_line(current, BOTTLE_PREGRASP_FULL, gp3, pp3, cloud, "s4 lift")
    add(path, "s4 lift", bottle_id, bottle_tf)
    current = path[-1]

    # ── Stage 5: nav to coffee table ──
    print("\n── stage 5: nav → coffee table ──")
    path = plan_base(current, BASE_NEAR_COFFEE, cloud, "s5 nav→coffee")
    add(path, "s5 nav→coffee", bottle_id, bottle_tf)
    current = path[-1]

    # ── Stage 6: place bottle on coffee table ──
    print("\n── stage 6: place bottle on coffee table ──")
    gp4, pp4 = grasp_line_endpoints(
        BOTTLE_MESH_PLACE + [0, 0, BOTTLE_GRASP_Z], BOTTLE_APPROACH
    )

    path = plan_arm_free(current, BOTTLE_PREPLACE_FULL, cloud, "s6 free→preplace")
    add(path, "s6 carry", bottle_id, bottle_tf)
    current = path[-1]

    path = plan_arm_line(current, BOTTLE_PLACE_FULL, pp4, gp4, cloud, "s6 lower")
    add(path, "s6 lower", bottle_id, bottle_tf)
    current = path[-1]

    path = plan_arm_line(current, BOTTLE_PREPLACE_FULL, gp4, pp4, cloud, "s6 retreat")
    add(path, "s6 retreat")
    current = path[-1]

    # Reset for playback
    env.set_configuration(segs[0].path[0])
    client.resetBasePositionAndOrientation(
        apple_id, APPLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    client.resetBasePositionAndOrientation(
        bottle_id, BOTTLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )

    total = sum(s.path.shape[0] for s in segs)
    print(f"\n── ready: {total} total frames across {len(segs)} segments ──")
    if not visualize:
        return
    play_segments(env, segs, gripper_link)


if __name__ == "__main__":
    Fire(main)
