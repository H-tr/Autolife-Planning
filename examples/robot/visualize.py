"""Interactive path visualizer for autolife_planner.plan_to_joints.

Plans a path for every group listed in *groups* (default: all supported
groups) and lets you step through them one by one in a PyBullet GUI.

Usage::

    pixi run python robot/visualize.py
    pixi run python robot/visualize.py --group autolife_left_arm
    pixi run python robot/visualize.py --planner rrtc --time_limit 1.0
    pixi run python robot/visualize.py --fps 30

Controls inside the viewer:
    SPACE       toggle auto-play / pause
    ← / →       step one waypoint back / forward (while paused)
    n           advance to the next group
    close       quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from fire import Fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.autolife_planner import SUPPORTED_GROUPS, AutolifePlanner  # noqa: E402

from autolife_planning.autolife import (  # noqa: E402
    HOME_JOINTS,
    VIZ_URDF_PATH,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Default goals — one hand-crafted 24-DOF target per supported group.
#
# Layout: [base(3), legs(2), waist(2), left_arm(7), neck(3), right_arm(7)]
#
# Only the joints owned by the group differ from HOME; every frozen joint
# keeps its HOME value so plan_to_joints never returns "not same".
# ---------------------------------------------------------------------------

# Start segments (frozen joints always stay here).
_BASE = HOME_JOINTS[0:3].tolist()  # [0, 0, 0]
_LEGS = [-0.2, 0.3]  # slight squat (HOME is [0, 0])
_WAIST = [0.2, 0.0]  # slight pitch (HOME is [0.00, -0.14])
_L_ARM = HOME_JOINTS[7:14].tolist()  # [0.70, -0.14, -0.09,  2.31,  0.04, -0.40, 0.0]
_NECK = HOME_JOINTS[14:17].tolist()  # [0, 0, 0]
_R_ARM = HOME_JOINTS[17:24].tolist()  # [-0.70, 0.14, -0.09, -2.31, -0.04, -0.40, 0.0]

# Goal segments (active joints moved to natural poses).
_GOAL_L_ARM = [0.6, -0.50, 0.30, 1.80, -0.20, -0.60, 0.50]  # left arm reach
_GOAL_R_ARM = [
    -0.6,
    0.50,
    0.30,
    -1.80,
    0.20,
    -0.60,
    -0.50,
]  # right arm reach (symmetric)
_GOAL_WAIST = [0.6, 0.5]  # slight pitch + yaw
_GOAL_LEGS = [0.78, 1.40]  # mid-squat (ankle, knee)


def _cfg(*segments: list) -> np.ndarray:
    return np.array([v for seg in segments for v in seg])


# Each value is a full 24-DOF goal config; only the group's active DOFs differ.
_DEFAULT_GOALS: dict[str, np.ndarray] = {
    "autolife_left_arm": _cfg(_BASE, _LEGS, _WAIST, _GOAL_L_ARM, _NECK, _R_ARM),
    "autolife_right_arm": _cfg(_BASE, _LEGS, _WAIST, _L_ARM, _NECK, _GOAL_R_ARM),
    "autolife_dual_arm": _cfg(_BASE, _LEGS, _WAIST, _GOAL_L_ARM, _NECK, _GOAL_R_ARM),
    "autolife_torso_left_arm": _cfg(
        _BASE, _LEGS, _GOAL_WAIST, _GOAL_L_ARM, _NECK, _R_ARM
    ),
    "autolife_torso_right_arm": _cfg(
        _BASE, _LEGS, _GOAL_WAIST, _L_ARM, _NECK, _GOAL_R_ARM
    ),
    "autolife_torso_dual_arm": _cfg(
        _BASE, _LEGS, _GOAL_WAIST, _GOAL_L_ARM, _NECK, _GOAL_R_ARM
    ),
    "autolife_leg_torso_dual_arm": _cfg(
        _BASE, _GOAL_LEGS, _GOAL_WAIST, _GOAL_L_ARM, _NECK, _GOAL_R_ARM
    ),
}

# Ordered for a natural demo progression (fast single-arm → whole-body).
_DEFAULT_GROUP_ORDER = [
    "autolife_left_arm",
    "autolife_right_arm",
    "autolife_dual_arm",
    "autolife_torso_left_arm",
    "autolife_torso_right_arm",
    "autolife_torso_dual_arm",
    "autolife_leg_torso_dual_arm",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(group: str, result, t_ms: float) -> None:
    if result is None:
        status = "NO PATH (timeout)"
    elif isinstance(result, str):
        status = f"SKIPPED ({result})"
    else:
        status = f"OK  {result.shape[0]} waypoints"
    print(f"  [{group}]  {status}  ({t_ms:.0f} ms)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    group: str | None = None,
    planner: str = "bitstar",
    time_limit: float = 2.0,
    fps: float = 50.0,
) -> None:
    """Visualize planned paths for one or all supported planning groups.

    Args:
        group:      Name of a single group to visualize.  Omit to cycle
                    through all supported groups in sequence.
        planner:    OMPL planner name (rrtc, rrtstar, bitstar, …).
        time_limit: Planning timeout in seconds per group.
        fps:        Playback frame rate while auto-playing.
    """
    if group is not None:
        if group not in SUPPORTED_GROUPS:
            print(
                f"Unknown group {group!r}.\n"
                f"Supported: {', '.join(sorted(SUPPORTED_GROUPS))}"
            )
            sys.exit(1)
        groups = [group]
    else:
        groups = [g for g in _DEFAULT_GROUP_ORDER if g in SUPPORTED_GROUPS]

    env = PyBulletEnv(
        autolife_robot_config, visualize=True, viz_urdf_path=VIZ_URDF_PATH
    )

    print(
        f"\n── autolife path visualizer ──\n"
        f"  planner={planner}  time_limit={time_limit}s\n"
        f"  {len(groups)} group(s): {', '.join(groups)}\n"
    )

    ap = AutolifePlanner(planner_name=planner, time_limit=time_limit)
    start = _BASE + _LEGS + _WAIST + _L_ARM + _NECK + _R_ARM
    env.set_configuration(start)

    for g in groups:
        goal = _DEFAULT_GOALS[g]

        t0 = time.perf_counter()
        result = ap.plan_to_joints(g, start, goal)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        _banner(g, result, elapsed_ms)

        if isinstance(result, np.ndarray):
            _, path = ap.time_parameterize(result)
            cont = env.animate_path(path, fps=fps, next_key="n")
        else:
            env.set_configuration(start)
            msg = (
                "  (no path found — press 'n' for next group, close to quit)"
                if result is None
                else "  (skipped: frozen joints differ — press 'n' for next)"
            )
            env.wait_key("n", msg)
            cont = True

        if not cont:
            break

    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
