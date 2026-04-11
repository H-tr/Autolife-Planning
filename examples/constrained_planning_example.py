"""Constrained motion planning demo: gripper on a surface + obstacle sphere.

A curobo-style "end effector on a manifold around an obstacle" demo
running on OMPL's ProjectedStateSpace:

* The **left** gripper is constrained to slide across an imaginary
  horizontal plane (1 holonomic equation:  z = z0), so the gripper
  acts like a finger sliding across a frictionless tabletop.
* A solid obstacle sphere sits in the natural sweep of the left
  arm, so the planner has to bend the arm around it without ever
  letting the gripper leave the plane.

The whole pipeline is symbolic + JIT:

* CasADi autodiffs the FK-based residual + Jacobian.
* CasADi codegens the constraint to C, ``c++ -O3`` compiles it,
  and the planner mmaps the resulting ``.so`` for callbacks.
* OMPL's RRTConnect runs on the projected manifold while
  VAMP's SIMD collision checker validates each sampled state.

A second IPOPT pass solves a constrained IK for the *goal*
configuration so it lives on the manifold and inside the joint
bounds without any hand tuning of joint values.

First run compiles the constraint to
``~/.cache/autolife_planning/constraints/``; subsequent runs are
a fast cache hit.

Note on scope: a fully dual-arm variant (left-gripper-on-plane +
right-gripper-on-line at the same time) is tempting but the
combined 3-equation manifold is extremely narrow on this robot's
HOME pose — OMPL's projector can't reliably traverse it.  This
demo therefore keeps the constraint on a single end effector,
which is more than enough to showcase the plumbing: swap
``autolife_left_arm`` for any other subgroup and the constraint
code is unchanged.

    pixi run python examples/constrained_planning_example.py
"""

import casadi as ca
import numpy as np
import pybullet as pb
from fire import Fire

from autolife_planning.config.robot_config import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import (
    Constraint,
    SymbolicContext,
    create_planner,
)
from autolife_planning.types import PlannerConfig

# ── obstacle helpers ───────────────────────────────────────────────────


def sample_ball(
    center: np.ndarray, radius: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample *n* points uniformly inside a solid ball.

    The obstacle is handed to VAMP as a point cloud — volumetric
    sampling (rather than just the surface) leaves no interior
    pocket a fast motion can slip through.
    """
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = radius * rng.random(n) ** (1.0 / 3.0)
    return (dirs * r[:, None] + center).astype(np.float32)


def add_sphere_visual(
    env: PyBulletEnv,
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float] = (0.95, 0.30, 0.30, 0.55),
) -> int:
    """Drop a translucent visual-only sphere into the PyBullet scene."""
    client = env.sim.client
    vis_id = client.createVisualShape(
        shapeType=pb.GEOM_SPHERE, radius=radius, rgbaColor=list(color)
    )
    return client.createMultiBody(
        baseVisualShapeIndex=vis_id,
        basePosition=center.tolist(),
        baseMass=0.0,
    )


# ── manifold projection for the goal configuration ────────────────────


def project_onto_manifold(
    q_sym: ca.SX,
    residual: ca.SX,
    q_init: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Project *q_init* onto the constraint manifold via IPOPT.

    Solves::

        min_q  ‖q - q_init‖²
        s.t.   residual(q) = 0
               lower ≤ q ≤ upper

    Using the same CasADi residual the planner sees, so the returned
    config is guaranteed to satisfy the OMPL tolerance check.  Joint
    bounds are respected, which a naive Newton iteration might
    otherwise blow past.
    """
    cost = ca.dot(q_sym - ca.DM(q_init.tolist()), q_sym - ca.DM(q_init.tolist()))
    nlp = {"x": q_sym, "f": cost, "g": residual}
    solver = ca.nlpsol(
        "project",
        "ipopt",
        nlp,
        {
            "print_time": False,
            "ipopt": {"print_level": 0, "sb": "yes", "tol": 1e-10},
        },
    )
    sol = solver(
        x0=q_init.tolist(),
        lbx=lower.tolist(),
        ubx=upper.tolist(),
        lbg=0.0,
        ubg=0.0,
    )
    if not solver.stats().get("success", False):
        raise RuntimeError(
            f"Manifold projection failed: "
            f"{solver.stats().get('return_status', 'unknown')}"
        )
    return np.asarray(sol["x"]).flatten()


# ── main ───────────────────────────────────────────────────────────────

# A known-good goal joint vector for the left-arm subgroup, landing
# with the gripper far to the robot's front-right on the target plane.
# This seed was discovered by random search; we project it onto the
# constraint manifold at runtime so the planner sees an exactly
# feasible goal regardless of the hardcoded digits.
_GOAL_SEED_LEFT_ARM = np.array(
    [-1.2608, 0.0682, -1.0594, 0.0971, -2.4225, 0.4973, -0.2851]
)


def main(planner_name: str = "rrtc", time_limit: float = 10.0):
    env = PyBulletEnv(autolife_robot_config, visualize=True)

    # ── Symbolic context for the left-arm subgroup (7 DOF) ─────────────
    ctx = SymbolicContext("autolife_left_arm")
    q = ctx.q
    start = HOME_JOINTS[ctx.active_indices].copy()

    # Lock the manifold plane to the home gripper height, so the start
    # configuration is trivially feasible with no projection needed.
    left_home = ctx.evaluate_link_pose("Link_Left_Gripper", start)[:3, 3]
    z_surface = float(left_home[2])

    print("manifold target")
    print(f"  left gripper plane :  z = {z_surface:+.3f} m")

    # ── Single holonomic constraint: left gripper on a horizontal plane
    left_pos = ctx.link_translation("Link_Left_Gripper")
    left_plane = Constraint(
        residual=left_pos[2] - z_surface,
        q_sym=q,
        name="left_gripper_on_plane",
    )

    # ── Joint bounds (for the bounded projection below) ────────────────
    probe = create_planner(
        "autolife_left_arm", config=PlannerConfig(time_limit=time_limit)
    )
    lower = np.array(probe._planner.lower_bounds()) + 0.02
    upper = np.array(probe._planner.upper_bounds()) - 0.02

    # ── Project the hardcoded goal seed onto the manifold ──────────────
    goal = project_onto_manifold(
        q, left_plane.residual, _GOAL_SEED_LEFT_ARM, lower, upper
    )
    left_goal = ctx.evaluate_link_pose("Link_Left_Gripper", goal)[:3, 3]
    res_fn = ca.Function("res", [q], [ca.reshape(left_plane.residual, -1, 1)])
    goal_res = float(np.linalg.norm(np.asarray(res_fn(goal)).flatten()))

    print("gripper positions")
    print(
        f"  start : ({left_home[0]:+.3f}, {left_home[1]:+.3f}, " f"{left_home[2]:+.3f})"
    )
    print(
        f"  goal  : ({left_goal[0]:+.3f}, {left_goal[1]:+.3f}, " f"{left_goal[2]:+.3f})"
    )
    print(f"  goal manifold residual: {goal_res:.2e}")

    # ── Obstacle sphere planted in the swept arc ───────────────────────
    # Centred slightly below the plane so the sphere's equator sits
    # right at the gripper's height — the arm has to curve around it.
    sphere_center = np.array(
        [
            float(0.5 * (left_home[0] + left_goal[0])),
            float(0.5 * (left_home[1] + left_goal[1])),
            z_surface - 0.03,
        ]
    )
    sphere_radius = 0.09
    add_sphere_visual(env, sphere_center, sphere_radius)
    cloud = sample_ball(
        sphere_center, sphere_radius, n=800, rng=np.random.default_rng(0)
    )
    print(
        f"obstacle sphere: center=({sphere_center[0]:+.3f}, "
        f"{sphere_center[1]:+.3f}, {sphere_center[2]:+.3f}) "
        f"radius={sphere_radius:.3f} m"
    )

    # ── Build the constrained planner and solve ────────────────────────
    config = PlannerConfig(
        planner_name=planner_name,
        time_limit=time_limit,
        point_radius=0.012,
    )
    planner = create_planner(
        "autolife_left_arm",
        config=config,
        pointcloud=cloud,
        constraints=[left_plane],
    )

    print(
        f"planning {planner_name} on a {planner.num_dof}-DOF left arm "
        f"with 1 manifold equation + 1 sphere obstacle ..."
    )
    result = planner.plan(start, goal)
    n_wp = result.path.shape[0] if result.path is not None else 0
    print(f"  status         : {result.status.value}")
    print(f"  waypoints      : {n_wp}")
    print(f"  planning time  : {result.planning_time_ns / 1e6:.1f} ms")
    print(f"  path cost      : {result.path_cost:.3f}")

    if result.success and result.path is not None:
        env.animate_path(planner.embed_path(result.path), fps=60)
    env.wait_key("q", "press 'q' to quit")


if __name__ == "__main__":
    Fire(main)
