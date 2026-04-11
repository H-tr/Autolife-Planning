# Autolife Planning

A planning library for the **Autolife robot** — inverse kinematics
(TRAC-IK and Pink QP), motion planning (OMPL front end, VAMP SIMD
collision backend), and task-space constrained planning, all behind
a unified Python API.

<video controls autoplay loop muted playsinline width="100%">
  <source src="assets/constrained_plane_obstacle.mp4" type="video/mp4">
</video>

## Features

- **Inverse Kinematics** — TRAC-IK (unconstrained) and Pink (QP-based)
  solvers. Pink composes primary end-effector tasks with secondary
  objectives: centre-of-mass stability, camera-frame stabilization,
  self-collision avoidance.
- **Motion Planning** — OMPL planner family (RRT*, BIT*, AIT*, EIT*,
  PRM, KPIECE, …) with a VAMP-accelerated SIMD collision checker;
  first-class support for point-cloud obstacles.
- **Subgroup Planning** — Plan over any slice of the robot (single
  arm, dual arm, torso + arm, height chain, mobile base, whole body)
  while the remaining joints stay pinned to a user-supplied pose.
- **Constrained Planning** — Hard task-space equality constraints
  written as CasADi expressions (planes, rails, couplings,
  orientation locks) running on OMPL's ``ProjectedStateSpace``.
- **Rotation Utilities** — Conversions between quaternion, RPY,
  axis-angle, and rotation matrices.

## Quick Install

Pre-built wheels are available for **Python 3.10–3.12** on Linux x86_64. No local compilation required:

```bash
pip install autolife-planning
```

See the [Getting Started](getting-started.md) guide for full details.
