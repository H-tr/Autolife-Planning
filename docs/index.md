# Autolife Planning

A planning library for the **Autolife robot**. It provides inverse kinematics (TRAC-IK), motion planning ([VAMP](https://github.com/KavrakiLab/vamp)), and collision-aware planning through a unified Python interface.

## Features

- **Inverse Kinematics** — TRAC-IK (unconstrained) and Pink (constrained, QP-based) solvers with CoM stability, camera stabilization, and collision avoidance
- **Motion Planning** — VAMP-based planner with collision checking and path validation
- **Subgroup Planning** — Plan for individual subgroups of the robot (single arm, dual arm, torso+arm, whole body) with automatic joint mapping
- **Collision Geometry** — Spherized URDF representations for efficient collision detection
- **Rotation Utilities** — Conversions between quaternion, RPY, axis-angle, and rotation matrices

## Quick Install

Pre-built wheels are available for **Python 3.10–3.12** on Linux x86_64. No local compilation required:

```bash
pip install autolife-planning
```

See the [Getting Started](getting-started.md) guide for full details.
