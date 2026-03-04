# Autolife Planning

A planning library for the **Autolife robot**. It integrates motion planning ([VAMP](https://github.com/KavrakiLab/vamp)), inverse kinematics (Cricket / TRAC-IK), and collision-aware planning (Foam) through a unified Python interface managed by [pixi](https://pixi.sh).

## Features

- **Inverse Kinematics** — TRAC-IK based solver with multiple solve strategies (speed, distance, manipulability)
- **Motion Planning** — VAMP-based planner with collision checking and path validation
- **Collision Geometry** — Spherized URDF representations for efficient collision detection
- **Rotation Utilities** — Conversions between quaternion, RPY, axis-angle, and rotation matrices
- **Environment Interface** — Abstract base with PyBullet simulation support

## Quick Start

```bash
git clone --recursive https://github.com/H-tr/Autolife-Planning.git
cd Autolife-Planning
bash scripts/setup.sh
```

This will install pixi (if needed), set up the environment, build the C++ dependencies, and download assets.

## Project Structure

```
autolife_planning/   # Core Python package
third_party/
  cricket/           # Inverse kinematics library
  foam/              # Collision-aware planning
  vamp/              # Motion planning (installed as editable PyPI dep)
scripts/             # Setup and utility scripts
examples/            # Example scripts
```
