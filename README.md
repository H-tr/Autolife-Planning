# Autolife-Planning

<div align="center">

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=for-the-badge&logo=materialformkdocs)](https://h-tr.github.io/Autolife-Planning/)
[![Build Docs](https://img.shields.io/github/actions/workflow/status/H-tr/Autolife-Planning/docs.yml?branch=main&style=for-the-badge&label=docs%20build&logo=github)](https://github.com/H-tr/Autolife-Planning/actions/workflows/docs.yml)

**[Documentation](https://h-tr.github.io/Autolife-Planning/)** | **[API Reference](https://h-tr.github.io/Autolife-Planning/api/kinematics/)** | **[Examples](https://h-tr.github.io/Autolife-Planning/examples/)**

</div>

A planning library for the Autolife robot. It provides inverse kinematics (TRAC-IK and Pink), motion planning ([VAMP](https://github.com/KavrakiLab/vamp)), and collision-aware planning through a unified Python interface.

## Features

- **Inverse Kinematics** — TRAC-IK (unconstrained) and Pink (QP-based constrained) solvers with CoM stability, camera stabilization, and self-collision avoidance
- **Motion Planning** — VAMP-based planner with collision checking, path validation, and subgroup planning
- **Collision Geometry** — Spherized URDF representations for efficient collision detection, pointcloud obstacle support

## Quick Start

```bash
pip install autolife-planning
```

For development with the full C++ toolchain:

```bash
git clone --recursive https://github.com/H-tr/Autolife-Planning.git
cd Autolife-Planning
bash scripts/setup.sh
```

See the [Getting Started](https://h-tr.github.io/Autolife-Planning/getting-started/) guide for detailed installation options.

## Usage

```bash
pixi run python examples/ik_example.py
pixi run python examples/planning_example.py

# Constrained IK with CoM stability (requires dev environment)
pixi run -e dev python examples/constrained_ik_example_vis.py

# With PyBullet visualization
pixi run -e dev python examples/ik_example_vis.py
```

## Project Structure

```
autolife_planning/   # Core Python package
third_party/
  cricket/           # FK code generator
  foam/              # Collision geometry processing
  vamp/              # Motion planning (installed as editable PyPI dep)
scripts/             # Setup and build scripts
examples/            # Example scripts
resources/           # Robot URDF and mesh files
```
