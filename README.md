# Autolife-Planning

<div align="center">

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=for-the-badge&logo=materialformkdocs)](https://h-tr.github.io/Autolife-Planning/)
[![Build Docs](https://img.shields.io/github/actions/workflow/status/H-tr/Autolife-Planning/docs.yml?branch=main&style=for-the-badge&label=docs%20build&logo=github)](https://github.com/H-tr/Autolife-Planning/actions/workflows/docs.yml)
[![CI](https://img.shields.io/github/actions/workflow/status/H-tr/Autolife-Planning/ci.yml?branch=main&style=for-the-badge&label=CI&logo=github)](https://github.com/H-tr/Autolife-Planning/actions/workflows/ci.yml)

**[Documentation](https://h-tr.github.io/Autolife-Planning/)** | **[API Reference](https://h-tr.github.io/Autolife-Planning/api/kinematics/)** | **[Examples](https://h-tr.github.io/Autolife-Planning/examples/)**

https://github.com/H-tr/Autolife-Planning/raw/main/docs/assets/rls_pick_place.mp4

</div>

A planning library for the Autolife robot. It provides inverse kinematics (TRAC-IK and Pink), motion planning ([VAMP](https://github.com/KavrakiLab/vamp)), and collision-aware planning through a unified Python interface.

## Features

- **Inverse Kinematics** — TRAC-IK (unconstrained) and Pink (QP-based constrained) solvers with CoM stability, camera stabilization, and self-collision avoidance
- **Motion Planning** — VAMP-based planner with collision checking, path validation, and subgroup planning
- **Collision Geometry** — Spherized URDF representations for efficient collision detection, pointcloud obstacle support

## Quick Start

**Platform**: Linux, Python 3.11+ (see `pixi.toml`).

For inference — running the planners and IK solvers — just pip install:

```bash
git clone --recursive https://github.com/H-tr/Autolife-Planning.git
cd Autolife-Planning
pip install -e .
```

For development — rebuilding URDFs, regenerating FK headers, running the C++ toolchain end-to-end — use the setup script, which also installs pixi and the conda-forge deps (pinocchio, orocos-kdl, eigen, boost, ...):

```bash
bash scripts/setup.sh
```

See the [Getting Started](https://h-tr.github.io/Autolife-Planning/getting-started/) guide for detailed installation options.

## Usage

```bash
# Inverse kinematics
pixi run python examples/ik/basic.py
pixi run -e dev python examples/ik/basic_vis.py           # PyBullet visualization
pixi run -e dev python examples/ik/constrained_vis.py     # Pink QP with CoM stability

# Motion planning
pixi run python examples/planning/motion.py
pixi run python examples/planning/subgroup.py
pixi run -e dev python examples/planning/constrained/plane.py
pixi run -e dev python examples/planning/cost/orientation_lock.py

# End-to-end pick-and-place demo (the video above)
pixi run -e dev python examples/demos/rls_pick_place.py

# Tests
pixi run -e dev test
```

## Project Structure

```
autolife_planning/     # Core Python package
  kinematics/          # TRAC-IK + Pink IK, FK, collision checking
  planning/            # VAMP motion planning, cost + constrained planners
  envs/                # Simulation environments (PyBullet)
  types/               # Shared dataclasses (Pose, JointState, ...)
  resources/           # Packaged URDFs and asset loaders
third_party/
  cricket/             # FK code generator
  foam/                # Collision geometry processing
  vamp/                # Motion planning (installed as editable PyPI dep)
examples/
  ik/                  # TRAC-IK + Pink examples
  planning/            # Motion, subgroup, cost, constrained planning demos
  demos/               # End-to-end scenarios (rls_pick_place, ...)
tests/                 # Pytest suite (CI)
scripts/
  render_videos/       # Docs/demo video pipeline
  ...                  # Setup, build, spherize, FK codegen
resources/             # Robot URDF and mesh files
assets/                # Scene pointclouds and env meshes
docs/                  # MkDocs site sources (GitHub Pages)
```
