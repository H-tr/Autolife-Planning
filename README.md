# Autolife-Planning

A planning library for the Autolife robot. It integrates motion planning (VAMP), inverse kinematics (Cricket), and collision-aware planning (Foam) through a unified Python interface managed by [pixi](https://pixi.sh).

## Prerequisites

- Linux (x86_64)
- [pixi](https://pixi.sh) package manager

## Quick Start

```bash
git clone --recursive https://github.com/H-tr/Autolife-Planning.git
cd Autolife-Planning
bash scripts/setup.sh
```

This will install pixi (if needed), set up the environment, build the C++ dependencies, and download assets.

## Manual Installation

If you prefer to set things up step by step:

```bash
# Clone with submodules
git clone --recursive https://github.com/H-tr/Autolife-Planning.git
cd Autolife-Planning

# Install the pixi environment (Python, C++ toolchain, and all dependencies)
pixi install

# Build C++ third-party libraries
pixi run cricket-build
pixi run foam-build

# Download robot assets
bash scripts/download_assets.sh
```

## Usage

Run examples inside the pixi environment:

```bash
pixi run python examples/random_dance_around_table.py
pixi run python examples/ik_solver_example.py
```

## Project Structure

```
autolife_planning/   # Core Python package
third_party/
  cricket/           # Inverse kinematics library
  foam/              # Collision-aware planning
  vamp/              # Motion planning (installed as editable PyPI dep)
scripts/             # Setup and utility scripts
examples/            # Example scripts
resources/           # Robot URDF and mesh files
```
