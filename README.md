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

## Building the Robot Description

When the robot URDF changes, rebuild the full pipeline from URDF to FK code:

```bash
pixi run generate-fk
```

This runs the full chain automatically:

1. **foam-build** — Build the foam collision library
2. **build-robot** — Process raw URDF into planning-ready descriptions (simple, base, SRDF)
3. **spherize-robot** — Generate spherized URDF for collision checking
4. **cricket-build** — Build the cricket FK code generator
5. **generate-fk** — Generate FK C++ code and install it into vamp

After generating, rebuild vamp to pick up the new FK:

```bash
pixi install
```

You can also run individual steps:

```bash
pixi run build-robot          # Steps 1-2 only
pixi run spherize-robot       # Steps 1-3
pixi run generate-fk          # Steps 5
```

## Usage

Run examples inside the pixi environment:

```bash
pixi run python examples/ik_example.py
pixi run python examples/planning_example.py

# With PyBullet visualization (requires dev environment)
pixi run -e dev python examples/ik_example_vis.py
pixi run -e dev python examples/random_dance_around_table.py
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
