# Getting Started

## Prerequisites

- **Linux** (x86_64)
- [pixi](https://pixi.sh) package manager

## Installation

### Quick Setup

```bash
git clone --recursive https://github.com/H-tr/Autolife-Planning.git
cd Autolife-Planning
bash scripts/setup.sh
```

### Manual Installation

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

When the robot URDF changes, rebuild the full pipeline:

```bash
pixi run generate-fk
```

This runs the full chain automatically:

1. **foam-build** — Build the foam collision library
2. **build-robot** — Process raw URDF into planning-ready descriptions
3. **spherize-robot** — Generate spherized URDF for collision checking
4. **generate-fk** — Generate FK C++ code and install it into VAMP

After generating, rebuild VAMP to pick up the new FK:

```bash
pixi install
```

You can also run individual steps:

```bash
pixi run build-robot          # Steps 1-2 only
pixi run spherize-robot       # Steps 1-3
pixi run generate-fk          # Full pipeline
```

## Running Examples

```bash
pixi run python examples/ik_example.py
pixi run python examples/planning_example.py

# With PyBullet visualization (requires dev environment)
pixi run -e dev python examples/ik_example_vis.py
pixi run -e dev python examples/random_dance_around_table.py
```
