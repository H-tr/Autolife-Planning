#!/usr/bin/env bash
# Build planning-ready robot description from v2.0 URDF.
# All project-specific paths live here; the Python script is a pure tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/build_robot_description.py" \
    --urdf "$ROOT/assets/autolife_description/urdf/robot_v2_0.urdf" \
    --mesh-dir "$ROOT/assets/autolife_description/meshes/robot_v2_0" \
    --output-dir "$ROOT/resources/robot/autolife" \
    --repair-meshes \
    --distribute-to \
        "$ROOT/third_party/cricket/resources/autolife" \
        "$ROOT/third_party/vamp/resources/autolife" \
    "$@"
