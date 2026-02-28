#!/usr/bin/env bash
# Spherize the simple URDF using foam.
# All project-specific paths live here; foam's script is the tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

python "$ROOT/third_party/foam/scripts/generate_sphere_urdf.py" \
    "$ROOT/resources/robot/autolife/autolife_base_simple.urdf" \
    --output "$ROOT/resources/robot/autolife/autolife_spherized.urdf" \
    --database "$ROOT/third_party/foam/sphere_database.json" \
    "$@"
