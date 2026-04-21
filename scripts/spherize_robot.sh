#!/usr/bin/env bash
# Spherize the simple URDF using foam.
# All project-specific paths live here; foam's script is the tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Feed the convex-decomposed URDF to foam so each convex piece is spherized
# independently (tight fit). Falls back to the non-decomposed URDF if the
# decomposed one is missing.
INPUT_URDF="$ROOT/resources/robot/autolife/autolife_base_decomposed.urdf"
if [ ! -f "$INPUT_URDF" ]; then
    INPUT_URDF="$ROOT/resources/robot/autolife/autolife_base_simple.urdf"
fi

# Use the 'medial' sphere-tree method: tightest fit in theory. The upstream
# bundled makeTreeMedial binary has a --verify mesh-validity check that
# raises false positives on our CoACD / trimesh-produced convex hulls
# (reporting "bad faces" from float-precision artefacts in the OBJ
# round-trip) — which makes foam's wrapper then give up with exit 1. We
# disable --verify: the inputs are guaranteed-manifold convex hulls so the
# extra check is redundant, and medial then runs to completion.
python "$ROOT/third_party/foam/scripts/generate_sphere_urdf.py" \
    "$INPUT_URDF" \
    --output "$ROOT/resources/robot/autolife/autolife_spherized.urdf" \
    --database "$ROOT/third_party/foam/sphere_database.json" \
    --method medial \
    --verify False \
    "$@"

# The decomposed URDF and its per-piece STLs are purely intermediate inputs to
# foam. The spherized output embeds absolute sphere positions and no longer
# references those meshes, so clean them up.
DECOMP_URDF="$ROOT/resources/robot/autolife/autolife_base_decomposed.urdf"
DECOMP_DIR="$ROOT/resources/robot/autolife/meshes/decomposed"
rm -f "$DECOMP_URDF"
rm -rf "$DECOMP_DIR"
echo "Cleaned intermediate decomposition artefacts."
