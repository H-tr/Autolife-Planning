#!/usr/bin/env bash
# Generate FK/collision-checking C++ code for vamp using cricket.
# Only generates the full-body autolife and autolife_body_coupled models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CRICKET="$ROOT/third_party/cricket"
VAMP="$ROOT/third_party/vamp"
RESOURCES="$ROOT/resources/robot/autolife"

ROBOTS=(autolife autolife_body_coupled)

# 1. Distribute robot descriptions to cricket + vamp resources
echo "[1/3] Distributing robot descriptions..."
for DEST in "$CRICKET/resources/autolife" "$VAMP/resources/autolife"; do
    cp "$RESOURCES/autolife_spherized.urdf" "$DEST/autolife_spherized.urdf"
    cp "$RESOURCES/autolife.srdf" "$DEST/autolife.srdf"
    if [ -f "$RESOURCES/autolife_body_coupled_spherized.urdf" ]; then
        cp "$RESOURCES/autolife_body_coupled_spherized.urdf" "$DEST/autolife_body_coupled_spherized.urdf"
    fi
    echo "  copied to $DEST"
done

# 2. Run cricket FK code generation
echo "[2/3] Generating FK code..."
for name in "${ROBOTS[@]}"; do
    CONFIG="$CRICKET/resources/${name}.json"
    if [ -f "$CONFIG" ]; then
        echo "  generating FK for $name..."
        "$CRICKET/build/fkcc_gen" "$CONFIG"
    fi
done

# 3. Copy generated headers to vamp
echo "[3/3] Installing FK headers into vamp..."
for name in "${ROBOTS[@]}"; do
    HEADER="${name}_fk.hh"
    if [ -f "$HEADER" ]; then
        cp "$HEADER" "$VAMP/src/impl/vamp/robots/${name}.hh"
        echo "  installed ${name}.hh"
    fi
done

echo "Done! Rebuild vamp to use the new FK code."
