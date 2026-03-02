#!/usr/bin/env bash
# Generate FK/collision-checking C++ code for vamp using cricket.
# All project-specific paths live here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CRICKET="$ROOT/third_party/cricket"
VAMP="$ROOT/third_party/vamp"
RESOURCES="$ROOT/resources/robot/autolife"

# 1. Distribute existing robot description to cricket + vamp resources
echo "[1/3] Distributing robot description..."
for DEST in "$CRICKET/resources/autolife" "$VAMP/resources/autolife"; do
    cp "$RESOURCES/autolife_spherized.urdf" "$DEST/autolife_spherized.urdf"
    cp "$RESOURCES/autolife.srdf" "$DEST/autolife.srdf"
    echo "  copied to $DEST"
done

# 2. Run cricket FK code generation
echo "[2/3] Generating FK code..."
"$CRICKET/build/fkcc_gen" "$CRICKET/resources/autolife.json"

# 3. Copy generated header to vamp
echo "[3/3] Installing FK header into vamp..."
cp "autolife_fk.hh" \
   "$VAMP/src/impl/vamp/robots/autolife.hh"
echo "  installed autolife.hh"

echo "Done! Rebuild vamp to use the new FK code."
