#!/bin/bash
# Setup script for MuJoCo export with USD support.
#
# Creates a separate virtual environment without bpy (Blender) to avoid
# the pxr (OpenUSD) library conflict between bpy and mujoco-usd-converter.
#
# Usage:
#   ./scripts/setup_mujoco_export.sh
#   source .mujoco_venv/bin/activate
#   python scripts/export_scene_to_mujoco.py <scene_dir> -o <output_dir> --usd

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.mujoco_venv"

echo "Creating virtual environment at $VENV_DIR..."
uv venv --python 3.12 "$VENV_DIR"

echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"
uv pip install --prerelease=allow -r "$SCRIPT_DIR/requirements-mujoco-export.txt"

echo "Installing scenesmith (without bpy)..."
uv pip install --no-deps -e "$PROJECT_DIR"

echo ""
echo "Setup complete!"
echo ""
echo "To use the MuJoCo export with USD support:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Export a generated scene:"
echo "  python scripts/export_scene_to_mujoco.py outputs/.../scene_000 -o mujoco_export --usd"
echo ""
echo "Export a standalone SDF model:"
echo "  python scripts/export_scene_to_mujoco.py --sdf /path/to/robot.sdf -o mujoco_export --usd"
