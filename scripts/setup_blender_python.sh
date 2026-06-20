#!/bin/bash

# Set up the separate Blender server environment.
#
# SceneSmith's main environment uses Python 3.12 for Drake and SAM3D/Open3D.
# Blender's bpy 5.1 wheels require Python 3.13, so the Blender HTTP server runs
# in this small side environment and imports SceneSmith code via PYTHONPATH.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BLENDER_PROJECT="${PROJECT_ROOT}/scripts/blender_server"
BLENDER_VENV="${SCENESMITH_BLENDER_VENV:-${PROJECT_ROOT}/.venv-blender}"
BLENDER_PYTHON="${SCENESMITH_BLENDER_PYTHON_VERSION:-3.13}"

UV_PROJECT_ENVIRONMENT="${BLENDER_VENV}" \
    uv sync \
    --project "${BLENDER_PROJECT}" \
    --python "${BLENDER_PYTHON}" \
    --frozen \
    --no-dev

"${BLENDER_VENV}/bin/python" - <<'PY'
import bpy
import importlib.metadata as metadata

print("bpy", bpy.app.version_string)
print("drake", metadata.version("drake"))
PY
