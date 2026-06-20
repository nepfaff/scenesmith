#!/bin/bash

# Non-interactive SAM3D installation for Docker builds.
# Differences from install_sam3d.sh:
# - No interactive prompts (auto-accept everything).
# - Skips CUDA detection/installation (already in base image).
# - Skips HuggingFace checkpoint download (mounted at runtime).
# - Keeps: repo cloning, dependency installation, CUDA package builds.

set -euo pipefail

SAM3D_OBJECTS_COMMIT="${SAM3D_OBJECTS_COMMIT:-81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd}"
SAM3_COMMIT="${SAM3_COMMIT:-11dec2936de97f2857c1f76b66d982d5a001155d}"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [ -x ".venv/bin/python" ]; then
    export VIRTUAL_ENV="$(pwd)/.venv"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

echo "========================================="
echo "SAM3D Docker Installation"
echo "========================================="
echo ""

# CUDA is pre-installed in the Docker image.
echo "Using CUDA_HOME: ${CUDA_HOME}"
nvcc --version

echo ""
echo "Step 1: Cloning repositories..."

mkdir -p external
cd external

# Clone SAM 3D Objects repository.
if [ ! -d "sam-3d-objects" ]; then
    git clone https://github.com/facebookresearch/sam-3d-objects.git
    echo "Cloned sam-3d-objects"
else
    echo "sam-3d-objects already exists"
fi
echo "Checking out SAM 3D Objects commit: ${SAM3D_OBJECTS_COMMIT}"
git -C sam-3d-objects fetch origin
git -C sam-3d-objects checkout --detach "${SAM3D_OBJECTS_COMMIT}"

# Clone SAM3 repository.
if [ ! -d "SAM3" ]; then
    git clone https://github.com/facebookresearch/sam3.git SAM3
    echo "Cloned SAM3"
else
    echo "SAM3 already exists"
fi
echo "Checking out SAM3 commit: ${SAM3_COMMIT}"
git -C SAM3 fetch origin
git -C SAM3 checkout --detach "${SAM3_COMMIT}"

echo ""
echo "Step 2: Installing SAM3..."
cd SAM3
# Install SAM3 without asking it to resolve dependencies, then install the
# runtime pieces SceneSmith uses explicitly. This keeps the main uv project in
# charge of core pins such as torch, torchvision, numpy, and Drake.
uv pip install -e . --no-deps
uv pip install \
    "iopath>=0.1.10" \
    "pycocotools>=2.0.7" \
    "decord>=0.6.0" \
    "scikit-learn>=1.4" \
    "ftfy>=6.1.1" \
    "timm>=1.0.17"
cd ..
echo "SAM3 installed"

echo ""
echo "Step 3: Installing SAM 3D Objects dependencies..."

cd sam-3d-objects

# Install the non-CUDA runtime dependencies used by SceneSmith's SAM3D path.
echo "Installing sam-3d-objects core dependencies..."
uv pip install \
    astor \
    easydict \
    einops-exts \
    fvcore \
    loguru \
    optree \
    roma \
    rootutils \
    OpenEXR \
    pymeshfix \
    igraph \
    "lightning==2.3.3" \
    plotly \
    plyfile \
    pyvista \
    psutil \
    "spconv-cu121==2.3.8" \
    "open3d>=0.19.0" \
    "numpy>=1.26,<2.0"

# Install CUDA-dependent packages with --no-build-isolation.
echo ""
echo "Installing gsplat..."
uv pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

echo ""
echo "Installing nvdiffrast..."
uv pip install --reinstall --no-cache --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"
uv pip install "numpy>=1.26,<2.0"

echo ""
echo "Pre-compiling nvdiffrast CUDA extensions..."
if "${PYTHON_BIN}" << 'PYEOF'
import sys

try:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available - pre-compilation will happen on first use")
        sys.exit(0)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print("Compiling nvdiffrast CUDA kernels...")

    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    print(f"SUCCESS: {type(ctx).__name__} initialized")

except Exception as e:
    print(f"Pre-compilation failed: {e}")
    print("NOTE: nvdiffrast may need to be rebuilt for this Torch/CUDA environment")
    sys.exit(1)
PYEOF
then
    echo "nvdiffrast pre-compiled successfully"
else
    echo "nvdiffrast pre-compilation skipped"
fi

echo ""
echo "Installing kaolin 0.17.0..."
uv pip install --no-build-isolation \
    "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0"

echo ""
echo "Installing pytorch3d from source..."
uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

echo ""
echo "Installing inference dependencies..."
uv pip install seaborn==0.13.2 gradio==5.49.0 imageio
uv pip install utils3d --no-deps
uv pip install "numpy>=1.26,<2.0"

echo ""
echo "Installing MoGe depth model..."
uv pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

cd ..

echo ""
echo "========================================="
echo "SAM3D Docker Installation Complete!"
echo "========================================="
echo ""
echo "Checkpoints must be mounted at runtime:"
echo "  -v ./external/checkpoints:/app/external/checkpoints"
echo ""
