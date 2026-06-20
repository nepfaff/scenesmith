#!/bin/bash

# Installation script for SAM3D backend.
# Works with any CUDA 12.x installation (system-wide, conda, or custom).
# See https://github.com/facebookresearch/sam-3d-objects for more information.

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
echo "SAM3D Installation Script"
echo "========================================="
echo ""

# Check for Python development headers (required for nvdiffrast JIT compilation).
echo "Step 0: Checking system dependencies..."

PYTHON_VERSION=$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_HEADER=$("${PYTHON_BIN}" - <<'PY'
import sysconfig

print(sysconfig.get_config_h_filename() or "")
PY
)

if [ -z "$PYTHON_HEADER" ] || [ ! -f "$PYTHON_HEADER" ]; then
    echo "⚠️  Python development headers not found at $PYTHON_HEADER"
    echo "   These are required for nvdiffrast JIT compilation (texture baking)."
    echo ""
    read -p "Install libpython${PYTHON_VERSION}-dev? (requires sudo) [Y/n]: " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Installing Python development headers..."
        sudo apt-get update && sudo apt-get install -y libpython${PYTHON_VERSION}-dev
        echo "✓ Installed libpython${PYTHON_VERSION}-dev"
    else
        echo "⚠️  Skipping. nvdiffrast texture baking may fail without Python headers."
        echo "   Install manually: sudo apt-get install libpython${PYTHON_VERSION}-dev"
    fi
else
    echo "✓ Python development headers found at $PYTHON_HEADER"
fi

echo ""

# Auto-detect and validate CUDA installation.
echo "Step 1: Detecting CUDA installation..."

# First check if nvcc is in PATH.
if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    FOUND_IN_PATH=true
else
    # Check common CUDA installation locations.
    FOUND_IN_PATH=false
    for cuda_path in /usr/local/cuda-12.4 /usr/local/cuda-12.* /usr/local/cuda ~/miniforge3 ~/miniconda3 ~/anaconda3; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            echo "✓ Found CUDA installation at $cuda_path"
            NVCC_PATH="$cuda_path/bin/nvcc"
            export PATH="$cuda_path/bin:$PATH"
            FOUND_IN_PATH=true
            break
        fi
    done
fi

if [ "$FOUND_IN_PATH" = true ]; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+")
    echo "✓ Found CUDA $CUDA_VERSION"

    # Verify CUDA 12.x.
    if [[ ! "$CUDA_VERSION" =~ ^12\. ]]; then
        echo "✗ Error: CUDA $CUDA_VERSION found, but SAM3D requires CUDA 12.x"
        echo ""
        echo "Please install CUDA 12.x:"
        echo "  - System-wide: https://developer.nvidia.com/cuda-downloads"
        echo "  - Conda: conda install cuda-toolkit=12.4 -c nvidia"
        exit 1
    fi

    # Auto-detect CUDA_HOME from nvcc location.
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    echo "✓ Using CUDA_HOME: $CUDA_HOME"

    # Set LD_LIBRARY_PATH (handle both lib64 and lib for conda compatibility).
    if [ -d "$CUDA_HOME/lib64" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "✓ Added $CUDA_HOME/lib64 to LD_LIBRARY_PATH"
    elif [ -d "$CUDA_HOME/lib" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "✓ Added $CUDA_HOME/lib to LD_LIBRARY_PATH"
    fi

else
    echo "✗ nvcc not found in PATH"
    echo ""
    echo "SAM3D requires CUDA 12.x toolkit to build dependencies (pytorch3d, gsplat, etc.)"
    echo ""
    read -p "Install CUDA 12.4 system-wide? [y/N]: " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading CUDA 12.4 installer..."
        CUDA_INSTALLER="cuda_12.4.0_550.54.14_linux.run"
        CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/${CUDA_INSTALLER}"

        if ! wget -q --show-progress "$CUDA_URL"; then
            echo "✗ Download failed"
            exit 1
        fi

        echo ""
        echo "Installing CUDA 12.4 toolkit (requires sudo)..."
        echo "This will install to /usr/local/cuda-12.4"
        echo ""

        if sudo sh "$CUDA_INSTALLER" --silent --toolkit; then
            echo ""
            echo "✓ CUDA 12.4 installed successfully"

            # Set environment variables for this session.
            export CUDA_HOME="/usr/local/cuda-12.4"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Verify installation.
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+")
                echo "✓ Verified: nvcc $CUDA_VERSION available"
            else
                echo "✗ Installation verification failed"
                exit 1
            fi

            # Clean up installer.
            rm "$CUDA_INSTALLER"
            echo "✓ Cleaned up installer"
        else
            echo "✗ Installation failed"
            rm -f "$CUDA_INSTALLER"
            exit 1
        fi

    else
        echo ""
        echo "Manual installation options:"
        echo "  1. System-wide: https://developer.nvidia.com/cuda-downloads"
        echo "  2. Conda: conda install cuda-toolkit=12.4 -c nvidia"
        echo ""
        echo "After installation, ensure 'nvcc' is in your PATH and re-run this script."
        exit 1
    fi
fi

echo ""
echo "Step 2: Cloning repositories..."

# Create external directory if it doesn't exist.
mkdir -p external
cd external

# Clone SAM 3D Objects repository.
if [ ! -d "sam-3d-objects" ]; then
    echo "Cloning SAM 3D Objects repository..."
    git clone https://github.com/facebookresearch/sam-3d-objects.git
    echo "✓ Cloned sam-3d-objects"
else
    echo "✓ sam-3d-objects already exists"
fi
echo "Checking out SAM 3D Objects commit: ${SAM3D_OBJECTS_COMMIT}"
git -C sam-3d-objects fetch origin
git -C sam-3d-objects checkout --detach "${SAM3D_OBJECTS_COMMIT}"

# Clone SAM3 repository.
if [ ! -d "SAM3" ]; then
    echo "Cloning SAM3 repository..."
    git clone https://github.com/facebookresearch/sam3.git SAM3
    echo "✓ Cloned SAM3"
else
    echo "✓ SAM3 already exists"
fi
echo "Checking out SAM3 commit: ${SAM3_COMMIT}"
git -C SAM3 fetch origin
git -C SAM3 checkout --detach "${SAM3_COMMIT}"

echo ""
echo "Step 3: Installing SAM3..."
cd SAM3

# Install SAM3 without asking it to resolve dependencies, then install the
# runtime pieces SceneSmith uses explicitly. This keeps the main uv project in
# charge of core pins such as torch, torchvision, numpy, and Drake.
echo "Installing SAM3 with inference dependencies..."
uv pip install -e . --no-deps
uv pip install \
    "iopath>=0.1.10" \
    "pycocotools>=2.0.7" \
    "decord>=0.6.0" \
    "scikit-learn>=1.4" \
    "ftfy>=6.1.1" \
    "timm>=1.0.17"
cd ..
echo "✓ SAM3 installed"

echo ""
echo "Step 4: Installing SAM 3D Objects dependencies..."
echo "This will install dependencies from requirements.txt and build CUDA packages."
echo "This may take 10-20 minutes..."
echo ""

cd sam-3d-objects

# Install the non-CUDA runtime dependencies used by SceneSmith's SAM3D path.
# The upstream requirements file contains many training, notebook, and cloud
# packages with old native pins; installing it wholesale is fragile across
# Python versions.
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

# Now install CUDA-dependent packages with --no-build-isolation.
echo ""
echo "Installing gsplat (requires PyTorch at build time)..."
uv pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

echo ""
echo "Installing nvdiffrast (requires CUDA)..."
uv pip install --reinstall --no-cache --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"
uv pip install "numpy>=1.26,<2.0"

echo ""
echo "Pre-compiling nvdiffrast CUDA extensions..."
echo "(This triggers PyTorch JIT compilation - may take 1-2 minutes)"

# Pre-compilation script - ensures nvdiffrast is ready to use.
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
    echo "✓ nvdiffrast pre-compiled successfully"
else
    echo "⚠️  nvdiffrast pre-compilation skipped (will compile on first use)"
fi

echo ""
echo "Installing kaolin 0.17.0 (requires CUDA, building from source)..."
uv pip install --no-build-isolation \
    "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0"

echo ""
echo "Installing pytorch3d from source..."
uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# Install inference-specific requirements.
echo ""
echo "Installing inference dependencies..."
uv pip install seaborn==0.13.2 gradio==5.49.0 imageio
uv pip install utils3d --no-deps
uv pip install "numpy>=1.26,<2.0"

# Install MoGe (depth model used by SAM 3D Objects).
echo ""
echo "Installing MoGe depth model..."
uv pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

cd ..

echo ""
echo "✓ All dependencies installed"

echo ""
echo "Step 5: Downloading model checkpoints..."
echo ""

if [ -f "checkpoints/sam3.pt" ] && [ -f "checkpoints/pipeline.yaml" ]; then
    echo "✓ SAM3D checkpoints already exist; skipping HuggingFace download"
    cd ..
    echo ""
    echo "========================================="
    echo "SAM3D Installation Complete!"
    echo "========================================="
    echo ""
    echo "Checkpoints located in: external/checkpoints/"
    echo "  SAM3: external/checkpoints/sam3.pt"
    echo "  SAM 3D Objects: external/checkpoints/*.{ckpt,pt,yaml}"
    echo ""
    exit 0
fi

echo "⚠️  Important: HuggingFace authentication required!"
echo "    1. Request access: https://huggingface.co/facebook/sam3"
echo "    2. Request access: https://huggingface.co/facebook/sam-3d-objects"
echo "    3. Login: hf auth login (or huggingface-cli login)"
echo ""
read -p "Have you requested access and logged in? [y/N]: " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please complete authentication steps above and re-run the script."
    exit 1
fi

# Create checkpoints directory.
mkdir -p checkpoints

# Download SAM3 checkpoint.
if [ ! -f "checkpoints/sam3.pt" ]; then
    echo "Downloading SAM3 checkpoint (sam3.pt)..."
    hf download facebook/sam3 sam3.pt --local-dir checkpoints
    echo "✓ Downloaded sam3.pt"
else
    echo "✓ sam3.pt already exists"
fi

# Download SAM 3D Objects checkpoints (entire checkpoints folder).
if [ ! -f "checkpoints/.sam3d_objects_downloaded" ]; then
    echo "Downloading SAM 3D Objects checkpoints..."
    hf download facebook/sam-3d-objects \
        --repo-type model \
        --local-dir checkpoints/sam-3d-objects-download \
        --include "checkpoints/*"

    # Move checkpoints to correct location.
    mv checkpoints/sam-3d-objects-download/checkpoints/* checkpoints/
    rm -rf checkpoints/sam-3d-objects-download
    touch checkpoints/.sam3d_objects_downloaded
    echo "✓ Downloaded SAM 3D Objects checkpoints"
else
    echo "✓ SAM 3D Objects checkpoints already exist"
fi

cd ..

echo ""
echo "========================================="
echo "SAM3D Installation Complete!"
echo "========================================="
echo ""
echo "Checkpoints located in: external/checkpoints/"
echo "  SAM3: external/checkpoints/sam3.pt"
echo "  SAM 3D Objects: external/checkpoints/*.{ckpt,pt,yaml}"
echo ""
echo "To use SAM3D backend, update your config:"
echo "  asset_manager:"
echo "    backend: \"sam3d\""
echo "    sam3d:"
echo "      sam3_checkpoint: \"external/checkpoints/sam3.pt\""
echo "      sam3d_checkpoint: \"external/checkpoints/pipeline.yaml\""
echo ""
echo "Note: SAM 3D Objects uses pipeline.yaml which references other checkpoints."
echo ""
