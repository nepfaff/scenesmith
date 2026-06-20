# Multi-stage Dockerfile for scenesmith.
# Single-container setup: all servers (geometry generation, retrieval, blender,
# etc.) are auto-managed by the pipeline inside the container.

# =============================================================================
# Stage 1: Base system with Python 3.12/3.13 and system dependencies.
# =============================================================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 for SceneSmith and Python 3.13 for the bpy 5.x server.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libpython3.12-dev \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    libpython3.13-dev \
    git \
    git-lfs \
    wget \
    unzip \
    bubblewrap \
    cmake \
    build-essential \
    # X11/EGL libs for headless Blender rendering.
    libgl1 \
    libegl1 \
    libxrender1 \
    libxkbcommon0 \
    libsm6 \
    libxext6 \
    libxi6 \
    libxxf86vm1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default SceneSmith runtime.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv package manager.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set environment variables.
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /app

# =============================================================================
# Stage 2: Python dependencies.
# =============================================================================
FROM base AS deps

COPY pyproject.toml uv.lock .python-version README.md ./
COPY scripts/blender_server/pyproject.toml scripts/blender_server/uv.lock scripts/blender_server/
RUN uv sync --frozen --no-dev \
    && UV_PROJECT_ENVIRONMENT=/app/.venv-blender \
        uv sync \
        --project scripts/blender_server \
        --python /usr/bin/python3.13 \
        --frozen \
        --no-dev

# =============================================================================
# Stage 3: SAM3D backend (optional but included in image).
# =============================================================================
FROM deps AS sam3d

COPY scripts/install_sam3d_docker.sh scripts/install_sam3d_docker.sh

# Disable uv project config to avoid hitting the Blender PyPI index
# (from pyproject.toml) which rate-limits during Docker builds.
# Set TORCH_CUDA_ARCH_LIST since no GPU is available during build.
# Covers Ampere (A100, A10), Ada Lovelace (L40S, RTX 4090), Hopper (H100).
RUN UV_NO_CONFIG=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    bash scripts/install_sam3d_docker.sh

# =============================================================================
# Stage 4: Application code.
# =============================================================================
FROM sam3d AS app

# Copy full repo source.
COPY . .

# Remove stale bytecode from earlier stages (SAM3D install imports
# scenesmith, creating .pyc that would shadow updated .py files).
RUN find /app/scenesmith -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

# Activate the virtual environment by prepending it to PATH.
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: print usage help.
CMD ["python", "-c", \
    "print('scenesmith Docker container\\n')\n\
print('Usage examples:\\n')\n\
print('  # Smoke test')\n\
print('  docker run --gpus all scenesmith python -c \"import torch; print(torch.cuda.is_available()); import scenesmith\"\\n')\n\
print('  # Run unit tests')\n\
print('  docker run --gpus all scenesmith pytest tests/unit/ -x\\n')\n\
print('  # Run scene generation (requires data volumes and API keys)')\n\
print('  docker compose up\\n')\n\
print('See README.md for full documentation.')"]
