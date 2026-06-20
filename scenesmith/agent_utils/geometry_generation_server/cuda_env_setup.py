"""CUDA environment setup for nvdiffrast JIT compilation.

This module MUST be imported before any code that uses nvdiffrast or
SAM 3D Objects. It configures environment variables needed for PyTorch
JIT compilation of CUDA kernels.

Usage:
    # At the top of any module that uses SAM3D (before other imports)
    from scenesmith.agent_utils.geometry_generation_server import cuda_env_setup
    cuda_env_setup.ensure_cuda_env()
"""

from __future__ import annotations

import logging
import os
import subprocess
import sysconfig

from glob import glob
from pathlib import Path

console_logger = logging.getLogger(__name__)

_cuda_env_configured = False


def detect_cuda_home() -> Path | None:
    """Auto-detect CUDA installation directory.

    Searches in order of priority:
    1. CUDA_HOME environment variable (user override)
    2. sam3d-objects conda environment (known working config for texture baking)
       The conda environment has bundled CUDA that's compatible with pip-installed
       Warp, which is critical for nvdiffrast JIT compilation.
    3. Common system CUDA installation paths (newest first)

    Returns:
        Path to CUDA installation, or None if not found.
    """
    # Priority 1: User-specified CUDA_HOME.
    if "CUDA_HOME" in os.environ:
        cuda_home = Path(os.environ["CUDA_HOME"])
        if (cuda_home / "bin" / "nvcc").exists():
            return cuda_home

    # Priority 2: sam3d-objects conda environment.
    # This environment has bundled CUDA toolkit that's compatible with the
    # pip-installed Warp package. Using this ensures nvdiffrast JIT compilation
    # works correctly. The demo_text_to_3d.py script uses this approach.
    conda_cuda_paths = [
        "~/miniforge3/envs/sam3d-objects",
    ]
    for conda_path in conda_cuda_paths:
        cuda_home = Path(conda_path).expanduser()
        # Check for CUDA headers (nvcc may not exist in conda env, but headers do).
        cuda_include = cuda_home / "targets" / "x86_64-linux" / "include"
        if cuda_include.exists() and (cuda_include / "cuda.h").exists():
            console_logger.info(
                f"Using conda environment CUDA at {cuda_home} "
                "(matches pip-installed Warp)"
            )
            return cuda_home
        # Also check standard include location.
        cuda_include_alt = cuda_home / "include"
        if cuda_include_alt.exists() and (cuda_include_alt / "cuda.h").exists():
            console_logger.info(
                f"Using conda environment CUDA at {cuda_home} "
                "(matches pip-installed Warp)"
            )
            return cuda_home

    # Priority 3: Search common system locations (newest CUDA 12.x first).
    search_patterns = [
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.*",
        "/usr/local/cuda",
        "/opt/cuda",
    ]

    for pattern in search_patterns:
        if "*" in pattern:
            # Glob pattern - sort descending to get newest first.
            matches = sorted(glob(pattern), reverse=True)
            for match in matches:
                cuda_home = Path(match)
                if (cuda_home / "bin" / "nvcc").exists():
                    return cuda_home
        else:
            cuda_home = Path(pattern)
            if cuda_home.exists() and (cuda_home / "bin" / "nvcc").exists():
                return cuda_home

    return None


def detect_gpu_compute_capability() -> str | None:
    """Detect GPU compute capability using nvidia-smi.

    Returns:
        Compute capability string (e.g., "8.9"), or None if detection fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse first GPU's compute capability (e.g., "8.9").
            compute_cap = result.stdout.strip().split("\n")[0]
            return compute_cap
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def ensure_cuda_env() -> bool:
    """Ensure CUDA environment variables are configured for JIT compilation.

    This function should be called before importing any nvdiffrast or
    SAM 3D Objects code. It is idempotent - multiple calls are safe.

    CRITICAL: All environment variables MUST be set BEFORE importing warp,
    because warp imports torch internally. If TORCH_CUDA_ARCH_LIST is not
    set before torch initializes, nvdiffrast JIT compilation will fail.

    Sets the following environment variables:
    - CUDA_HOME: Base CUDA installation directory
    - CPATH: CUDA include headers path (for GCC during JIT)
    - PATH: Add CUDA bin directory
    - CUDACXX: Path to nvcc compiler
    - TORCH_CUDA_ARCH_LIST: GPU architecture for compilation

    Then configures Warp to disable CUDA mempool for nvdiffrast compatibility.

    Returns:
        True if CUDA environment was configured successfully, False if CUDA
        is not available. When False, SAM3D features will not work.
    """
    global _cuda_env_configured

    if _cuda_env_configured:
        console_logger.debug("CUDA environment already configured, skipping.")
        return True

    # =========================================================================
    # STEP 1: Set ALL environment variables BEFORE any library imports.
    # This is critical because warp imports torch, and torch reads these
    # environment variables during initialization.
    # =========================================================================

    cuda_home = detect_cuda_home()
    if cuda_home is None:
        console_logger.warning(
            "CUDA installation not found. SAM3D features will be unavailable. "
            "Install CUDA toolkit or set CUDA_HOME to enable SAM3D."
        )
        return False

    cuda_home_str = str(cuda_home)

    # Set CUDA_HOME.
    os.environ["CUDA_HOME"] = cuda_home_str
    console_logger.info(f"Set CUDA_HOME to {cuda_home_str}")

    # Set CPATH for GCC to find headers during JIT compilation.
    # We need multiple paths:
    # 1. CUDA headers from the conda environment or system CUDA
    # 2. Python multiarch headers (required for nvdiffrast JIT compilation).
    #    Some system Python headers include a multiarch pyconfig.h path, so
    #    resolve it dynamically instead of assuming a specific Python version.
    cpath_entries = []

    # Add CUDA include path.
    cuda_include = cuda_home / "targets" / "x86_64-linux" / "include"
    if not cuda_include.exists():
        cuda_include = cuda_home / "include"

    if cuda_include.exists():
        cpath_entries.append(str(cuda_include))
    else:
        console_logger.warning(f"CUDA include directory not found at {cuda_include}")

    # Add Python headers for nvdiffrast JIT compilation. Use sysconfig so this
    # works for uv-managed Python builds as well as system Python packages.
    python_include = sysconfig.get_path("include")
    if python_include and Path(python_include).exists():
        cpath_entries.append(str(python_include))
        console_logger.info(f"Added Python include directory: {python_include}")

    python_config_header = sysconfig.get_config_h_filename()
    if python_config_header:
        header_parent = Path(python_config_header).parent
        if header_parent.exists() and str(header_parent) not in cpath_entries:
            cpath_entries.append(str(header_parent))
            console_logger.info(
                f"Added Python config header directory: {header_parent}"
            )

    multiarch = sysconfig.get_config_var("MULTIARCH")
    python_version = sysconfig.get_python_version()
    if multiarch:
        python_multiarch_header = Path(
            f"/usr/include/{multiarch}/python{python_version}/pyconfig.h"
        )
        if python_multiarch_header.exists():
            multiarch_shim = Path("/tmp/python_multiarch_headers")
            target_dir = multiarch_shim / multiarch / f"python{python_version}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_link = target_dir / "pyconfig.h"
            if not target_link.exists():
                target_link.symlink_to(python_multiarch_header)
            cpath_entries.append(str(multiarch_shim))
            console_logger.info(
                f"Created Python multiarch header shim at {multiarch_shim}"
            )

    if cpath_entries:
        os.environ["CPATH"] = ":".join(cpath_entries)
        console_logger.info(f"Set CPATH to {os.environ['CPATH']}")

    # Ensure nvcc is in PATH.
    cuda_bin = cuda_home / "bin"
    nvvm_bin = cuda_home / "nvvm" / "bin"
    path_additions = []
    current_path = os.environ.get("PATH", "")

    if str(cuda_bin) not in current_path:
        path_additions.append(str(cuda_bin))
    if nvvm_bin.exists() and str(nvvm_bin) not in current_path:
        path_additions.append(str(nvvm_bin))

    if path_additions:
        os.environ["PATH"] = ":".join(path_additions) + ":" + current_path
        console_logger.info(f"Added to PATH: {path_additions}")

    # Set CUDACXX to force correct nvcc.
    nvcc_path = cuda_bin / "nvcc"
    os.environ["CUDACXX"] = str(nvcc_path)
    console_logger.info(f"Set CUDACXX to {nvcc_path}")

    # Detect and set GPU architecture BEFORE importing torch (via warp).
    # This is critical for nvdiffrast JIT compilation to work correctly.
    compute_cap = detect_gpu_compute_capability()
    if compute_cap:
        os.environ["TORCH_CUDA_ARCH_LIST"] = compute_cap
        console_logger.info(f"Set TORCH_CUDA_ARCH_LIST to {compute_cap}")
    else:
        # Fallback for L40S GPU (common in this environment).
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
        console_logger.warning(
            "Could not detect GPU architecture, using default 8.9 (L40S)"
        )

    # =========================================================================
    # STEP 2: Now that all env vars are set, configure Warp.
    # Warp imports torch internally, so this must happen AFTER env var setup.
    # =========================================================================

    # Disable Warp's CUDA mempool to prevent conflicts with nvdiffrast.
    # When Warp's mempool is enabled, it can interfere with nvdiffrast's CUDA
    # context during texture baking, causing hangs or crashes.
    try:
        import warp as wp

        wp.config.enable_mempools_at_init = False
        wp.init()
        console_logger.info(
            "Initialized Warp with CUDA mempool disabled for nvdiffrast compatibility"
        )
    except ImportError:
        console_logger.debug("Warp not installed - skipping mempool configuration")

    _cuda_env_configured = True
    console_logger.info("CUDA environment configured successfully for nvdiffrast")
    return True
