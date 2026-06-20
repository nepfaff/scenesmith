import os


def has_gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def is_github_actions() -> bool:
    """Check if running in GitHub Actions CI."""
    return os.getenv("GITHUB_ACTIONS") == "true"


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_hunyuan3d_installed() -> bool:
    """Check if Hunyuan3D-2 is installed for 3D geometry generation."""
    try:
        from hy3dgen.shapegen.pipelines import export_to_trimesh  # noqa: F401

        return True
    except ImportError:
        return False


def has_sam3d_installed() -> bool:
    """Check if the default SAM3D backend assets are installed."""
    required_paths = [
        "external/SAM3/sam3",
        "external/sam-3d-objects/sam3d_objects",
        "external/checkpoints/sam3.pt",
        "external/checkpoints/pipeline.yaml",
    ]
    return all(os.path.exists(path) for path in required_paths)


def has_usd_export_env() -> bool:
    """Check if the MuJoCo/USD export test environment is available."""
    if os.getenv("SCENESMITH_RUN_USD_EXPORT_TESTS") != "1":
        return False

    try:
        import mujoco_usd_converter  # noqa: F401

        from pxr import Usd  # noqa: F401

        return True
    except ImportError:
        return False
