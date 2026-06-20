from .params import RenderParams

__all__ = ["RenderParams", "BlenderRenderer", "BlenderRenderApp", "BlenderServer"]


def __getattr__(name: str):
    if name == "BlenderRenderer":
        from .renderer import BlenderRenderer

        return BlenderRenderer
    if name == "BlenderRenderApp":
        from .server_app import BlenderRenderApp

        return BlenderRenderApp
    if name == "BlenderServer":
        from .server_manager import BlenderServer

        return BlenderServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
