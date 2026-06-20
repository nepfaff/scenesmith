"""Abstract base class for wall-mounted object placement agents.

This module defines the interface that all wall agents must implement,
following the same architectural patterns as furniture and manipuland agents.
"""

import logging
import os

from abc import ABC, abstractmethod

from omegaconf import DictConfig

from scenesmith.agent_utils.asset_manager import AssetManager
from scenesmith.agent_utils.blender import BlenderServer
from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionServer
from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.rendering_manager import RenderingManager
from scenesmith.agent_utils.room import AgentType, RoomScene
from scenesmith.agent_utils.scene_analyzer import SceneAnalyzer
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.prompts import prompt_registry
from scenesmith.utils.logging import BaseLogger
from scenesmith.wall_agents.tools.wall_surface import WallSurface, extract_wall_surfaces

console_logger = logging.getLogger(__name__)


class BaseWallAgent(ABC):
    """Abstract base class for wall-mounted object placement agents.

    WallAgents are responsible for placing wall-mounted objects (mirrors,
    artwork, shelves, clocks, etc.) on room walls. They follow the same
    architectural patterns as FurnitureAgents and ManipulandAgents.

    Pipeline order: floor_plan → furniture → wall-mounted → ceiling-mounted → manipulands

    NOTE: It is expected that a new WallAgent instance is created for each
    scene. Hence, this class does not need to manage session lifecycle and
    `add_wall_objects` should not be called multiple times.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        house_layout: HouseLayout,
        ceiling_height: float,
        wall_thickness: float = 0.05,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        num_workers: int = 1,
        render_gpu_id: int | None = None,
    ):
        """Initialize base wall agent.

        Args:
            cfg: Configuration object containing wall agent settings.
            logger: Logger instance for tracking operations.
            house_layout: HouseLayout containing wall geometry.
            ceiling_height: Height of ceiling in meters.
            wall_thickness: Wall thickness in meters for surface offset.
            geometry_server_host: Host for geometry generation server.
            geometry_server_port: Port for geometry generation server.
            hssd_server_host: Host for HSSD retrieval server.
            hssd_server_port: Port for HSSD retrieval server.
            articulated_server_host: Host for articulated retrieval server.
            articulated_server_port: Port for articulated retrieval server.
            materials_server_host: Host for materials retrieval server.
            materials_server_port: Port for materials retrieval server.
            num_workers: Number of parallel workers (for OMP thread allocation).
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.
        """
        self.cfg = cfg
        self.logger = logger
        self.house_layout = house_layout
        self.ceiling_height = ceiling_height
        self.wall_thickness = wall_thickness

        # Start BlenderServer for thread-safe validation rendering.
        # Required when router uses parallel generation (bpy requires main thread).
        self.blender_server: BlenderServer | None = None
        self.collision_server: ConvexDecompositionServer | None = None

        try:
            # Start BlenderServer for all bpy operations (rendering, validation,
            # canonicalization). Required because forked workers cannot safely use
            # embedded bpy - GPU/OpenGL state is corrupted by fork.
            console_logger.info("Starting BlenderServer for parallel asset validation")
            self.blender_server = BlenderServer(
                port_range=tuple(cfg.rendering.blender_server_port_range),
                server_startup_delay=cfg.rendering.server_startup_delay,
                port_cleanup_delay=cfg.rendering.port_cleanup_delay,
                gpu_id=render_gpu_id,
                log_file=logger.output_dir / "room.log",
            )
            self.blender_server.start()
            self.blender_server.wait_until_ready()

            # Start ConvexDecompositionServer for collision geometry generation.
            # This isolates OpenMP from ThreadPoolExecutor to prevent deadlocks.
            # Calculate OMP threads: each worker gets a fair share of CPU cores.
            cpu_count = os.cpu_count() or 1
            omp_threads = max(1, cpu_count // num_workers)
            console_logger.info(
                f"Starting ConvexDecompositionServer (omp_threads={omp_threads})"
            )
            self.collision_server = ConvexDecompositionServer(
                port_range=tuple(cfg.collision_geometry.server_port_range),
                omp_threads=omp_threads,
                log_file=logger.output_dir / "room.log",
            )
            self.collision_server.start()
            self.collision_server.wait_until_ready()
        except Exception:
            # Clean up any servers that were started before the failure.
            self.cleanup()
            raise

        service_tier = getattr(self.cfg.openai, "service_tier", None)
        get_api_timeout = getattr(self, "_get_api_timeout", lambda: None)
        self.vlm_service = VLMService(
            service_tier=service_tier, timeout=get_api_timeout()
        )
        self.asset_manager = AssetManager(
            logger=logger,
            vlm_service=self.vlm_service,
            blender_server=self.blender_server,
            collision_client=self.collision_server.get_client(),
            cfg=cfg,
            agent_type=AgentType.WALL_MOUNTED,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
            articulated_server_host=articulated_server_host,
            articulated_server_port=articulated_server_port,
            materials_server_host=materials_server_host,
            materials_server_port=materials_server_port,
        )
        self.rendering_manager = RenderingManager(
            cfg=cfg.rendering, logger=logger, subdirectory="wall"
        )
        self.prompt_registry = prompt_registry
        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.vlm_service,
            rendering_manager=self.rendering_manager,
            cfg=cfg,
            blender_server=self.blender_server,
        )

        self.scene: RoomScene | None = None
        self.wall_surfaces: list[WallSurface] = []

    def cleanup(self) -> None:
        """Cleanup resources held by the agent."""
        if self.blender_server is not None and self.blender_server.is_running():
            console_logger.info("Stopping BlenderServer")
            self.blender_server.stop()
            self.blender_server = None

        if self.collision_server is not None and self.collision_server.is_running():
            console_logger.info("Stopping ConvexDecompositionServer")
            self.collision_server.stop()

    def _extract_wall_surfaces(self, room_id: str) -> list[WallSurface]:
        """Extract wall surfaces for a room.

        Args:
            room_id: ID of the room to extract surfaces for.

        Returns:
            List of WallSurface objects (one per wall).
        """
        return extract_wall_surfaces(
            house_layout=self.house_layout,
            room_id=room_id,
            ceiling_height=self.ceiling_height,
            wall_thickness=self.wall_thickness,
        )

    @abstractmethod
    async def add_wall_objects(self, scene: RoomScene) -> None:
        """Add wall-mounted objects to a scene.

        Args:
            scene: RoomScene instance containing furniture to augment with
                wall-mounted objects. The scene is mutated in place.
        """
