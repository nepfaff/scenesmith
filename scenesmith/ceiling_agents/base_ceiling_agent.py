"""Abstract base class for ceiling-mounted object placement agents.

This module defines the interface that all ceiling agents must implement,
following the same architectural patterns as furniture, manipuland, and wall agents.
"""

import logging
import os

from abc import ABC, abstractmethod

from omegaconf import DictConfig

from scenesmith.agent_utils.asset_manager import AssetManager
from scenesmith.agent_utils.blender import BlenderServer
from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionServer
from scenesmith.agent_utils.rendering_manager import RenderingManager
from scenesmith.agent_utils.room import AgentType, RoomScene
from scenesmith.agent_utils.scene_analyzer import SceneAnalyzer
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.prompts import prompt_registry
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


class BaseCeilingAgent(ABC):
    """Abstract base class for ceiling-mounted object placement agents.

    CeilingAgents are responsible for placing ceiling-mounted objects (lights,
    fans, chandeliers, etc.) on the ceiling plane. They follow the same
    architectural patterns as FurnitureAgents, WallAgents, and ManipulandAgents.

    Pipeline order: floor_plan -> furniture -> wall-mounted -> ceiling-mounted -> manipulands

    NOTE: It is expected that a new CeilingAgent instance is created for each
    scene. Hence, this class does not need to manage session lifecycle and
    `add_ceiling_objects` should not be called multiple times.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        ceiling_height: float,
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
        """Initialize base ceiling agent.

        Args:
            cfg: Configuration object containing ceiling agent settings.
            logger: Logger instance for tracking operations.
            ceiling_height: Height of ceiling in meters.
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
        self.ceiling_height = ceiling_height

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
            agent_type=AgentType.CEILING_MOUNTED,
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
            cfg=cfg.rendering, logger=logger, subdirectory="ceiling"
        )
        self.prompt_registry = prompt_registry
        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.vlm_service,
            rendering_manager=self.rendering_manager,
            cfg=cfg,
            blender_server=self.blender_server,
        )

        self.scene: RoomScene | None = None
        self.room_bounds: tuple[float, float, float, float] | None = None

    def cleanup(self) -> None:
        """Cleanup resources held by the agent."""
        if self.blender_server is not None and self.blender_server.is_running():
            console_logger.info("Stopping BlenderServer")
            self.blender_server.stop()
            self.blender_server = None

        if self.collision_server is not None and self.collision_server.is_running():
            console_logger.info("Stopping ConvexDecompositionServer")
            self.collision_server.stop()

    def _extract_room_bounds(
        self, scene: RoomScene
    ) -> tuple[float, float, float, float]:
        """Extract room XY bounds from scene geometry.

        Args:
            scene: RoomScene to extract bounds from.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in meters.
        """
        # Compute bounds from room geometry dimensions.
        # Room is centered at origin, so bounds are +/- half dimensions.
        room_geom = scene.room_geometry
        if room_geom is not None and room_geom.length > 0 and room_geom.width > 0:
            half_length = room_geom.length / 2
            half_width = room_geom.width / 2
            return (-half_length, -half_width, half_length, half_width)

        raise RuntimeError(
            f"Room {scene.room_id} has no valid room_geometry dimensions. "
            f"Ensure room geometry is set at scene creation."
        )

    @abstractmethod
    async def add_ceiling_objects(self, scene: RoomScene) -> None:
        """Add ceiling-mounted objects to a scene.

        Args:
            scene: RoomScene instance containing furniture to augment with
                ceiling-mounted objects. The scene is mutated in place.
        """
