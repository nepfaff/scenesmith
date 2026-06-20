"""Base class for floor plan agents.

Provides shared functionality for floor plan design and geometry generation.
"""

import logging

from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.floor_plan_agents.tools.floor_plan_tools import DoorWindowConfig
from scenesmith.floor_plan_agents.tools.materials_resolver import MaterialsConfig
from scenesmith.floor_plan_agents.tools.room_placement import ScoringWeights
from scenesmith.prompts import prompt_registry
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


class BaseFloorPlanAgent(ABC):
    """Base class with shared functionality for floor plan agents.

    NOTE: A new FloorPlanAgent instance is created for each house/scene.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
    ):
        """Initialize base floor plan agent.

        Args:
            cfg: Configuration for floor plan generation.
            logger: Scene-specific logger from experiment.
            materials_server_host: Host for materials retrieval server.
            materials_server_port: Port for materials retrieval server.
        """
        self.cfg = cfg
        self.logger = logger
        self.materials_server_host = materials_server_host
        self.materials_server_port = materials_server_port

        # Floor plan mode: "room" (single room) or "house" (multi-room).
        self.mode = cfg.mode

        # Prompt registry for agent prompts.
        self.prompt_registry = prompt_registry

        # Layout being designed (set in subclass).
        self.layout: HouseLayout | None = None

    def _create_materials_config(self) -> MaterialsConfig:
        """Create materials configuration from config.

        Returns:
            MaterialsConfig with settings from cfg.

        Raises:
            ValueError: If use_retrieval_server is True but layout.house_dir is not set.
        """
        use_retrieval_server = self.cfg.materials.use_retrieval_server

        # Get output directory from layout if available.
        output_dir = None
        if self.layout and self.layout.house_dir:
            output_dir = self.layout.house_dir

        # Fail fast if server is enabled but no output directory.
        if use_retrieval_server and not output_dir:
            raise ValueError(
                "materials.use_retrieval_server=True requires layout.house_dir "
                "to be set. Materials from server need a persistent location."
            )

        return MaterialsConfig(
            use_retrieval_server=use_retrieval_server,
            default_wall_material=self.cfg.materials.default_wall_material,
            default_floor_material=self.cfg.materials.default_floor_material,
            output_dir=output_dir,
            retrieval_server_host=self.materials_server_host,
            retrieval_server_port=self.materials_server_port,
        )

    def _create_door_window_config(self) -> DoorWindowConfig:
        """Create door/window configuration from config.

        Returns:
            DoorWindowConfig with constraints from cfg.
        """
        doors_cfg = self.cfg.doors
        windows_cfg = self.cfg.windows

        return DoorWindowConfig(
            door_width_min=doors_cfg.width_range[0],
            door_width_max=doors_cfg.width_range[1],
            door_height_min=doors_cfg.height_range[0],
            door_height_max=doors_cfg.height_range[1],
            door_default_width=doors_cfg.default_width,
            door_default_height=doors_cfg.default_height,
            window_width_min=windows_cfg.width_range[0],
            window_width_max=windows_cfg.width_range[1],
            window_height_min=windows_cfg.height_range[0],
            window_height_max=windows_cfg.height_range[1],
            window_default_width=windows_cfg.default_width,
            window_default_height=windows_cfg.default_height,
            window_default_sill_height=windows_cfg.default_sill_height,
            window_segment_margin=windows_cfg.segment_margin,
            exterior_door_clearance_m=doors_cfg.exterior_clearance,
        )

    def _create_scoring_weights(self) -> ScoringWeights:
        """Create scoring weights for room placement from config.

        Returns:
            ScoringWeights with compactness and stability weights.
        """
        weights_cfg = self.cfg.room_placement.scoring_weights
        return ScoringWeights(
            compactness=weights_cfg.compactness,
            stability=weights_cfg.stability,
        )

    def cleanup(self) -> None:
        """Cleanup resources held by the agent.

        Override in subclass if there are resources to clean up.
        """

    @abstractmethod
    async def generate_house_layout(self, prompt: str, output_dir: Path) -> HouseLayout:
        """Generate a house layout with floor plan geometry.

        This is the main entry point for floor plan generation. It runs the agent trio
        to design the layout, then generates geometry for all rooms.

        Args:
            prompt: Description of the house/room to design.
            output_dir: Directory to save generated geometry files.

        Returns:
            HouseLayout with designed layout and generated RoomGeometry.
        """
        raise NotImplementedError
