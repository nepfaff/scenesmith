from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scenesmith.ceiling_agents.base_ceiling_agent import BaseCeilingAgent
from scenesmith.floor_plan_agents.base_floor_plan_agent import BaseFloorPlanAgent
from scenesmith.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenesmith.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from scenesmith.utils.logging import BaseLogger
from scenesmith.wall_agents.base_wall_agent import BaseWallAgent


class BaseExperiment(ABC):
    """
    Abstract base class for scene generation experiments.

    Experiments define tasks (generate_scenes, evaluate_scenes) that are executed
    sequentially. Each experiment specifies compatible floor plan agents,
    furniture agents, etc., enabling flexible composition of different generation
    strategies.

    New experiments inherit this base and register in the exp_registry.
    """

    # Each key has to be a yaml file under
    # configurations/floor_plan_agent/<key>.yaml.
    compatible_floor_plan_agents: dict[str, type] = {}

    # Each key has to be a yaml file under
    # configurations/furniture_agent/<key>.yaml.
    compatible_furniture_agents: dict[str, type] = {}

    # Each key has to be a yaml file under
    # configurations/manipuland_agent/<key>.yaml.
    compatible_manipuland_agents: dict[str, type] = {}

    # Each key has to be a yaml file under
    # configurations/wall_agent/<key>.yaml.
    compatible_wall_agents: dict[str, type] = {}

    # Each key has to be a yaml file under
    # configurations/ceiling_agent/<key>.yaml.
    compatible_ceiling_agents: dict[str, type] = {}

    # List of task names to execute in order.
    tasks: list[str] = []

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.experiment.output_dir)

    @staticmethod
    def build_floor_plan_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
        render_gpu_id: int | None = None,
    ) -> BaseFloorPlanAgent:
        """Build floor plan agent from config dictionary.

        Args:
            cfg_dict: Configuration as dictionary or DictConfig.
            compatible_agents: Dictionary mapping agent names to classes.
            logger: Logger instance to use.
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.

        Returns:
            Floor plan agent instance.
        """
        # Extract config as dict if needed.
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )

        agent_config = config_dict["floor_plan_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Floor plan agent {agent_name} not found in "
                "compatible_floor_plan_agents for this Experiment class. Make sure "
                "you define compatible_floor_plan_agents correctly and make sure "
                "that each key has same name as yaml file under "
                "'[project_root]/configurations/floor_plan_agent' without .yaml "
                "suffix."
            )

        # Extract materials retrieval server config from experiment config.
        experiment_config = OmegaConf.create(config_dict["experiment"])
        materials_server_config = experiment_config.materials_retrieval_server

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
            materials_server_host=materials_server_config.host,
            materials_server_port=materials_server_config.port,
            render_gpu_id=render_gpu_id,
        )

    @staticmethod
    def build_furniture_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
        render_gpu_id: int | None = None,
    ) -> BaseFurnitureAgent:
        """Build furniture agent from config dictionary.

        Args:
            cfg_dict: Configuration as dictionary or DictConfig
            compatible_agents: Dictionary mapping agent names to classes
            logger: Logger instance to use
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.

        Returns:
            Furniture agent instance
        """
        # Extract config as dict if needed.
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )

        agent_config = config_dict["furniture_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Furniture agent {agent_name} not found in "
                "compatible_furniture_agents for this Experiment class. Make sure "
                "you define compatible_furniture_agents correctly and make sure "
                "that each key has same name as yaml file under "
                "'[project_root]/configurations/furniture_agent' without .yaml "
                "suffix."
            )

        # Extract geometry generation server config from experiment config.
        experiment_config = OmegaConf.create(config_dict["experiment"])
        geometry_server_config = experiment_config.geometry_generation_server

        # Extract HSSD retrieval server config from experiment config.
        hssd_server_config = experiment_config.hssd_retrieval_server

        # Extract articulated retrieval server config from experiment config.
        articulated_server_config = experiment_config.articulated_retrieval_server

        # Extract materials retrieval server config from experiment config.
        materials_server_config = experiment_config.materials_retrieval_server

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
            geometry_server_host=geometry_server_config.host,
            geometry_server_port=geometry_server_config.port,
            hssd_server_host=hssd_server_config.host,
            hssd_server_port=hssd_server_config.port,
            articulated_server_host=articulated_server_config.host,
            articulated_server_port=articulated_server_config.port,
            materials_server_host=materials_server_config.host,
            materials_server_port=materials_server_config.port,
            num_workers=experiment_config.num_workers,
            render_gpu_id=render_gpu_id,
        )

    @staticmethod
    def build_manipuland_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
        render_gpu_id: int | None = None,
    ) -> BaseManipulandAgent:
        """Build manipuland agent from config dictionary.

        Args:
            cfg_dict: Configuration as dictionary or DictConfig
            compatible_agents: Dictionary mapping agent names to classes
            logger: Logger instance to use
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.

        Returns:
            Manipuland agent instance
        """
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )

        agent_config = config_dict["manipuland_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Manipuland agent {agent_name} not found in "
                "compatible_manipuland_agents for this Experiment class. Make sure "
                "you define compatible_manipuland_agents correctly and make sure "
                "that each key has same name as yaml file under "
                "'[project_root]/configurations/manipuland_agent' without .yaml "
                "suffix."
            )

        # Extract geometry generation server config from experiment config.
        experiment_config = OmegaConf.create(config_dict["experiment"])
        geometry_server_config = experiment_config.geometry_generation_server

        # Extract HSSD retrieval server config from experiment config.
        hssd_server_config = experiment_config.hssd_retrieval_server

        # Extract articulated retrieval server config from experiment config.
        articulated_server_config = experiment_config.articulated_retrieval_server

        # Extract materials retrieval server config from experiment config.
        materials_server_config = experiment_config.materials_retrieval_server

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
            geometry_server_host=geometry_server_config.host,
            geometry_server_port=geometry_server_config.port,
            hssd_server_host=hssd_server_config.host,
            hssd_server_port=hssd_server_config.port,
            articulated_server_host=articulated_server_config.host,
            articulated_server_port=articulated_server_config.port,
            materials_server_host=materials_server_config.host,
            materials_server_port=materials_server_config.port,
            num_workers=experiment_config.num_workers,
            render_gpu_id=render_gpu_id,
        )

    @staticmethod
    def build_wall_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
        house_layout: "HouseLayout",
        ceiling_height: float,
        wall_thickness: float = 0.05,
        render_gpu_id: int | None = None,
    ) -> BaseWallAgent:
        """Build wall agent from config dictionary.

        Args:
            cfg_dict: Configuration as dictionary or DictConfig.
            compatible_agents: Dictionary mapping agent names to classes.
            logger: Logger instance to use.
            house_layout: HouseLayout containing wall geometry.
            ceiling_height: Height of ceiling in meters.
            wall_thickness: Wall thickness in meters for surface offset.
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.

        Returns:
            Wall agent instance.
        """
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )

        agent_config = config_dict["wall_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Wall agent {agent_name} not found in "
                "compatible_wall_agents for this Experiment class. Make sure "
                "you define compatible_wall_agents correctly and make sure "
                "that each key has same name as yaml file under "
                "'[project_root]/configurations/wall_agent' without .yaml "
                "suffix."
            )

        # Extract geometry generation server config from experiment config.
        experiment_config = OmegaConf.create(config_dict["experiment"])
        geometry_server_config = experiment_config.geometry_generation_server

        # Extract HSSD retrieval server config from experiment config.
        hssd_server_config = experiment_config.hssd_retrieval_server

        # Extract articulated retrieval server config from experiment config.
        articulated_server_config = experiment_config.articulated_retrieval_server

        # Extract materials retrieval server config from experiment config.
        materials_server_config = experiment_config.materials_retrieval_server

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
            house_layout=house_layout,
            ceiling_height=ceiling_height,
            wall_thickness=wall_thickness,
            geometry_server_host=geometry_server_config.host,
            geometry_server_port=geometry_server_config.port,
            hssd_server_host=hssd_server_config.host,
            hssd_server_port=hssd_server_config.port,
            articulated_server_host=articulated_server_config.host,
            articulated_server_port=articulated_server_config.port,
            materials_server_host=materials_server_config.host,
            materials_server_port=materials_server_config.port,
            num_workers=experiment_config.num_workers,
            render_gpu_id=render_gpu_id,
        )

    @staticmethod
    def build_ceiling_agent(
        cfg_dict: dict | DictConfig,
        compatible_agents: dict[str, type],
        logger: BaseLogger,
        ceiling_height: float,
        render_gpu_id: int | None = None,
    ) -> BaseCeilingAgent:
        """Build ceiling agent from config dictionary.

        Args:
            cfg_dict: Configuration as dictionary or DictConfig.
            compatible_agents: Dictionary mapping agent names to classes.
            logger: Logger instance to use.
            ceiling_height: Height of ceiling in meters.
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.

        Returns:
            Ceiling agent instance.
        """
        config_dict = (
            OmegaConf.to_container(cfg_dict, resolve=True)
            if isinstance(cfg_dict, DictConfig)
            else cfg_dict
        )

        agent_config = config_dict["ceiling_agent"]
        agent_name = agent_config["_name"]

        if agent_name not in compatible_agents:
            raise ValueError(
                f"Ceiling agent {agent_name} not found in "
                "compatible_ceiling_agents for this Experiment class. Make sure "
                "you define compatible_ceiling_agents correctly and make sure "
                "that each key has same name as yaml file under "
                "'[project_root]/configurations/ceiling_agent' without .yaml "
                "suffix."
            )

        # Extract geometry generation server config from experiment config.
        experiment_config = OmegaConf.create(config_dict["experiment"])
        geometry_server_config = experiment_config.geometry_generation_server

        # Extract HSSD retrieval server config from experiment config.
        hssd_server_config = experiment_config.hssd_retrieval_server

        # Extract articulated retrieval server config from experiment config.
        articulated_server_config = experiment_config.articulated_retrieval_server

        # Extract materials retrieval server config from experiment config.
        materials_server_config = experiment_config.materials_retrieval_server

        return compatible_agents[agent_name](
            cfg=OmegaConf.create(agent_config),
            logger=logger,
            ceiling_height=ceiling_height,
            geometry_server_host=geometry_server_config.host,
            geometry_server_port=geometry_server_config.port,
            hssd_server_host=hssd_server_config.host,
            hssd_server_port=hssd_server_config.port,
            articulated_server_host=articulated_server_config.host,
            articulated_server_port=articulated_server_config.port,
            materials_server_host=materials_server_config.host,
            materials_server_port=materials_server_config.port,
            num_workers=experiment_config.num_workers,
            render_gpu_id=render_gpu_id,
        )

    def exec_task(self, task_name: str) -> None:
        """Execute a specific task by name."""
        if hasattr(self, task_name):
            task_method = getattr(self, task_name)
            task_method()
        else:
            raise ValueError(
                f"Task '{task_name}' not found in experiment "
                f"'{self.__class__.__name__}'"
            )

    @abstractmethod
    def generate_scenes(self) -> None:
        """Generate scenes for the experiment."""

    @abstractmethod
    def evaluate_scenes(self) -> None:
        """Evaluate the generated scenes."""
