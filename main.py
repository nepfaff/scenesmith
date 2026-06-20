"""
Main file for the project. This will create and run new experiments.
"""

import logging
import os
import time

from datetime import timedelta
from pathlib import Path

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from scenesmith.utils.logging import FileLoggingContext
from scenesmith.utils.omegaconf import register_resolvers
from scenesmith.utils.print_utils import cyan

console_logger = logging.getLogger(__name__)


def run_local(cfg: DictConfig):
    # Delay some imports in case they are not needed in non-local envs for submission.
    from scenesmith.experiments import build_experiment

    start_time = time.time()

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["floor_plan_agent"] is not None:
            cfg.floor_plan_agent._name = cfg_choice["floor_plan_agent"]
        if cfg_choice["furniture_agent"] is not None:
            cfg.furniture_agent._name = cfg_choice["furniture_agent"]
        if cfg_choice["wall_agent"] is not None:
            cfg.wall_agent._name = cfg_choice["wall_agent"]
        if cfg_choice["ceiling_agent"] is not None:
            cfg.ceiling_agent._name = cfg_choice["ceiling_agent"]
        if cfg_choice["manipuland_agent"] is not None:
            cfg.manipuland_agent._name = cfg_choice["manipuland_agent"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    with open_dict(cfg):
        cfg.experiment.output_dir = output_dir

    # Set up experiment-level logging to file while preserving stdout.
    experiment_log_path = output_dir / "experiment.log"
    experiment_log_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLoggingContext(log_file_path=experiment_log_path, suppress_stdout=False):
        console_logger.info(f"Outputs will be saved to: {output_dir}")
        print(cyan(f"Outputs will be saved to:"), output_dir)

        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(
            output_dir, target_is_directory=True
        )

        # Log and save resolved configuration.
        resolved_config_yaml = OmegaConf.to_yaml(cfg)
        console_logger.info("Resolved configuration:\n" + resolved_config_yaml)
        print(cyan("Resolved configuration:"))
        print(resolved_config_yaml)

        # Save config to output directory for reproducibility.
        config_file = output_dir / "resolved_config.yaml"
        with open(config_file, "w") as f:
            f.write(resolved_config_yaml)
        console_logger.info(f"Saved resolved config to: {config_file}")
        print(cyan(f"Saved resolved config to: {config_file}"))

        # Launch experiment.
        console_logger.info("Starting experiment execution")
        experiment = build_experiment(cfg=cfg)
        for task in cfg.experiment.tasks:
            console_logger.info(f"Executing task: {task}")
            experiment.exec_task(task)
            console_logger.info(f"Completed task: {task}")

        console_logger.info(
            "Experiment execution completed in "
            f"{timedelta(seconds=time.time() - start_time)}"
        )


@hydra.main(version_base=None, config_path="configurations", config_name="config")
def run(cfg: DictConfig):
    if "name" not in cfg:
        raise ValueError(
            "Must specify a name for the run with command line argument '+name=[name]'"
        )

    # Configure logging level from LOGLEVEL environment variable.
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configure separate tracing API key if provided.
    tracing_api_key = os.environ.get("OPENAI_TRACING_KEY")
    if tracing_api_key:
        from agents.tracing import set_tracing_export_api_key

        set_tracing_export_api_key(tracing_api_key)
        console_logger.info("Using separate API key for tracing exports")

    run_local(cfg)


if __name__ == "__main__":
    run()
