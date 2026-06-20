import logging
import shutil
import socket
import tempfile
import unittest

from pathlib import Path

from omegaconf import OmegaConf

from scenesmith.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
)
from tests.integration.common import (
    has_gpu_available,
    has_openai_key,
    has_sam3d_installed,
    is_github_actions,
)


def _get_free_port() -> int:
    """Return an available localhost TCP port for this test process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@unittest.skipIf(
    not has_openai_key()
    or not has_gpu_available()
    or not has_sam3d_installed()
    or is_github_actions(),
    "Requires OpenAI API key, GPU, SAM3D checkpoints, and non-CI environment",
)
class TestParallelSceneGeneration(unittest.TestCase):
    """Integration test for parallel scene generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "parallel_test"
        self.output_dir.mkdir(exist_ok=True)
        self.service_ports = {
            "geometry_generation_server": _get_free_port(),
            "hssd_retrieval_server": _get_free_port(),
            "articulated_retrieval_server": _get_free_port(),
            "objaverse_retrieval_server": _get_free_port(),
            "materials_retrieval_server": _get_free_port(),
        }

        # Print output directory for progress tracking.
        print(f"\n{'='*60}")
        print(f"Test output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # Load base experiment configuration.
        experiment_config_path = (
            Path(__file__).parent.parent.parent
            / "configurations/experiment/base_experiment.yaml"
        )
        base_experiment_config = OmegaConf.load(experiment_config_path)

        # Add required agent and generator configs.
        # Load base furniture agent config first.
        base_furniture_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/furniture_agent/base_furniture_agent.yaml"
        )

        # Load stateful furniture agent config.
        stateful_furniture_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/furniture_agent/stateful_furniture_agent.yaml"
        )

        # Merge furniture agent configs.
        furniture_agent_config = OmegaConf.merge(
            base_furniture_agent_config, stateful_furniture_agent_config
        )
        # Load floor plan agent configs.
        base_floor_plan_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/floor_plan_agent/base_floor_plan_agent.yaml"
        )
        stateful_floor_plan_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/floor_plan_agent/stateful_floor_plan_agent.yaml"
        )

        # Merge floor plan agent configs.
        floor_plan_config = OmegaConf.merge(
            base_floor_plan_agent_config, stateful_floor_plan_agent_config
        )
        base_manipuland_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/manipuland_agent/base_manipuland_agent.yaml"
        )
        stateful_manipuland_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/manipuland_agent/stateful_manipuland_agent.yaml"
        )
        manipuland_agent_config = OmegaConf.merge(
            base_manipuland_agent_config, stateful_manipuland_agent_config
        )

        # Load wall agent configs.
        base_wall_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/wall_agent/base_wall_agent.yaml"
        )
        stateful_wall_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/wall_agent/stateful_wall_agent.yaml"
        )
        wall_agent_config = OmegaConf.merge(
            base_wall_agent_config, stateful_wall_agent_config
        )

        # Load ceiling agent configs.
        base_ceiling_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/ceiling_agent/base_ceiling_agent.yaml"
        )
        stateful_ceiling_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configurations/ceiling_agent/stateful_ceiling_agent.yaml"
        )
        ceiling_agent_config = OmegaConf.merge(
            base_ceiling_agent_config, stateful_ceiling_agent_config
        )

        # Add _name fields to configs as Hydra would.
        furniture_agent_config._name = "stateful_furniture_agent"
        floor_plan_config._name = "stateful_floor_plan_agent"
        manipuland_agent_config._name = "stateful_manipuland_agent"
        wall_agent_config._name = "stateful_wall_agent"
        ceiling_agent_config._name = "stateful_ceiling_agent"

        # Create complete base config structure with proper nesting.
        self.base_config = OmegaConf.create(
            {
                "experiment": base_experiment_config,
                "furniture_agent": furniture_agent_config,
                "floor_plan_agent": floor_plan_config,
                "manipuland_agent": manipuland_agent_config,
                "wall_agent": wall_agent_config,
                "ceiling_agent": ceiling_agent_config,
            }
        )

    def _dump_scene_diagnostics(self, scene_dir: Path) -> str:
        """Get diagnostic info about a scene directory for debugging.

        Only called on test failures to provide context about what went wrong.

        Args:
            scene_dir: Path to the scene directory to diagnose.

        Returns:
            Formatted diagnostic string with directory contents and log tail.
        """
        lines = [f"\n=== Scene Diagnostics: {scene_dir.name} ==="]

        # List top-level contents.
        if scene_dir.exists():
            lines.append(f"Contents: {[f.name for f in scene_dir.iterdir()]}")

            # Check generated_assets subdirs.
            assets_dir = scene_dir / "generated_assets"
            if assets_dir.exists():
                for subdir in ["images", "geometry", "sdf", "debug"]:
                    subdir_path = assets_dir / subdir
                    count = (
                        len(list(subdir_path.iterdir())) if subdir_path.exists() else 0
                    )
                    lines.append(f"  {subdir}: {count} files")

            # Show last 10 lines of scene.log if it exists.
            log_file = scene_dir / "scene.log"
            if log_file.exists():
                lines.append("\nLast 10 lines of scene.log:")
                lines.extend(log_file.read_text().splitlines()[-10:])

        return "\n".join(lines)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_parallel_scene_generation(self):
        """Test generating 2 scenes in parallel.

        Verifies that multiple scenes can be generated concurrently with:
        - Proper scene isolation (separate directories and logs)
        - Asset server communication working correctly
        - Expected output files created for each scene
        """

        # Define test overrides.
        test_overrides = {
            "name": "test_parallel_scene_generation",
            "experiment": {
                "name": "test_parallel_scene_generation",  # Required by experiment
                "num_workers": 2,  # Test parallel execution
                "prompts": [
                    "A compact dining room with one simple table and one chair. No other objects.",
                    "A compact office with one simple desk and one office chair. No other objects.",
                ],
                "output_dir": self.output_dir,
                "pipeline": {
                    "start_stage": "floor_plan",
                    "stop_stage": "furniture",
                },
                "geometry_generation_server": {
                    "port": self.service_ports["geometry_generation_server"],
                    "preload_pipeline": False,
                },
                "hssd_retrieval_server": {
                    "port": self.service_ports["hssd_retrieval_server"],
                },
                "articulated_retrieval_server": {
                    "port": self.service_ports["articulated_retrieval_server"],
                },
                "objaverse_retrieval_server": {
                    "port": self.service_ports["objaverse_retrieval_server"],
                },
                "materials_retrieval_server": {
                    "port": self.service_ports["materials_retrieval_server"],
                },
            },
            "floor_plan_agent": {
                "max_critique_rounds": 0,
            },
            "furniture_agent": {
                "max_critique_rounds": 0,
            },
            "openai": {
                "model": "gpt-4o-mini",  # Cheaper model for testing
                "service_tier": "default",
                "reasoning_effort": {
                    "planner": "low",  # Faster for tests
                    "designer": "low",
                    "critic": "low",
                },
                "verbosity": {
                    "planner": "low",
                    "designer": "low",
                    "critic": "low",
                },
            },
            "rendering": {
                "image_size": 256,  # Smaller for faster testing
            },
        }

        # Merge configurations (base config provides all other values).
        test_config = OmegaConf.merge(self.base_config, test_overrides)

        # Run experiment.
        experiment = IndoorSceneGenerationExperiment(cfg=test_config)
        experiment.generate_scenes()

        # Log generation completion summary.
        scene_count = len(list(self.output_dir.glob("scene_*")))
        logging.info(f"Generation complete. Found {scene_count} scene directories")

        # Verify results - should have 2 scene directories.
        scene_dirs = list(self.output_dir.glob("scene_*"))
        self.assertEqual(len(scene_dirs), 2, "Should generate 2 scenes")

        # Check each scene has required files.
        for scene_dir in scene_dirs:
            self.assertTrue(
                scene_dir.is_dir(), f"Scene directory should exist: {scene_dir}"
            )

            # Check log file exists and has content.
            log_file = scene_dir / "scene.log"
            self.assertTrue(log_file.exists(), f"Scene log should exist: {scene_dir}")

            # Check floor plan files exist.
            floor_plan_files = list((scene_dir / "room_geometry").glob("*.sdf"))
            self.assertGreater(
                len(floor_plan_files),
                0,
                f"Floor plan SDF should exist: {scene_dir}",
            )

            room_dirs = [
                path
                for path in scene_dir.glob("room_*")
                if path.is_dir() and (path / "room.log").exists()
            ]
            self.assertGreater(
                len(room_dirs), 0, f"Room output should exist: {scene_dir}"
            )

            for room_dir in room_dirs:
                # Check furniture assets. These may come from generated geometry,
                # articulated retrieval, or another configured backend.
                generated_assets_dir = room_dir / "generated_assets" / "furniture"
                self.assertTrue(
                    generated_assets_dir.exists() and generated_assets_dir.is_dir(),
                    f"Furniture assets directory should exist: {room_dir}",
                )

                self.assertTrue(
                    (generated_assets_dir / "asset_registry.json").exists(),
                    f"Furniture asset registry should exist: {room_dir}",
                )
                self.assertGreater(
                    len(list((generated_assets_dir / "sdf").rglob("*.sdf"))),
                    0,
                    f"At least one furniture SDF should exist: {room_dir}",
                )

                # Check furniture renders and furniture-stage scene state.
                scene_renders_dir = room_dir / "scene_renders" / "furniture"
                self.assertTrue(
                    scene_renders_dir.exists() and scene_renders_dir.is_dir(),
                    f"Furniture renders directory should exist: {room_dir}",
                )

                furniture_state_dir = (
                    room_dir / "scene_states" / "scene_after_furniture"
                )
                self.assertTrue(
                    furniture_state_dir.exists() and furniture_state_dir.is_dir(),
                    f"Furniture scene state directory should exist: {room_dir}",
                )

                for filename in ["scene_state.json", "scene.dmd.yaml", "scene.blend"]:
                    self.assertTrue(
                        (furniture_state_dir / filename).exists(),
                        f"{filename} should exist in furniture state: {room_dir}",
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
