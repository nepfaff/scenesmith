import asyncio
import atexit
import logging
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path
from unittest.mock import Mock

import yaml

from omegaconf import OmegaConf

# isort: off
# Need to import bpy first to avoid potential symbol loading issues.
try:
    import bpy  # noqa: F401
except ImportError:
    bpy = None

# isort: on

from scenesmith.agent_utils.geometry_generation_server import GeometryGenerationServer
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import RoomScene
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.utils.logging import ConsoleLogger
from tests.integration.common import (
    has_gpu_available,
    has_hunyuan3d_installed,
    has_openai_key,
    is_github_actions,
)

console_logger = logging.getLogger(__name__)


@unittest.skipIf(
    not has_openai_key()
    or not has_gpu_available()
    or not has_hunyuan3d_installed()
    or is_github_actions()
    or bpy is None
    or isinstance(bpy, Mock),
    "Requires OpenAI API key, GPU, Hunyuan3D-2, and non-CI environment",
)
class TestFurnitureAgentIntegration(unittest.TestCase):
    """Integration test for FurnitureAgent complete workflow."""

    @classmethod
    def setUpClass(cls):
        """Set up class fixtures - runs once before all tests."""
        super().setUpClass()
        # Start geometry generation server once for all tests.
        cls.geometry_server = GeometryGenerationServer(
            host="127.0.0.1", port_range=(7000, 7050)
        )
        cls.geometry_server.start()
        cls.geometry_server.wait_until_ready(timeout_s=30.0)

        # Register cleanup handler to ensure server is stopped even if tearDownClass fails.
        atexit.register(cls._cleanup_server)

    @classmethod
    def tearDownClass(cls):
        """Clean up class fixtures - runs once after all tests."""
        cls._cleanup_server()
        super().tearDownClass()

    @classmethod
    def _cleanup_server(cls):
        """Clean up geometry server - used by tearDownClass and atexit handler."""
        if hasattr(cls, "geometry_server") and cls.geometry_server.is_running():
            cls.geometry_server.stop()

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "furniture_agent_integration"
        self.output_dir.mkdir(exist_ok=True)

        # Define common test overrides.
        self.test_overrides = {
            "rendering": {
                "image_size": 256,  # Smaller for faster testing
            },
            "max_critique_rounds": 2,  # Fewer rounds for testing
            "max_fix_attempts_per_critique": 1,  # Single attempt
            "openai": {
                "model": "gpt-4o-mini",  # Cheaper model for testing
            },
        }

        # Create test scene.
        self.test_scene = self._create_test_scene()

        # Create logger.
        self.logger = ConsoleLogger(output_dir=self.output_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _load_test_config(self, config_filename: str):
        """Load and merge test configuration for a specific agent type."""
        base_path = (
            Path(__file__).parent.parent.parent / "configurations/furniture_agent"
        )

        # Load base configuration first.
        base_config = OmegaConf.load(base_path / "base_furniture_agent.yaml")

        # Load agent-specific configuration.
        agent_config = OmegaConf.load(base_path / config_filename)

        # Merge: base <- agent <- test_overrides.
        merged_config = OmegaConf.merge(base_config, agent_config, self.test_overrides)

        return merged_config

    def _create_test_scene(self) -> RoomScene:
        """Create a simple test scene using existing test data."""
        # Use existing test data floor plan.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        floor_plan_path = test_data_dir / "simple_room_geometry.sdf"

        # Read the existing SDF file.
        with open(floor_plan_path, "r") as f:
            floor_plan_sdf = f.read()

        # Create RoomGeometry object.
        room_geometry_tree = ET.ElementTree(ET.fromstring(floor_plan_sdf))
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=floor_plan_path,
        )

        # Create a simple scene for integration testing.
        return RoomScene(
            room_geometry=room_geometry,
            text_description="A room with a single table in the center. No other "
            "items are in the room.",
            scene_dir=self.output_dir,
        )

    def test_complete_furniture_placement_workflow_integration_stateful(self):
        """
        Test the complete StatefulFurnitureAgent workflow.
        """
        # Load configuration for stateful agent.
        test_config = self._load_test_config("stateful_furniture_agent.yaml")

        # Create the agent.
        agent = StatefulFurnitureAgent(cfg=test_config, logger=self.logger)

        # Verify basic initialization contract.
        self.assertIsInstance(agent, StatefulFurnitureAgent)
        self.assertIsNotNone(agent.asset_manager, "Asset manager should be initialized")
        self.assertIsNotNone(
            agent.designer_session, "Designer session should be initialized"
        )
        self.assertIsNotNone(
            agent.critic_session, "Critic session should be initialized"
        )

        initial_object_count = len(self.test_scene.objects)

        try:
            # Execute the main workflow.
            asyncio.run(agent.add_furniture(self.test_scene))

            # The scene should have been modified (furniture added).
            final_object_count = len(self.test_scene.objects)
            self.assertGreater(
                final_object_count,
                initial_object_count,
                f"Initial scene objects: {initial_object_count}, Final scene "
                f"objects: {final_object_count}",
            )

            # Verify scores were saved.
            render_dirs = sorted((self.output_dir / "scene_renders").glob("renders_*"))
            self.assertGreater(len(render_dirs), 0, "No render directories found")
            last_render_dir = render_dirs[-1]
            scores_path = last_render_dir / "scores.yaml"
            self.assertTrue(
                scores_path.exists(),
                f"scores.yaml not found in {last_render_dir}",
            )

            # Verify YAML structure.
            with open(scores_path) as f:
                scores = yaml.safe_load(f)
            required_categories = [
                "realism",
                "functionality",
                "layout",
                "completion",
            ]
            for category in required_categories:
                self.assertIn(category, scores, f"Missing category: {category}")
                self.assertIn(
                    "grade",
                    scores[category],
                    f"{category} missing grade",
                )
                self.assertIn(
                    "comment",
                    scores[category],
                    f"{category} missing comment",
                )

        except Exception as e:
            self.fail(
                f"Stateful integration test failed: {e}\n"
                f"Initial scene objects: {initial_object_count}\n"
                f"Final scene objects: {len(self.test_scene.objects)}\n"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
