import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from omegaconf import OmegaConf

from scenesmith.agent_utils.asset_manager import (
    AssetGenerationRequest,
    AssetGenerationResult,
    AssetManager,
    AssetPathConfig,
    FailedAsset,
)
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerResponse,
)
from scenesmith.agent_utils.image_generation import (
    AssetOperationType,
    OpenAIImageGenerator,
)
from scenesmith.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenesmith.agent_utils.room import AgentType, ObjectType, SceneObject
from tests.unit.mock_utils import create_mock_logger


def create_mock_cfg():
    """Create mock configuration for AssetManager tests.

    Uses the config merge pattern to load actual config and override for testing.
    """
    # Load base configuration from actual config file.
    config_path = (
        Path(__file__).parent.parent.parent
        / "configurations/furniture_agent/base_furniture_agent.yaml"
    )
    base_config = OmegaConf.load(config_path)

    # Define test overrides for fast testing.
    test_overrides = {
        "openai": {
            "model": "gpt-4o-mini",  # Cheaper model for testing
        },
        "asset_manager": {
            "general_asset_source": "generated",  # Avoid HSSD client initialization
            "reset_registry_based_on_style_change": True,  # Enable for testing
            "image_generation": {
                "parallel": False,  # Use sequential mode for tests
            },
            "router": {
                "enabled": False,  # Disable router for non-router tests
            },
        },
    }

    # Merge configurations (base config provides all other values).
    return OmegaConf.merge(base_config, test_overrides)


class TestAssetManager(unittest.TestCase):
    """Test AssetManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(self.temp_dir)
        self.mock_logger = create_mock_logger(self.output_dir)

        # Start persistent patches.
        self.patcher_image_gen = patch(
            "scenesmith.agent_utils.asset_manager.create_image_generator"
        )
        self.patcher_geo_client = patch(
            "scenesmith.agent_utils.asset_manager.GeometryGenerationClient"
        )
        # Patch the entire mesh-to-simulation pipeline to avoid complex file setup.
        self.patcher_mesh_conversion = patch.object(
            AssetManager, "_convert_mesh_to_simulation_asset"
        )

        self.patcher_image_gen.start()
        self.patcher_geo_client.start()
        mock_mesh_conversion = self.patcher_mesh_conversion.start()

        # Mock mesh conversion pipeline to return fake SDF path, bounding box, and scale.
        mock_mesh_conversion.return_value = (
            Path("/test/asset.sdf"),
            Path("/test/asset.gltf"),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            1.0,  # initial_scale
        )

        self.asset_manager = AssetManager(
            logger=self.mock_logger,
            vlm_service=MagicMock(),
            blender_server=MagicMock(),
            collision_client=MagicMock(),
            cfg=create_mock_cfg(),
            agent_type=AgentType.FURNITURE,
        )

        # Replace with proper mocks.
        self.mock_image_generator = MagicMock(spec=OpenAIImageGenerator)
        self.asset_manager.image_generator = self.mock_image_generator

        self.mock_geometry_client = MagicMock()
        self.asset_manager.geometry_client = self.mock_geometry_client

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop all patchers.
        self.patcher_mesh_conversion.stop()
        self.patcher_geo_client.stop()
        self.patcher_image_gen.stop()

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_asset_generation_request_creation(self):
        """Test creating AssetGenerationRequest instances."""
        request = AssetGenerationRequest(
            object_descriptions=["Modern sofa", "Coffee table"],
            short_names=["modern_sofa", "coffee_table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.8], [1.2, 0.6, 0.45]],
            style_context="Modern minimalist living room",
            operation_type=AssetOperationType.INITIAL,
        )

        self.assertEqual(len(request.object_descriptions), 2)
        self.assertEqual(request.object_type, ObjectType.FURNITURE)
        self.assertEqual(request.style_context, "Modern minimalist living room")
        self.assertEqual(request.operation_type, AssetOperationType.INITIAL)

    def test_create_scene_object(self):
        """Test creating SceneObject from asset paths."""
        config = AssetPathConfig(
            description="A comfortable test chair",
            short_name="test_chair",
            image_path=Path("/test/image.png"),
            geometry_path=Path("/test/geometry.glb"),
            sdf_dir=Path("/test/sdf"),
        )
        sdf_path = Path("/test/asset.sdf")

        scene_obj = self.asset_manager._create_scene_object(
            config=config,
            object_type=ObjectType.FURNITURE,
            sdf_path=sdf_path,
            final_gltf_path=Path("/test/asset.gltf"),
        )

        self.assertIsInstance(scene_obj, SceneObject)
        self.assertEqual(scene_obj.name, "test_chair")
        self.assertEqual(scene_obj.description, "A comfortable test chair")
        self.assertEqual(scene_obj.object_type, ObjectType.FURNITURE)
        self.assertEqual(scene_obj.image_path, config.image_path)
        self.assertEqual(scene_obj.geometry_path, Path("/test/asset.gltf"))
        self.assertEqual(scene_obj.sdf_path, sdf_path)

    def test_initialization(self):
        """Test AssetManager initialization."""
        self.assertEqual(self.asset_manager.output_dir, self.output_dir)
        self.assertEqual(self.asset_manager.logger, self.mock_logger)

    def test_create_asset_paths_uses_unique_basenames_for_duplicate_short_names(self):
        """Duplicate short names in one batch should not share output paths."""
        asset_paths = self.asset_manager._create_asset_paths(
            object_descriptions=["bed"] * 4,
            short_names=["bed"] * 4,
        )

        self.assertEqual(len({config.image_path for config in asset_paths}), 4)
        self.assertEqual(len({config.geometry_path for config in asset_paths}), 4)
        self.assertEqual(len({config.sdf_dir for config in asset_paths}), 4)
        for config in asset_paths:
            self.assertTrue(config.image_path.name.startswith("bed_"))
            self.assertTrue(config.geometry_path.name.startswith("bed_"))
            self.assertTrue(config.sdf_dir.name.startswith("bed_"))

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_initial_operation(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test generate_assets with INITIAL operation type."""
        # Mock SDF file discovery - return one SDF file.
        mock_sdf_path = Path("/test/asset.sdf")
        mock_scale_mesh.return_value = (mock_sdf_path, 1.0)
        mock_glob.return_value = [mock_sdf_path]

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock asset client to return a valid response object.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [(0, GeometryGenerationServerResponse(geometry_path=str(mock_sdf_path)))]
        )

        request = AssetGenerationRequest(
            object_descriptions=["Modern sofa"],
            short_names=["modern_sofa"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.8]],
            style_context="Modern minimalist living room",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Verify image generation was called with correct contract.
        self.mock_image_generator.generate_images.assert_called_once()
        call_args = self.mock_image_generator.generate_images.call_args
        self.assertEqual(
            call_args.kwargs["style_prompt"], "Modern minimalist living room"
        )
        self.assertEqual(call_args.kwargs["object_descriptions"], ["Modern sofa"])
        self.assertEqual(len(call_args.kwargs["output_paths"]), 1)

        # Verify asset client was called.
        self.mock_geometry_client.generate_geometries.assert_called_once()

        # Verify result contract.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 1)
        self.assertIsInstance(result.successful_assets[0], SceneObject)
        self.assertEqual(result.successful_assets[0].name, "modern_sofa")
        self.assertEqual(result.successful_assets[0].object_type, ObjectType.FURNITURE)
        self.assertIn("generation_timestamp", result.successful_assets[0].metadata)

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_multiple_items(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test generate_assets with multiple items in a batch."""
        # Mock SDF file discovery - return one SDF file per call.
        mock_sdf_paths = [Path("/test/asset1.sdf"), Path("/test/asset2.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock asset client to return valid response objects.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(mock_sdf_paths)
            ]
        )

        descriptions = ["Modern sofa", "Coffee table"]
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=["modern_sofa", "coffee_table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.8], [1.2, 0.6, 0.45]],
            style_context="Modern style",
            operation_type=AssetOperationType.ADDITION,
        )

        result = self.asset_manager.generate_assets(request)

        # Verify batch processing contract.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 2)
        self.mock_geometry_client.generate_geometries.assert_called_once()

        # Verify all items were processed with correct names.
        result_names = {obj.name for obj in result.successful_assets}
        expected_names = {"modern_sofa", "coffee_table"}
        self.assertEqual(result_names, expected_names)

        # Verify all results are proper SceneObjects.
        for obj in result.successful_assets:
            self.assertIsInstance(obj, SceneObject)
            self.assertEqual(obj.object_type, ObjectType.FURNITURE)

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_different_operation_types(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test generate_assets with different operation types."""
        # Mock SDF file discovery.
        mock_glob.return_value = [Path("/test/asset.sdf")]
        mock_scale_mesh.return_value = (Path("/test/asset.sdf"), 1.0)

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        for op_type in [
            AssetOperationType.INITIAL,
            AssetOperationType.ADDITION,
            AssetOperationType.REPLACEMENT,
        ]:
            with self.subTest(operation_type=op_type):
                # Set up mock for this iteration.
                self.mock_geometry_client.generate_geometries.return_value = iter(
                    [
                        (
                            0,
                            GeometryGenerationServerResponse(
                                geometry_path="/test/asset.sdf"
                            ),
                        )
                    ]
                )

                request = AssetGenerationRequest(
                    object_descriptions=["Test item"],
                    short_names=["test_item"],
                    object_type=ObjectType.FURNITURE,
                    desired_dimensions=[[1.0, 1.0, 1.0]],
                    operation_type=op_type,
                )

                result = self.asset_manager.generate_assets(request)

                # Verify contract: returns AssetGenerationResult with successful assets.
                self.assertIsInstance(result, AssetGenerationResult)
                self.assertTrue(result.all_succeeded)
                self.assertEqual(len(result.successful_assets), 1)
                self.assertIsInstance(result.successful_assets[0], SceneObject)

                # Verify image generation was called.
                self.mock_image_generator.generate_images.assert_called()

                # Reset mocks for next iteration.
                self.mock_image_generator.reset_mock()
                self.mock_geometry_client.reset_mock()

    def test_generate_assets_error_handling(self):
        """Test error handling in generate_assets."""
        # Mock image generation to fail.
        self.mock_image_generator.generate_images.side_effect = Exception(
            "Image generation failed"
        )

        request = AssetGenerationRequest(
            object_descriptions=["Test item"],
            short_names=["test_item"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[1.0, 1.0, 1.0]],
        )

        with self.assertRaises(Exception) as context:
            self.asset_manager.generate_assets(request)

        # Verify original error message is preserved (not wrapped).
        self.assertIn("Image generation failed", str(context.exception))

    def test_no_sdf_file_error(self):
        """Test error when no SDF file is generated."""
        # Create empty directory.
        sdf_dir = self.temp_dir / "empty_sdf"
        sdf_dir.mkdir()

        with self.assertRaises(RuntimeError) as context:
            self.asset_manager._find_sdf_file(sdf_dir)

        self.assertIn("No SDF file generated", str(context.exception))

    def test_multiple_sdf_files_error(self):
        """Test error when multiple SDF files are found."""
        # Create directory with multiple SDF files.
        sdf_dir = self.temp_dir / "multi_sdf"
        sdf_dir.mkdir()
        (sdf_dir / "asset1.sdf").touch()
        (sdf_dir / "asset2.sdf").touch()

        with self.assertRaises(RuntimeError) as context:
            self.asset_manager._find_sdf_file(sdf_dir)

        self.assertIn("Multiple SDF files generated", str(context.exception))

    def test_3d_generation_failure(self):
        """Test handling of 3D geometry generation failures."""
        self.mock_geometry_client.generate_geometries.side_effect = RuntimeError(
            "3D generation failed"
        )

        request = AssetGenerationRequest(
            object_descriptions=["Test chair"],
            short_names=["test_chair"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[0.5, 0.5, 0.9]],
        )

        with self.assertRaises(RuntimeError) as context:
            self.asset_manager.generate_assets(request)

        # Verify original error message is preserved (not wrapped at top level).
        self.assertIn("3D generation failed", str(context.exception))

    def test_asset_registry_integration(self):
        """Test that AssetManager integrates with AssetRegistry."""
        # Verify registry is initialized.
        self.assertIsNotNone(self.asset_manager.registry)
        self.assertEqual(self.asset_manager.registry.size(), 0)

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_assets_registered_after_generation(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that generated assets are automatically registered."""
        # Mock SDF file discovery.
        mock_sdf_path = Path("/test/asset.sdf")
        mock_scale_mesh.return_value = (mock_sdf_path, 1.0)
        mock_glob.return_value = [mock_sdf_path]

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock asset client to return a valid response object.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [(0, GeometryGenerationServerResponse(geometry_path=str(mock_sdf_path)))]
        )

        request = AssetGenerationRequest(
            object_descriptions=["Test chair"],
            short_names=["test_chair"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[0.5, 0.5, 0.9]],
        )

        result = self.asset_manager.generate_assets(request)

        # Verify asset was registered.
        self.assertEqual(self.asset_manager.registry.size(), 1)
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 1)

        generated_asset = result.successful_assets[0]
        retrieved_asset = self.asset_manager.get_asset_by_id(generated_asset.object_id)
        self.assertEqual(retrieved_asset, generated_asset)

    @patch("trimesh.load")
    def test_extract_bounds_from_visual_mesh_success(self, mock_trimesh_load):
        """Test that bounds extraction returns correct values from GLTF mesh."""
        # Mock trimesh mesh with known bounds and make it look like real Trimesh.
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_mesh.bounds = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]]
        mock_trimesh_load.return_value = mock_mesh

        # Create required file structure.
        sdf_path = self.temp_dir / "test_asset" / "test_asset.sdf"
        sdf_path.parent.mkdir(parents=True, exist_ok=True)
        sdf_path.write_text("<sdf></sdf>")

        # GLTF file should be alongside the SDF file.
        gltf_path = sdf_path.with_suffix(".gltf")
        gltf_path.write_text("{}")

        # Test the contract: returns tuple of min/max bounds.
        bbox_min, bbox_max = self.asset_manager._extract_bounds_from_visual_mesh(
            sdf_path
        )

        np.testing.assert_array_equal(bbox_min, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(bbox_max, [1.0, 2.0, 0.5])

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_with_duplicates_same_dimensions(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that duplicates with same dimensions are detected and removed."""
        # Mock SDF file discovery - return one SDF file per unique item (3 total).
        mock_sdf_paths = [
            Path("/test/desk.sdf"),
            Path("/test/chair.sdf"),
            Path("/test/printer.sdf"),
        ]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client to return only unique items (3).
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(mock_sdf_paths)
            ]
        )

        # Create request with duplicates.
        descriptions = [
            "Modern office desk",
            "Modern office desk",  # Duplicate of index 0
            "Ergonomic office chair",
            "Ergonomic office chair",  # Duplicate of index 2
            "Commercial laser printer",
        ]
        dimensions = [
            [1.5, 0.75, 0.75],
            [1.5, 0.75, 0.75],  # Same as index 0
            [0.6, 0.6, 1.0],
            [0.6, 0.6, 1.0],  # Same as index 2
            [0.5, 0.5, 0.4],
        ]
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=[
                "office_desk",
                "office_desk_2",
                "office_chair",
                "office_chair_2",
                "laser_printer",
            ],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=dimensions,
            style_context="Modern office",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Verify only unique items were generated (3 instead of 5).
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 3)

        # Verify duplicate info was stored.
        self.assertIsNotNone(self.asset_manager.last_duplicate_info)
        self.assertEqual(len(self.asset_manager.last_duplicate_info), 2)

        # Verify correct duplicates were detected.
        self.assertIn("Modern office desk", self.asset_manager.last_duplicate_info)
        self.assertIn("Ergonomic office chair", self.asset_manager.last_duplicate_info)
        self.assertEqual(
            self.asset_manager.last_duplicate_info["Modern office desk"], [1]
        )
        self.assertEqual(
            self.asset_manager.last_duplicate_info["Ergonomic office chair"], [3]
        )

        # Verify returned objects are correct.
        result_names = {obj.name for obj in result.successful_assets}
        expected_names = {"office_desk", "office_chair", "laser_printer"}
        self.assertEqual(result_names, expected_names)

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_with_duplicates_different_dimensions(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that duplicates with different dimensions are NOT deduplicated."""
        # Mock SDF file discovery.
        mock_sdf_paths = [Path("/test/table1.sdf"), Path("/test/table2.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(mock_sdf_paths)
            ]
        )

        # Same description but different dimensions.
        descriptions = ["Dining table", "Dining table"]
        dimensions = [[1.8, 0.9, 0.75], [2.0, 1.0, 0.75]]  # Different widths
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=["dining_table_1", "dining_table_2"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=dimensions,
            style_context="Modern dining room",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Both should be generated (no deduplication).
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 2)

        # No duplicates should be detected.
        self.assertIsNone(self.asset_manager.last_duplicate_info)

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_no_duplicates(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that no duplicates are detected when all items are unique."""
        # Mock SDF file discovery.
        mock_sdf_paths = [Path("/test/sofa.sdf"), Path("/test/table.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(mock_sdf_paths)
            ]
        )

        descriptions = ["Modern sofa", "Coffee table"]
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=["modern_sofa", "coffee_table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.8], [1.2, 0.6, 0.45]],
            style_context="Modern style",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # All items should be generated.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 2)

        # No duplicates should be detected.
        self.assertIsNone(self.asset_manager.last_duplicate_info)

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_multiple_duplicates_of_same_item(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test detection of multiple duplicates of the same item."""
        # Mock SDF file discovery.
        mock_sdf_paths = [Path("/test/chair.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (
                    0,
                    GeometryGenerationServerResponse(
                        geometry_path=str(mock_sdf_paths[0])
                    ),
                )
            ]
        )

        # Four identical chairs.
        descriptions = ["Dining chair"] * 4
        dimensions = [[0.5, 0.5, 0.9]] * 4
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=["chair_1", "chair_2", "chair_3", "chair_4"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=dimensions,
            style_context="Dining room",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Only one unique item should be generated.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 1)

        # Verify duplicate info.
        self.assertIsNotNone(self.asset_manager.last_duplicate_info)
        self.assertEqual(len(self.asset_manager.last_duplicate_info), 1)
        self.assertIn("Dining chair", self.asset_manager.last_duplicate_info)

        # Three duplicates should be detected (indices 1, 2, 3).
        self.assertEqual(
            self.asset_manager.last_duplicate_info["Dining chair"], [1, 2, 3]
        )

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_partial_success_continues_processing(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that partial success is handled gracefully.

        When one asset fails during conversion, remaining assets should still be
        processed.
        """
        # Mock SDF file discovery for successful assets.
        mock_sdf_paths = [Path("/test/bed.sdf"), Path("/test/chair.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client to return 3 geometries.
        all_geometry_paths = [
            Path("/test/bed.glb"),
            Path("/test/nightstand.glb"),
            Path("/test/chair.glb"),
        ]
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(all_geometry_paths)
            ]
        )

        # Mock _convert_mesh_to_simulation_asset to fail for index 1 (nightstand).
        original_convert = self.asset_manager._convert_mesh_to_simulation_asset

        def mock_convert_with_failure(
            geometry_path, config, object_type, desired_dimensions
        ):
            if "nightstand" in str(geometry_path):
                raise RuntimeError(
                    "Degenerate mesh: Z dimension is too small (0.000028m)"
                )
            # For successful assets, use the original mock behavior from setUp.
            return original_convert(
                geometry_path, config, object_type, desired_dimensions
            )

        with patch.object(
            self.asset_manager,
            "_convert_mesh_to_simulation_asset",
            side_effect=mock_convert_with_failure,
        ):
            request = AssetGenerationRequest(
                object_descriptions=["King bed", "Nightstand", "Accent chair"],
                short_names=["king_bed", "nightstand", "accent_chair"],
                object_type=ObjectType.FURNITURE,
                desired_dimensions=[[2.0, 2.0, 1.0], [0.5, 0.5, 0.6], [0.7, 0.7, 0.9]],
                style_context="Bedroom furniture",
                operation_type=AssetOperationType.INITIAL,
            )

            result = self.asset_manager.generate_assets(request)

        # Verify partial success structure.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertFalse(result.all_succeeded)
        self.assertTrue(result.has_failures)

        # Verify 2 assets succeeded (bed and chair).
        self.assertEqual(len(result.successful_assets), 2)
        success_names = {obj.name for obj in result.successful_assets}
        self.assertEqual(success_names, {"king_bed", "accent_chair"})

        # Verify 1 asset failed (nightstand).
        self.assertEqual(len(result.failed_assets), 1)
        failed_asset = result.failed_assets[0]
        self.assertIsInstance(failed_asset, FailedAsset)
        self.assertEqual(failed_asset.index, 1)
        self.assertEqual(failed_asset.description, "Nightstand")
        self.assertIn("Degenerate mesh", failed_asset.error_message)

        # Verify ALL geometries were attempted (critical benefit of issue #86).
        # The geometry client should have streamed all 3 geometries.
        self.mock_geometry_client.generate_geometries.assert_called_once()

    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_multiple_failures_collected(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that multiple failures are collected and reported.

        Verifies that when multiple assets fail, all failures are collected and
        returned in the result.
        """
        # Mock SDF file discovery for the one successful asset.
        mock_sdf_path = Path("/test/table.sdf")
        mock_glob.side_effect = [[mock_sdf_path]]
        mock_scale_mesh.side_effect = [(mock_sdf_path, 1.0)]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client to return 4 geometries.
        all_geometry_paths = [
            Path("/test/wardrobe.glb"),
            Path("/test/table.glb"),
            Path("/test/dresser.glb"),
            Path("/test/tv_stand.glb"),
        ]
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(all_geometry_paths)
            ]
        )

        # Mock _convert_mesh_to_simulation_asset to fail for indices 0, 2, 3.
        original_convert = self.asset_manager._convert_mesh_to_simulation_asset

        def mock_convert_with_failures(
            geometry_path, config, object_type, desired_dimensions
        ):
            path_str = str(geometry_path)
            if "wardrobe" in path_str:
                raise RuntimeError("Mesh too thin in X dimension")
            elif "dresser" in path_str:
                raise RuntimeError("VLM analysis failed")
            elif "tv_stand" in path_str:
                raise RuntimeError("CoACD decomposition timeout")
            # Table succeeds.
            return original_convert(
                geometry_path, config, object_type, desired_dimensions
            )

        with patch.object(
            self.asset_manager,
            "_convert_mesh_to_simulation_asset",
            side_effect=mock_convert_with_failures,
        ):
            request = AssetGenerationRequest(
                object_descriptions=["Wardrobe", "Coffee table", "Dresser", "TV stand"],
                short_names=["wardrobe", "coffee_table", "dresser", "tv_stand"],
                object_type=ObjectType.FURNITURE,
                desired_dimensions=[
                    [2.0, 0.6, 2.0],
                    [1.2, 0.6, 0.45],
                    [1.5, 0.5, 1.0],
                    [1.8, 0.4, 0.6],
                ],
                style_context="Modern furniture",
                operation_type=AssetOperationType.INITIAL,
            )

            result = self.asset_manager.generate_assets(request)

        # Verify partial success structure.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertFalse(result.all_succeeded)
        self.assertTrue(result.has_failures)

        # Verify 1 asset succeeded (coffee table).
        self.assertEqual(len(result.successful_assets), 1)
        self.assertEqual(result.successful_assets[0].name, "coffee_table")

        # Verify 3 assets failed.
        self.assertEqual(len(result.failed_assets), 3)

        # Verify failure details for each failed asset.
        failed_by_index = {fa.index: fa for fa in result.failed_assets}
        self.assertEqual(set(failed_by_index.keys()), {0, 2, 3})

        # Check wardrobe failure (index 0).
        self.assertEqual(failed_by_index[0].description, "Wardrobe")
        self.assertIn("too thin", failed_by_index[0].error_message)

        # Check dresser failure (index 2).
        self.assertEqual(failed_by_index[2].description, "Dresser")
        self.assertIn("VLM analysis", failed_by_index[2].error_message)

        # Check TV stand failure (index 3).
        self.assertEqual(failed_by_index[3].description, "TV stand")
        self.assertIn("CoACD", failed_by_index[3].error_message)


class TestAssetManagerDimensionControl(unittest.TestCase):
    """Test AssetManager dimension control functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(self.temp_dir)
        self.mock_logger = create_mock_logger(self.output_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_asset_generation_request_with_dimensions(self):
        """Test AssetGenerationRequest with desired_dimensions."""
        request = AssetGenerationRequest(
            object_descriptions=["Modern sofa", "Coffee table"],
            short_names=["modern_sofa", "coffee_table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.85], [1.2, 0.6, 0.45]],
        )

        self.assertEqual(len(request.desired_dimensions), 2)
        self.assertEqual(request.desired_dimensions[0], [2.0, 0.9, 0.85])
        self.assertEqual(request.desired_dimensions[1], [1.2, 0.6, 0.45])

    def test_validate_dimensions_mismatch(self):
        """Test validation error when dimensions don't match descriptions."""
        with (
            patch("scenesmith.agent_utils.asset_manager.create_image_generator"),
            patch("scenesmith.agent_utils.asset_manager.GeometryGenerationClient"),
        ):
            asset_manager = AssetManager(
                logger=self.mock_logger,
                vlm_service=MagicMock(),
                blender_server=MagicMock(),
                collision_client=MagicMock(),
                cfg=create_mock_cfg(),
                agent_type=AgentType.FURNITURE,
            )

        # Create request with mismatched dimensions.
        request = AssetGenerationRequest(
            object_descriptions=["Sofa", "Table"],
            short_names=["sofa", "table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[
                (2.0, 0.9, 0.85)
            ],  # Only one dimension for two objects.
        )

        with self.assertRaises(ValueError) as context:
            asset_manager.generate_assets(request)

        self.assertIn("Mismatch between desired_dimensions", str(context.exception))

    @patch("scenesmith.agent_utils.asset_manager.generate_drake_sdf")
    @patch("scenesmith.agent_utils.asset_manager.canonicalize_mesh")
    @patch("scenesmith.agent_utils.asset_manager.analyze_mesh_orientation_and_material")
    @patch("scenesmith.agent_utils.asset_manager.scale_mesh_uniformly_to_dimensions")
    def test_mesh_scaling_when_dimensions_provided(
        self,
        mock_scale_mesh,
        mock_analyze,
        mock_canon,
        mock_sdf,
    ):
        """Test that mesh scaling is called when dimensions are provided."""
        # Mock VLM analysis.
        mock_analyze.return_value = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        with (
            patch("scenesmith.agent_utils.asset_manager.create_image_generator"),
            patch("scenesmith.agent_utils.asset_manager.GeometryGenerationClient"),
        ):
            asset_manager = AssetManager(
                logger=self.mock_logger,
                vlm_service=MagicMock(),
                blender_server=MagicMock(),
                collision_client=MagicMock(),
                cfg=create_mock_cfg(),
                agent_type=AgentType.FURNITURE,
            )

        # Mock geometry client to return a geometry path.
        mock_response = GeometryGenerationServerResponse(
            geometry_path=str(self.temp_dir / "test.glb")
        )
        asset_manager.geometry_client.generate_geometries = MagicMock(
            return_value=[(0, mock_response)]
        )

        # Mock image generator.
        asset_manager.image_generator.generate_images = MagicMock()

        # Create actual test geometry file.
        test_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        test_geometry_path = Path(mock_response.geometry_path)
        test_geometry_path.parent.mkdir(parents=True, exist_ok=True)
        test_mesh.export(test_geometry_path)

        # Mock the SDF creation and mesh files.
        sdf_dir = self.temp_dir / "generated_assets" / "sdf" / "test_1234567890"
        sdf_dir.mkdir(parents=True, exist_ok=True)
        sdf_path = sdf_dir / "test.sdf"
        gltf_path = sdf_dir / "test.gltf"

        # Create dummy SDF and GLTF files.
        sdf_path.write_text("<sdf></sdf>")
        test_mesh.export(gltf_path)

        # Mock canonicalize_mesh to create canonical file.
        def mock_canon_side_effect(gltf_path, output_path, **kwargs):
            test_mesh.export(output_path)

        mock_canon.side_effect = mock_canon_side_effect

        # Mock scale_mesh_uniformly_to_dimensions to create scaled file.
        def mock_scale_side_effect(
            mesh_path, desired_dimensions, output_path, **kwargs
        ):
            test_mesh.export(output_path)
            return (output_path, 1.5)

        mock_scale_mesh.side_effect = mock_scale_side_effect

        # Mock blender_server.convert_glb_to_gltf to create GLTF file.
        def mock_convert_side_effect(input_path, output_path, export_yup=False):
            test_mesh.export(output_path)
            return output_path

        asset_manager.blender_server.convert_glb_to_gltf.side_effect = (
            mock_convert_side_effect
        )

        # collision_client is already mocked in AssetManager init.

        # Mock _find_sdf_file and _extract_bounds_from_visual_mesh.
        with (
            patch.object(asset_manager, "_find_sdf_file", return_value=sdf_path),
            patch.object(
                asset_manager,
                "_extract_bounds_from_visual_mesh",
                return_value=(
                    np.array([0, 0, 0]),
                    np.array([1, 1, 1]),
                ),
            ),
        ):
            # Create request with dimensions.
            request = AssetGenerationRequest(
                object_descriptions=["Test object"],
                short_names=["test"],
                object_type=ObjectType.FURNITURE,
                desired_dimensions=[[1.8, 0.9, 0.75]],
            )

            # Generate assets.
            asset_manager.generate_assets(request)

        # Verify scale_mesh_uniformly_to_dimensions was called.
        mock_scale_mesh.assert_called_once()
        call_args = mock_scale_mesh.call_args
        self.assertEqual(call_args[1]["desired_dimensions"], [1.8, 0.9, 0.75])
        self.assertFalse(call_args[1]["preserve_aspect_ratio"])
        self.assertEqual(call_args[1]["desired_dimensions_frame"], "drake_yup_to_zup")


if __name__ == "__main__":
    unittest.main()
