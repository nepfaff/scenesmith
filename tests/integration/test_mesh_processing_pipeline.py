"""Integration test for the mesh processing pipeline.

This test validates the end-to-end pipeline from raw mesh to Drake SDF:
1. VLM analysis for orientation, material, and mass
2. Mesh canonicalization with coordinate conversion
3. Uniform scaling to desired dimensions
4. Collision geometry generation via convex decomposition
5. Drake SDF generation

This is a pure integration test with no mocking.
"""

import logging
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import trimesh

from omegaconf import OmegaConf

from scenesmith.agent_utils.blender.server_manager import BlenderServer
from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionServer
from scenesmith.agent_utils.mesh_canonicalization import canonicalize_mesh
from scenesmith.agent_utils.mesh_physics_analyzer import (
    analyze_mesh_orientation_and_material,
)
from scenesmith.agent_utils.mesh_utils import scale_mesh_uniformly_to_dimensions
from scenesmith.agent_utils.room import ObjectType
from scenesmith.agent_utils.sdf_generator import generate_drake_sdf
from scenesmith.agent_utils.vlm_service import VLMService
from tests.integration.common import has_openai_key


@unittest.skipIf(not has_openai_key(), "Requires OpenAI API key")
class TestMeshProcessingPipeline(unittest.TestCase):
    """Integration test for the complete mesh processing pipeline."""

    @staticmethod
    def _verify_collision_visual_alignment(
        visual_mesh_path: Path, sdf_dir: Path
    ) -> None:
        """Verify collision and visual meshes are in the same coordinate system.

        This is a regression test for coordinate system bugs. Drake auto-converts
        Y-up GLTF visual meshes to Z-up but leaves OBJ collision meshes as-is.
        Collision OBJs must be exported in Z-up to match.

        Args:
            visual_mesh_path: Path to visual GLTF mesh (Y-up on disk).
            sdf_dir: Directory containing collision OBJ files.

        Raises:
            AssertionError: If meshes are not aligned in the same coordinate system.
        """
        # Load visual mesh and transform to Z-up (simulate Drake's conversion).
        visual_mesh = trimesh.load(visual_mesh_path, force="mesh")
        yup_to_zup = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        visual_mesh.apply_transform(yup_to_zup)

        visual_dims = visual_mesh.bounds[1] - visual_mesh.bounds[0]
        visual_center = (visual_mesh.bounds[0] + visual_mesh.bounds[1]) / 2

        # Load and combine collision OBJs (should be in Z-up).
        collision_files = list(sdf_dir.glob("*_collision_*.obj"))
        assert len(collision_files) > 0, "Should have collision OBJ files"

        collision_meshes = [trimesh.load(f, force="mesh") for f in collision_files]
        combined_collision = trimesh.util.concatenate(collision_meshes)

        collision_dims = combined_collision.bounds[1] - combined_collision.bounds[0]
        collision_center = (
            combined_collision.bounds[0] + combined_collision.bounds[1]
        ) / 2

        # Verify dimensions match on corresponding axes (not just sorted).
        # If coordinate systems don't match, axes would be swapped.
        # CoACD collision geometry can be slightly larger than visual geometry
        # due to convex hull approximation, so use 10% tolerance.
        np.testing.assert_allclose(
            collision_dims,
            visual_dims,
            rtol=0.10,
            err_msg=(
                f"Collision dims {collision_dims} should match visual dims "
                f"{visual_dims} on corresponding axes (coordinate system test)."
            ),
        )

        # Verify centers are close.
        center_diff = np.abs(visual_center - collision_center)
        max_deviation = 0.1 * np.max(visual_dims)
        assert np.all(center_diff < max_deviation), (
            f"Collision center {collision_center} should align with visual center "
            f"{visual_center} (max deviation: {max_deviation})."
        )

    @staticmethod
    def _verify_sdf_structure(sdf_path: Path, expected_mass: float) -> None:
        """Verify SDF file has valid structure and physics properties.

        Args:
            sdf_path: Path to SDF file.
            expected_mass: Expected mass value from VLM analysis.

        Raises:
            AssertionError: If SDF structure is invalid or mass doesn't match.
        """
        tree = ET.parse(sdf_path)
        root = tree.getroot()

        assert root.tag == "sdf", "Root element should be 'sdf'"
        model = root.find("model")
        assert model is not None, "SDF should contain model element"

        link = model.find("link")
        assert link is not None, "Model should contain link element"

        # Check for required components.
        inertial = link.find("inertial")
        visual = link.find("visual")
        collision = link.find("collision")

        assert inertial is not None, "Link should have inertial properties"
        assert visual is not None, "Link should have visual geometry"
        assert collision is not None, "Link should have collision geometry"

        # Verify mass matches VLM analysis.
        mass_elem = inertial.find("mass")
        assert mass_elem is not None, "Inertial should have mass element"
        mass_value = float(mass_elem.text)
        np.testing.assert_almost_equal(
            mass_value,
            expected_mass,
            decimal=3,
            err_msg="SDF mass should match VLM analysis",
        )

    @staticmethod
    def _verify_uniform_scaling(
        scaled_mesh_path: Path, expected_aspect_ratios: tuple[float, float, float]
    ) -> None:
        """Verify mesh was scaled uniformly (aspect ratios preserved).

        Args:
            scaled_mesh_path: Path to the scaled mesh file.
            expected_aspect_ratios: Expected aspect ratios as (smallest:middle,
                smallest:largest, middle:largest) tuple. For example, a mesh with
                original dimensions [1.0, 0.5, 2.0] has ratios (2.0, 4.0, 2.0).

        Raises:
            AssertionError: If aspect ratios are not preserved.
        """
        scaled_mesh = trimesh.load(scaled_mesh_path, force="mesh")
        actual_dimensions = scaled_mesh.bounds[1] - scaled_mesh.bounds[0]

        # Sort dimensions to compare aspect ratios independent of rotation.
        sorted_dims = np.sort(actual_dimensions)
        smallest, middle, largest = sorted_dims

        # Verify aspect ratios are preserved.
        middle_to_smallest, largest_to_smallest, largest_to_middle = (
            expected_aspect_ratios
        )

        np.testing.assert_almost_equal(
            middle / smallest,
            middle_to_smallest,
            decimal=1,
            err_msg=(
                f"Uniform scaling should preserve middle:smallest ratio "
                f"({middle_to_smallest}:1)"
            ),
        )
        np.testing.assert_almost_equal(
            largest / smallest,
            largest_to_smallest,
            decimal=1,
            err_msg=(
                f"Uniform scaling should preserve largest:smallest ratio "
                f"({largest_to_smallest}:1)"
            ),
        )
        np.testing.assert_almost_equal(
            largest / middle,
            largest_to_middle,
            decimal=1,
            err_msg=(
                f"Uniform scaling should preserve largest:middle ratio "
                f"({largest_to_middle}:1)"
            ),
        )

    def setUp(self):
        """Set up test configuration and temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Start Blender server for mesh rendering in VLM analysis.
        self.blender_server = BlenderServer(
            host="127.0.0.1",
            port_range=(8010, 8020),
        )
        self.blender_server.start()

        # Start convex decomposition server for collision geometry generation.
        self.collision_server = ConvexDecompositionServer(
            port_range=(7100, 7150), omp_threads=4
        )
        self.collision_server.start()
        self.collision_server.wait_until_ready()
        self.collision_client = self.collision_server.get_client()

        # Load base config and merge test overrides.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configurations/furniture_agent"
            / "base_furniture_agent.yaml"
        )
        base_config = OmegaConf.load(config_path)

        # Test overrides for faster testing.
        test_overrides = {
            "openai": {
                "model": "gpt-4o-mini",
            },
        }

        self.config = OmegaConf.merge(base_config, test_overrides)

        # Create VLM service for mesh analysis.
        self.vlm_service = VLMService()

    def tearDown(self):
        """Clean up temporary directory and servers."""
        if self.blender_server is not None and self.blender_server.is_running():
            self.blender_server.stop()
        if self.collision_server is not None and self.collision_server.is_running():
            self.collision_server.stop()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_test_rectangular_mesh(self) -> Path:
        """Create a rectangular box with distinct dimensions for coordinate testing.

        Uses different dimensions on each axis so that coordinate system bugs are
        detectable. A missing Y-up to Z-up transformation would swap Y/Z dimensions,
        causing collision and visual meshes to have mismatched bounding boxes.

        Returns:
            Path to the saved GLB file.
        """
        # Create rectangular box with distinct dimensions [width, depth, height].
        box = trimesh.primitives.Box(extents=[1.0, 0.5, 2.0])

        # Save as GLB.
        mesh_path = self.temp_dir / "test_rectangle.glb"
        box.export(mesh_path)
        return mesh_path

    def test_complete_mesh_processing_pipeline(self):
        """Test the complete pipeline from raw mesh to Drake SDF."""
        # Create test mesh with distinct dimensions for coordinate testing.
        test_mesh_path = self._create_test_rectangular_mesh()
        self.assertTrue(test_mesh_path.exists(), "Test mesh should be created")

        # Convert GLB to Y-up GLTF (required before analysis and canonicalization).
        gltf_path = self.temp_dir / "test_rectangle.gltf"
        self.blender_server.convert_glb_to_gltf(
            input_path=test_mesh_path,
            output_path=gltf_path,
            export_yup=True,
        )
        self.assertTrue(gltf_path.exists(), "Converted GLTF should be created")

        # VLM analysis for orientation, material, and mass.
        analysis = analyze_mesh_orientation_and_material(
            mesh_path=gltf_path,
            vlm_service=self.vlm_service,
            cfg=self.config,
            elevation_degrees=self.config.asset_manager.side_view_elevation_degrees,
            blender_server=self.blender_server,
        )

        # Verify analysis results are reasonable.
        axes = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        self.assertIn(analysis.up_axis, axes, "VLM should predict a valid up axis")
        self.assertIn(
            analysis.front_axis, axes, "VLM should predict a valid front axis"
        )
        self.assertIsInstance(analysis.material, str)
        self.assertGreater(len(analysis.material), 0, "Material should not be empty")
        self.assertGreater(analysis.mass_kg, 0, "Mass should be positive")

        # Canonicalize mesh.
        canonical_path = self.temp_dir / "canonical.gltf"
        canonicalize_mesh(
            gltf_path=gltf_path,
            output_path=canonical_path,
            up_axis=analysis.up_axis,
            front_axis=analysis.front_axis,
            blender_server=self.blender_server,
            object_type=ObjectType.FURNITURE,
        )

        self.assertTrue(canonical_path.exists(), "Canonical mesh should be created")

        # Verify canonicalization.
        # Note: GLTF file is Y-up (Drake will convert to Z-up on load).
        # In Y-up: furniture bottom at y=0, centered in XZ.
        canonical_mesh = trimesh.load(canonical_path, force="mesh")
        bounds = canonical_mesh.bounds
        bbox_min = bounds[0]
        bbox_max = bounds[1]

        # For furniture in Y-up GLTF, bottom should be at y=0.
        self.assertAlmostEqual(
            bbox_min[1],
            0.0,
            places=4,
            msg="Furniture bottom should be at y=0 in Y-up GLTF (Drake converts to Z-up)",
        )

        # Mesh should be centered in XZ plane (Y-up coordinate system).
        center_x = (bbox_min[0] + bbox_max[0]) / 2
        center_z = (bbox_min[2] + bbox_max[2]) / 2
        self.assertAlmostEqual(
            center_x, 0.0, places=4, msg="Mesh should be centered in X"
        )
        self.assertAlmostEqual(
            center_z, 0.0, places=4, msg="Mesh should be centered in Z (Y-up system)"
        )

        # Scale to desired dimensions.
        desired_dimensions = [0.5, 0.5, 0.9]
        scaled_path = self.temp_dir / "scaled.glb"
        scale_mesh_uniformly_to_dimensions(
            mesh_path=canonical_path,
            desired_dimensions=desired_dimensions,
            output_path=scaled_path,
        )

        self.assertTrue(scaled_path.exists(), "Scaled mesh should be created")

        # Verify uniform scaling preserved aspect ratios.
        self._verify_uniform_scaling(
            scaled_mesh_path=scaled_path,
            expected_aspect_ratios=(2.0, 4.0, 2.0),
        )

        # Generate collision geometry via convex decomposition server.
        collision_pieces = self.collision_client.generate_collision_geometry(
            mesh_path=scaled_path, method="coacd", threshold=0.05
        )
        scaled_mesh = trimesh.load(scaled_path, force="mesh")

        # Verify collision pieces were generated.
        self.assertGreater(
            len(collision_pieces),
            0,
            "Should generate at least one collision piece",
        )

        # Each piece should be a valid trimesh.
        for piece in collision_pieces:
            self.assertIsInstance(piece, trimesh.Trimesh)
            self.assertGreater(len(piece.vertices), 0)
            self.assertGreater(len(piece.faces), 0)

        # Generate Drake SDF.
        sdf_path = self.temp_dir / "test_asset.sdf"
        generate_drake_sdf(
            visual_mesh_path=scaled_path,
            collision_pieces=collision_pieces,
            physics_analysis=analysis,
            output_path=sdf_path,
        )

        # Verify SDF is valid and collision/visual meshes are aligned.
        self.assertTrue(sdf_path.exists(), "SDF file should be created")
        self._verify_collision_visual_alignment(scaled_path, sdf_path.parent)
        self._verify_sdf_structure(sdf_path, analysis.mass_kg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
