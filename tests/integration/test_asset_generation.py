import logging
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import trimesh

from omegaconf import OmegaConf

# isort: off
# Need to import bpy first to avoid potential symbol loading issues.
try:
    import bpy  # noqa: F401
except ImportError:
    bpy = None

# isort: on

from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionServer
from scenesmith.agent_utils.geometry_generation_server.geometry_generation import (
    generate_geometry_from_image,
)
from scenesmith.agent_utils.image_generation import create_image_generator
from scenesmith.agent_utils.mesh_canonicalization import canonicalize_mesh
from scenesmith.agent_utils.mesh_physics_analyzer import (
    analyze_mesh_orientation_and_material,
)
from scenesmith.agent_utils.mesh_utils import (
    convert_glb_to_gltf,
    scale_mesh_uniformly_to_dimensions,
)
from scenesmith.agent_utils.room import ObjectType
from scenesmith.agent_utils.sdf_generator import generate_drake_sdf
from scenesmith.agent_utils.vlm_service import VLMService
from tests.integration.common import (
    has_gpu_available,
    has_hunyuan3d_installed,
    has_openai_key,
    is_github_actions,
)


@unittest.skipIf(
    not has_openai_key()
    or not has_gpu_available()
    or not has_hunyuan3d_installed()
    or is_github_actions()
    or bpy is None
    or isinstance(bpy, Mock),
    "Requires OpenAI API key, GPU, Hunyuan3D-2, and non-CI environment",
)
class TestAssetGenerationIntegration(unittest.TestCase):
    """Integration test for the complete asset generation pipeline.

    Tests the entire workflow from asset_generation.py:
    1. Contextual image generation with style consistency
    2. 3D geometry generation from images
    3. Conversion to Drake simulation assets
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.debug_dir = self.temp_dir / "debug_asset_gen"
        self.debug_dir.mkdir(exist_ok=True)
        self.vlm_service = VLMService()

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

    def tearDown(self):
        """Clean up test fixtures."""
        if self.collision_server is not None and self.collision_server.is_running():
            self.collision_server.stop()
        shutil.rmtree(self.temp_dir)

    def test_complete_asset_generation_pipeline(self):
        """Test the complete asset generation pipeline end-to-end.

        Verifies that the entire pipeline completes without errors
        and produces the expected output files.
        """
        # Define file paths.
        table_path = self.debug_dir / "table.png"
        chair_path = self.debug_dir / "chair.png"
        table_glb_path = self.debug_dir / "table.glb"

        style_prompt = "A modern kitchen with a table for four people made of wood."

        # Step 1: Generate images for assets using configured backend.
        image_gen_config = self.config.asset_manager.image_generation
        generator = create_image_generator(
            backend=image_gen_config.backend,
            config=image_gen_config,
        )
        generator.generate_images(
            style_prompt=style_prompt,
            object_descriptions=["A table", "A chair"],
            output_paths=[table_path, chair_path],
        )

        # Verify images were created.
        self.assertTrue(table_path.exists(), "Table image should be created")
        self.assertTrue(chair_path.exists(), "Chair image should be created")
        self.assertGreater(
            table_path.stat().st_size, 0, "Table image should not be empty"
        )
        self.assertGreater(
            chair_path.stat().st_size, 0, "Chair image should not be empty"
        )

        # Step 2: Generate 3D geometry from image.
        generate_geometry_from_image(
            image_path=table_path,
            output_path=table_glb_path,
            debug_folder=self.debug_dir,
        )

        # Verify geometry was created.
        self.assertTrue(table_glb_path.exists(), "3D geometry file should be created")
        self.assertGreater(
            table_glb_path.stat().st_size, 0, "Geometry file should not be empty"
        )

        # Verify debug image was created.
        debug_image = self.debug_dir / f"{table_path.stem}_without_background.png"
        self.assertTrue(
            debug_image.exists(), "Debug background-removed image should be created"
        )

        # Step 4: Convert geometry to Drake simulation asset.
        try:
            # Convert GLB to Y-up GLTF (required before canonicalization).
            gltf_path = self.debug_dir / "table.gltf"
            convert_glb_to_gltf(
                input_path=table_glb_path,
                output_path=gltf_path,
                export_yup=True,
            )

            # VLM analysis for orientation, material, and mass.
            analysis = analyze_mesh_orientation_and_material(
                mesh_path=gltf_path,
                vlm_service=self.vlm_service,
                cfg=self.config,
                elevation_degrees=self.config.asset_manager.side_view_elevation_degrees,
            )

            # Canonicalize mesh.
            canonical_path = self.debug_dir / "table_canonical.gltf"
            canonicalize_mesh(
                gltf_path=gltf_path,
                output_path=canonical_path,
                up_axis=analysis.up_axis,
                front_axis=analysis.front_axis,
                object_type=ObjectType.FURNITURE,
            )

            # Scale to desired dimensions (example: 1.0m x 0.6m x 0.75m table).
            scaled_path = self.debug_dir / "table_scaled.glb"
            scale_mesh_uniformly_to_dimensions(
                mesh_path=canonical_path,
                desired_dimensions=[1.0, 0.6, 0.75],
                output_path=scaled_path,
            )

            # Generate collision geometry via convex decomposition server.
            collision_pieces = self.collision_client.generate_collision_geometry(
                mesh_path=scaled_path, method="coacd", threshold=0.05
            )
            scaled_mesh = trimesh.load(scaled_path, force="mesh")

            # Verify bbox extracted from mesh matches mesh geometry.
            # Extract bbox in Y-up coordinates (GLTF native) like asset_manager does.
            # The bbox is kept in Y-up to match what trimesh loads from the GLTF.
            bounds = scaled_mesh.bounds
            bbox_min = bounds[0]
            bbox_max = bounds[1]

            # Now reload mesh and verify vertices are within bbox.
            test_mesh = trimesh.load(scaled_path, force="mesh")
            vertices = test_mesh.vertices

            # Check if all vertices are within the extracted bbox.
            vertices_inside = np.all(
                (vertices >= bbox_min) & (vertices <= bbox_max), axis=1
            )
            num_outside = np.sum(~vertices_inside)
            if num_outside > 0:
                # Provide diagnostic info.
                outside_vertices = vertices[~vertices_inside]
                logging.error(f"\n{'='*60}")
                logging.error(f"BBOX-MESH MISMATCH DETECTED!")
                logging.error(f"{'='*60}")
                logging.error(f"BBox: min={bbox_min}, max={bbox_max}")
                logging.error(f"Vertices outside bbox: {num_outside}/{len(vertices)}")
                logging.error(f"Example outside vertices (first 5):")
                for i, v in enumerate(outside_vertices[:5]):
                    logging.error(f"  {i}: {v}")
                logging.error(f"Mesh bounds: {test_mesh.bounds}")
                logging.error(f"{'='*60}\n")
            self.assertEqual(
                num_outside, 0, f"{num_outside}/{len(vertices)} vertices outside bbox."
            )

            # Generate Drake SDF.
            sdf_path = self.debug_dir / "table.sdf"
            generate_drake_sdf(
                visual_mesh_path=scaled_path,
                collision_pieces=collision_pieces,
                physics_analysis=analysis,
                output_path=sdf_path,
            )

            # Verify SDF was created.
            self.assertTrue(sdf_path.exists(), "SDF file should be created")

        except Exception as e:
            self.fail(f"Unexpected error in mesh processing pipeline: {e}")

        # Final verification: All expected files exist.
        self.assertTrue(table_path.exists())
        self.assertTrue(chair_path.exists())
        self.assertTrue(table_glb_path.exists())

    def test_vhacd_collision_geometry(self):
        """Test V-HACD convex decomposition produces valid collision geometry.

        Ensures vhacdx is properly installed and working. This test was added
        after a production failure where missing vhacdx caused 500 errors.
        """
        # Create a simple test mesh (box with dimensions 0.5x0.3x0.4m).
        test_mesh = trimesh.creation.box(extents=[0.5, 0.3, 0.4])
        mesh_path = self.debug_dir / "test_box.glb"
        test_mesh.export(mesh_path)

        # Generate collision geometry using V-HACD.
        collision_pieces = self.collision_client.generate_collision_geometry(
            mesh_path=mesh_path, method="vhacd", max_convex_hulls=8
        )

        # Verify we got valid collision geometry.
        self.assertGreater(len(collision_pieces), 0, "V-HACD should produce pieces")
        for piece in collision_pieces:
            self.assertIsInstance(piece, trimesh.Trimesh)
            self.assertTrue(piece.is_convex, "V-HACD pieces should be convex")

    def test_coacd_collision_geometry(self):
        """Test CoACD convex decomposition produces valid collision geometry.

        Ensures coacd is properly installed and working alongside V-HACD.
        """
        # Create a simple test mesh (box with dimensions 0.5x0.3x0.4m).
        test_mesh = trimesh.creation.box(extents=[0.5, 0.3, 0.4])
        mesh_path = self.debug_dir / "test_box_coacd.glb"
        test_mesh.export(mesh_path)

        # Generate collision geometry using CoACD.
        collision_pieces = self.collision_client.generate_collision_geometry(
            mesh_path=mesh_path, method="coacd", threshold=0.05
        )

        # Verify we got valid collision geometry.
        self.assertGreater(len(collision_pieces), 0, "CoACD should produce pieces")
        for piece in collision_pieces:
            self.assertIsInstance(piece, trimesh.Trimesh)
            self.assertTrue(piece.is_convex, "CoACD pieces should be convex")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
