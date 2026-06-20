import logging
import shutil
import tempfile
import time
import unittest

from pathlib import Path

import trimesh

from PIL import Image, ImageDraw

# isort: off
# Configure CUDA environment before any CUDA-dependent imports.
# This is required for nvdiffrast JIT compilation used by SAM 3D Objects.
try:
    from scenesmith.agent_utils.geometry_generation_server.cuda_env_setup import (
        ensure_cuda_env,
    )

    ensure_cuda_env()
except (ImportError, RuntimeError) as e:
    # CUDA setup may not be available in CI or on systems without CUDA.
    # Note: console_logger not yet defined, use inline logger here.
    logging.getLogger(__name__).warning(f"CUDA environment setup skipped: {e}")
# isort: on

from scenesmith.agent_utils.blender.server_manager import BlenderServer
from scenesmith.agent_utils.geometry_generation_server.geometry_generation import (
    generate_geometry_from_image,
)
from scenesmith.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
    SAM3DPipelineManager,
)
from scenesmith.agent_utils.mesh_utils import load_mesh_as_trimesh
from tests.integration.common import has_gpu_available, is_github_actions

console_logger = logging.getLogger(__name__)


@unittest.skipIf(
    not has_gpu_available() or is_github_actions(),
    "Requires GPU for SAM3D generation and non-CI environment",
)
class TestSAM3DPipelineIntegration(unittest.TestCase):
    """Integration test for the complete SAM3D pipeline.

    Tests the entire SAM3D workflow:
    1. Image segmentation using SAM3
    2. 3D mesh generation from mask using SAM 3D Objects
    3. Output validation
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.debug_dir = self.temp_dir / "debug_sam3d"
        self.debug_dir.mkdir(exist_ok=True)

        # Check if SAM3D is installed.
        sam3_checkpoint = Path("external/checkpoints/sam3.pt")
        sam3d_checkpoint = Path("external/checkpoints/pipeline.yaml")

        if not sam3_checkpoint.exists() or not sam3d_checkpoint.exists():
            self.skipTest(
                "SAM3D checkpoints not found. Run scripts/install_sam3d.sh first."
            )

        self.sam3d_config = {
            "sam3_checkpoint": str(sam3_checkpoint),
            "sam3d_checkpoint": str(sam3d_checkpoint),
            "mode": "foreground",
            "text_prompt": None,
            "threshold": 0.5,
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_sam3d_foreground_segmentation(self):
        """Test SAM3D pipeline with foreground segmentation mode.

        Verifies that the SAM3D pipeline can generate a 3D mesh from an image
        using automatic foreground segmentation.
        """
        # Create a simple test image with object on white background.
        test_image = Image.new("RGB", (512, 512), color=(255, 255, 255))
        # Draw a red square in the center (foreground object).
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([(156, 156), (356, 356)], fill=(255, 0, 0))

        image_path = self.debug_dir / "test_object.png"
        test_image.save(image_path)

        output_path = self.debug_dir / "output_foreground.glb"

        # Generate 3D mesh using SAM3D backend.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            backend="sam3d",
            sam3d_config=self.sam3d_config,
            debug_folder=self.debug_dir,
        )

        # Verify output file exists.
        self.assertTrue(output_path.exists(), "Output mesh file should exist")

        # Verify mesh is valid.
        mesh = load_mesh_as_trimesh(output_path)
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertGreater(len(mesh.vertices), 0, "Mesh should have vertices")
        self.assertGreater(len(mesh.faces), 0, "Mesh should have faces")

        # Verify debug outputs.
        mask_path = self.debug_dir / "test_object_mask.png"
        masked_path = self.debug_dir / "test_object_masked.png"
        self.assertTrue(mask_path.exists(), "Mask debug image should exist")
        self.assertTrue(masked_path.exists(), "Masked image should exist")

    def test_sam3d_text_segmentation(self):
        """Test SAM3D pipeline with object description segmentation mode.

        Verifies that the SAM3D pipeline can generate a 3D mesh from an image
        using object description-based segmentation with a custom prompt.

        Note: We draw a blue circle (ellipse) and use "circle" as the prompt
        since SAM3 correctly identifies 2D shapes. Using "ball" would fail
        because a ball is a 3D sphere concept that requires shading to depict.
        SAM3 recognizes "circle" with 0.917 confidence.
        """
        # Create a test image.
        test_image = Image.new("RGB", (512, 512), color=(255, 255, 255))

        draw = ImageDraw.Draw(test_image)
        # Draw a blue circle (SAM3 recognizes as "circle", not "ball").
        draw.ellipse([(156, 156), (356, 356)], fill=(0, 0, 255))

        image_path = self.debug_dir / "test_circle.png"
        test_image.save(image_path)

        output_path = self.debug_dir / "output_text.glb"

        # Update config for object_description mode.
        text_config = self.sam3d_config.copy()
        text_config["mode"] = "object_description"
        text_config["object_description"] = "circle"

        # Generate 3D mesh using SAM3D backend with text prompt.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            backend="sam3d",
            sam3d_config=text_config,
            debug_folder=self.debug_dir,
        )

        # Verify output file exists.
        self.assertTrue(output_path.exists(), "Output mesh file should exist")

        # Verify mesh is valid.
        mesh = load_mesh_as_trimesh(output_path)
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertGreater(len(mesh.vertices), 0, "Mesh should have vertices")
        self.assertGreater(len(mesh.faces), 0, "Mesh should have faces")

    def test_sam3d_pipeline_caching(self):
        """Test that SAM3D pipeline caching works correctly.

        Verifies that subsequent generations reuse cached pipelines
        for improved performance.
        """
        # Reset pipelines to start fresh.
        SAM3DPipelineManager.reset_pipelines()
        self.assertFalse(SAM3DPipelineManager.are_pipelines_loaded())

        # Create test image.
        test_image = Image.new("RGB", (256, 256), color=(200, 200, 200))

        draw = ImageDraw.Draw(test_image)
        draw.rectangle([(78, 78), (178, 178)], fill=(100, 100, 255))

        image_path = self.debug_dir / "test_cache.png"
        test_image.save(image_path)

        output_path1 = self.debug_dir / "output_cache1.glb"

        # First generation should load pipelines.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path1,
            backend="sam3d",
            sam3d_config=self.sam3d_config,
            use_pipeline_caching=True,
        )

        # Pipelines should now be loaded.
        self.assertTrue(SAM3DPipelineManager.are_pipelines_loaded())

        # Second generation should reuse cached pipelines.
        output_path2 = self.debug_dir / "output_cache2.glb"
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path2,
            backend="sam3d",
            sam3d_config=self.sam3d_config,
            use_pipeline_caching=True,
        )

        # Verify both outputs exist.
        self.assertTrue(output_path1.exists())
        self.assertTrue(output_path2.exists())

        # Clean up pipelines.
        SAM3DPipelineManager.reset_pipelines()
        self.assertFalse(SAM3DPipelineManager.are_pipelines_loaded())

    def _render_mesh_to_image(self, mesh_path: Path, output_path: Path) -> None:
        """Render mesh to PNG using the Blender server render path."""
        render_dir = output_path.parent / "office_shelf_render_views"
        server = BlenderServer(port_range=(8000, 8350))
        try:
            server.start()
            image_paths = server.render_multiview_for_analysis(
                mesh_path=mesh_path,
                output_dir=render_dir,
                elevation_degrees=20.0,
                num_side_views=4,
                include_vertical_views=True,
            )
            self.assertGreater(len(image_paths), 0, "Render should produce images")
            shutil.copy2(image_paths[0], output_path)
        finally:
            server.stop()

    def test_sam3d_real_image_with_rendering(self):
        """Test SAM3D pipeline with real office shelf image and render output.

        Uses a real product image instead of synthetic shapes to better
        validate the SAM3D pipeline quality. Outputs are saved to a persistent
        directory for visual inspection.
        """
        # Use persistent output directory for visual inspection.
        test_output_dir = Path(__file__).parent / "test_outputs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = test_output_dir / f"sam3d_office_shelf_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use real test image.
        image_path = Path(__file__).parent.parent / "test_data" / "office_shelf.png"
        self.assertTrue(image_path.exists(), f"Test image not found: {image_path}")

        output_path = output_dir / "office_shelf_3d.glb"

        # Generate 3D mesh using SAM3D with foreground segmentation.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            backend="sam3d",
            sam3d_config=self.sam3d_config,
            debug_folder=output_dir,
        )

        # Verify output mesh exists.
        self.assertTrue(output_path.exists(), "Output mesh should exist")

        # Load and validate mesh.
        mesh = load_mesh_as_trimesh(output_path)
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertGreater(len(mesh.vertices), 0, "Mesh should have vertices")
        self.assertGreater(len(mesh.faces), 0, "Mesh should have faces")

        # Render mesh to image for visual inspection.
        render_path = output_dir / "office_shelf_render.png"
        self._render_mesh_to_image(mesh_path=output_path, output_path=render_path)
        self.assertTrue(render_path.exists(), "Render should exist")

        # Log output location for manual inspection.
        console_logger.info(f"Test outputs saved to: {output_dir}")


if __name__ == "__main__":
    unittest.main()
