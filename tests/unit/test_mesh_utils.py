import shutil
import tempfile
import unittest

from pathlib import Path

import numpy as np
import trimesh

from scenesmith.agent_utils.blender.canonicalization import choose_fallback_front_axis
from scenesmith.agent_utils.mesh_utils import (
    remove_mesh_floaters,
    scale_mesh_uniformly_to_dimensions,
)
from scenesmith.agent_utils.sdf_generator import YUP_TO_ZUP_TRANSFORM


class TestScaleMeshToDimensions(unittest.TestCase):
    """Test the scale_mesh_uniformly_to_dimensions function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_scale_mesh_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        # Create a test mesh.
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        input_path = self.temp_path / "cube.glb"
        mesh.export(input_path)

        # Test zero dimension.
        with self.assertRaises(ValueError) as context:
            scale_mesh_uniformly_to_dimensions(
                mesh_path=input_path,
                desired_dimensions=[1.0, 0.0, 1.0],
                output_path=self.temp_path / "output.glb",
            )

        self.assertIn("All dimensions must be positive", str(context.exception))

        # Test negative dimension.
        with self.assertRaises(ValueError) as context:
            scale_mesh_uniformly_to_dimensions(
                mesh_path=input_path,
                desired_dimensions=[1.0, 1.0, -1.0],
                output_path=self.temp_path / "output.glb",
            )

        self.assertIn("All dimensions must be positive", str(context.exception))

    def test_scale_mesh_with_scene_object(self):
        """Test scaling a mesh file that loads as a Scene (multiple meshes)."""
        # Create two separate meshes that will be part of a scene.
        mesh1 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        mesh2 = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
        mesh2.apply_translation([0.6, 0, 0])  # Offset second box.

        # Create a scene.
        scene = trimesh.Scene([mesh1, mesh2])
        input_path = self.temp_path / "multi_mesh_scene.glb"
        scene.export(input_path)

        # Scale the entire scene.
        desired_dims = [2.0, 1.0, 1.0]
        output_path = self.temp_path / "scene_scaled.glb"

        scale_mesh_uniformly_to_dimensions(
            mesh_path=input_path,
            desired_dimensions=desired_dims,
            output_path=output_path,
        )

        # Load and verify dimensions fit within bounds.
        scaled_mesh = trimesh.load(output_path, force="mesh")
        scaled_dims = scaled_mesh.bounds[1] - scaled_mesh.bounds[0]

        # Dimensions should fit within desired bounds.
        self.assertTrue(
            np.all(scaled_dims <= np.array(desired_dims) + 1e-5),
            f"Scene dimensions {scaled_dims} exceed desired {desired_dims}",
        )

    def test_uniform_scaling_preserves_proportions(self):
        """Test that uniform scaling preserves the original mesh proportions."""
        # Create a mesh with specific proportions (2:1:0.5 ratio).
        mesh = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
        input_path = self.temp_path / "proportional_box.glb"
        mesh.export(input_path)

        # Request different dimensions but uniform scaling should preserve ratios.
        desired_dims = [4.0, 3.0, 2.0]
        output_path = self.temp_path / "uniform_scaled.glb"

        scale_mesh_uniformly_to_dimensions(
            mesh_path=input_path,
            desired_dimensions=desired_dims,
            output_path=output_path,
        )

        # Load and verify.
        scaled_mesh = trimesh.load(output_path, force="mesh")
        scaled_dims = scaled_mesh.bounds[1] - scaled_mesh.bounds[0]

        # Calculate expected uniform scale (average ratio).
        # Original: [2.0, 1.0, 0.5]
        # Desired: [4.0, 3.0, 2.0]
        # Ratios: [2.0, 3.0, 4.0]
        # Average ratio: 3.0
        # Expected result: [6.0, 3.0, 1.5]

        expected_dims = np.array([6.0, 3.0, 1.5])
        np.testing.assert_allclose(
            scaled_dims,
            expected_dims,
            rtol=1e-5,
            err_msg="Uniform scaling did not preserve proportions",
        )

        # Verify proportions are preserved (2:1:0.5 ratio).
        original_ratios = np.array([2.0, 1.0, 0.5]) / 0.5
        scaled_ratios = scaled_dims / scaled_dims[2]
        np.testing.assert_allclose(
            scaled_ratios,
            original_ratios,
            rtol=1e-5,
            err_msg="Proportions were not preserved",
        )

    def test_uniform_scaling_scales_by_average_factor(self):
        """Test that uniform scaling uses the average scale factor."""
        # Create a mesh.
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        input_path = self.temp_path / "cube.glb"
        mesh.export(input_path)

        # Request asymmetric dimensions.
        desired_dims = [3.0, 2.0, 4.0]
        output_path = self.temp_path / "avg_scaled.glb"

        scale_mesh_uniformly_to_dimensions(
            mesh_path=input_path,
            desired_dimensions=desired_dims,
            output_path=output_path,
        )

        # Load and verify.
        scaled_mesh = trimesh.load(output_path, force="mesh")
        scaled_dims = scaled_mesh.bounds[1] - scaled_mesh.bounds[0]

        # Average scale factor is 3.0 (mean of [3.0, 2.0, 4.0]).
        # Expected result: [3.0, 3.0, 3.0] (all scaled by 3.0).
        expected_dims = np.array([3.0, 3.0, 3.0])
        np.testing.assert_allclose(
            scaled_dims,
            expected_dims,
            rtol=1e-5,
            err_msg="Did not scale by average factor",
        )

    def test_exact_drake_frame_scaling_for_yup_gltf(self):
        """Test per-axis scaling for GLTFs that Drake converts from Y-up to Z-up."""
        # In a Y-up GLTF, mesh Y becomes Drake Z and mesh Z becomes Drake Y.
        mesh = trimesh.creation.box(extents=[1.0, 0.25, 0.8])
        input_path = self.temp_path / "yup_box.glb"
        mesh.export(input_path)

        desired_drake_dims = [0.85, 0.85, 0.42]
        output_path = self.temp_path / "scaled_yup_box.glb"

        scale_mesh_uniformly_to_dimensions(
            mesh_path=input_path,
            desired_dimensions=desired_drake_dims,
            output_path=output_path,
            preserve_aspect_ratio=False,
            desired_dimensions_frame="drake_yup_to_zup",
        )

        scaled_mesh = trimesh.load(output_path, force="mesh")
        scaled_mesh.apply_transform(YUP_TO_ZUP_TRANSFORM)
        scaled_drake_dims = scaled_mesh.bounds[1] - scaled_mesh.bounds[0]

        np.testing.assert_allclose(
            scaled_drake_dims,
            np.array(desired_drake_dims),
            rtol=1e-5,
            err_msg="Y-up GLTF scaling did not match Drake-frame dimensions",
        )


class TestRemoveMeshFloaters(unittest.TestCase):
    """Test the remove_mesh_floaters function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_remove_floaters_single_component(self):
        """Test that a single component mesh remains unchanged."""
        # Create a simple cube mesh.
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        input_path = self.temp_path / "single_cube.glb"
        mesh.export(input_path)

        # Apply floater removal.
        output_path = self.temp_path / "cleaned.glb"
        result_path = remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path,
            distance_threshold=0.05,
        )

        # Load the result.
        cleaned_mesh = trimesh.load(result_path, force="mesh")

        # Verify the mesh is essentially the same.
        self.assertEqual(result_path, output_path)
        np.testing.assert_allclose(
            mesh.bounds, cleaned_mesh.bounds, rtol=1e-5, err_msg="Mesh was modified"
        )

    def test_remove_floaters_removes_far_floaters(self):
        """Test that floaters far from main mesh are removed."""
        # Create main mesh (large cube).
        main_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        # Create small floater (sphere with much smaller volume).
        # Place it 1.5m away (far beyond the 0.05m default threshold).
        floater = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
        floater.apply_translation([1.5, 0, 0])

        # Combine into scene and export.
        combined = trimesh.util.concatenate([main_mesh, floater])
        input_path = self.temp_path / "with_floater.glb"
        combined.export(input_path)

        # Apply floater removal with 5cm distance threshold.
        output_path = self.temp_path / "without_floater.glb"
        remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path,
            distance_threshold=0.05,
        )

        # Load cleaned mesh.
        cleaned_mesh = trimesh.load(output_path, force="mesh")

        # Verify floater was removed (volume should be close to main mesh only).
        main_volume = main_mesh.volume
        cleaned_volume = cleaned_mesh.volume

        # Cleaned volume should be very close to main mesh volume.
        self.assertAlmostEqual(
            cleaned_volume,
            main_volume,
            delta=0.01,
            msg="Far floater was not removed",
        )

        # Verify the cleaned mesh only has one connected component.
        components = cleaned_mesh.split()
        self.assertEqual(len(components), 1, "Cleaned mesh should have one component")

    def test_remove_floaters_keeps_close_components(self):
        """Test that components close to main mesh are preserved."""
        # Create two meshes close together (should both be kept).
        mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh2 = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
        # Place mesh2 only 0.02m away (within 0.05m threshold).
        mesh2.apply_translation([0.52, 0, 0])

        # Combine meshes.
        combined = trimesh.util.concatenate([mesh1, mesh2])
        combined_volume = combined.volume

        input_path = self.temp_path / "two_close.glb"
        combined.export(input_path)

        # Apply floater removal with 5cm distance threshold.
        # mesh2 is close (0.02m away) so should be kept.
        output_path = self.temp_path / "both_kept.glb"
        remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path,
            distance_threshold=0.05,
        )

        # Load cleaned mesh.
        cleaned_mesh = trimesh.load(output_path, force="mesh")

        # Both components should be kept.
        self.assertAlmostEqual(
            cleaned_mesh.volume,
            combined_volume,
            delta=0.01,
            msg="Close component was incorrectly removed",
        )

        # Verify we still have two components.
        components = cleaned_mesh.split()
        self.assertEqual(len(components), 2, "Should keep both close components")

    def test_remove_floaters_different_thresholds(self):
        """Test that different distance threshold values work correctly."""
        # Create main mesh (1x1x1 cube, bounds: [-0.5, 0.5] on each axis).
        main_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        # Create component at medium distance (0.3x0.3x0.3 cube).
        medium_mesh = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
        # Place it to create 0.1m gap on x-axis.
        # Main mesh x-max is 0.5, medium mesh x-extent is 0.3 (half-width 0.15).
        # To get 0.1m gap: position = 0.5 + 0.1 + 0.15 = 0.75
        medium_mesh.apply_translation([0.75, 0, 0])

        combined = trimesh.util.concatenate([main_mesh, medium_mesh])
        input_path = self.temp_path / "medium_distance.glb"
        combined.export(input_path)

        # Test with 0.15m threshold - should keep component (gap is 0.1m).
        output_path_15cm = self.temp_path / "threshold_15cm.glb"
        remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path_15cm,
            distance_threshold=0.15,
        )
        cleaned_15cm = trimesh.load(output_path_15cm, force="mesh")
        components_15cm = cleaned_15cm.split()
        self.assertEqual(
            len(components_15cm), 2, "0.15m threshold should keep component at 0.1m gap"
        )

        # Test with 0.05m threshold - should remove component (gap is 0.1m).
        output_path_5cm = self.temp_path / "threshold_5cm.glb"
        remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path_5cm,
            distance_threshold=0.05,
        )
        cleaned_5cm = trimesh.load(output_path_5cm, force="mesh")
        components_5cm = cleaned_5cm.split()
        self.assertEqual(
            len(components_5cm),
            1,
            "0.05m threshold should remove component at 0.1m gap",
        )

    def test_remove_floaters_handles_scene_objects(self):
        """Test that function properly handles trimesh.Scene objects."""
        # Create meshes and combine into a scene.
        mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        floater = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
        # Place floater 1.5m away (far beyond 0.05m threshold).
        floater.apply_translation([1.5, 0, 0])

        # Create and export a scene.
        scene = trimesh.Scene([mesh1, floater])
        input_path = self.temp_path / "scene_with_floater.glb"
        scene.export(input_path)

        # Apply floater removal.
        output_path = self.temp_path / "scene_cleaned.glb"
        remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path,
            distance_threshold=0.05,
        )

        # Load and verify.
        cleaned_mesh = trimesh.load(output_path, force="mesh")

        # Should have removed the far floater.
        components = cleaned_mesh.split()
        self.assertEqual(
            len(components), 1, "Scene with floater should be cleaned to one component"
        )

    def test_remove_floaters_large_threshold(self):
        """Test that very large threshold keeps all components."""
        # Create main mesh and far floater.
        main_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        floater = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
        floater.apply_translation([2.0, 0, 0])

        combined = trimesh.util.concatenate([main_mesh, floater])
        combined_volume = combined.volume

        input_path = self.temp_path / "with_floater.glb"
        combined.export(input_path)

        # Apply with very large threshold (1000m - effectively infinite).
        output_path = self.temp_path / "large_threshold.glb"
        remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path,
            distance_threshold=1000.0,
        )

        # Load and verify all components kept.
        cleaned_mesh = trimesh.load(output_path, force="mesh")

        # Both components should be kept.
        self.assertAlmostEqual(
            cleaned_mesh.volume,
            combined_volume,
            delta=0.01,
            msg="Large threshold should keep all components",
        )

    def test_remove_floaters_distance_based_removal(self):
        """Test distance-based removal: keeps close small parts, removes far large parts.

        This test demonstrates the key improvement of the distance-based approach
        over volume-based removal:
        1. Small components close to the main mesh are KEPT (e.g., handles, knobs)
        2. Large components far from the main mesh are REMOVED (actual floaters)
        """
        # Create main mesh: 1x1x1 cube (bounds: [-0.5, 0.5] on each axis).
        main_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        # Create small component close to main mesh (simulates a handle/knob).
        # This should be KEPT despite being small.
        small_close = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
        # Place it 0.02m gap from main mesh on positive Y axis.
        # Main mesh y-max is 0.5, sphere radius is 0.03.
        # To get 0.02m gap: position = 0.5 + 0.02 + 0.03 = 0.55
        small_close.apply_translation([0, 0.55, 0])

        # Create large component far from both main mesh and small component.
        # This should be REMOVED despite being large (40% of main mesh volume).
        large_far = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
        # Place it 0.1m gap from main mesh on positive X axis (far from both).
        # Main mesh x-max is 0.5, large box half-width is 0.2.
        # To get 0.1m gap: position = 0.5 + 0.1 + 0.2 = 0.8
        large_far.apply_translation([0.8, 0, 0])

        # Combine all components.
        combined = trimesh.util.concatenate([main_mesh, small_close, large_far])
        input_path = self.temp_path / "mixed_floaters.glb"
        combined.export(input_path)

        # Apply floater removal with 5cm distance threshold.
        output_path = self.temp_path / "distance_cleaned.glb"
        remove_mesh_floaters(
            mesh_path=input_path,
            output_path=output_path,
            distance_threshold=0.05,
        )

        # Load cleaned mesh.
        cleaned_mesh = trimesh.load(output_path, force="mesh")

        # Verify results.
        components = cleaned_mesh.split()

        # Should have exactly 2 components: main mesh + small close component.
        self.assertEqual(
            len(components),
            2,
            "Should keep main mesh and close small component, remove far large component",
        )

        # Verify total volume is main + small (large far component removed).
        expected_volume = main_mesh.volume + small_close.volume
        cleaned_volume = cleaned_mesh.volume

        self.assertAlmostEqual(
            cleaned_volume,
            expected_volume,
            delta=0.01,
            msg="Should keep main mesh + small close component, remove large far floater",
        )

        # Verify the large far component was actually removed.
        self.assertLess(
            cleaned_volume,
            combined.volume - 0.01,
            msg="Large far component should have been removed",
        )


class TestChooseFallbackFrontAxis(unittest.TestCase):
    """Test fallback front axis selection when VLM predicts parallel axes."""

    def test_fallback_z_up(self):
        """Test fallback front axis selection when up is +Z."""
        self.assertEqual(choose_fallback_front_axis("+Z"), "+Y")
        self.assertEqual(choose_fallback_front_axis("-Z"), "+Y")

    def test_fallback_y_up(self):
        """Test fallback front axis selection when up is +Y."""
        self.assertEqual(choose_fallback_front_axis("+Y"), "+X")
        self.assertEqual(choose_fallback_front_axis("-Y"), "+X")

    def test_fallback_x_up(self):
        """Test fallback front axis selection when up is +X."""
        self.assertEqual(choose_fallback_front_axis("+X"), "+Y")
        self.assertEqual(choose_fallback_front_axis("-X"), "+Y")


if __name__ == "__main__":
    unittest.main()
