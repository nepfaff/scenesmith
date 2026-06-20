"""Unit tests for surface occlusion detection.

Tests the raycast-based occlusion detection used for placing surface labels
on multi-surface furniture.
"""

import math
import unittest

from pathlib import Path
from unittest.mock import Mock

import bpy
import numpy as np

try:
    from mathutils import Vector
except ImportError:
    Vector = None

from scenesmith.agent_utils.blender.surface_utils import (
    find_best_label_position,
    is_point_occluded,
)


@unittest.skipIf(
    isinstance(bpy, Mock) or Vector is None,
    "Requires Blender's bpy and mathutils Python modules",
)
class TestSurfaceOcclusion(unittest.TestCase):
    """Tests for surface occlusion detection contracts."""

    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        cls.test_data_dir = (
            Path(__file__).parent.parent / "test_data" / "support_surface_algorithm"
        )
        cls.artistic_shelf_path = cls.test_data_dir / "artistic_shelf_canonical.gltf"

        if not cls.artistic_shelf_path.exists():
            raise FileNotFoundError(f"Test data not found: {cls.artistic_shelf_path}")

    def setUp(self):
        """Reset Blender scene before each test."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    def _load_shelf(self) -> dict:
        """Load the test shelf and return geometry info.

        Returns:
            Dict with center, min, max, and dimensions of shelf geometry.
        """
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.gltf(filepath=str(self.artistic_shelf_path))

        mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
        all_corners = []
        for obj in mesh_objects:
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                all_corners.append(world_corner)

        all_corners = np.array(all_corners)
        bbox_min = all_corners.min(axis=0)
        bbox_max = all_corners.max(axis=0)
        center = (bbox_min + bbox_max) / 2

        return {
            "center": Vector(center),
            "min": Vector(bbox_min),
            "max": Vector(bbox_max),
            "dimensions": Vector(bbox_max - bbox_min),
        }

    def _create_camera(self, location: Vector, look_at: Vector) -> bpy.types.Object:
        """Create a camera at specified location looking at target."""
        camera_data = bpy.data.cameras.new("TestCamera")
        camera_obj = bpy.data.objects.new("TestCamera", camera_data)
        bpy.context.scene.collection.objects.link(camera_obj)
        camera_obj.location = location

        direction = look_at - location
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera_obj.rotation_euler = rot_quat.to_euler()
        bpy.context.scene.camera = camera_obj

        return camera_obj

    def test_point_occluded_from_back_view(self):
        """Point inside shelf should be occluded when camera is behind."""
        shelf = self._load_shelf()
        center = shelf["center"]
        dims = shelf["dimensions"]

        # Camera behind shelf (negative Y), looking at center.
        camera_dist = max(dims) * 2
        camera_loc = Vector((center.x, center.y - camera_dist, center.z))
        camera = self._create_camera(camera_loc, center)

        bpy.context.view_layer.update()

        # Point at shelf center should be blocked by back panel.
        is_blocked = is_point_occluded(camera, center, "test")

        self.assertTrue(is_blocked, "Point inside shelf should be occluded from back")

    def test_point_visible_from_front_view(self):
        """Point inside shelf should be visible when camera is in front."""
        shelf = self._load_shelf()
        center = shelf["center"]
        dims = shelf["dimensions"]

        # Camera in front of shelf (positive Y).
        camera_dist = max(dims) * 2
        camera_loc = Vector((center.x, center.y + camera_dist, center.z))
        camera = self._create_camera(camera_loc, center)

        bpy.context.view_layer.update()

        # Point at shelf center should be visible (compartments open toward +Y).
        is_blocked = is_point_occluded(camera, center, "test")

        self.assertFalse(is_blocked, "Point inside shelf should be visible from front")

    def test_top_surface_visible_from_side_angles(self):
        """Top surface should be visible from various side angles."""
        shelf = self._load_shelf()
        center = shelf["center"]
        dims = shelf["dimensions"]
        top_z = shelf["max"].z

        target = Vector((center.x, center.y, top_z))
        camera_dist = max(dims) * 2

        # Test from 4 cardinal angles.
        for angle_deg in [0, 90, 180, 270]:
            self.setUp()
            shelf = self._load_shelf()

            rad = math.radians(angle_deg)
            camera_loc = Vector(
                (
                    center.x + camera_dist * math.sin(rad),
                    center.y + camera_dist * math.cos(rad),
                    center.z + dims.z * 0.5,
                )
            )
            camera = self._create_camera(camera_loc, center)
            bpy.context.view_layer.update()

            is_blocked = is_point_occluded(camera, target, "test")

            self.assertFalse(
                is_blocked,
                f"Top surface should be visible from {angle_deg} degrees",
            )

    def test_find_best_label_returns_none_for_occluded_surface(self):
        """find_best_label_position should return None for occluded surfaces."""
        shelf = self._load_shelf()
        center = shelf["center"]
        dims = shelf["dimensions"]

        # Camera behind shelf.
        camera_dist = max(dims) * 2
        camera_loc = Vector((center.x, center.y - camera_dist, center.z))
        camera = self._create_camera(camera_loc, center)

        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.view_layer.update()

        # Internal surface vertices (inside the shelf).
        half = dims.x * 0.3
        surface_verts = np.array(
            [
                [center.x - half, center.y - 0.1, center.z],
                [center.x + half, center.y - 0.1, center.z],
                [center.x - half, center.y + 0.1, center.z],
                [center.x + half, center.y + 0.1, center.z],
            ]
        )

        result = find_best_label_position(
            convex_hull_vertices=surface_verts,
            camera_obj=camera,
            img_width=512,
            img_height=512,
            grid_size=5,
            surface_id="internal",
        )

        self.assertIsNone(result, "Should return None for occluded internal surface")

    def test_find_best_label_returns_position_for_visible_surface(self):
        """find_best_label_position should return valid position for visible surfaces."""
        shelf = self._load_shelf()
        center = shelf["center"]
        dims = shelf["dimensions"]
        top_z = shelf["max"].z

        # Elevated side camera.
        camera_dist = max(dims) * 2
        camera_loc = Vector(
            (
                center.x + camera_dist * 0.7,
                center.y + camera_dist * 0.7,
                center.z + dims.z,
            )
        )
        camera = self._create_camera(camera_loc, center)

        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.view_layer.update()

        # Top surface vertices.
        half = dims.x * 0.4
        surface_verts = np.array(
            [
                [center.x - half, center.y - half, top_z],
                [center.x + half, center.y - half, top_z],
                [center.x - half, center.y + half, top_z],
                [center.x + half, center.y + half, top_z],
            ]
        )

        result = find_best_label_position(
            convex_hull_vertices=surface_verts,
            camera_obj=camera,
            img_width=512,
            img_height=512,
            grid_size=5,
            surface_id="top",
        )

        self.assertIsNotNone(result, "Should return position for visible top surface")

        # Position should be within image bounds.
        px, py = result
        self.assertTrue(0 <= px <= 512 and 0 <= py <= 512, "Position within bounds")


if __name__ == "__main__":
    unittest.main()
