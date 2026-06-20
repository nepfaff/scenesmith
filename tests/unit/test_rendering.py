"""Tests for Drake plant and scene graph setup from RoomScene objects."""

import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import MultibodyPlant, RigidTransform, SceneGraph

from scenesmith.agent_utils.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.rendering import _generate_angled_drawer_view
from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID


class TestGenerateAngledDrawerView(unittest.TestCase):
    """Test pure drawer view helper used outside the Blender Python process."""

    def test_generates_normalized_direction_without_bpy(self):
        view = _generate_angled_drawer_view(
            surface={"surface_id": "drawer_surface"},
            joint_name="drawer_joint",
            drawer_direction=[0.0, 2.0, 0.0],
        )

        direction = np.array(view["direction"])

        self.assertEqual(view["name"], "drawer_drawer_joint_drawer_surface")
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0)
        self.assertGreater(direction[1], 0.0)
        self.assertGreater(direction[2], 0.0)


class TestCreateDrakePlantAndSceneGraphFromScene(unittest.TestCase):
    """Test cases for create_drake_plant_and_scene_graph_from_scene function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.floor_plan_path = self.test_data_dir / "simple_room_geometry.sdf"
        self.box_sdf_path = self.test_data_dir / "simple_box.sdf"
        self.sphere_sdf_path = self.test_data_dir / "simple_sphere.sdf"

        # Create RoomGeometry from SDF file.
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(
            room_geometry=room_geometry, scene_dir=self.test_data_dir
        )

    def test_setup_empty_scene(self):
        """Test setting up plant with only floor plan."""
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(self.scene)

        self.assertIsInstance(plant, MultibodyPlant)
        self.assertIsInstance(scene_graph, SceneGraph)
        self.assertTrue(plant.is_finalized())

        # Should have at least the floor plan model.
        self.assertGreater(plant.num_model_instances(), 1)

    def test_setup_scene_with_furniture(self):
        """Test setting up plant with furniture objects."""
        furniture_obj = SceneObject(
            object_id=UniqueID("test_box"),
            object_type=ObjectType.FURNITURE,
            name="Test Box",
            description="A test box",
            transform=RigidTransform(np.array([1.0, 0.0, 0.5])),
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(furniture_obj)

        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(self.scene)

        self.assertIsInstance(plant, MultibodyPlant)
        self.assertIsInstance(scene_graph, SceneGraph)
        self.assertTrue(plant.is_finalized())

        # Should have floor plan + furniture models.
        self.assertGreater(plant.num_model_instances(), 2)

    def test_setup_scene_with_manipuland(self):
        """Test setting up plant with manipuland objects."""
        manipuland_obj = SceneObject(
            object_id=UniqueID("test_sphere"),
            object_type=ObjectType.MANIPULAND,
            name="Test Sphere",
            description="A test sphere",
            transform=RigidTransform(np.array([0.0, 1.0, 1.0])),
            sdf_path=self.sphere_sdf_path,
        )
        self.scene.add_object(manipuland_obj)

        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(self.scene)

        self.assertIsInstance(plant, MultibodyPlant)
        self.assertIsInstance(scene_graph, SceneGraph)
        self.assertTrue(plant.is_finalized())

        # Should have floor plan + manipuland models
        self.assertGreater(plant.num_model_instances(), 2)


if __name__ == "__main__":
    unittest.main()
