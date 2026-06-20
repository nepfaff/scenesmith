"""Unit tests for physical feasibility post-processing module."""

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path
from unittest.mock import patch

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix

from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.physical_feasibility import (
    _apply_floor_penetration_fallback,
    _get_colliding_object_ids,
    apply_forward_simulation,
    apply_non_penetration_projection,
    apply_physical_feasibility_postprocessing,
    compute_tilt_angle_degrees,
)
from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

# Path to test data.
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


class PhysicalFeasibilityTestCase(unittest.TestCase):
    """Base test case with shared fixtures for physical feasibility tests."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up shared test fixtures."""
        floor_plan_sdf_path = TEST_DATA_DIR / "simple_room_geometry.sdf"
        cls.room_geometry = RoomGeometry(
            sdf_tree=ET.parse(floor_plan_sdf_path),
            sdf_path=floor_plan_sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=10.0,
            length=10.0,
        )

    def _create_overlapping_boxes_scene(self, scene_dir: Path) -> RoomScene:
        """Create a scene with two overlapping boxes.

        Uses simple_box.sdf (0.5x0.5x0.5m boxes). Boxes overlap when centers
        are less than 0.5m apart.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with overlapping boxes",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        box1 = SceneObject(
            object_id=UniqueID("box_1"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 1",
            transform=RigidTransform(p=[0.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        box2 = SceneObject(
            object_id=UniqueID("box_2"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 2",
            transform=RigidTransform(p=[0.3, 0.01, 0.25]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(box1)
        scene.add_object(box2)

        return scene

    def _create_non_overlapping_boxes_scene(self, scene_dir: Path) -> RoomScene:
        """Create a scene with two non-overlapping boxes (2m apart)."""
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with non-overlapping boxes",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        box1 = SceneObject(
            object_id=UniqueID("box_1"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 1",
            transform=RigidTransform(p=[1.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        box2 = SceneObject(
            object_id=UniqueID("box_2"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 2",
            transform=RigidTransform(p=[-1.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(box1)
        scene.add_object(box2)

        return scene

    def _create_scene_with_manipuland(self, scene_dir: Path) -> RoomScene:
        """Create a scene with furniture and a manipuland (sphere on box)."""
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with furniture and manipuland",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        sphere_sdf_path = TEST_DATA_DIR / "simple_sphere.sdf"

        furniture = SceneObject(
            object_id=UniqueID("table_0"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(p=[0.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        manipuland = SceneObject(
            object_id=UniqueID("ball_0"),
            object_type=ObjectType.MANIPULAND,
            name="ball",
            description="Test ball",
            transform=RigidTransform(p=[0.0, 0.0, 0.7]),
            sdf_path=sphere_sdf_path,
        )

        scene.add_object(furniture)
        scene.add_object(manipuland)

        return scene

    def _create_scene_with_stack(self, scene_dir: Path) -> RoomScene:
        """Create a scene with a stack (composite object).

        Uses simple_box.sdf (0.5x0.5x0.5m boxes) to create a two-box stack.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with stack",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        # Create a stack with 2 boxes stacked vertically.
        stack = SceneObject(
            object_id=UniqueID("stack_0"),
            object_type=ObjectType.FURNITURE,
            name="stack",
            description="Test stack",
            transform=RigidTransform(p=[0.0, 0.0, 0.0]),
            sdf_path=None,  # Composite objects don't have their own SDF.
            metadata={
                "composite_type": "stack",
                "member_assets": [
                    {
                        "name": "bottom_box",
                        "asset_id": "asset_bottom123",
                        "sdf_path": str(box_sdf_path),
                        "transform": {
                            "translation": [0.0, 0.0, 0.25],
                            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    },
                    {
                        "name": "top_box",
                        "asset_id": "asset_top456",
                        "sdf_path": str(box_sdf_path),
                        "transform": {
                            "translation": [0.0, 0.0, 0.75],
                            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    },
                ],
            },
        )
        scene.add_object(stack)

        return scene


class TestApplyNonPenetrationProjection(PhysicalFeasibilityTestCase):
    """Tests for apply_non_penetration_projection function."""

    def test_overlapping_boxes_separated_snopt(self) -> None:
        """Test that overlapping boxes are separated by projection using SNOPT."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="snopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            self.assertTrue(success)

            box1_after = projected_scene.get_object(UniqueID("box_1"))
            box2_after = projected_scene.get_object(UniqueID("box_2"))

            pos1_changed = not np.allclose(
                box1_after.transform.translation(), initial_pos1, atol=0.01
            )
            pos2_changed = not np.allclose(
                box2_after.transform.translation(), initial_pos2, atol=0.01
            )
            self.assertTrue(pos1_changed or pos2_changed)

    def test_non_overlapping_boxes_unchanged(self) -> None:
        """Test that non-overlapping boxes remain roughly unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            self.assertTrue(success)

            box1_after = projected_scene.get_object(UniqueID("box_1"))
            box2_after = projected_scene.get_object(UniqueID("box_2"))

            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos1, atol=0.1)
            )
            self.assertTrue(
                np.allclose(box2_after.transform.translation(), initial_pos2, atol=0.1)
            )

    def test_fix_rotation_constraint(self) -> None:
        """Test that fix_rotation=True keeps rotations unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_rot1 = box1.transform.rotation().ToQuaternion().wxyz()
            initial_rot2 = box2.transform.rotation().ToQuaternion().wxyz()

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            if success:
                box1_after = projected_scene.get_object(UniqueID("box_1"))
                box2_after = projected_scene.get_object(UniqueID("box_2"))

                final_rot1 = box1_after.transform.rotation().ToQuaternion().wxyz()
                final_rot2 = box2_after.transform.rotation().ToQuaternion().wxyz()

                self.assertTrue(np.allclose(final_rot1, initial_rot1, atol=1e-3))
                self.assertTrue(np.allclose(final_rot2, initial_rot2, atol=1e-3))

    def test_xy_only_constraint(self) -> None:
        """Test that xy_only=True keeps Z position fixed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_z1 = box1.transform.translation()[2]
            initial_z2 = box2.transform.translation()[2]

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=False,
                    xy_only=True,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            if success:
                box1_after = projected_scene.get_object(UniqueID("box_1"))
                box2_after = projected_scene.get_object(UniqueID("box_2"))

                self.assertTrue(
                    np.isclose(
                        box1_after.transform.translation()[2], initial_z1, atol=1e-3
                    )
                )
                self.assertTrue(
                    np.isclose(
                        box2_after.transform.translation()[2], initial_z2, atol=1e-3
                    )
                )

    def test_weld_furniture_keeps_furniture_fixed(self) -> None:
        """Test that weld_furniture=True keeps furniture fixed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_pos = furniture.transform.translation().copy()

            try:
                projected_scene, _ = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=True,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            furniture_after = projected_scene.get_object(UniqueID("table_0"))
            self.assertTrue(
                np.allclose(
                    furniture_after.transform.translation(),
                    initial_furniture_pos,
                    atol=1e-6,
                )
            )

    def test_empty_scene_returns_success(self) -> None:
        """Test that empty scene returns success."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Empty test scene",
            )

            _, success = apply_non_penetration_projection(
                scene=scene,
                influence_distance=0.03,
                solver_name="snopt",
                iteration_limit=100,
                weld_furniture=False,
                xy_only=True,
                fix_rotation=True,
            )

            self.assertTrue(success)

    def test_stack_members_maintain_relative_positions(self) -> None:
        """Test that stack members maintain relative positions during projection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_stack(Path(tmp_dir))

            # Add overlapping box to force projection.
            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"
            overlapping_box = SceneObject(
                object_id=UniqueID("box_0"),
                object_type=ObjectType.FURNITURE,
                name="box",
                description="Overlapping box",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )
            scene.add_object(overlapping_box)

            stack = scene.get_object(UniqueID("stack_0"))
            members_before = stack.metadata["member_assets"]
            initial_bottom_pos = np.array(members_before[0]["transform"]["translation"])
            initial_top_pos = np.array(members_before[1]["transform"]["translation"])
            initial_z_diff = initial_top_pos[2] - initial_bottom_pos[2]

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            self.assertTrue(success)

            stack_after = projected_scene.get_object(UniqueID("stack_0"))
            members_after = stack_after.metadata["member_assets"]
            final_bottom_pos = np.array(members_after[0]["transform"]["translation"])
            final_top_pos = np.array(members_after[1]["transform"]["translation"])
            final_z_diff = final_top_pos[2] - final_bottom_pos[2]

            # Verify stack actually moved to resolve collision.
            bottom_moved = not np.allclose(
                final_bottom_pos, initial_bottom_pos, atol=0.01
            )
            top_moved = not np.allclose(final_top_pos, initial_top_pos, atol=0.01)
            self.assertTrue(
                bottom_moved and top_moved,
                f"Stack should have moved to resolve collision. "
                f"Bottom moved: {bottom_moved}, Top moved: {top_moved}",
            )

            # Stack members should maintain their relative Z distance.
            self.assertAlmostEqual(initial_z_diff, final_z_diff, places=3)


class TestApplyForwardSimulation(PhysicalFeasibilityTestCase):
    """Tests for apply_forward_simulation function."""

    def test_simulation_runs_without_error(self) -> None:
        """Test that simulation runs without errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.5,
                time_step_s=1e-3,
                timeout_s=30.0,
                weld_furniture=False,
            )

            self.assertIsNotNone(simulated_scene)
            self.assertEqual(removed_ids, [])

    def test_simulation_with_timeout(self) -> None:
        """Test that simulation respects timeout and returns scene unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            initial_pos1 = box1.transform.translation().copy()

            simulated_scene, _ = apply_forward_simulation(
                scene=scene,
                simulation_time_s=10.0,
                time_step_s=1e-3,
                timeout_s=1e-16,
                weld_furniture=False,
            )

            box1_after = simulated_scene.get_object(UniqueID("box_1"))
            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos1, atol=0.1)
            )

    def test_simulation_with_welded_furniture(self) -> None:
        """Test that welded furniture doesn't move during simulation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_pos = furniture.transform.translation().copy()

            simulated_scene, _ = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.5,
                time_step_s=1e-3,
                timeout_s=30.0,
                weld_furniture=True,
            )

            furniture_after = simulated_scene.get_object(UniqueID("table_0"))
            self.assertTrue(
                np.allclose(
                    furniture_after.transform.translation(),
                    initial_furniture_pos,
                    atol=1e-6,
                )
            )

    def test_empty_scene_simulation(self) -> None:
        """Test that simulation on empty scene succeeds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Empty test scene",
            )

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
            )

            self.assertIsNotNone(simulated_scene)
            self.assertEqual(removed_ids, [])


class TestApplyPhysicalFeasibilityPostprocessing(PhysicalFeasibilityTestCase):
    """Tests for the combined post-processing pipeline."""

    def test_projection_followed_by_simulation(self) -> None:
        """Test applying projection followed by simulation (full pipeline)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            # Use SNOPT instead of IPOPT - IPOPT has numerical issues with
            # Drake's box-box gradient computation in edge cases.
            try:
                processed_scene, success, removed_ids = (
                    apply_physical_feasibility_postprocessing(
                        scene=scene,
                        weld_furniture=False,
                        projection_enabled=True,
                        projection_influence_distance=0.03,
                        projection_solver_name="snopt",
                        projection_iteration_limit=5000,
                        projection_xy_only=False,
                        projection_fix_rotation=True,
                        simulation_enabled=True,
                        simulation_time_s=0.5,
                        simulation_time_step_s=1e-3,
                        simulation_timeout_s=30.0,
                    )
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            self.assertTrue(success)
            self.assertEqual(removed_ids, [])

            box1_after = processed_scene.get_object(UniqueID("box_1"))
            box2_after = processed_scene.get_object(UniqueID("box_2"))

            pos1_changed = not np.allclose(
                box1_after.transform.translation(), initial_pos1, atol=0.01
            )
            pos2_changed = not np.allclose(
                box2_after.transform.translation(), initial_pos2, atol=0.01
            )
            self.assertTrue(pos1_changed or pos2_changed)

    def test_disabled_projection(self) -> None:
        """Test that disabled projection skips projection stage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            initial_pos = box1.transform.translation().copy()

            processed_scene, success, _ = apply_physical_feasibility_postprocessing(
                scene=scene,
                weld_furniture=False,
                projection_enabled=False,
                simulation_enabled=False,
            )

            self.assertTrue(success)

            box1_after = processed_scene.get_object(UniqueID("box_1"))
            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos)
            )

    def test_disabled_simulation(self) -> None:
        """Test that disabled simulation skips simulation stage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            try:
                processed_scene, _, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    projection_solver_name="ipopt",
                    projection_iteration_limit=100,
                    simulation_enabled=False,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            self.assertIsNotNone(processed_scene)

    def test_weld_furniture_in_pipeline(self) -> None:
        """Test weld_furniture flag in full pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_pos = furniture.transform.translation().copy()

            try:
                processed_scene, _, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=True,
                    projection_enabled=True,
                    projection_solver_name="ipopt",
                    projection_iteration_limit=1000,
                    simulation_enabled=True,
                    simulation_time_s=0.5,
                    simulation_time_step_s=1e-3,
                    simulation_timeout_s=30.0,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            furniture_after = processed_scene.get_object(UniqueID("table_0"))
            self.assertTrue(
                np.allclose(
                    furniture_after.transform.translation(),
                    initial_furniture_pos,
                    atol=1e-6,
                )
            )


class TestComputeTiltAngle(unittest.TestCase):
    """Tests for compute_tilt_angle_degrees function."""

    def test_upright_object_zero_tilt(self) -> None:
        """Test that an upright object has zero tilt angle."""
        transform = RigidTransform(p=[1.0, 2.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 0.0, places=5)

    def test_yaw_rotation_zero_tilt(self) -> None:
        """Test that yaw rotation (turning in place) gives zero tilt."""
        # Rotate 90 degrees around Z-axis (yaw).
        rotation = RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2))
        transform = RigidTransform(rotation, [1.0, 2.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 0.0, places=5)

    def test_45_degree_pitch_tilt(self) -> None:
        """Test that 45 degree pitch gives 45 degree tilt."""
        # Rotate 45 degrees around Y-axis (pitch).
        rotation = RotationMatrix(RollPitchYaw(0.0, np.pi / 4, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 45.0, places=3)

    def test_45_degree_roll_tilt(self) -> None:
        """Test that 45 degree roll gives 45 degree tilt."""
        # Rotate 45 degrees around X-axis (roll).
        rotation = RotationMatrix(RollPitchYaw(np.pi / 4, 0.0, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 45.0, places=3)

    def test_90_degree_tilt_horizontal(self) -> None:
        """Test that 90 degree pitch gives horizontal object (90 degree tilt)."""
        # Rotate 90 degrees around Y-axis.
        rotation = RotationMatrix(RollPitchYaw(0.0, np.pi / 2, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 90.0, places=3)

    def test_combined_roll_pitch_tilt(self) -> None:
        """Test combined roll and pitch gives correct tilt angle."""
        # Small roll and pitch should give a combined tilt.
        rotation = RotationMatrix(RollPitchYaw(np.pi / 6, np.pi / 6, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        # Combined tilt should be greater than either individual angle.
        self.assertGreater(tilt, 30.0)
        self.assertLess(tilt, 60.0)


class TestFallenFurnitureRemoval(PhysicalFeasibilityTestCase):
    """Tests for fallen furniture removal functionality."""

    def _create_tilted_furniture_scene(
        self, scene_dir: Path, tilt_degrees: float
    ) -> RoomScene:
        """Create a scene with a tilted furniture piece."""
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with tilted furniture",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        # Create a tilted box (tilt around Y-axis).
        tilt_rad = np.radians(tilt_degrees)
        rotation = RotationMatrix(RollPitchYaw(0.0, tilt_rad, 0.0))

        tilted_box = SceneObject(
            object_id=UniqueID("tilted_box"),
            object_type=ObjectType.FURNITURE,
            name="tilted_box",
            description="A tilted box",
            transform=RigidTransform(rotation, [0.0, 0.0, 0.5]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(tilted_box)
        return scene

    def test_fallen_furniture_removed_above_threshold(self) -> None:
        """Test that furniture tilted above threshold is removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with 50 degree tilt (above 45 degree threshold).
            scene = self._create_tilted_furniture_scene(Path(tmp_dir), tilt_degrees=50)

            # Verify object exists before.
            self.assertIsNotNone(scene.get_object(UniqueID("tilted_box")))

            # Run simulation with fallen furniture removal enabled.
            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
                remove_fallen_furniture=True,
                fallen_tilt_threshold_degrees=45.0,
            )

            # Object should be removed.
            self.assertEqual(len(removed_ids), 1)
            self.assertEqual(removed_ids[0], UniqueID("tilted_box"))
            self.assertIsNone(simulated_scene.get_object(UniqueID("tilted_box")))

    def test_upright_furniture_not_removed(self) -> None:
        """Test that upright furniture is not removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with slight tilt (below threshold).
            scene = self._create_tilted_furniture_scene(Path(tmp_dir), tilt_degrees=20)

            # Run simulation with fallen furniture removal enabled.
            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
                remove_fallen_furniture=True,
                fallen_tilt_threshold_degrees=45.0,
            )

            # Object should NOT be removed.
            self.assertEqual(len(removed_ids), 0)
            self.assertIsNotNone(simulated_scene.get_object(UniqueID("tilted_box")))

    def test_fallen_removal_disabled_by_default(self) -> None:
        """Test that fallen furniture removal is disabled by default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with 50 degree tilt.
            scene = self._create_tilted_furniture_scene(Path(tmp_dir), tilt_degrees=50)

            # Run simulation WITHOUT enabling fallen furniture removal.
            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
                # remove_fallen_furniture defaults to False
            )

            # Object should NOT be removed (feature disabled).
            self.assertEqual(len(removed_ids), 0)
            self.assertIsNotNone(simulated_scene.get_object(UniqueID("tilted_box")))


class TestFallenManipulandRemoval(PhysicalFeasibilityTestCase):
    """Tests for fallen manipuland removal functionality."""

    def _create_manipuland_scene(
        self,
        scene_dir: Path,
        manipuland_z: float,
        pre_sim_z: float | None = None,
        manipuland_xy: tuple[float, float] = (2.0, 0.0),
    ) -> RoomScene:
        """Create a scene with furniture and a manipuland at specified Z.

        Args:
            scene_dir: Directory for scene files.
            manipuland_z: Z position for the manipuland (post-simulation).
            pre_sim_z: If provided, the Z position before simulation (for z_delta).
                       If None, defaults to manipuland_z (no displacement).
            manipuland_xy: XY position for manipuland. Default (2, 0) places it
                away from the table at origin to allow free falling.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with manipuland",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        sphere_sdf_path = TEST_DATA_DIR / "simple_sphere.sdf"

        # Add furniture (will be welded during simulation).
        furniture = SceneObject(
            object_id=UniqueID("table_0"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(p=[0.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        # Add manipuland at specified Z and XY.
        # Use pre_sim_z for initial position if testing z_delta.
        initial_z = pre_sim_z if pre_sim_z is not None else manipuland_z
        manipuland = SceneObject(
            object_id=UniqueID("ball_0"),
            object_type=ObjectType.MANIPULAND,
            name="ball",
            description="Test ball",
            transform=RigidTransform(p=[manipuland_xy[0], manipuland_xy[1], initial_z]),
            sdf_path=sphere_sdf_path,
            bbox_min=np.array([-0.2, -0.2, -0.2]),
            bbox_max=np.array([0.2, 0.2, 0.2]),
        )

        scene.add_object(furniture)
        scene.add_object(manipuland)

        # If we need different post-sim Z, update transform after adding.
        # This simulates what happens during simulation.
        if pre_sim_z is not None and pre_sim_z != manipuland_z:
            manipuland.transform = RigidTransform(
                p=[manipuland_xy[0], manipuland_xy[1], manipuland_z]
            )

        return scene

    def test_floor_penetration_removed(self) -> None:
        """Test that manipuland below floor_z threshold is removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with manipuland at z=-1.0 (below -0.5 threshold).
            scene = self._create_manipuland_scene(
                Path(tmp_dir), manipuland_z=-1.0, pre_sim_z=0.7
            )

            self.assertIsNotNone(scene.get_object(UniqueID("ball_0")))

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.01,  # Short sim, object already positioned.
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=True,
                remove_fallen_manipulands=True,
                fallen_manipuland_floor_z=-0.5,
                fallen_manipuland_near_floor_z=0.02,
                fallen_manipuland_z_displacement=0.3,
            )

            # Object should be removed (fell through floor).
            self.assertIn(UniqueID("ball_0"), removed_ids)
            self.assertIsNone(simulated_scene.get_object(UniqueID("ball_0")))

    def test_fell_to_floor_removed(self) -> None:
        """Test that manipuland that fell to floor (big z_delta) is removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with manipuland high in the air (z=0.7).
            # During simulation, it will fall to the floor (z~0).
            scene = self._create_manipuland_scene(
                Path(tmp_dir), manipuland_z=0.7, pre_sim_z=None
            )

            # Verify starting position.
            ball = scene.get_object(UniqueID("ball_0"))
            self.assertIsNotNone(ball)
            self.assertAlmostEqual(ball.transform.translation()[2], 0.7, places=2)

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=2.0,  # Enough time for object to fall.
                time_step_s=1e-3,
                timeout_s=30.0,
                weld_furniture=True,
                remove_fallen_manipulands=True,
                fallen_manipuland_floor_z=-0.5,
                fallen_manipuland_near_floor_z=0.1,  # Object on floor after falling.
                fallen_manipuland_z_displacement=0.3,  # Will have delta < -0.3.
            )

            # Object should be removed (fell from z=0.7 to floor).
            self.assertIn(UniqueID("ball_0"), removed_ids)
            self.assertIsNone(simulated_scene.get_object(UniqueID("ball_0")))

    def test_floor_placed_not_removed(self) -> None:
        """Test that floor-placed manipuland (no z_delta) is NOT removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene: started at z=0.05, still at z=0.05 (no displacement).
            scene = self._create_manipuland_scene(
                Path(tmp_dir), manipuland_z=0.05, pre_sim_z=0.05
            )

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.01,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=True,
                remove_fallen_manipulands=True,
                fallen_manipuland_floor_z=-0.5,
                fallen_manipuland_near_floor_z=0.1,  # On floor (bottom_z=0).
                fallen_manipuland_z_displacement=0.3,  # delta=0, not < -0.3.
            )

            # Object should NOT be removed (floor-placed intentionally).
            self.assertNotIn(UniqueID("ball_0"), removed_ids)
            self.assertIsNotNone(simulated_scene.get_object(UniqueID("ball_0")))


class TestApplyFloorPenetrationFallback(PhysicalFeasibilityTestCase):
    """Tests for _apply_floor_penetration_fallback function."""

    def _create_floor_penetrating_scene(
        self, scene_dir: Path, penetration_depth: float
    ) -> RoomScene:
        """Create a scene with furniture penetrating the floor.

        Args:
            scene_dir: Directory for scene files.
            penetration_depth: How far below Z=0 the bottom of the box should be.
                               Positive values mean penetration.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with floor-penetrating furniture",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        # Box is 0.5x0.5x0.5m. Center at z=0.25 places bottom at z=0 (on floor).
        # Center at z=(0.25 - penetration_depth) places bottom at z=-penetration_depth.
        box_z = 0.25 - penetration_depth

        penetrating_box = SceneObject(
            object_id=UniqueID("box_0"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Floor-penetrating box",
            transform=RigidTransform(p=[0.0, 0.0, box_z]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(penetrating_box)
        return scene

    def test_penetrating_furniture_lifted(self) -> None:
        """Test that furniture penetrating the floor is lifted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            penetration = 0.05  # 5cm penetration.
            scene = self._create_floor_penetrating_scene(
                Path(tmp_dir), penetration_depth=penetration
            )

            box = scene.get_object(UniqueID("box_0"))
            initial_z = box.transform.translation()[2]

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            self.assertEqual(lifted_count, 1)

            box_after = updated_scene.get_object(UniqueID("box_0"))
            final_z = box_after.transform.translation()[2]

            # Box should be lifted by at least the penetration depth.
            lift_amount = final_z - initial_z
            self.assertGreaterEqual(lift_amount, penetration)

    def test_non_penetrating_furniture_unchanged(self) -> None:
        """Test that furniture not penetrating the floor is unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_z1 = box1.transform.translation()[2]
            initial_z2 = box2.transform.translation()[2]

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            self.assertEqual(lifted_count, 0)

            box1_after = updated_scene.get_object(UniqueID("box_1"))
            box2_after = updated_scene.get_object(UniqueID("box_2"))

            self.assertAlmostEqual(
                box1_after.transform.translation()[2], initial_z1, places=6
            )
            self.assertAlmostEqual(
                box2_after.transform.translation()[2], initial_z2, places=6
            )

    def test_only_furniture_processed(self) -> None:
        """Test that only furniture objects are processed, not manipulands."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            # Move manipuland to penetrate floor (this should NOT be lifted).
            manipuland = scene.get_object(UniqueID("ball_0"))
            manipuland.transform = RigidTransform(p=[0.0, 0.0, -0.1])

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_z = furniture.transform.translation()[2]
            initial_manipuland_z = manipuland.transform.translation()[2]

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            # No furniture penetrating floor, so nothing lifted.
            self.assertEqual(lifted_count, 0)

            # Manipuland should be unchanged (function ignores non-furniture).
            manipuland_after = updated_scene.get_object(UniqueID("ball_0"))
            self.assertAlmostEqual(
                manipuland_after.transform.translation()[2],
                initial_manipuland_z,
                places=6,
            )

            # Furniture should also be unchanged.
            furniture_after = updated_scene.get_object(UniqueID("table_0"))
            self.assertAlmostEqual(
                furniture_after.transform.translation()[2],
                initial_furniture_z,
                places=6,
            )

    def test_empty_scene_returns_zero_lifted(self) -> None:
        """Test that empty scene returns zero lifted objects."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Empty test scene",
            )

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            self.assertEqual(lifted_count, 0)
            self.assertIsNotNone(updated_scene)

    def test_wall_penetration_ignored(self) -> None:
        """Test that furniture penetrating walls is NOT lifted (only floor matters)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Test scene with wall-penetrating furniture",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Place box penetrating wall_1 (at y=5) but NOT the floor.
            # Box center at y=4.8 with size 0.5 means edge at y=5.05 (penetrating wall).
            wall_penetrating_box = SceneObject(
                object_id=UniqueID("box_0"),
                object_type=ObjectType.FURNITURE,
                name="box",
                description="Wall-penetrating box",
                transform=RigidTransform(p=[0.0, 4.8, 0.25]),  # On floor, near wall.
                sdf_path=box_sdf_path,
            )

            scene.add_object(wall_penetrating_box)

            box = scene.get_object(UniqueID("box_0"))
            initial_pos = box.transform.translation().copy()

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            # Wall penetration should be ignored - nothing lifted.
            self.assertEqual(lifted_count, 0)

            # Position should be unchanged.
            box_after = updated_scene.get_object(UniqueID("box_0"))
            self.assertTrue(
                np.allclose(box_after.transform.translation(), initial_pos, atol=1e-6)
            )


class TestGetCollidingObjectIds(PhysicalFeasibilityTestCase):
    """Tests for _get_colliding_object_ids helper function."""

    def test_no_collisions_returns_empty_set(self) -> None:
        """Scene with no collisions should return empty set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))
            colliding_ids = _get_colliding_object_ids(scene)
            self.assertEqual(colliding_ids, set())

    def test_two_penetrating_objects_returns_both_ids(self) -> None:
        """Two overlapping objects should both be in result."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))
            colliding_ids = _get_colliding_object_ids(scene)

            self.assertEqual(len(colliding_ids), 2)
            self.assertIn(UniqueID("box_1"), colliding_ids)
            self.assertIn(UniqueID("box_2"), colliding_ids)

    def test_chain_collision_returns_all_involved(self) -> None:
        """A-B collision and B-C collision should return A, B, C."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Chain collision test",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Box A at origin.
            box_a = SceneObject(
                object_id=UniqueID("box_a"),
                object_type=ObjectType.FURNITURE,
                name="box_a",
                description="Box A",
                transform=RigidTransform(p=[0.0, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # Box B overlapping with A (shifted 0.3m in X).
            box_b = SceneObject(
                object_id=UniqueID("box_b"),
                object_type=ObjectType.FURNITURE,
                name="box_b",
                description="Box B",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # Box C overlapping with B but not A (shifted 0.6m in X).
            box_c = SceneObject(
                object_id=UniqueID("box_c"),
                object_type=ObjectType.FURNITURE,
                name="box_c",
                description="Box C",
                transform=RigidTransform(p=[0.6, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            scene.add_object(box_a)
            scene.add_object(box_b)
            scene.add_object(box_c)

            colliding_ids = _get_colliding_object_ids(scene)

            # A-B collide, B-C collide → all three should be in the set.
            self.assertEqual(len(colliding_ids), 3)
            self.assertIn(UniqueID("box_a"), colliding_ids)
            self.assertIn(UniqueID("box_b"), colliding_ids)
            self.assertIn(UniqueID("box_c"), colliding_ids)

    def test_isolated_non_colliding_object_not_included(self) -> None:
        """Object not in collision should not be in the result."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Mixed collision test",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Two overlapping boxes.
            box1 = SceneObject(
                object_id=UniqueID("box_1"),
                object_type=ObjectType.FURNITURE,
                name="box_1",
                description="Box 1",
                transform=RigidTransform(p=[0.0, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )
            box2 = SceneObject(
                object_id=UniqueID("box_2"),
                object_type=ObjectType.FURNITURE,
                name="box_2",
                description="Box 2",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # One isolated box far away (but not near walls - room is 10x10).
            box_isolated = SceneObject(
                object_id=UniqueID("box_isolated"),
                object_type=ObjectType.FURNITURE,
                name="box_isolated",
                description="Isolated box",
                transform=RigidTransform(p=[3.0, 3.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            scene.add_object(box1)
            scene.add_object(box2)
            scene.add_object(box_isolated)

            colliding_ids = _get_colliding_object_ids(scene)

            # Only the overlapping pair should be in result (isolated box not colliding).
            self.assertEqual(len(colliding_ids), 2)
            self.assertIn(UniqueID("box_1"), colliding_ids)
            self.assertIn(UniqueID("box_2"), colliding_ids)
            self.assertNotIn(UniqueID("box_isolated"), colliding_ids)


class TestLargeSceneOptimization(PhysicalFeasibilityTestCase):
    """Tests for threshold-based DOF reduction optimization."""

    def test_small_scene_uses_all_free_objects(self) -> None:
        """Scene below threshold uses original path (all objects free)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            # Capture initial positions before projection.
            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            # 2 objects, threshold is 100 → small scene path.
            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="snopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                    large_scene_optimization_threshold=100,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            # At least one box should have moved to resolve collision.
            self.assertTrue(success)
            box1_after = projected_scene.get_object(UniqueID("box_1"))
            box2_after = projected_scene.get_object(UniqueID("box_2"))

            pos1_changed = not np.allclose(
                box1_after.transform.translation(),
                initial_pos1,
                atol=0.01,
            )
            pos2_changed = not np.allclose(
                box2_after.transform.translation(),
                initial_pos2,
                atol=0.01,
            )
            self.assertTrue(pos1_changed or pos2_changed)

    def test_large_scene_no_collisions_skips_projection(self) -> None:
        """Large scene with no collisions returns early with success."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            # Set threshold to 1 so 2 objects triggers large scene path.
            result_scene, success = apply_non_penetration_projection(
                scene=scene,
                influence_distance=0.03,
                solver_name="snopt",
                iteration_limit=5000,
                weld_furniture=False,
                xy_only=False,
                fix_rotation=True,
                large_scene_optimization_threshold=1,  # 2 objects > 1 threshold.
            )

            # Should succeed with no changes (early return, no collisions).
            self.assertTrue(success)

            box1_after = result_scene.get_object(UniqueID("box_1"))
            box2_after = result_scene.get_object(UniqueID("box_2"))

            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos1, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(box2_after.transform.translation(), initial_pos2, atol=1e-6)
            )

    def test_small_scene_no_collisions_skips_projection(self) -> None:
        """Small scenes with no collisions return before building an IK plant."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            with patch(
                "scenesmith.agent_utils.physical_feasibility._create_drake_plant_for_ik"
            ) as create_plant:
                result_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="snopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                    large_scene_optimization_threshold=100,
                )

            self.assertTrue(success)
            create_plant.assert_not_called()

            box1_after = result_scene.get_object(UniqueID("box_1"))
            box2_after = result_scene.get_object(UniqueID("box_2"))

            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos1, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(box2_after.transform.translation(), initial_pos2, atol=1e-6)
            )

    def test_large_scene_only_colliding_objects_move(self) -> None:
        """Large scene optimization only allows colliding objects to move."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Large scene optimization test",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Two overlapping boxes.
            box1 = SceneObject(
                object_id=UniqueID("box_1"),
                object_type=ObjectType.FURNITURE,
                name="box_1",
                description="Box 1",
                transform=RigidTransform(p=[0.0, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )
            box2 = SceneObject(
                object_id=UniqueID("box_2"),
                object_type=ObjectType.FURNITURE,
                name="box_2",
                description="Box 2",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # One isolated box far away (but not near walls - room is 10x10).
            box_isolated = SceneObject(
                object_id=UniqueID("box_isolated"),
                object_type=ObjectType.FURNITURE,
                name="box_isolated",
                description="Isolated box",
                transform=RigidTransform(p=[3.0, 3.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            scene.add_object(box1)
            scene.add_object(box2)
            scene.add_object(box_isolated)

            initial_isolated_pos = box_isolated.transform.translation().copy()

            # Set threshold to 1 so 3 objects triggers large scene path.
            try:
                result_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="snopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                    large_scene_optimization_threshold=1,  # 3 objects > 1.
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            self.assertTrue(success)

            # Isolated box should NOT have moved (welded in optimization).
            isolated_after = result_scene.get_object(UniqueID("box_isolated"))
            self.assertTrue(
                np.allclose(
                    isolated_after.transform.translation(),
                    initial_isolated_pos,
                    atol=1e-6,
                ),
                f"Isolated box should not move. Initial: {initial_isolated_pos}, "
                f"Final: {isolated_after.transform.translation()}",
            )


if __name__ == "__main__":
    unittest.main()
