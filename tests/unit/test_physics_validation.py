import math
import unittest

from pathlib import Path
from types import SimpleNamespace

import lxml.etree as ET
import numpy as np

from pydrake.all import RigidTransform
from pydrake.math import RollPitchYaw

from scenesmith.agent_utils.clearance_zones import (
    DoorClearanceViolation,
    OpenConnectionBlockedViolation,
    WallHeightExceededViolation,
    WindowClearanceViolation,
    compute_wall_height_violations,
)
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.physics_validation import (
    CollisionPair,
    ThinCoveringBoundaryViolation,
    ThinCoveringOverlap,
    _get_furniture_id_for_manipuland,
    compute_scene_collisions,
    compute_thin_covering_boundary_violations,
    filter_collisions_by_agent,
    filter_door_violations_by_agent,
    filter_open_connection_violations_by_agent,
    filter_thin_covering_boundary_violations_by_agent,
    filter_thin_covering_overlaps_by_agent,
    filter_wall_height_violations_by_agent,
    filter_window_violations_by_agent,
)
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
    serialize_rigid_transform,
)
from scenesmith.manipuland_agents.tools.physics_utils import (
    load_collision_bounds_for_scene_object,
)
from scenesmith.manipuland_agents.tools.stacking import (
    compute_initial_stack_transforms,
    simulate_stack_stability,
)


class TestCollisionPair(unittest.TestCase):
    """Test CollisionPair dataclass."""

    def test_to_description_with_meaningful_penetration(self):
        """Test description formatting for meaningful penetration depth."""
        collision = CollisionPair(
            object_a_name="dining chair",
            object_a_id="dining_chair_a3f2e8b1",
            object_b_name="dining table",
            object_b_id="dining_table_5c9d7e2f",
            penetration_depth=0.05,  # 5cm penetration
        )

        expected = (
            "dining_chair_a3f2e8b1 collides with "
            "dining_table_5c9d7e2f (5.0cm penetration)"
        )
        self.assertEqual(collision.to_description(), expected)

    def test_to_description_with_minimal_penetration(self):
        """Test description formatting for sub-millimeter penetration."""
        collision = CollisionPair(
            object_a_name="chair",
            object_a_id="chair_12345678",
            object_b_name="table",
            object_b_id="table_87654321",
            penetration_depth=0.0001,  # 0.01cm penetration
        )

        expected = "chair_12345678 collides with table_87654321 (touching)"
        self.assertEqual(collision.to_description(), expected)


class TestComputeSceneCollisions(unittest.TestCase):
    """Test compute_scene_collisions function with real physics."""

    def setUp(self):
        """Set up test fixtures with real scene data."""
        # Create base scene with floor plan.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        self.box_sdf_path = test_data_dir / "simple_box.sdf"
        self.sphere_sdf_path = test_data_dir / "simple_sphere.sdf"

        # Create room geometry.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

    def test_no_collisions_with_separated_objects(self):
        """Test that separated objects don't report collisions."""
        # Add two boxes clearly separated.
        box1 = SceneObject(
            object_id=UniqueID("box1"),
            object_type=ObjectType.FURNITURE,
            name="Box 1",
            description="Test box 1",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),  # At origin
            sdf_path=self.box_sdf_path,
        )
        box2 = SceneObject(
            object_id=UniqueID("box2"),
            object_type=ObjectType.FURNITURE,
            name="Box 2",
            description="Test box 2",
            transform=RigidTransform(np.array([3.0, 0.0, 0.5])),  # 3m away
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box1)
        self.scene.add_object(box2)

        # Test actual physics.
        collisions = compute_scene_collisions(self.scene)

        # Filter out floor plan internal collisions (walls, etc.) and focus on furniture.
        furniture_collisions = [
            c
            for c in collisions
            if not (
                c.object_a_name.startswith("floor plan")
                and c.object_b_name.startswith("floor plan")
            )
        ]
        self.assertEqual(
            len(furniture_collisions),
            0,
            "Should not detect collisions between separated furniture objects",
        )

    def test_furniture_to_furniture_multiple_collisions(self):
        """Test multiple furniture collisions with deduplication.

        Creates 3 overlapping boxes to test:
        1. Multiple collisions are detected
        2. Deduplication works (no A-B and B-A duplicates)
        """
        # Create 3 boxes in a row with overlap.
        for i in range(3):
            box = SceneObject(
                object_id=UniqueID(f"box{i}"),
                object_type=ObjectType.FURNITURE,
                name=f"Box {i}",
                description=f"Test box {i}",
                transform=RigidTransform(
                    np.array([i * 0.3, 0.0, 0.5])
                ),  # 0.2m overlap each
                sdf_path=self.box_sdf_path,
            )
            self.scene.add_object(box)

        collisions = compute_scene_collisions(self.scene)

        # Filter for furniture-to-furniture collisions.
        furniture_collisions = [
            c
            for c in collisions
            if "Box" in c.object_a_name and "Box" in c.object_b_name
        ]

        # Should detect exactly 2 collision pairs: (0,1) and (1,2).
        # Box 0 and Box 2 don't overlap.
        self.assertEqual(
            len(furniture_collisions),
            2,
            f"Expected 2 furniture collision pairs, got {len(furniture_collisions)}",
        )

        # Verify no duplicates (each pair should appear only once).
        collision_pairs = set()
        for c in furniture_collisions:
            pair = tuple(sorted([c.object_a_name, c.object_b_name]))
            self.assertNotIn(pair, collision_pairs, "Duplicate collision pair detected")
            collision_pairs.add(pair)

    def test_exact_touching_objects_no_collision(self):
        """Test that exactly touching objects don't report collision.

        Objects with faces exactly touching (0 penetration) should not
        be reported as colliding.
        """
        # Place two 0.5m boxes exactly 0.5m apart (touching faces).
        box1 = SceneObject(
            object_id=UniqueID("box1"),
            object_type=ObjectType.FURNITURE,
            name="Box 1",
            description="Test box 1",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=self.box_sdf_path,
        )
        box2 = SceneObject(
            object_id=UniqueID("box2"),
            object_type=ObjectType.FURNITURE,
            name="Box 2",
            description="Test box 2",
            transform=RigidTransform(np.array([0.5, 0.0, 0.5])),  # Exactly touching
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box1)
        self.scene.add_object(box2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for box-to-box collisions.
        box_collisions = [
            c
            for c in collisions
            if "Box" in c.object_a_name and "Box" in c.object_b_name
        ]

        # Should not detect collision for exactly touching objects.
        self.assertEqual(
            len(box_collisions),
            0,
            "Should not report collision for exactly touching objects",
        )

    def test_floor_penetration_tolerance_ignored(self):
        """Test that floor collision tolerance works correctly.

        Objects slightly penetrating the floor (< 5cm) should not be reported.
        Objects deeply penetrating the floor (> 5cm) should be reported.
        """
        # Test object slightly penetrating floor (2cm).
        box_slight = SceneObject(
            object_id=UniqueID("box_slight"),
            object_type=ObjectType.FURNITURE,
            name="Box Slight",
            description="Slightly penetrating box",
            transform=RigidTransform(np.array([0.0, 0.0, 0.23])),  # 2cm into floor
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box_slight)

        collisions = compute_scene_collisions(
            self.scene, floor_penetration_tolerance=0.05  # 5cm tolerance
        )

        # Should not report slight floor penetration.
        floor_collisions_slight = [
            c
            for c in collisions
            if (
                "floor" in c.object_a_name.lower() or "floor" in c.object_b_name.lower()
            )
            and "Slight" in (c.object_a_name + c.object_b_name)
        ]
        self.assertEqual(
            len(floor_collisions_slight),
            0,
            "Should not report floor collision with penetration < tolerance",
        )

        # Now test deep penetration.
        box_deep = SceneObject(
            object_id=UniqueID("box_deep"),
            object_type=ObjectType.FURNITURE,
            name="Box Deep",
            description="Deeply penetrating box",
            transform=RigidTransform(np.array([2.0, 0.0, 0.15])),  # 10cm into floor
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box_deep)

        collisions = compute_scene_collisions(
            self.scene, floor_penetration_tolerance=0.05  # 5cm tolerance
        )

        # Should report deep floor penetration.
        floor_collisions_deep = [
            c
            for c in collisions
            if (
                "floor" in c.object_a_name.lower() or "floor" in c.object_b_name.lower()
            )
            and "Deep" in (c.object_a_name + c.object_b_name)
        ]
        self.assertGreater(
            len(floor_collisions_deep),
            0,
            "Should report floor collision with penetration > tolerance",
        )

    def test_ceiling_to_ceiling_collision_detected(self):
        """Test collision detection works for CEILING_MOUNTED objects.

        This is a regression test for a bug where ceiling objects were always
        welded even during collision checking, which caused Drake's broadphase
        to miss ceiling-to-ceiling collisions.
        """
        # Add two overlapping ceiling-mounted objects (e.g., track lights).
        ceiling1 = SceneObject(
            object_id=UniqueID("track_light_1"),
            object_type=ObjectType.CEILING_MOUNTED,
            name="Track Light 1",
            description="First ceiling track light",
            transform=RigidTransform(np.array([0.0, 0.0, 3.0])),  # At ceiling height
            sdf_path=self.box_sdf_path,
        )
        ceiling2 = SceneObject(
            object_id=UniqueID("track_light_2"),
            object_type=ObjectType.CEILING_MOUNTED,
            name="Track Light 2",
            description="Second ceiling track light",
            transform=RigidTransform(np.array([0.3, 0.0, 3.0])),  # 0.2m overlap
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(ceiling1)
        self.scene.add_object(ceiling2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for ceiling-to-ceiling collisions.
        ceiling_collisions = [
            c
            for c in collisions
            if "Track Light" in c.object_a_name and "Track Light" in c.object_b_name
        ]

        # Should detect the collision between the two ceiling objects.
        self.assertGreater(
            len(ceiling_collisions),
            0,
            "Should detect collision between overlapping CEILING_MOUNTED objects",
        )

        # Verify penetration depth is approximately 0.2m.
        if ceiling_collisions:
            penetration = ceiling_collisions[0].penetration_depth
            self.assertAlmostEqual(penetration, 0.2, places=1)

    def test_wall_mounted_to_wall_mounted_collision_detected(self):
        """Test collision detection works for WALL_MOUNTED objects.

        This is a regression test for a bug where wall-mounted objects were
        always welded even during collision checking.
        """
        # Add two overlapping wall-mounted objects.
        wall1 = SceneObject(
            object_id=UniqueID("painting_1"),
            object_type=ObjectType.WALL_MOUNTED,
            name="Painting 1",
            description="First wall painting",
            transform=RigidTransform(np.array([0.0, 2.0, 1.5])),  # On wall
            sdf_path=self.box_sdf_path,
        )
        wall2 = SceneObject(
            object_id=UniqueID("painting_2"),
            object_type=ObjectType.WALL_MOUNTED,
            name="Painting 2",
            description="Second wall painting",
            transform=RigidTransform(np.array([0.3, 2.0, 1.5])),  # 0.2m overlap
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(wall1)
        self.scene.add_object(wall2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for wall-to-wall object collisions.
        wall_object_collisions = [
            c
            for c in collisions
            if "Painting" in c.object_a_name and "Painting" in c.object_b_name
        ]

        # Should detect the collision between the two wall-mounted objects.
        self.assertGreater(
            len(wall_object_collisions),
            0,
            "Should detect collision between overlapping WALL_MOUNTED objects",
        )

    def test_collision_detection_by_object_type(self):
        """Test collision detection works for both FURNITURE and MANIPULAND object types."""
        # Test both object types to ensure collision detection works regardless of
        # welding behavior.
        for object_type in [ObjectType.FURNITURE, ObjectType.MANIPULAND]:
            with self.subTest(object_type=object_type):
                # Clear scene between subtests
                test_data_dir = Path(__file__).parent.parent / "test_data"
                self.scene = RoomScene(
                    room_geometry=self.scene.room_geometry, scene_dir=test_data_dir
                )

                # Add two overlapping objects of the specified type
                box1 = SceneObject(
                    object_id=UniqueID("box1"),
                    object_type=object_type,
                    name=f"{object_type.value.title()} 1",
                    description=f"Test {object_type.value} 1",
                    transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
                    sdf_path=self.box_sdf_path,
                )
                box2 = SceneObject(
                    object_id=UniqueID("box2"),
                    object_type=object_type,
                    name=f"{object_type.value.title()} 2",
                    description=f"Test {object_type.value} 2",
                    transform=RigidTransform(np.array([0.3, 0.0, 0.5])),  # 0.2m overlap
                    sdf_path=self.box_sdf_path,
                )
                self.scene.add_object(box1)
                self.scene.add_object(box2)

                collisions = compute_scene_collisions(self.scene)

                # Should detect collision between the two objects.
                object_collisions = [
                    c
                    for c in collisions
                    if object_type.value.title() in c.object_a_name
                    and object_type.value.title() in c.object_b_name
                ]
                self.assertGreater(
                    len(object_collisions),
                    0,
                    f"Should detect collision between overlapping {object_type.value} "
                    "objects",
                )

                # Verify penetration depth is approximately 0.2m.
                if object_collisions:
                    penetration = object_collisions[0].penetration_depth
                    self.assertAlmostEqual(penetration, 0.2, places=1)


class TestCollisionFiltering(unittest.TestCase):
    """Test collision filtering for false positives."""

    def setUp(self):
        """Set up test fixtures with floor plan that has walls."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        self.box_sdf_path = test_data_dir / "simple_box.sdf"

        # Create room geometry with walls.
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

    def test_wall_to_wall_collisions_filtered(self):
        """Test that wall-to-wall collisions are filtered out.

        Adjacent walls in a floor plan naturally intersect at corners.
        These should not be reported as collisions.
        """
        # Test with just the floor plan (no furniture).
        collisions = compute_scene_collisions(self.scene)

        # Filter for wall-to-wall collisions.
        wall_collisions = [
            c
            for c in collisions
            if ("wall" in c.object_a_name.lower() and "wall" in c.object_b_name.lower())
        ]

        # Should not detect wall-to-wall collisions.
        self.assertEqual(
            len(wall_collisions),
            0,
            f"Wall-to-wall collisions should be filtered out, but found: {wall_collisions}",
        )

    def test_self_collisions_filtered(self):
        """Test that self-collisions are filtered out.

        Objects with multiple collision geometries should not report
        collisions between their own geometries.
        """
        # Add a furniture object with multiple collision geometries.
        multi_collision_sdf_path = (
            Path(__file__).parent.parent / "test_data" / "multi_collision_object.sdf"
        )

        chair = SceneObject(
            object_id=UniqueID("office_chair"),
            object_type=ObjectType.FURNITURE,
            name="Office Chair",
            description="Chair with multiple collision geometries",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=multi_collision_sdf_path,
        )
        self.scene.add_object(chair)

        collisions = compute_scene_collisions(self.scene)

        # Filter for self-collisions (same object ID on both sides).
        self_collisions = [
            c
            for c in collisions
            if c.object_a_id == c.object_b_id and c.object_a_id != "room_geometry"
        ]

        # Should not detect self-collisions.
        self.assertEqual(
            len(self_collisions),
            0,
            f"Self-collisions should be filtered out, but found: {self_collisions}",
        )

    def test_legitimate_furniture_collisions_preserved(self):
        """Test that legitimate furniture-to-furniture collisions are still detected.

        Real collisions between different furniture objects should not be filtered.
        """
        # Add two overlapping furniture objects.
        desk1 = SceneObject(
            object_id=UniqueID("desk1"),
            object_type=ObjectType.FURNITURE,
            name="Modern Office Desk",
            description="First desk",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=self.box_sdf_path,
        )
        desk2 = SceneObject(
            object_id=UniqueID("desk2"),
            object_type=ObjectType.FURNITURE,
            name="Modern Office Desk",
            description="Second desk",
            transform=RigidTransform(np.array([0.3, 0.0, 0.5])),  # Overlapping
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(desk1)
        self.scene.add_object(desk2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for desk-to-desk collisions.
        desk_collisions = [
            c
            for c in collisions
            if (
                "Modern Office Desk" in c.object_a_name
                and "Modern Office Desk" in c.object_b_name
            )
            and c.object_a_id != c.object_b_id  # Different objects
        ]

        # Should detect the legitimate collision.
        self.assertGreater(
            len(desk_collisions),
            0,
            "Legitimate furniture-to-furniture collisions should be preserved",
        )

        # Verify penetration depth is reasonable.
        if desk_collisions:
            penetration = desk_collisions[0].penetration_depth
            self.assertGreater(penetration, 0.1, "Should detect significant overlap")


class TestStackCollisionFiltering(unittest.TestCase):
    """Test that intra-stack collisions are correctly filtered."""

    def setUp(self):
        """Set up test fixtures with real stacking assets."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        stacking_assets_dir = test_data_dir / "stacking_assets"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        self.bread_plate_sdf = stacking_assets_dir / "bread_plate" / "bread_plate_2.sdf"

        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

    def test_no_intra_stack_collisions_reported(self):
        """Test that collisions between members of the same stack are NOT reported.

        This tests the fix for the bug where XY proximity matching incorrectly
        mapped stack members to different parent stacks, causing false collision
        reports between members of the same physical stack.
        """
        # Skip if test asset not available.
        if not self.bread_plate_sdf.exists():
            self.skipTest(f"Test asset not found: {self.bread_plate_sdf}")

        # Create stack members as SceneObjects for simulation.
        members = []
        for i in range(3):
            member = SceneObject(
                object_id=UniqueID(f"plate_{i:08x}"),
                object_type=ObjectType.MANIPULAND,
                name=f"Bread Plate",
                description="Test plate",
                transform=RigidTransform(),
                sdf_path=self.bread_plate_sdf,
            )
            members.append(member)

        # Get collision bounds and compute stack transforms.
        bounds_list = [load_collision_bounds_for_scene_object(m) for m in members]
        base_transform = RigidTransform(np.array([0.0, 0.0, 0.0]))
        initial_transforms = compute_initial_stack_transforms(
            bounds_list, base_transform
        )

        # Simulate to get final transforms.
        sim_result = simulate_stack_stability(
            scene_objects=members,
            initial_transforms=initial_transforms,
            ground_xyz=(0.0, 0.0, 0.0),
            simulation_time=1.0,
            simulation_time_step=0.001,
            position_threshold=0.1,
        )
        self.assertTrue(sim_result.is_stable, "Stack should be stable")

        # Build member_assets metadata (same structure as manipuland_tools.py).
        member_assets = []
        for i, (member, final_transform) in enumerate(
            zip(members, sim_result.final_transforms)
        ):
            member_assets.append(
                {
                    "asset_id": str(member.object_id),
                    "name": member.name,
                    "transform": serialize_rigid_transform(final_transform),
                    "sdf_path": str(member.sdf_path.absolute()),
                    "geometry_path": str(member.sdf_path.absolute()),
                }
            )

        # Create the stack scene object with proper metadata structure.
        # Use realistic stack ID like "stack_1" (real stacks use incrementing counters).
        # This gives suffix "1" (1 char), not "test" (4 chars).
        stack = SceneObject(
            object_id=UniqueID("stack_1"),
            object_type=ObjectType.MANIPULAND,
            name="stack_3",
            description="Stack of 3 plates",
            transform=sim_result.final_transforms[0],
            sdf_path=self.bread_plate_sdf,  # Not used for stacks.
            metadata={
                "composite_type": "stack",
                "member_assets": member_assets,
                "num_members": len(members),
            },
        )
        self.scene.add_object(stack)

        # Run collision detection.
        collisions = compute_scene_collisions(self.scene)

        # Filter for collisions involving the stack.
        stack_collisions = [
            c
            for c in collisions
            if "stack" in c.object_a_id.lower() or "stack" in c.object_b_id.lower()
        ]

        # Filter for intra-stack collisions (same stack ID on both sides).
        intra_stack_collisions = [
            c for c in stack_collisions if c.object_a_id == c.object_b_id
        ]

        self.assertEqual(
            len(intra_stack_collisions),
            0,
            f"Should not report intra-stack collisions, but found: {intra_stack_collisions}",
        )


class TestThinCoveringBoundaryViolation(unittest.TestCase):
    """Test ThinCoveringBoundaryViolation dataclass."""

    def test_to_description_single_boundary(self):
        """Test description formatting for single boundary violation."""
        violation = ThinCoveringBoundaryViolation(
            covering_id="rug_12345678",
            exceeded_boundaries=["west"],
        )
        expected = "Thin covering [rug_12345678] extends beyond west boundary"
        self.assertEqual(violation.to_description(), expected)

    def test_to_description_multiple_boundaries(self):
        """Test description formatting for multiple boundary violations."""
        violation = ThinCoveringBoundaryViolation(
            covering_id="rug_87654321",
            exceeded_boundaries=["east", "north"],
        )
        expected = "Thin covering [rug_87654321] extends beyond east, north boundaries"
        self.assertEqual(violation.to_description(), expected)


class TestComputeThinCoveringBoundaryViolations(unittest.TestCase):
    """Test compute_thin_covering_boundary_violations function."""

    def setUp(self):
        """Set up test fixtures with a room geometry."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"

        # Create room geometry with 5m x 5m room.
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
            length=5.0,  # x-dimension
            width=5.0,  # y-dimension
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)
        self.wall_thickness = 0.05  # 5cm walls

    def _create_thin_covering(
        self,
        object_id: str,
        x: float,
        y: float,
        width_m: float,
        depth_m: float,
        shape: str = "rectangular",
        yaw: float = 0.0,
    ) -> SceneObject:
        """Helper to create a thin covering SceneObject.

        Uses FURNITURE type with asset_source="thin_covering" metadata,
        matching how thin coverings are created in production.
        """
        transform = RigidTransform(
            RollPitchYaw(0, 0, yaw).ToRotationMatrix(), np.array([x, y, 0.01])
        )
        return SceneObject(
            object_id=UniqueID(object_id),
            object_type=ObjectType.FURNITURE,  # Thin coverings keep agent's type.
            name=f"Test Rug {object_id}",
            description="Test rug",
            transform=transform,
            sdf_path=Path(
                "/fake/path.sdf"
            ),  # Not used for thin covering boundary check.
            metadata={
                "asset_source": "thin_covering",  # Identified via metadata.
                "width_m": width_m,
                "depth_m": depth_m,
                "shape": shape,
            },
        )

    def test_thin_covering_within_bounds_no_violation(self):
        """Test thin covering fully within bounds reports no violation."""
        # Room is 5m x 5m, wall takes 0.025m on each side, so usable is ~4.95m x 4.95m.
        # Center a 2m x 2m thin covering at origin - should be well within bounds.
        covering = self._create_thin_covering(
            object_id="rug_00000001",
            x=0.0,
            y=0.0,
            width_m=2.0,
            depth_m=2.0,
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(
            len(violations), 0, "Thin covering within bounds should have no violations"
        )

    def test_thin_covering_exceeds_west_boundary(self):
        """Test thin covering extending beyond west boundary."""
        # Room x ranges from -2.5 to 2.5. With wall_thickness=0.05, inner bounds
        # are -2.475 to 2.475. Place a 2m wide thin covering centered at x=-2.0.
        # Left edge at x=-3.0 < -2.475 -> west violation.
        covering = self._create_thin_covering(
            object_id="rug_00000002",
            x=-2.0,
            y=0.0,
            width_m=2.0,
            depth_m=1.0,
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].covering_id, "rug_00000002")
        self.assertIn("west", violations[0].exceeded_boundaries)

    def test_thin_covering_exceeds_multiple_boundaries(self):
        """Test thin covering at corner extending beyond two boundaries."""
        # Place a 2m x 2m thin covering at corner position where it exceeds NE corner.
        # With inner bounds at ±2.475, a 2m thin covering centered at (2.0, 2.0)
        # has edges at x=3.0 > 2.475 and y=3.0 > 2.475.
        covering = self._create_thin_covering(
            object_id="rug_00000003",
            x=2.0,
            y=2.0,
            width_m=2.0,
            depth_m=2.0,
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].covering_id, "rug_00000003")
        # Should have both east and north (sorted alphabetically).
        self.assertIn("east", violations[0].exceeded_boundaries)
        self.assertIn("north", violations[0].exceeded_boundaries)

    def test_circular_thin_covering_within_bounds(self):
        """Test circular thin covering within bounds."""
        covering = self._create_thin_covering(
            object_id="rug_circular_01",
            x=0.0,
            y=0.0,
            width_m=2.0,
            depth_m=2.0,
            shape="circular",
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 0)

    def test_circular_thin_covering_exceeds_boundary(self):
        """Test circular thin covering extending beyond boundary."""
        # Circular thin covering with radius=1.0, centered at x=2.0.
        # Right edge at x=3.0 > 2.475 -> east violation.
        covering = self._create_thin_covering(
            object_id="rug_circular_02",
            x=2.0,
            y=0.0,
            width_m=2.0,  # radius = 1.0
            depth_m=2.0,
            shape="circular",
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 1)
        self.assertIn("east", violations[0].exceeded_boundaries)

    def test_rotated_thin_covering_boundary_check(self):
        """Test rotated rectangular thin covering boundary check using OBB corners."""
        # A 3m x 1m thin covering rotated 45 degrees at origin.
        # After rotation, corners extend further than the unrotated extents.
        # At 45 degrees, a 3x1 thin covering's effective bounding box is approximately:
        # diagonal extent = sqrt((1.5)^2 + (0.5)^2) ≈ 1.58m from center.
        # Place at (1.5, 1.5) with 45 degree rotation.
        covering = self._create_thin_covering(
            object_id="rug_rotated_01",
            x=1.5,
            y=1.5,
            width_m=3.0,
            depth_m=1.0,
            yaw=math.pi / 4,  # 45 degrees
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        # The rotated thin covering extends to ~3.08 from center along the diagonal.
        # From (1.5, 1.5), corners reach beyond 2.475 on north/east.
        self.assertEqual(len(violations), 1)
        self.assertTrue(
            len(violations[0].exceeded_boundaries) >= 1,
            "Rotated thin covering should exceed at least one boundary",
        )

    def test_non_thin_covering_objects_ignored(self):
        """Test that objects without asset_source=thin_covering are ignored."""
        # Create a furniture object (not a thin covering).
        furniture = SceneObject(
            object_id=UniqueID("sofa_001"),
            object_type=ObjectType.FURNITURE,
            name="Sofa",
            description="A sofa",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=Path("/fake/path.sdf"),
            metadata={},  # No asset_source metadata.
        )
        self.scene.add_object(furniture)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )
        # Furniture without thin_covering metadata should be ignored.
        self.assertEqual(len(violations), 0)


class _ObjectWithBounds:
    def __init__(self, object_id: str, object_type: ObjectType, top_height: float):
        self.object_id = UniqueID(object_id)
        self.object_type = object_type
        self._top_height = top_height

    def compute_world_bounds(self):
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, self._top_height],
            ]
        )


class TestWallHeightViolations(unittest.TestCase):
    """Test wall-height checks used by physics feedback."""

    def test_allows_small_numeric_tolerance_at_wall_height(self):
        scene = RoomScene(
            room_geometry=SimpleNamespace(wall_height=2.5),
            scene_dir=Path(__file__).parent.parent / "test_data",
        )
        furniture = _ObjectWithBounds(
            "bookcase_001", ObjectType.FURNITURE, top_height=2.505
        )
        scene.objects[furniture.object_id] = furniture

        violations = compute_wall_height_violations(scene)

        self.assertEqual(violations, [])

    def test_reports_non_ceiling_object_above_tolerance(self):
        scene = RoomScene(
            room_geometry=SimpleNamespace(wall_height=2.5),
            scene_dir=Path(__file__).parent.parent / "test_data",
        )
        furniture = _ObjectWithBounds(
            "bookcase_001", ObjectType.FURNITURE, top_height=2.52
        )
        scene.objects[furniture.object_id] = furniture

        violations = compute_wall_height_violations(scene)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].object_id, "bookcase_001")

    def test_ignores_ceiling_mounted_objects_at_ceiling_plane(self):
        scene = RoomScene(
            room_geometry=SimpleNamespace(wall_height=2.5),
            scene_dir=Path(__file__).parent.parent / "test_data",
        )
        light = _ObjectWithBounds(
            "ceiling_light_001", ObjectType.CEILING_MOUNTED, top_height=2.52
        )
        scene.objects[light.object_id] = light

        violations = compute_wall_height_violations(scene)

        self.assertEqual(violations, [])


class TestClearanceViolationFiltering(unittest.TestCase):
    """Test clearance violation filtering by agent type."""

    def setUp(self):
        """Set up test fixtures with mock scene containing different object types."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.scene = RoomScene(room_geometry=None, scene_dir=test_data_dir)

        # Use an existing SDF file for tests.
        sdf_path = test_data_dir / "test_box.sdf"

        # Add furniture object.
        self.furniture_id = UniqueID("sofa_12345678")
        furniture = SceneObject(
            object_id=self.furniture_id,
            object_type=ObjectType.FURNITURE,
            name="Sofa",
            description="Test sofa",
            transform=RigidTransform(np.array([1.0, 0.0, 0.0])),
            sdf_path=sdf_path,
        )
        self.scene.add_object(furniture)

        # Add ceiling-mounted object.
        self.ceiling_id = UniqueID("chandelier_87654321")
        ceiling_obj = SceneObject(
            object_id=self.ceiling_id,
            object_type=ObjectType.CEILING_MOUNTED,
            name="Chandelier",
            description="Test chandelier",
            transform=RigidTransform(np.array([2.0, 0.0, 2.5])),
            sdf_path=sdf_path,
        )
        self.scene.add_object(ceiling_obj)

        # Add wall-mounted object.
        self.wall_id = UniqueID("mirror_11112222")
        wall_obj = SceneObject(
            object_id=self.wall_id,
            object_type=ObjectType.WALL_MOUNTED,
            name="Mirror",
            description="Test mirror",
            transform=RigidTransform(np.array([0.0, 2.0, 1.5])),
            sdf_path=sdf_path,
        )
        self.scene.add_object(wall_obj)

    def test_filter_door_violations_furniture_agent(self):
        """FurnitureAgent sees only door violations from FURNITURE objects."""
        violations = [
            DoorClearanceViolation(
                furniture_id=str(self.furniture_id),
                door_label="door_1",
                penetration_depth=0.1,
            ),
            DoorClearanceViolation(
                furniture_id=str(self.ceiling_id),
                door_label="door_2",
                penetration_depth=0.2,
            ),
        ]

        filtered = filter_door_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].furniture_id, str(self.furniture_id))

    def test_filter_door_violations_ceiling_agent(self):
        """CeilingAgent sees only door violations from CEILING_MOUNTED objects."""
        violations = [
            DoorClearanceViolation(
                furniture_id=str(self.furniture_id),
                door_label="door_1",
                penetration_depth=0.1,
            ),
            DoorClearanceViolation(
                furniture_id=str(self.ceiling_id),
                door_label="door_2",
                penetration_depth=0.2,
            ),
        ]

        filtered = filter_door_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].furniture_id, str(self.ceiling_id))

    def test_filter_door_violations_floor_plan_agent_gets_none(self):
        """FloorPlanAgent sees no door violations (can't move objects)."""
        violations = [
            DoorClearanceViolation(
                furniture_id=str(self.furniture_id),
                door_label="door_1",
                penetration_depth=0.1,
            ),
        ]

        filtered = filter_door_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FLOOR_PLAN,
        )

        self.assertEqual(len(filtered), 0)

    def test_filter_window_violations_by_agent(self):
        """Window violations are filtered by object type."""
        violations = [
            WindowClearanceViolation(
                furniture_id=str(self.furniture_id),
                window_label="window_1",
                furniture_top_height=1.5,
                sill_height=1.0,
            ),
            WindowClearanceViolation(
                furniture_id=str(self.wall_id),
                window_label="window_2",
                furniture_top_height=2.0,
                sill_height=1.2,
            ),
        ]

        # FurnitureAgent sees furniture violations.
        furniture_filtered = filter_window_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].furniture_id, str(self.furniture_id))

        # WallAgent sees wall-mounted violations.
        wall_filtered = filter_window_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].furniture_id, str(self.wall_id))

    def test_filter_open_connection_violations_by_agent(self):
        """Open connection violations filter by any blocking furniture of agent type."""
        violations = [
            OpenConnectionBlockedViolation(
                opening_label="open_living_kitchen",
                blocking_furniture_ids=[str(self.furniture_id), str(self.ceiling_id)],
                required_passage_size=0.8,
            ),
            OpenConnectionBlockedViolation(
                opening_label="open_hallway",
                blocking_furniture_ids=[str(self.wall_id)],
                required_passage_size=0.8,
            ),
        ]

        # FurnitureAgent sees first violation (has furniture in blocking list).
        furniture_filtered = filter_open_connection_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].opening_label, "open_living_kitchen")

        # CeilingAgent also sees first violation (has ceiling object in list).
        ceiling_filtered = filter_open_connection_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 1)
        self.assertEqual(ceiling_filtered[0].opening_label, "open_living_kitchen")

        # WallAgent sees second violation (has wall-mounted object).
        wall_filtered = filter_open_connection_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].opening_label, "open_hallway")

    def test_filter_wall_height_violations_by_agent(self):
        """Wall height violations filter by object type matching agent."""
        violations = [
            WallHeightExceededViolation(
                object_id=str(self.furniture_id),
                object_top_height=3.2,
                wall_height=3.0,
            ),
            WallHeightExceededViolation(
                object_id=str(self.ceiling_id),
                object_top_height=3.5,
                wall_height=3.0,
            ),
            WallHeightExceededViolation(
                object_id=str(self.wall_id),
                object_top_height=3.1,
                wall_height=3.0,
            ),
        ]

        # FurnitureAgent sees only furniture height violation.
        furniture_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].object_id, str(self.furniture_id))

        # CeilingAgent sees only ceiling object height violation.
        ceiling_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 1)
        self.assertEqual(ceiling_filtered[0].object_id, str(self.ceiling_id))

        # WallAgent sees only wall-mounted object height violation.
        wall_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].object_id, str(self.wall_id))

        # FloorPlanAgent sees none (no object type).
        floor_plan_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FLOOR_PLAN,
        )
        self.assertEqual(len(floor_plan_filtered), 0)

    def test_filter_collisions_by_agent(self):
        """Collisions filter to show only those involving agent's object type."""
        collisions = [
            CollisionPair(
                object_a_name="Sofa",
                object_a_id=str(self.furniture_id),
                object_b_name="Chandelier",
                object_b_id=str(self.ceiling_id),
                penetration_depth=0.05,
            ),
            CollisionPair(
                object_a_name="Mirror",
                object_a_id=str(self.wall_id),
                object_b_name="Chandelier",
                object_b_id=str(self.ceiling_id),
                penetration_depth=0.03,
            ),
        ]

        # FurnitureAgent sees first collision (involves furniture).
        furniture_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].object_a_id, str(self.furniture_id))

        # CeilingAgent sees both (chandelier in both).
        ceiling_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 2)

        # WallAgent sees second collision (involves mirror).
        wall_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].object_a_id, str(self.wall_id))

        # FloorPlanAgent sees none.
        floor_plan_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.FLOOR_PLAN,
        )
        self.assertEqual(len(floor_plan_filtered), 0)

    def test_filter_thin_covering_overlaps_by_agent(self):
        """Thin covering overlaps filter by owner agent type."""
        # Both furniture_id objects are FURNITURE type, so both overlaps
        # are owned by FurnitureAgent.
        overlaps = [
            ThinCoveringOverlap(
                covering_a_name="Rug",
                covering_a_id=str(self.furniture_id),
                covering_b_name="Carpet",
                covering_b_id=str(self.furniture_id),
            ),
        ]

        # FurnitureAgent sees overlap (furniture-owned thin coverings).
        furniture_filtered = filter_thin_covering_overlaps_by_agent(
            overlaps=overlaps,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)

        # CeilingAgent sees none (no ceiling-owned thin coverings).
        ceiling_filtered = filter_thin_covering_overlaps_by_agent(
            overlaps=overlaps,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 0)

        # WallAgent sees none (no wall-owned thin coverings).
        wall_filtered = filter_thin_covering_overlaps_by_agent(
            overlaps=overlaps,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 0)

    def test_filter_thin_covering_boundary_violations_by_agent(self):
        """Thin covering boundary violations only shown to FurnitureAgent."""
        violations = [
            ThinCoveringBoundaryViolation(
                covering_id="rug_123",
                exceeded_boundaries=["north", "east"],
            ),
        ]

        # FurnitureAgent sees floor covering boundary violations.
        furniture_filtered = filter_thin_covering_boundary_violations_by_agent(
            violations=violations,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)

        # Other agents see none (only floor coverings have boundary constraints).
        for agent_type in [
            AgentType.CEILING_MOUNTED,
            AgentType.WALL_MOUNTED,
            AgentType.MANIPULAND,
            AgentType.FLOOR_PLAN,
        ]:
            filtered = filter_thin_covering_boundary_violations_by_agent(
                violations=violations,
                agent_type=agent_type,
            )
            self.assertEqual(len(filtered), 0)


class TestWallMountedParentLookup(unittest.TestCase):
    """Test that manipulands on wall-mounted objects are correctly handled."""

    def setUp(self):
        """Set up test fixtures with a wall-mounted object and manipuland."""
        test_data_dir = Path(__file__).parent.parent / "test_data"
        self.scene = RoomScene(room_geometry=None, scene_dir=test_data_dir)

        # Create a wall-mounted shelf with a support surface.
        self.wall_shelf_id = UniqueID("wall_shelf_0")
        self.surface_id = UniqueID("S_5")

        wall_shelf = SceneObject(
            object_id=self.wall_shelf_id,
            object_type=ObjectType.WALL_MOUNTED,
            name="Wall Shelf",
            description="A floating wall shelf",
            transform=RigidTransform(np.array([2.0, 0.0, 1.5])),
            sdf_path=Path("/fake/wall_shelf.sdf"),
            support_surfaces=[
                SupportSurface(
                    surface_id=self.surface_id,
                    bounding_box_min=np.array([-0.5, -0.12, 0.0]),
                    bounding_box_max=np.array([0.5, 0.12, 0.05]),
                    transform=RigidTransform(np.array([2.0, 0.0, 1.55])),
                )
            ],
        )
        self.scene.add_object(wall_shelf)

        # Create a manipuland placed on the wall shelf.
        self.manipuland_id = UniqueID("book_0")
        manipuland = SceneObject(
            object_id=self.manipuland_id,
            object_type=ObjectType.MANIPULAND,
            name="Book",
            description="A book on the shelf",
            transform=RigidTransform(np.array([2.0, 0.0, 1.6])),
            sdf_path=Path("/fake/book.sdf"),
            placement_info=PlacementInfo(
                parent_surface_id=self.surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        self.scene.add_object(manipuland)

    def test_wall_mounted_parent_found(self):
        """Test that manipulands on wall-mounted objects find their parent.

        This is a regression test for the bug where _get_furniture_id_for_manipuland
        only checked for FURNITURE and FLOOR types, missing WALL_MOUNTED.
        """
        parent_id = _get_furniture_id_for_manipuland(
            manipuland_id=str(self.manipuland_id),
            scene=self.scene,
        )

        self.assertEqual(
            parent_id,
            str(self.wall_shelf_id),
            "Manipuland on wall-mounted object should find its parent",
        )

    def test_manipuland_without_placement_info_returns_none(self):
        """Test that manipulands without placement_info return None."""
        # Create manipuland without placement_info.
        orphan_id = UniqueID("orphan_book")
        orphan = SceneObject(
            object_id=orphan_id,
            object_type=ObjectType.MANIPULAND,
            name="Orphan Book",
            description="A book without placement info",
            transform=RigidTransform(np.array([0.0, 0.0, 0.0])),
            sdf_path=Path("/fake/book.sdf"),
        )
        self.scene.add_object(orphan)

        parent_id = _get_furniture_id_for_manipuland(
            manipuland_id=str(orphan_id),
            scene=self.scene,
        )

        self.assertIsNone(parent_id)


class TestFloorManipulandParentLookup(unittest.TestCase):
    """Test that manipulands on the floor find their parent via room_geometry.floor."""

    def setUp(self):
        """Set up test fixtures with a floor and manipuland."""
        from unittest.mock import Mock

        test_data_dir = Path(__file__).parent.parent / "test_data"

        # Create mock room_geometry with a floor object.
        self.floor_id = UniqueID("floor_bedroom")
        self.floor_surface_id = "S_floor"

        floor_obj = SceneObject(
            object_id=self.floor_id,
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Floor surface",
            transform=RigidTransform(),
            sdf_path=None,
            support_surfaces=[
                SupportSurface(
                    surface_id=self.floor_surface_id,
                    bounding_box_min=np.array([-2.0, -2.0, 0.0]),
                    bounding_box_max=np.array([2.0, 2.0, 0.1]),
                    transform=RigidTransform(np.array([0.0, 0.0, 0.01])),
                )
            ],
        )

        room_geometry = Mock()
        room_geometry.floor = floor_obj

        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Create a manipuland placed on the floor.
        self.manipuland_id = UniqueID("backpack_0")
        manipuland = SceneObject(
            object_id=self.manipuland_id,
            object_type=ObjectType.MANIPULAND,
            name="Backpack",
            description="A backpack on the floor",
            transform=RigidTransform(np.array([1.0, 0.5, 0.1])),
            sdf_path=Path("/fake/backpack.sdf"),
            placement_info=PlacementInfo(
                parent_surface_id=self.floor_surface_id,
                position_2d=np.array([1.0, 0.5]),
                rotation_2d=0.0,
            ),
        )
        self.scene.add_object(manipuland)

    def test_floor_manipuland_parent_found_via_room_geometry(self):
        """Test that manipulands on floor find their parent via room_geometry.floor.

        This is a regression test for the bug where _get_furniture_id_for_manipuland
        only searched scene.objects but the floor is in room_geometry.floor.
        """
        parent_id = _get_furniture_id_for_manipuland(
            manipuland_id=str(self.manipuland_id),
            scene=self.scene,
        )

        self.assertEqual(
            parent_id,
            str(self.floor_id),
            "Manipuland on floor should find its parent via room_geometry.floor",
        )


if __name__ == "__main__":
    unittest.main()
