"""Clearance zone computation and violation detection for door/window/opening awareness.

This module provides:
- Violation dataclasses for physics constraint feedback
- Opening data computation from PlacedRoom walls
- AABB-based violation detection for doors and windows
- Sweep-based passability check for open connections
"""

import logging

from dataclasses import dataclass

from scenesmith.agent_utils.house import (
    ClearanceOpeningData,
    OpeningType,
    PlacedRoom,
    Wall,
    WallDirection,
)
from scenesmith.agent_utils.room import ObjectType, RoomScene

console_logger = logging.getLogger(__name__)

WALL_HEIGHT_TOLERANCE_M = 0.01


@dataclass
class DoorClearanceViolation:
    """Violation when furniture blocks a door clearance zone."""

    furniture_id: str
    """ID of the furniture blocking the door."""

    door_label: str
    """Label of the blocked door (e.g., 'door_1')."""

    penetration_depth: float
    """How far the furniture penetrates the clearance zone in meters."""

    def to_description(self) -> str:
        """Format violation for physics output."""
        return (
            f"Furniture {self.furniture_id} blocks {self.door_label} "
            f"({self.penetration_depth:.2f}m penetration)"
        )


@dataclass
class WindowClearanceViolation:
    """Warning when furniture blocks a window above sill height."""

    furniture_id: str
    """ID of the furniture blocking the window."""

    window_label: str
    """Label of the blocked window (e.g., 'window_1')."""

    furniture_top_height: float
    """Height of the top of the blocking furniture in meters."""

    sill_height: float
    """Height of the window sill in meters."""

    def to_description(self) -> str:
        """Format violation for physics output."""
        return (
            f"Furniture {self.furniture_id} blocks {self.window_label} "
            f"(furniture top: {self.furniture_top_height:.2f}m, "
            f"sill: {self.sill_height:.2f}m)"
        )


@dataclass
class OpenConnectionBlockedViolation:
    """Violation when an open connection is completely blocked by furniture."""

    opening_label: str
    """Label of the blocked opening (e.g., 'open_living_kitchen')."""

    blocking_furniture_ids: list[str]
    """IDs of furniture pieces blocking the opening."""

    required_passage_size: float
    """Required passage size in meters."""

    def to_description(self) -> str:
        """Format violation for physics output."""
        return (
            f"Open connection {self.opening_label} is completely blocked - "
            f"no {self.required_passage_size}m passage available for robot exit"
        )


@dataclass
class WallHeightExceededViolation:
    """Violation when an object exceeds the wall height."""

    object_id: str
    """ID of the object exceeding wall height."""

    object_top_height: float
    """Height of the top of the object in meters."""

    wall_height: float
    """Height of the walls in meters."""

    def to_description(self) -> str:
        """Format violation for physics output."""
        return (
            f"Object {self.object_id} exceeds wall height "
            f"(object top: {self.object_top_height:.2f}m, wall: {self.wall_height:.2f}m). "
            f"Remove or replace with shorter object."
        )


def _compute_opening_center(
    wall: Wall,
    position_along_wall: float,
    opening_width: float,
    sill_height: float,
    opening_height: float,
) -> list[float]:
    """Compute world-space center of an opening.

    Args:
        wall: Wall containing the opening.
        position_along_wall: Distance from wall start to opening LEFT EDGE.
        opening_width: Width of the opening.
        sill_height: Height from floor to bottom of opening.
        opening_height: Height of the opening itself.

    Returns:
        [x, y, z] world coordinates of opening center.
    """
    # Wall direction vector (normalized).
    dx = wall.end_point[0] - wall.start_point[0]
    dy = wall.end_point[1] - wall.start_point[1]
    length = wall.length if wall.length > 0 else 1.0
    dir_x = dx / length
    dir_y = dy / length

    # Opening center along wall (position_along_wall is LEFT EDGE, add half width).
    center_position = position_along_wall + opening_width / 2.0
    center_x = wall.start_point[0] + dir_x * center_position
    center_y = wall.start_point[1] + dir_y * center_position

    # Z center: midpoint of opening height.
    center_z = sill_height + opening_height / 2.0

    return [center_x, center_y, center_z]


def _compute_clearance_bbox(
    wall: Wall,
    position_along_wall: float,
    width: float,
    clearance_height: float,
    clearance_distance: float,
) -> tuple[list[float], list[float]]:
    """Compute clearance zone AABB for a door or window.

    The clearance zone extends from the opening INTO the room.

    Args:
        wall: Wall containing the opening.
        position_along_wall: Distance from wall start to opening LEFT EDGE.
        width: Opening width.
        clearance_height: Height of the clearance zone (use door/window height,
            not wall height, so objects above openings aren't flagged).
        clearance_distance: How far the zone extends into the room.

    Returns:
        (bbox_min, bbox_max) as [x, y, z] lists.
    """
    # Wall direction vector.
    dx = wall.end_point[0] - wall.start_point[0]
    dy = wall.end_point[1] - wall.start_point[1]
    length = wall.length if wall.length > 0 else 1.0
    dir_x = dx / length
    dir_y = dy / length

    # Opening center along wall (position_along_wall is LEFT EDGE, add half width).
    half_width = width / 2.0
    center_position = position_along_wall + half_width
    center_x = wall.start_point[0] + dir_x * center_position
    center_y = wall.start_point[1] + dir_y * center_position

    # Compute AABB based on wall direction.

    if wall.direction in (WallDirection.NORTH, WallDirection.SOUTH):
        # Wall runs along X axis.
        min_x = center_x - half_width
        max_x = center_x + half_width
        # Clearance extends in Y direction (normal direction).
        if wall.direction == WallDirection.NORTH:
            # North wall at +Y, clearance extends into room (-Y).
            min_y = center_y - clearance_distance
            max_y = center_y
        else:
            # South wall at -Y, clearance extends into room (+Y).
            min_y = center_y
            max_y = center_y + clearance_distance
    else:
        # Wall runs along Y axis (EAST or WEST).
        min_y = center_y - half_width
        max_y = center_y + half_width
        # Clearance extends in X direction (normal direction).
        if wall.direction == WallDirection.EAST:
            # East wall at +X, clearance extends into room (-X).
            min_x = center_x - clearance_distance
            max_x = center_x
        else:
            # West wall at -X, clearance extends into room (+X).
            min_x = center_x
            max_x = center_x + clearance_distance

    # Clearance zone height (floor to opening top).
    min_z = 0.0
    max_z = clearance_height

    return [min_x, min_y, min_z], [max_x, max_y, max_z]


def compute_openings_data(
    placed_room: PlacedRoom,
    wall_height: float,
    door_clearance_distance: float,
    window_clearance_distance: float,
) -> list[ClearanceOpeningData]:
    """Compute opening data for all doors/windows/open connections.

    This function is called once in the floor plan agent and the result
    is stored in RoomGeometry.openings.

    Coordinates are transformed from house coordinates (room corner at position)
    to room-local coordinates (room center at origin) since furniture placement
    and rendering use room-local coordinates.

    Args:
        placed_room: Room with computed wall positions.
        wall_height: Height of the walls in meters.
        door_clearance_distance: Clearance distance for doors.
        window_clearance_distance: Clearance distance for windows.

    Returns:
        List of ClearanceOpeningData with physics and rendering data.
    """
    openings_data: list[ClearanceOpeningData] = []

    # Compute offset to transform from house coords to room-local coords.
    # House coords: room corner at placed_room.position.
    # Room-local coords: room center at origin.
    offset_x = placed_room.position[0] + placed_room.width / 2
    offset_y = placed_room.position[1] + placed_room.depth / 2

    for wall in placed_room.walls:
        for opening in wall.openings:
            # For OPEN type, use wall_height since opening.height is not stored.
            effective_height = (
                wall_height
                if opening.opening_type == OpeningType.OPEN
                else opening.height
            )

            # Compute opening center in house coordinates.
            center_house = _compute_opening_center(
                wall=wall,
                position_along_wall=opening.position_along_wall,
                opening_width=opening.width,
                sill_height=opening.sill_height,
                opening_height=effective_height,
            )

            # Transform to room-local coordinates.
            center = [
                center_house[0] - offset_x,
                center_house[1] - offset_y,
                center_house[2],  # Z stays the same.
            ]

            # Compute clearance zone (None for OPEN type).
            clearance_bbox_min = None
            clearance_bbox_max = None

            if opening.opening_type == OpeningType.DOOR:
                # Use door height, not wall height. Objects above door height
                # (e.g., wall clocks) don't block passage through the door.
                bbox_min_house, bbox_max_house = _compute_clearance_bbox(
                    wall=wall,
                    position_along_wall=opening.position_along_wall,
                    width=opening.width,
                    clearance_height=opening.height,
                    clearance_distance=door_clearance_distance,
                )
                # Transform clearance bbox to room-local coordinates.
                clearance_bbox_min = [
                    bbox_min_house[0] - offset_x,
                    bbox_min_house[1] - offset_y,
                    bbox_min_house[2],
                ]
                clearance_bbox_max = [
                    bbox_max_house[0] - offset_x,
                    bbox_max_house[1] - offset_y,
                    bbox_max_house[2],
                ]
            elif opening.opening_type == OpeningType.WINDOW:
                # Use window height, not wall height. Objects above windows
                # shouldn't be flagged as blocking the window.
                bbox_min_house, bbox_max_house = _compute_clearance_bbox(
                    wall=wall,
                    position_along_wall=opening.position_along_wall,
                    width=opening.width,
                    clearance_height=opening.sill_height + opening.height,
                    clearance_distance=window_clearance_distance,
                )
                # Transform clearance bbox to room-local coordinates.
                clearance_bbox_min = [
                    bbox_min_house[0] - offset_x,
                    bbox_min_house[1] - offset_y,
                    bbox_min_house[2],
                ]
                clearance_bbox_max = [
                    bbox_max_house[0] - offset_x,
                    bbox_max_house[1] - offset_y,
                    bbox_max_house[2],
                ]
            # OPEN type: no clearance zone, uses sweep algorithm.

            # Transform wall start/end to room-local coordinates.
            wall_start_local = [
                wall.start_point[0] - offset_x,
                wall.start_point[1] - offset_y,
            ]
            wall_end_local = [
                wall.end_point[0] - offset_x,
                wall.end_point[1] - offset_y,
            ]

            opening_data = ClearanceOpeningData(
                opening_id=opening.opening_id,
                opening_type=opening.opening_type.value,
                wall_direction=wall.direction.value,
                center_world=center,
                width=opening.width,
                sill_height=opening.sill_height,
                height=opening.height,
                clearance_bbox_min=clearance_bbox_min,
                clearance_bbox_max=clearance_bbox_max,
                wall_start=wall_start_local,
                wall_end=wall_end_local,
                position_along_wall=opening.position_along_wall,
            )
            openings_data.append(opening_data)

    return openings_data


def _aabb_intersects(
    obj_min: list[float],
    obj_max: list[float],
    zone_min: list[float],
    zone_max: list[float],
) -> bool:
    """Check if two AABBs intersect.

    Args:
        obj_min: Object AABB minimum [x, y, z].
        obj_max: Object AABB maximum [x, y, z].
        zone_min: Zone AABB minimum [x, y, z].
        zone_max: Zone AABB maximum [x, y, z].

    Returns:
        True if AABBs overlap.
    """
    return (
        obj_min[0] < zone_max[0]
        and obj_max[0] > zone_min[0]
        and obj_min[1] < zone_max[1]
        and obj_max[1] > zone_min[1]
        and obj_min[2] < zone_max[2]
        and obj_max[2] > zone_min[2]
    )


def _compute_penetration_depth(
    obj_min: list[float],
    obj_max: list[float],
    zone_min: list[float],
    zone_max: list[float],
) -> float:
    """Compute how deep an object penetrates a zone.

    Returns the minimum penetration depth across all axes.
    """
    penetration_x = min(obj_max[0] - zone_min[0], zone_max[0] - obj_min[0])
    penetration_y = min(obj_max[1] - zone_min[1], zone_max[1] - obj_min[1])
    penetration_z = min(obj_max[2] - zone_min[2], zone_max[2] - obj_min[2])
    return min(penetration_x, penetration_y, penetration_z)


def compute_door_clearance_violations(
    scene: RoomScene,
) -> list[DoorClearanceViolation]:
    """Check furniture AABB intersection with door clearance zones.

    Args:
        scene: RoomScene with furniture objects.

    Returns:
        List of door clearance violations.
    """
    violations = []
    room_geom = scene.room_geometry
    if not room_geom or not room_geom.openings:
        return violations

    for opening in room_geom.openings:
        if opening.opening_type != "door":
            continue

        zone_min = opening.clearance_bbox_min
        zone_max = opening.clearance_bbox_max
        if zone_min is None or zone_max is None:
            continue

        for obj in scene.objects.values():
            # Skip structural elements and thin coverings - they don't block doors.
            if obj.object_type in (ObjectType.WALL, ObjectType.FLOOR):
                continue
            if obj.metadata.get("asset_source") == "thin_covering":
                continue

            # Compute object world-space AABB using existing method.
            world_bounds = obj.compute_world_bounds()
            if world_bounds is None:
                continue
            obj_min = list(world_bounds[0])
            obj_max = list(world_bounds[1])

            if _aabb_intersects(
                obj_min=obj_min,
                obj_max=obj_max,
                zone_min=zone_min,
                zone_max=zone_max,
            ):
                penetration = _compute_penetration_depth(
                    obj_min=obj_min,
                    obj_max=obj_max,
                    zone_min=zone_min,
                    zone_max=zone_max,
                )
                violations.append(
                    DoorClearanceViolation(
                        furniture_id=str(obj.object_id),
                        door_label=opening.opening_id,
                        penetration_depth=penetration,
                    )
                )

    return violations


def compute_window_clearance_violations(
    scene: RoomScene,
) -> list[WindowClearanceViolation]:
    """Check furniture above sill height intersecting window clearance zones.

    Only reports violations where furniture top exceeds window sill height.

    Args:
        scene: RoomScene with furniture objects.

    Returns:
        List of window clearance violations (warnings, not failures).
    """
    violations = []
    room_geom = scene.room_geometry
    if not room_geom or not room_geom.openings:
        return violations

    for opening in room_geom.openings:
        if opening.opening_type != "window":
            continue

        zone_min = opening.clearance_bbox_min
        zone_max = opening.clearance_bbox_max
        sill_height = opening.sill_height
        if zone_min is None or zone_max is None:
            continue

        for obj in scene.objects.values():
            # Skip structural elements and thin coverings - they don't block windows.
            if obj.object_type in (ObjectType.WALL, ObjectType.FLOOR):
                continue
            if obj.metadata.get("asset_source") == "thin_covering":
                continue

            # Compute object world-space AABB using existing method.
            world_bounds = obj.compute_world_bounds()
            if world_bounds is None:
                continue
            obj_min = list(world_bounds[0])
            obj_max = list(world_bounds[1])

            # Check if furniture top exceeds sill height.
            furniture_top = obj_max[2]
            if furniture_top <= sill_height:
                continue  # Furniture below window sill is OK.

            # Check XY intersection with clearance zone.
            if (
                obj_min[0] < zone_max[0]
                and obj_max[0] > zone_min[0]
                and obj_min[1] < zone_max[1]
                and obj_max[1] > zone_min[1]
            ):
                violations.append(
                    WindowClearanceViolation(
                        furniture_id=str(obj.object_id),
                        window_label=opening.opening_id,
                        furniture_top_height=furniture_top,
                        sill_height=sill_height,
                    )
                )

    return violations


def compute_open_connection_blocked_violations(
    scene: RoomScene, passage_size: float, open_connection_clearance: float
) -> list[OpenConnectionBlockedViolation]:
    """Check if open connections have at least one robot-passable corridor.

    NOTE: This check is REQUIRED but NOT SUFFICIENT for ensuring robot passage
    through open connections. We can only verify that the opening is accessible
    from within this room, but we have no access to the connecting room's
    furniture data.

    Args:
        scene: RoomScene with furniture objects.
        passage_size: Required robot passage width/depth in meters.
        open_connection_clearance: How far into room to check for clearance.

    Returns:
        List of open connection blocked violations.
    """
    violations = []
    room_geom = scene.room_geometry
    if not room_geom or not room_geom.openings:
        return violations

    # Get furniture footprints (XY projection).
    furniture_footprints = []
    for obj in scene.objects.values():
        # Skip structural elements and thin coverings - they don't block passages.
        if obj.object_type in (ObjectType.WALL, ObjectType.FLOOR):
            continue
        if obj.metadata.get("asset_source") == "thin_covering":
            continue

        # Compute object world-space AABB using existing method.
        world_bounds = obj.compute_world_bounds()
        if world_bounds is None:
            continue
        obj_min, obj_max = world_bounds
        footprint = {
            "id": str(obj.object_id),
            "min_x": obj_min[0],
            "max_x": obj_max[0],
            "min_y": obj_min[1],
            "max_y": obj_max[1],
        }
        furniture_footprints.append(footprint)

    for opening in room_geom.openings:
        if opening.opening_type != "open":
            continue

        # Get opening geometry.
        wall_direction = WallDirection(opening.wall_direction)
        opening_width = opening.width
        position_along_wall = opening.position_along_wall
        wall_start = opening.wall_start
        wall_end = opening.wall_end

        # Compute opening center (position_along_wall is LEFT EDGE, add half width).
        dx = wall_end[0] - wall_start[0]
        dy = wall_end[1] - wall_start[1]
        wall_length = (dx**2 + dy**2) ** 0.5
        if wall_length == 0:
            continue
        dir_x = dx / wall_length
        dir_y = dy / wall_length

        center_position = position_along_wall + opening_width / 2.0
        opening_center_x = wall_start[0] + dir_x * center_position
        opening_center_y = wall_start[1] + dir_y * center_position

        # Sweep a passage_size square across the opening width.
        # Check if any position has a clear corridor into the room.
        num_positions = max(1, int((opening_width - passage_size) / 0.1) + 1)
        found_clear_passage = False

        for i in range(num_positions):
            # Position along opening.
            offset = -opening_width / 2 + passage_size / 2 + i * 0.1
            if offset + passage_size / 2 > opening_width / 2:
                offset = opening_width / 2 - passage_size / 2

            # Corridor center.
            corridor_center_x = opening_center_x + dir_x * offset
            corridor_center_y = opening_center_y + dir_y * offset

            # Corridor extends from wall into room.
            # Check a rectangle: passage_size × open_connection_clearance.
            if wall_direction in (WallDirection.NORTH, WallDirection.SOUTH):
                corridor_min_x = corridor_center_x - passage_size / 2
                corridor_max_x = corridor_center_x + passage_size / 2
                if wall_direction == WallDirection.NORTH:
                    corridor_min_y = opening_center_y - open_connection_clearance
                    corridor_max_y = opening_center_y
                else:
                    corridor_min_y = opening_center_y
                    corridor_max_y = opening_center_y + open_connection_clearance
            else:
                corridor_min_y = corridor_center_y - passage_size / 2
                corridor_max_y = corridor_center_y + passage_size / 2
                if wall_direction == WallDirection.EAST:
                    corridor_min_x = opening_center_x - open_connection_clearance
                    corridor_max_x = opening_center_x
                else:
                    corridor_min_x = opening_center_x
                    corridor_max_x = opening_center_x + open_connection_clearance

            # Check if corridor is clear of furniture.
            corridor_clear = True
            for footprint in furniture_footprints:
                if (
                    footprint["min_x"] < corridor_max_x
                    and footprint["max_x"] > corridor_min_x
                    and footprint["min_y"] < corridor_max_y
                    and footprint["max_y"] > corridor_min_y
                ):
                    corridor_clear = False
                    break

            if corridor_clear:
                found_clear_passage = True
                break

        if not found_clear_passage and furniture_footprints:
            # Find which furniture pieces are blocking.
            blocking_ids = []
            for footprint in furniture_footprints:
                # Check if furniture is near the opening.
                # Use full opening width for blocking check.
                if wall_direction in (WallDirection.NORTH, WallDirection.SOUTH):
                    opening_min_x = opening_center_x - opening_width / 2
                    opening_max_x = opening_center_x + opening_width / 2
                    if wall_direction == WallDirection.NORTH:
                        opening_min_y = opening_center_y - open_connection_clearance
                        opening_max_y = opening_center_y
                    else:
                        opening_min_y = opening_center_y
                        opening_max_y = opening_center_y + open_connection_clearance
                else:
                    opening_min_y = opening_center_y - opening_width / 2
                    opening_max_y = opening_center_y + opening_width / 2
                    if wall_direction == WallDirection.EAST:
                        opening_min_x = opening_center_x - open_connection_clearance
                        opening_max_x = opening_center_x
                    else:
                        opening_min_x = opening_center_x
                        opening_max_x = opening_center_x + open_connection_clearance

                if (
                    footprint["min_x"] < opening_max_x
                    and footprint["max_x"] > opening_min_x
                    and footprint["min_y"] < opening_max_y
                    and footprint["max_y"] > opening_min_y
                ):
                    blocking_ids.append(footprint["id"])

            if blocking_ids:
                violations.append(
                    OpenConnectionBlockedViolation(
                        opening_label=opening.opening_id,
                        blocking_furniture_ids=blocking_ids,
                        required_passage_size=passage_size,
                    )
                )

    return violations


def compute_wall_height_violations(
    scene: RoomScene,
) -> list[WallHeightExceededViolation]:
    """Check if any non-ceiling objects exceed wall height.

    Args:
        scene: RoomScene with objects.

    Returns:
        List of wall height exceeded violations.
    """
    violations = []
    room_geom = scene.room_geometry
    if not room_geom:
        return violations

    wall_height = room_geom.wall_height
    if wall_height <= 0:
        return violations

    for obj in scene.objects.values():
        if obj.object_type == ObjectType.CEILING_MOUNTED:
            # Ceiling-mounted assets are canonicalized with their top at z=0 and
            # then placed at the ceiling plane. Treat that flush contact as valid.
            continue

        # Compute object world-space AABB using existing method.
        world_bounds = obj.compute_world_bounds()
        if world_bounds is None:
            continue
        obj_max = world_bounds[1]

        # Object top height.
        obj_top = obj_max[2]

        if obj_top > wall_height + WALL_HEIGHT_TOLERANCE_M:
            violations.append(
                WallHeightExceededViolation(
                    object_id=str(obj.object_id),
                    object_top_height=obj_top,
                    wall_height=wall_height,
                )
            )

    return violations
