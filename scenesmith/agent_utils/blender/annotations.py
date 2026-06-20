"""Utilities for adding annotations to Blender scenes and rendered images."""

import logging

from pathlib import Path

import bpy
import numpy as np

from mathutils import Matrix, Vector
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from scenesmith.agent_utils.blender.camera_utils import get_pixel_coordinates
from scenesmith.agent_utils.blender.font_utils import load_annotation_font
from scenesmith.agent_utils.house import ClearanceOpeningData

console_logger = logging.getLogger(__name__)


def _quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float):
    """Convert quaternion to 3x3 rotation matrix."""
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    return np.array(
        [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ]
    )


def _wall_local_to_world(local_pos: np.ndarray, transform: list[float]) -> np.ndarray:
    """Convert wall-local position to world coordinates.

    Wall local coordinate system:
    - X: along wall (from start to end)
    - Y: into room (wall normal)
    - Z: up (vertical)

    Args:
        local_pos: Position in wall-local frame [x, y, z].
        transform: [x, y, z, qw, qx, qy, qz] pose of wall origin in world frame.

    Returns:
        Position in world coordinates.
    """
    wall_origin = np.array(transform[:3])
    qw, qx, qy, qz = transform[3:7]
    rotation_matrix = _quaternion_to_rotation_matrix(qw, qx, qy, qz)
    return wall_origin + rotation_matrix @ local_pos


def _compute_wall_center(
    transform: list[float], wall_length: float, wall_height: float
) -> np.ndarray:
    """Compute wall center in world coordinates using quaternion rotation.

    Uses the quaternion from the transform to rotate the local center
    position to world coordinates. This correctly handles any wall origin
    convention (corner-based, edge-based, centered, etc.).

    Wall local frame:
        x = along wall (0 at origin end, wall_length at far end)
        y = 0 (on wall surface)
        z = height above floor

    Args:
        transform: [x, y, z, qw, qx, qy, qz] pose of wall origin in world frame.
        wall_length: Wall length in meters.
        wall_height: Wall height in meters.

    Returns:
        Wall center position in world coordinates.
    """
    wall_origin = np.array(transform[:3])
    qw, qx, qy, qz = transform[3], transform[4], transform[5], transform[6]

    # Build rotation matrix from quaternion.
    # This gives us the local-to-world rotation.
    rot_matrix = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

    # Local center: half length along X, zero along Y, half height along Z.
    local_center = np.array([wall_length / 2, 0, wall_height / 2])

    # Transform to world: rotate local center then add origin.
    world_offset = rot_matrix @ local_center
    return wall_origin + world_offset


def annotate_image_with_coordinates(
    image_path: Path,
    marks: dict[tuple[float, float], tuple[int, int]],
    dot_radius: int = 3,
    base_font_size_divisor: float = 60,
    text_color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Add coordinate labels and markers to rendered image.

    Args:
        image_path: Path to the rendered image file.
        marks: Markers with labels (world coords -> pixel coords).
        dot_radius: The radius of the dots in pixels.
        base_font_size_divisor: The divisor of the width of the image to determine
            the base font size.
        text_color: The color of the text and markers.
    """
    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)
    font = load_annotation_font(pil_image.size[0], base_font_size_divisor)

    # Draw markers with labels.
    draw_coordinate_annotations(
        draw=draw,
        visual_marks=marks,
        font=font,
        dot_radius=dot_radius,
        text_color=text_color,
        image_size=pil_image.size,
    )

    pil_image.save(str(image_path))


def draw_coordinate_annotations(
    draw: ImageDraw.ImageDraw,
    visual_marks: dict[tuple[float, float], tuple[int, int]],
    font: ImageFont.ImageFont,
    dot_radius: int,
    text_color: tuple[int, int, int],
    image_size: tuple[int, int],
) -> None:
    """Draw coordinate dots and labels on the image."""
    width, height = image_size

    for (world_x, world_y), (pixel_x, pixel_y) in visual_marks.items():
        dot_x, dot_y = int(pixel_x), int(pixel_y)

        # Draw dot.
        draw.ellipse(
            [
                dot_x - dot_radius,
                dot_y - dot_radius,
                dot_x + dot_radius,
                dot_y + dot_radius,
            ],
            fill=text_color,
        )

        # Draw text label (limited to 2 decimal places, drop trailing zeros).
        # Format with 2 decimals then strip trailing zeros and decimal point.
        x_str = f"{world_x:.2f}".rstrip("0").rstrip(".")
        y_str = f"{world_y:.2f}".rstrip("0").rstrip(".")
        text = f"({x_str},{y_str})"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = max(0, min(dot_x - text_width // 2, width - text_width))
        text_y = max(0, min(dot_y + dot_radius + 2, height - text_height))

        draw.text((text_x, text_y), text, fill=text_color, font=font)


def add_set_of_mark_labels_pil(
    image_path: Path,
    scene_objects: list[dict],
    camera_obj: bpy.types.Camera,
    rendering_mode: str = "furniture",
    current_surface_id: str | None = None,
    annotate_object_types: list[str] | None = None,
) -> None:
    """Add set-of-mark labels as PIL overlays (always on top).

    Uses existing camera projection utilities (get_pixel_coordinates)
    to convert 3D bbox centers to 2D pixel positions.

    Args:
        image_path: Path to rendered image.
        scene_objects: List of scene object metadata dicts.
        camera_obj: Blender camera for 3D-to-2D projection.
        rendering_mode: Rendering mode ("furniture", "manipuland", "wall").
        current_surface_id: Optional surface ID to filter manipulands by.
            If provided, only show labels for manipulands on this surface.
        annotate_object_types: Optional list of object types to annotate.
            If provided, only objects of these types get labels.
            Example: ["wall_mounted"] to only label wall objects.
    """
    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)

    # Smaller font to match 3D text scale (0.15).
    # 3D text was quite compact, so use larger divisor.
    font = load_annotation_font(pil_image.size[0], base_font_size_divisor=60)

    scene = bpy.context.scene

    for obj_meta in scene_objects:
        obj_type = obj_meta.get("object_type", "")

        # Skip room geometry and floor objects. Keep wall ones as they are useful
        # for snapping and facing checks.
        if obj_type in ["room_geometry", "floor"]:
            continue

        # Filter by annotate_object_types if provided.
        if annotate_object_types is not None and obj_type not in annotate_object_types:
            continue

        # In manipuland mode, only show manipuland labels (exclude all other types).
        if rendering_mode == "manipuland" and obj_type != "manipuland":
            continue

        # Filter by surface if current_surface_id is provided.
        if current_surface_id is not None:
            parent_surface_id = obj_meta.get("parent_surface_id")
            if parent_surface_id != current_surface_id:
                continue

        # Use object_id directly for labels (already short with 2-char UUID).
        obj_name = obj_meta.get("object_id", "object")
        bbox = obj_meta.get("bounding_box")
        if not bbox:
            continue

        # Determine object type for positioning.
        obj_type = obj_meta.get("object_type", "")
        is_furniture = obj_type == "furniture"

        if is_furniture:
            # For furniture (support surfaces), place label at surface center.
            # Use XY position of bbox center but Z at support surface height (bbox bottom).
            extents = bbox["extents"]
            surface_z = bbox["center"][2] - extents[2] / 2  # Bottom of bbox.
            label_3d_pos = Vector((bbox["center"][0], bbox["center"][1], surface_z))
        else:
            # For other objects (manipulands, etc.), use bbox center.
            label_3d_pos = Vector(bbox["center"])

        pixel_x, pixel_y = get_pixel_coordinates(
            scene=scene, camera=camera_obj, world_coord=label_3d_pos
        )

        # Apply vertical offset for non-furniture objects to avoid occlusion.
        # Furniture labels stay at surface level for better association.
        if not is_furniture:
            label_offset_pixels = 30  # Pixels above object center.
            pixel_y -= label_offset_pixels  # Move up in screen space.

        # Measure text size.
        text_bbox = draw.textbbox((0, 0), obj_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw blue background rectangle.
        padding = 5
        bg_left = int(pixel_x - text_width // 2 - padding)
        bg_top = int(pixel_y - text_height // 2 - padding)
        bg_right = int(pixel_x + text_width // 2 + padding)
        bg_bottom = int(pixel_y + text_height // 2 + padding)

        draw.rectangle(
            [bg_left, bg_top, bg_right, bg_bottom],
            fill=(77, 153, 255),  # Blue color for all labels (furniture and objects).
        )

        # Draw white text centered on background.
        text_x = int(pixel_x - text_width // 2)
        text_y = int(pixel_y - text_height // 2)
        draw.text((text_x, text_y), obj_name, fill=(255, 255, 255), font=font)

    pil_image.save(str(image_path))


def add_opening_labels_pil(
    image_path: Path, openings: list[ClearanceOpeningData], camera_obj: bpy.types.Camera
) -> None:
    """Add door/window/opening labels as PIL overlays.

    Uses camera projection to convert 3D opening centers to 2D pixel positions.
    Labels use distinct colors by opening type for visual clarity.

    Args:
        image_path: Path to rendered image.
        openings: List of ClearanceOpeningData from RoomGeometry.openings.
        camera_obj: Blender camera for 3D-to-2D projection.
    """
    if not openings:
        return

    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)

    # Use slightly smaller font for opening labels.
    font = load_annotation_font(pil_image.size[0], base_font_size_divisor=70)

    scene = bpy.context.scene

    # Color scheme by opening type.
    type_colors = {
        "door": (34, 139, 34),  # Forest green for doors.
        "window": (255, 165, 0),  # Orange for windows.
        "open": (138, 43, 226),  # Blue-violet for open connections.
    }
    default_color = (100, 100, 100)  # Gray fallback.

    for opening in openings:
        label = opening.opening_id
        center_world = opening.center_world
        opening_type = opening.opening_type.lower()

        if not center_world:
            continue

        # Use actual opening center position for projection.
        label_3d_pos = Vector(center_world)
        pixel_x, pixel_y = get_pixel_coordinates(
            scene=scene, camera=camera_obj, world_coord=label_3d_pos
        )

        # Skip if projection failed (behind camera or out of frame).
        if pixel_x < 0 or pixel_y < 0:
            continue
        if pixel_x > pil_image.size[0] or pixel_y > pil_image.size[1]:
            continue

        # Measure text size.
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle with type-specific color.
        bg_color = type_colors.get(opening_type, default_color)
        padding = 4
        bg_left = int(pixel_x - text_width // 2 - padding)
        bg_top = int(pixel_y - text_height // 2 - padding)
        bg_right = int(pixel_x + text_width // 2 + padding)
        bg_bottom = int(pixel_y + text_height // 2 + padding)

        draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_color)

        # Draw white text centered on background.
        text_x = int(pixel_x - text_width // 2)
        text_y = int(pixel_y - text_height // 2)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    pil_image.save(str(image_path))


def create_annotation_collection() -> bpy.types.Collection:
    """Create or get the annotations collection for organizing annotation objects.

    Returns:
        Blender collection for annotations.
    """
    collection_name = "SceneAnnotations"

    # Check if collection already exists.
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]

    # Create new collection.
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    return collection


def add_bounding_box_annotation(
    bbox: dict,
    rotation_matrix: list[list[float]],
    collection: bpy.types.Collection,
    object_name: str,
) -> None:
    """Add a 3D wireframe bounding box annotation.

    Args:
        bbox: Bounding box dict with 'center' and 'extents'.
        rotation_matrix: 3x3 rotation matrix as list of lists.
        collection: Collection to add annotation to.
        object_name: Name for the annotation object.
    """
    # Create cube primitive.
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    bbox_obj = bpy.context.active_object
    bbox_obj.name = f"bbox_{object_name}"

    # Set position and scale.
    center = Vector(bbox["center"])
    extents = bbox["extents"]
    bbox_obj.location = center
    bbox_obj.scale = (extents[0], extents[1], extents[2])

    # Apply rotation from object.
    rot_matrix_3x3 = Matrix(rotation_matrix)
    bbox_obj.rotation_euler = rot_matrix_3x3.to_euler()

    # Add wireframe modifier (SceneWeaver pattern) with thick lines for visibility.
    mod = bbox_obj.modifiers.new("Wireframe", type="WIREFRAME")
    mod.thickness = 0.05
    mod.use_even_offset = True

    # Set display type to wireframe.
    bbox_obj.display_type = "WIRE"

    # Create blue Principled BSDF material (SceneWeaver color).
    mat = bpy.data.materials.new(name=f"mat_bbox_{object_name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (
            0.0,
            0.3,
            1.0,
            1.0,
        )  # Match PIL label color exactly.
        # No emission to match flat PIL color appearance.
        bsdf.inputs["Emission Strength"].default_value = 0.0
    bbox_obj.data.materials.append(mat)

    # Link to annotations collection.
    if bbox_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(bbox_obj)
    collection.objects.link(bbox_obj)

    console_logger.debug(
        f"Created bbox: {bbox_obj.name}, location={bbox_obj.location}, "
        f"scale={bbox_obj.scale}, hide_render={bbox_obj.hide_render}, "
        f"in_collection={bbox_obj.name in collection.objects}"
    )


def add_set_of_mark_label_annotation(
    position: tuple[float, float, float],
    text: str,
    collection: bpy.types.Collection,
    object_name: str,
) -> None:
    """Add a 3D text label annotation with background plane.

    Args:
        position: (x, y, z) position for the label.
        text: Text to display.
        collection: Collection to add annotation to.
        object_name: Name for the annotation object.
    """
    # Create 3D text object.
    bpy.ops.object.text_add(location=position)
    text_obj = bpy.context.active_object
    text_obj.name = f"label_{object_name}"

    # Configure text properties (default alignment, not centered).
    text_obj.data.body = text

    # Scale text.
    scale = 0.15
    text_obj.scale = (scale, scale, scale)

    # Update view layer to get accurate text dimensions before positioning.
    bpy.context.view_layer.update()

    # Position text above bbox center with offset to center left-aligned text.
    text_offset = Vector((-text_obj.dimensions[0] * text_obj.scale[0] / 2, 0.1, 0.5))
    text_obj.location = Vector(position) + text_offset

    # Create bright white material for text with emission.
    text_mat = bpy.data.materials.new(name=f"mat_text_{object_name}")
    text_mat.use_nodes = True
    bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 5.0
    text_obj.data.materials.append(text_mat)

    # Create background plane.
    bpy.ops.mesh.primitive_plane_add(size=1)
    bg_plane = bpy.context.active_object
    bg_plane.name = f"label_bg_{object_name}"

    # Scale background based on text dimensions with padding.
    padding = 0.05
    text_size = text_obj.dimensions
    bg_plane.scale.x = (text_size.x + padding) / text_obj.scale[0]
    bg_plane.scale.y = (text_size.y + padding) / text_obj.scale[1]

    # Position background to center on left-aligned text.
    bg_plane.location.x = (text_size.x / 2) / text_obj.scale[0]
    bg_plane.location.y = (text_size.y / 2) / text_obj.scale[1]
    bg_plane.location.z = -0.01  # Slightly behind text.

    # Create blue material for background.
    bg_mat = bpy.data.materials.new(name=f"mat_label_bg_{object_name}")
    bg_mat.use_nodes = True
    bsdf = bg_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (
            0.0,
            0.3,
            1.0,
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = 1.0
    bg_plane.data.materials.append(bg_mat)

    # Parent background to text so they move together.
    bg_plane.parent = text_obj

    # Link both to annotations collection.
    if text_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(text_obj)
    collection.objects.link(text_obj)

    if bg_plane.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(bg_plane)
    collection.objects.link(bg_plane)

    console_logger.debug(
        f"Created label: {text_obj.name}, location={text_obj.location}, "
        f"text='{text_obj.data.body}', hide_render={text_obj.hide_render}, "
        f"in_collection={text_obj.name in collection.objects}"
    )
    console_logger.debug(
        f"Created label bg: {bg_plane.name}, location={bg_plane.location}, "
        f"hide_render={bg_plane.hide_render}, "
        f"in_collection={bg_plane.name in collection.objects}"
    )


def _calculate_arrow_dimensions(
    rendering_mode: str,
    y_direction: Vector,
    bbox_extents: tuple[float, float, float] | None,
) -> tuple[float, float, float, float]:
    """Calculate arrow dimensions based on rendering mode and bbox.

    Args:
        rendering_mode: Rendering mode - "furniture" or "manipuland".
        y_direction: Direction vector for the arrow (normalized Y-axis).
        bbox_extents: Optional bbox extents for furniture mode scaling.

    Returns:
        Tuple of (shaft_length, shaft_radius, head_length, head_radius).
    """
    if rendering_mode == "manipuland":
        # Smaller, thinner arrows for manipuland mode.
        return (0.05, 0.005, 0.02, 0.015)

    if bbox_extents:
        # Furniture mode with bbox: size based on object.
        # Compute bbox dimension along arrow direction (Y-axis).
        # For AABB, projection = sum of absolute dot products with each axis.
        projected_length = (
            abs(y_direction.x) * bbox_extents[0]
            + abs(y_direction.y) * bbox_extents[1]
            + abs(y_direction.z) * bbox_extents[2]
        )
        shaft_length = projected_length * 0.75
        return (shaft_length, 0.02, 0.1, 0.1)

    # Fallback to fixed size.
    return (0.6, 0.02, 0.1, 0.1)


def _create_arrow_geometry(
    shaft_length: float,
    shaft_radius: float,
    head_length: float,
    head_radius: float,
    object_name: str,
) -> bpy.types.Object:
    """Create arrow geometry from cylinder shaft and cone head.

    Args:
        shaft_length: Length of the arrow shaft.
        shaft_radius: Radius of the arrow shaft.
        head_length: Length of the arrow head (cone).
        head_radius: Radius of the arrow head base.
        object_name: Name for the arrow object.

    Returns:
        Combined arrow object (shaft + head joined).
    """
    # Create arrow shaft (cylinder).
    bpy.ops.mesh.primitive_cylinder_add(
        radius=shaft_radius,
        depth=shaft_length,
        location=(0, 0, 0),
    )
    shaft = bpy.context.active_object
    shaft.name = f"arrow_shaft_{object_name}"

    # Create arrowhead (cone).
    bpy.ops.mesh.primitive_cone_add(
        radius1=head_radius,
        radius2=0,
        depth=head_length,
        location=(0, 0, shaft_length / 2 + head_length / 2),
    )
    head = bpy.context.active_object
    head.name = f"arrow_head_{object_name}"

    # Select both and join.
    shaft.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join()

    # Combined arrow object.
    arrow_obj = bpy.context.active_object
    arrow_obj.name = f"arrow_{object_name}"
    return arrow_obj


def _create_cyan_emission_material(object_name: str) -> bpy.types.Material:
    """Create bright cyan material with strong emission for arrows.

    Args:
        object_name: Name for the material (used in naming).

    Returns:
        Configured cyan emission material.
    """
    arrow_mat = bpy.data.materials.new(name=f"mat_arrow_{object_name}")
    arrow_mat.use_nodes = True
    bsdf = arrow_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Color"].default_value = (0.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 10.0
    return arrow_mat


def add_direction_arrow_annotation(
    position: tuple[float, float, float],
    rotation_matrix: list[list[float]],
    collection: bpy.types.Collection,
    object_name: str,
    bbox_extents: tuple[float, float, float] | None = None,
    rendering_mode: str = "furniture",
) -> None:
    """Add a 3D direction arrow annotation showing Y-axis (forward) orientation.

    Args:
        position: (x, y, z) position for arrow base (bbox center).
        rotation_matrix: 3x3 rotation matrix as list of lists.
        collection: Collection to add annotation to.
        object_name: Name for the annotation object.
        bbox_extents: Optional bbox extents for positioning at top of bbox.
        rendering_mode: Rendering mode - "furniture" or "manipuland".
    """
    # Extract Y-axis direction from rotation matrix (Drake uses Y for "forward").
    rot_matrix = Matrix(rotation_matrix)
    y_direction = rot_matrix @ Vector((0, 1, 0))

    # Calculate arrow dimensions based on rendering mode.
    shaft_length, shaft_radius, head_length, head_radius = _calculate_arrow_dimensions(
        rendering_mode=rendering_mode,
        y_direction=y_direction,
        bbox_extents=bbox_extents,
    )

    # Create arrow geometry.
    arrow_obj = _create_arrow_geometry(
        shaft_length=shaft_length,
        shaft_radius=shaft_radius,
        head_length=head_length,
        head_radius=head_radius,
        object_name=object_name,
    )

    # Position at top of bbox.
    # Add half bbox height to Z coordinate to place arrow at top.
    arrow_location = Vector(position)
    if bbox_extents:
        arrow_location.z += bbox_extents[2] / 2

    # Offset arrow so it starts at bbox center (not centered there).
    # Move arrow forward by half its shaft length.
    arrow_obj.location = arrow_location + y_direction * (shaft_length / 2)

    # Rotate to align with Y-axis direction.
    # Default arrow points up (+Z), rotate to point in y_direction.
    default_up = Vector((0, 0, 1))
    rotation_quat = default_up.rotation_difference(y_direction)
    arrow_obj.rotation_euler = rotation_quat.to_euler()

    # Apply cyan emission material.
    arrow_mat = _create_cyan_emission_material(object_name=object_name)
    arrow_obj.data.materials.append(arrow_mat)

    # Link to annotations collection.
    if arrow_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(arrow_obj)
    collection.objects.link(arrow_obj)

    console_logger.debug(
        f"Created arrow: {arrow_obj.name}, location={arrow_obj.location}, "
        f"rotation={arrow_obj.rotation_euler}, hide_render={arrow_obj.hide_render}, "
        f"in_collection={arrow_obj.name in collection.objects}"
    )


def add_blender_scene_annotations(
    scene_objects: list[dict], annotations: DictConfig
) -> None:
    """Add Blender 3D annotation objects to the scene before rendering.

    Args:
        scene_objects: List of scene object metadata dictionaries.
        annotations: Annotation config flags.
    """
    # Create or get annotations collection.
    collection = create_annotation_collection()

    console_logger.info(f"Adding annotations for {len(scene_objects)} objects")

    # Get rendering mode from annotations.
    rendering_mode = "furniture"
    if hasattr(annotations, "rendering_mode"):
        rendering_mode = annotations.rendering_mode

    # Get annotate_object_types filter if specified.
    annotate_object_types = getattr(annotations, "annotate_object_types", None)

    # Process each scene object.
    for obj_meta in scene_objects:
        obj_type = obj_meta.get("object_type", "")

        # Skip room geometry, wall, and floor objects.
        if obj_type in ["room_geometry", "wall", "floor"]:
            continue

        # Filter by annotate_object_types if specified.
        if annotate_object_types is not None and obj_type not in annotate_object_types:
            continue

        # In manipuland mode, only annotate manipulands (skip furniture).
        if rendering_mode == "manipuland" and obj_type != "manipuland":
            continue

        obj_name = obj_meta.get("name", "object")
        position = obj_meta.get("position", [0, 0, 0])
        rotation_matrix = obj_meta.get(
            "rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        # Add bounding box if enabled.
        if annotations.enable_bounding_boxes:
            bbox = obj_meta.get("bounding_box")
            if bbox:
                console_logger.info(f"Adding bbox for {obj_name}: {bbox}")
                add_bounding_box_annotation(
                    bbox=bbox,
                    rotation_matrix=rotation_matrix,
                    collection=collection,
                    object_name=obj_name,
                )
            else:
                console_logger.warning(f"No bbox data for {obj_name}")

        # Add direction arrow if enabled.
        if annotations.enable_direction_arrows:
            console_logger.info(f"Adding arrow for {obj_name}")
            # Use bbox center and extents if available for proper positioning.
            arrow_position = position
            bbox_extents = None
            bbox = obj_meta.get("bounding_box")
            if bbox:
                arrow_position = bbox["center"]
                bbox_extents = tuple(bbox["extents"])
            add_direction_arrow_annotation(
                position=tuple(arrow_position),
                rotation_matrix=rotation_matrix,
                collection=collection,
                object_name=obj_name,
                bbox_extents=bbox_extents,
                rendering_mode=rendering_mode,
            )

    console_logger.debug(
        f"Annotation collection has {len(collection.objects)} objects total"
    )
    for obj in collection.objects:
        console_logger.debug(
            f"  - {obj.name}: location={obj.location}, "
            f"hide_render={obj.hide_render}, visible={not obj.hide_viewport}"
        )


def remove_annotation_objects() -> None:
    """Remove all annotation objects from the scene after rendering."""
    collection_name = "SceneAnnotations"

    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]

        # Remove all objects in the collection.
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

        # Remove the collection itself.
        bpy.data.collections.remove(collection)


def draw_wall_coordinate_grid(
    wall_surface_data: dict,
    grid_divisions: int = 5,
    line_color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.8),
    label_color: tuple[float, float, float, float] = (0.0, 0.0, 0.8, 1.0),
) -> list[bpy.types.Object]:
    """Draw coordinate grid on wall surface with position labels.

    Creates grid lines and corner/center labels for wall orthographic views.
    Grid lines are gray, labels are blue showing wall-local coordinates.

    Args:
        wall_surface_data: Wall surface info including length, height.
        grid_divisions: Number of grid divisions (default 5).
        line_color: RGBA color for grid lines.
        label_color: RGBA color for coordinate labels.

    Returns:
        List of created Blender objects (for cleanup).
    """
    wall_length = wall_surface_data.get("length", 4.0)
    wall_height = wall_surface_data.get("height", 2.5)

    created_objects = []

    # Create material for grid lines.
    line_mat = bpy.data.materials.new(name="WallGridLineMaterial")
    line_mat.use_nodes = True
    bsdf = line_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = line_color
        bsdf.inputs["Alpha"].default_value = line_color[3]
    line_mat.blend_method = "BLEND"

    # Create grid lines.
    for i in range(grid_divisions + 1):
        # Vertical lines (constant x).
        x = wall_length * i / grid_divisions
        v_line = _create_line_mesh(
            start=(x, 0.01, 0),
            end=(x, 0.01, wall_height),
            name=f"wall_grid_v_{i}",
        )
        if v_line:
            v_line.data.materials.append(line_mat)
            created_objects.append(v_line)

        # Horizontal lines (constant z).
        z = wall_height * i / grid_divisions
        h_line = _create_line_mesh(
            start=(0, 0.01, z),
            end=(wall_length, 0.01, z),
            name=f"wall_grid_h_{i}",
        )
        if h_line:
            h_line.data.materials.append(line_mat)
            created_objects.append(h_line)

    # Create label material.
    label_mat = bpy.data.materials.new(name="WallGridLabelMaterial")
    label_mat.use_nodes = True
    bsdf = label_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = label_color
        bsdf.inputs["Emission Color"].default_value = label_color
        bsdf.inputs["Emission Strength"].default_value = 2.0

    # Add coordinate labels at key positions (corners and center).
    label_positions = [
        (0, 0),  # Bottom-left.
        (wall_length, 0),  # Bottom-right.
        (0, wall_height),  # Top-left.
        (wall_length, wall_height),  # Top-right.
        (wall_length / 2, wall_height / 2),  # Center.
    ]

    for x, z in label_positions:
        label_text = f"({x:.1f}, {z:.1f})"
        label_obj = _create_text_label(
            position=(x, 0.02, z),
            text=label_text,
            size=0.1,
            name=f"wall_label_{x:.0f}_{z:.0f}",
        )
        if label_obj:
            label_obj.data.materials.append(label_mat)
            created_objects.append(label_obj)

    return created_objects


def draw_excluded_regions(
    wall_surface_data: dict,
    material_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.5),
) -> list[bpy.types.Object]:
    """Draw hatched rectangles for door/window regions.

    Creates semi-transparent gray planes with diagonal hatching pattern
    to indicate areas where wall objects cannot be placed.

    Args:
        wall_surface_data: Wall surface info including excluded_regions.
        material_color: RGBA color for excluded region overlay.

    Returns:
        List of created Blender objects (for cleanup).
    """
    excluded_regions = wall_surface_data.get("excluded_regions", [])
    if not excluded_regions:
        return []

    created_objects = []

    # Create material for excluded regions.
    excluded_mat = bpy.data.materials.new(name="ExcludedRegionMaterial")
    excluded_mat.use_nodes = True
    excluded_mat.blend_method = "BLEND"

    bsdf = excluded_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = material_color
        bsdf.inputs["Alpha"].default_value = material_color[3]

    for i, region in enumerate(excluded_regions):
        x_min, z_min, x_max, z_max = region

        # Create plane mesh for the region.
        width = x_max - x_min
        height = z_max - z_min
        center_x = x_min + width / 2
        center_z = z_min + height / 2

        bpy.ops.mesh.primitive_plane_add(
            size=1.0,
            location=(center_x, 0.005, center_z),  # Slight Y offset.
        )
        plane = bpy.context.active_object
        plane.name = f"excluded_region_{i}"

        # Scale to match region size.
        plane.scale = (width, height, 1)

        # Rotate to face into room (plane default faces up, need to face +Y).
        plane.rotation_euler = (1.5708, 0, 0)  # 90 degrees around X.

        # Apply material.
        plane.data.materials.append(excluded_mat)

        created_objects.append(plane)

    return created_objects


def _create_line_mesh(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    name: str,
    thickness: float = 0.005,
) -> bpy.types.Object | None:
    """Create a thin cylinder to represent a line.

    Args:
        start: Start point (x, y, z).
        end: End point (x, y, z).
        name: Object name.
        thickness: Line thickness (cylinder radius).

    Returns:
        Created cylinder object, or None if creation fails.
    """

    start_vec = Vector(start)
    end_vec = Vector(end)
    direction = end_vec - start_vec
    length = direction.length

    if length < 0.001:
        return None

    # Create cylinder.
    bpy.ops.mesh.primitive_cylinder_add(
        radius=thickness,
        depth=length,
        location=((start_vec + end_vec) / 2).to_tuple(),
    )
    line_obj = bpy.context.active_object
    line_obj.name = name

    # Rotate to align with direction.
    # Default cylinder is aligned with Z axis, need to rotate to align with direction.
    up = Vector((0, 0, 1))
    direction_normalized = direction.normalized()

    if abs(up.dot(direction_normalized)) < 0.9999:
        rotation_quat = up.rotation_difference(direction_normalized)
        line_obj.rotation_euler = rotation_quat.to_euler()

    return line_obj


def _create_text_label(
    position: tuple[float, float, float],
    text: str,
    size: float = 0.1,
    name: str = "label",
) -> bpy.types.Object | None:
    """Create a 3D text label.

    Args:
        position: Label position (x, y, z).
        text: Text to display.
        size: Text size.
        name: Object name.

    Returns:
        Created text object, or None if creation fails.
    """
    try:
        bpy.ops.object.text_add(location=position)
        text_obj = bpy.context.active_object
        text_obj.name = name
        text_obj.data.body = text
        text_obj.scale = (size, size, size)

        # Rotate to face camera (+Y direction) in wall orthographic view.
        text_obj.rotation_euler = (1.5708, 0, 0)  # 90 degrees around X.

        return text_obj
    except Exception as e:
        console_logger.warning(f"Failed to create text label: {e}")
        return None


def _add_wall_coordinate_frame(
    draw: ImageDraw.ImageDraw, image_size: tuple[int, int], font: ImageFont.FreeTypeFont
) -> None:
    """Add coordinate frame indicator for wall orthographic views.

    Draws X and Z axis arrows in bottom-left corner with labels indicating:
    - X: distance along wall (horizontal)
    - Z: height above floor (vertical)

    Args:
        draw: PIL ImageDraw object.
        image_size: (width, height) of image.
        font: Font for axis labels.
    """
    # Position in bottom-left corner with margin.
    margin = 40
    origin_x = margin + 10
    origin_y = image_size[1] - margin - 10
    arrow_length = 50

    # Colors matching the coordinate grid (red).
    axis_color = (200, 0, 0)
    label_color = (200, 0, 0)
    line_width = 2

    # Draw X axis (horizontal, pointing right).
    x_end = (origin_x + arrow_length, origin_y)
    draw.line([(origin_x, origin_y), x_end], fill=axis_color, width=line_width)
    # X arrowhead.
    draw.polygon(
        [
            (x_end[0], x_end[1]),
            (x_end[0] - 8, x_end[1] - 5),
            (x_end[0] - 8, x_end[1] + 5),
        ],
        fill=axis_color,
    )
    # X label.
    draw.text((x_end[0] + 5, x_end[1] - 8), "X", fill=label_color, font=font)

    # Draw Z axis (vertical, pointing up).
    z_end = (origin_x, origin_y - arrow_length)
    draw.line([(origin_x, origin_y), z_end], fill=axis_color, width=line_width)
    # Z arrowhead.
    draw.polygon(
        [
            (z_end[0], z_end[1]),
            (z_end[0] - 5, z_end[1] + 8),
            (z_end[0] + 5, z_end[1] + 8),
        ],
        fill=axis_color,
    )
    # Z label.
    draw.text((z_end[0] + 5, z_end[1] - 5), "Z", fill=label_color, font=font)

    # Add small labels for axis meanings.
    small_font = font  # Use same font for now.
    draw.text(
        (origin_x + arrow_length + 20, origin_y - 3),
        "(along wall)",
        fill=(100, 100, 100),
        font=small_font,
    )
    draw.text(
        (origin_x + 12, origin_y - arrow_length - 20),
        "(height)",
        fill=(100, 100, 100),
        font=small_font,
    )


def add_wall_grid_annotations_pil(
    image_path: Path,
    wall_surface_data: dict,
    camera_obj: bpy.types.Camera,
    num_markers: int = 5,
) -> None:
    """Add wall coordinate markers as PIL overlays (post-render).

    Uses wall direction to compute grid positions directly in world coordinates,
    then projects to pixels. Draws red coordinate points and labels matching
    the style of floor coordinate annotations.

    Wall coordinate system:
        x = distance along wall (0 at left when viewed from inside room)
        z = height above floor

    Args:
        image_path: Path to rendered image.
        wall_surface_data: Wall surface info (direction, length, height).
        camera_obj: Blender camera for 3D-to-2D projection.
        num_markers: Number of markers per axis (e.g., 5 gives 5x5=25 markers).
    """
    wall_length = wall_surface_data.get("length", 4.0)
    wall_height = wall_surface_data.get("height", 2.5)
    transform_data = wall_surface_data.get("transform")

    if not transform_data:
        console_logger.warning("Wall surface missing transform data")
        return

    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)
    scene = bpy.context.scene

    # Load font - smaller than furniture floor coordinates for wall ortho views.
    # Divisor 80 with min_font_size=10 gives ~10pt on 512px wall ortho images.
    font = load_annotation_font(
        pil_image.size[0], base_font_size_divisor=80, min_font_size=10
    )

    # Red color matching floor coordinate style.
    marker_color = (255, 0, 0)
    dot_radius = 2

    # Generate grid points using quaternion rotation for correct transformation.
    # Wall local coords: x = along wall (0 at start), z = height above floor.
    visual_marks = {}
    for i in range(num_markers):
        for j in range(num_markers):
            # Wall-local coordinates at exact positions.
            wall_x = wall_length * i / (num_markers - 1) if num_markers > 1 else 0
            wall_z = wall_height * j / (num_markers - 1) if num_markers > 1 else 0
            # Round for clean display labels.
            display_x = round(wall_x, 1)
            display_z = round(wall_z, 1)

            # Wall-local position (x along wall, y=0 on surface, z height).
            local_pos = np.array([wall_x, 0, wall_z])

            # Transform to world coordinates using quaternion rotation.
            world_pos = _wall_local_to_world(
                local_pos=local_pos, transform=transform_data
            )

            px = get_pixel_coordinates(
                scene=scene, camera=camera_obj, world_coord=Vector(world_pos.tolist())
            )

            if _is_valid_pixel(px=px, image_size=pil_image.size):
                # Store as (display_x, display_z) -> pixel position.
                # Use clean display values (0, 1, 2, ...) not margin-adjusted values.
                visual_marks[(display_x, display_z)] = px

    # Draw coordinate annotations using shared helper (same style as floor).
    draw_coordinate_annotations(
        draw=draw,
        visual_marks=visual_marks,
        font=font,
        dot_radius=dot_radius,
        text_color=marker_color,
        image_size=pil_image.size,
    )

    pil_image.save(str(image_path))


def _is_valid_pixel(px: tuple[float, float], image_size: tuple[int, int]) -> bool:
    """Check if pixel coordinates are within image bounds.

    Args:
        px: Pixel coordinates (x, y).
        image_size: Image size (width, height).

    Returns:
        True if pixel is within bounds, False otherwise.
    """
    if px[0] < 0 or px[1] < 0:
        return False
    if px[0] > image_size[0] or px[1] > image_size[1]:
        return False
    return True


def add_wall_surface_id_label(image_path: Path, wall_surface_id: str) -> None:
    """Add wall surface ID label to rendered wall orthographic image.

    Draws a label in the top-right corner showing the wall_surface_id that
    can be used with place/move tools.

    Args:
        image_path: Path to the rendered image file.
        wall_surface_id: Wall surface identifier to display.
    """
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Load font.
        # Divisor 25 gives ~20pt on 512px wall views for readable corner labels.
        font = load_annotation_font(img.width, base_font_size_divisor=25)

        # Build label text.
        label_text = f"Wall: {wall_surface_id}"

        # Get text bounding box.
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position in top-right corner with padding.
        padding = 8
        x = img.width - text_width - padding * 2
        y = padding

        # Draw background (blue for wall surfaces).
        bg_color = (70, 130, 180)  # Steel blue.
        draw.rectangle(
            [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ],
            fill=bg_color,
        )

        # Draw text in white.
        draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

        img.save(str(image_path))

    except Exception as e:
        console_logger.warning(f"Failed to add wall surface ID label: {e}")


def add_wall_labels_to_top_view(
    image_path: Path, camera_obj: bpy.types.Object, wall_surfaces: list[dict]
) -> None:
    """Add wall surface labels to a top-down view.

    Projects wall center positions to 2D and draws labels showing each
    wall's surface_id. Labels are positioned at wall midpoints.

    Args:
        image_path: Path to the rendered top-down image.
        camera_obj: Blender camera object for projection.
        wall_surfaces: List of wall surface dicts with surface_id, direction,
            length, height, and transform.
    """
    if not wall_surfaces:
        return

    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        scene = bpy.context.scene

        # Load font.
        font = load_annotation_font(image_width=img.width, base_font_size_divisor=70)

        for wall_data in wall_surfaces:
            surface_id = wall_data.get(
                "surface_id", wall_data.get("wall_id", "unknown")
            )
            wall_direction = wall_data.get("direction", "north")
            wall_length = wall_data.get("length", 4.0)
            wall_height = wall_data.get("height", 2.5)
            transform_data = wall_data.get("transform")

            if not transform_data:
                continue

            # Compute wall center from transform and dimensions.
            wall_center = _compute_wall_center(
                transform=transform_data,
                wall_length=wall_length,
                wall_height=wall_height,
            )

            # Project to pixel coordinates.
            px = get_pixel_coordinates(
                scene=scene,
                camera=camera_obj,
                world_coord=Vector(wall_center.tolist()),
            )

            # Debug: log wall center and pixel position.
            console_logger.info(
                f"Wall label {surface_id}: direction={wall_direction}, "
                f"transform={transform_data[:3]}, center={wall_center}, "
                f"px={px}, img_size={img.size}"
            )

            if not _is_valid_pixel(px=px, image_size=img.size):
                continue

            # Draw label.
            label_text = surface_id
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text on projected position.
            x = int(px[0] - text_width / 2)
            y = int(px[1] - text_height / 2)

            # Draw background.
            padding = 4
            bg_color = (70, 130, 180)  # Steel blue.
            draw.rectangle(
                [
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding,
                ],
                fill=bg_color,
            )

            # Draw text in white.
            draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

        img.save(str(image_path))

    except Exception as e:
        console_logger.warning(f"Failed to add wall labels to top view: {e}")
