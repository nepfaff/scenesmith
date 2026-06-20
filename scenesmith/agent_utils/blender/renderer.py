import logging
import math
import time

from pathlib import Path

import bpy
import numpy as np

from mathutils import Matrix, Vector
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from scenesmith.agent_utils.blender.annotations import (
    add_blender_scene_annotations,
    add_opening_labels_pil,
    add_set_of_mark_labels_pil,
    add_wall_grid_annotations_pil,
    add_wall_labels_to_top_view,
    add_wall_surface_id_label,
    annotate_image_with_coordinates,
    remove_annotation_objects,
)
from scenesmith.agent_utils.blender.camera_utils import (
    calculate_camera_distance,
    configure_camera_from_params,
    configure_metric_camera,
    look_at_target,
)
from scenesmith.agent_utils.blender.coordinate_frame import (
    add_coordinate_frame_top_view,
    add_coordinate_frame_wall_view,
    create_coordinate_frame,
    remove_coordinate_frame,
    remove_wall_coordinate_frame,
)
from scenesmith.agent_utils.blender.coordinate_grid_mixin import CoordinateGridMixin
from scenesmith.agent_utils.blender.image_overlays import (
    add_number_overlay,
    add_support_surface_debug_volume,
)
from scenesmith.agent_utils.blender.params import RenderParams
from scenesmith.agent_utils.blender.render_dataclasses import (
    ArticulatedRenderResult,
    LinkMeshInfo,
    OverlayRenderingSetup,
)
from scenesmith.agent_utils.blender.render_settings import (
    apply_image_type_settings,
    apply_render_settings,
    setup_metric_world,
)
from scenesmith.agent_utils.blender.scene_setup_mixin import SceneSetupMixin
from scenesmith.agent_utils.blender.scene_utils import (
    compute_scene_bounds,
    disable_backface_culling,
    get_floor_bounds,
)
from scenesmith.agent_utils.blender.surface_rendering_mixin import SurfaceRenderingMixin
from scenesmith.agent_utils.blender.surface_utils import (
    add_surface_id_label,
    add_surface_labels_to_side_view,
    generate_multi_surface_views,
    generate_surface_colors,
)
from scenesmith.agent_utils.blender.view_generation_mixin import ViewGenerationMixin
from scenesmith.agent_utils.blender.wall_utils import (
    looks_like_wall,
    restore_hidden_walls,
    should_hide_wall,
)
from scenesmith.agent_utils.house import ClearanceOpeningData
from scenesmith.utils.print_utils import suppress_stdout_stderr

console_logger = logging.getLogger(__name__)

# Rendering constants.
DEFAULT_LIGHT_ENERGY = 1000
DEFAULT_LIGHT_POSITION = (4.0, 1.0, 6.0)
DEFAULT_NUM_SIDE_VIEWS = 4
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
# EEVEE TAA samples for asset validation renders. Using 8 as a good balance
# between quality and speed. EEVEE is ~6x faster than CYCLES.
EEVEE_ASSET_VALIDATION_SAMPLES = 8
# CYCLES samples for offline CLIP embedding renders (higher quality, slower).
CYCLES_CLIP_SAMPLES = 20
VLM_ANALYSIS_LIGHT_ENERGY = 2000

# Camera constants.
DEFAULT_CAMERA_LENS_MM = 50
DEFAULT_CAMERA_SENSOR_WIDTH_MM = 36
DEFAULT_CAMERA_CLIP_START = 0.01
DEFAULT_CAMERA_CLIP_END = 100000
CAMERA_DISTANCE_MARGIN_MULTIPLIER = (
    1 / 0.8
)  # Scene occupies ~80% of image (10% margin per side).
LIGHT_DISTANCE_RATIO = 0.1
# Offset above lower surfaces for camera near-plane clipping (meters).
# This clips furniture geometry above lower surfaces so they're visible from top-down views.
LOWER_SURFACE_CLIP_OFFSET_M = 0.05

# Multi-view rendering constants.
COORDINATE_FRAME_SCALE_FACTOR = 0.01


def _set_eevee_render_engine(scene) -> str:
    """Select the Eevee render engine name supported by the active Blender build."""
    try:
        available_engines = {
            item.identifier
            for item in scene.render.bl_rna.properties["engine"].enum_items
        }
    except (AttributeError, KeyError, TypeError):
        available_engines = set()

    if "BLENDER_EEVEE_NEXT" in available_engines:
        engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in available_engines:
        engine = "BLENDER_EEVEE"
    else:
        engine = "CYCLES"

    scene.render.engine = engine
    return engine


def _apply_eevee_speed_settings(scene, keep_shadows: bool = True) -> None:
    """Apply EEVEE speed optimizations for faster rendering.

    Disables expensive effects that aren't needed for asset validation or scene
    observation. Keeps shadows by default since they help VLMs understand 3D form.

    Args:
        scene: Blender scene object.
        keep_shadows: If True, keep shadows enabled (helps VLM see 3D form).
            If False, disable shadows entirely for maximum speed.
    """
    # Disable bloom (lens effect, not needed for validation).
    try:
        scene.eevee.use_bloom = False
    except AttributeError:
        pass  # Not available in EEVEE_NEXT.

    # Disable screen space reflections (expensive, not needed).
    try:
        scene.eevee.use_ssr = False
    except AttributeError:
        pass

    # Disable ambient occlusion (adds render time, not critical for validation).
    try:
        scene.eevee.use_gtao = False
    except AttributeError:
        pass

    # Disable volumetric shadows (expensive atmospheric effect).
    try:
        scene.eevee.use_volumetric_shadows = False
    except AttributeError:
        pass

    if keep_shadows:
        # Reduce shadow quality for speed while keeping depth cues.
        try:
            scene.eevee.shadow_cube_size = "256"  # Reduced from default 1024.
        except AttributeError:
            pass
        try:
            scene.eevee.shadow_cascade_size = "256"
        except AttributeError:
            pass
    else:
        # Disable shadows entirely for maximum speed.
        try:
            scene.eevee.use_shadows = False
        except AttributeError:
            pass
        for light in bpy.data.lights:
            try:
                light.use_shadow = False
            except AttributeError:
                pass


def _composite_onto_grey(image_path: Path, grey_value: int = 128) -> None:
    """Composite RGBA image onto neutral grey background.

    Args:
        image_path: Path to RGBA PNG image (modified in place).
        grey_value: Grey background value 0-255 (default 128 = 50% grey).
    """
    img = Image.open(image_path).convert("RGBA")
    background = Image.new("RGB", img.size, (grey_value, grey_value, grey_value))
    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask.
    background.save(image_path)


def _compute_bounds_from_corners(corners: list[list[float]]) -> tuple:
    """Compute axis-aligned and oriented bounding box from corner points.

    Args:
        corners: List of 8 corner points [x, y, z] in Drake Z-up world coordinates.

    Returns:
        Tuple of (corners_array, bbox_min, bbox_max) where:
        - corners_array: numpy array of shape (8, 3) with corner positions
        - bbox_min: numpy array of AABB minimum bounds
        - bbox_max: numpy array of AABB maximum bounds
    """
    # Convert to numpy array for easier manipulation.
    corners_array = np.array(corners)

    # Compute axis-aligned bounding box for positioning coordinate frame.
    bbox_min = corners_array.min(axis=0)
    bbox_max = corners_array.max(axis=0)

    return corners_array, bbox_min, bbox_max


def _compute_wall_center_from_transform(
    transform: list[float], wall_length: float, wall_height: float
) -> np.ndarray:
    """Compute wall center in world coordinates using quaternion rotation.

    Uses the quaternion from the transform to rotate the local center
    position to world coordinates. This correctly handles any wall origin
    convention (corner-based, edge-based, centered, etc.).

    Wall local frame:
    - Local X is along wall (from origin to far end)
    - Local Y is outward from room (wall normal)
    - Local Z is up

    Args:
        transform: [x, y, z, qw, qx, qy, qz] pose of wall origin in world frame.
        wall_length: Wall length in meters.
        wall_height: Wall height in meters.

    Returns:
        Wall center position in world coordinates as numpy array.
    """
    wall_origin = np.array(transform[:3])
    qw, qx, qy, qz = transform[3], transform[4], transform[5], transform[6]

    # Build rotation matrix from quaternion.
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


class BlenderRenderer(
    SurfaceRenderingMixin,
    CoordinateGridMixin,
    ViewGenerationMixin,
    SceneSetupMixin,
):
    """Encapsulates access to Blender rendering functionality.

    Note that even though this is a class, bpy is a singleton so likewise you
    should only ever create one instance of this class.
    """

    def __init__(
        self,
        blend_file: Path | None = None,
        bpy_settings_file: Path | None = None,
    ) -> None:
        """Initialize the Blender renderer.

        Args:
            blend_file: Optional path to a .blend file to use as base scene.
            bpy_settings_file: Optional path to a .py file with Blender settings.
        """
        self._blend_file = blend_file
        self._bpy_settings_file = bpy_settings_file
        self._client_objects = None

    def add_default_light_source(self) -> None:
        """Add a default point light source to the scene."""
        # Create a new light data block.
        light_data = bpy.data.lights.new(name="DefaultLight", type="POINT")
        light_data.energy = DEFAULT_LIGHT_ENERGY

        # Create new object with the light datablock.
        light_object = bpy.data.objects.new(name="DefaultLight", object_data=light_data)

        # Link light object to scene collection.
        bpy.context.collection.objects.link(light_object)

        # Set light position.
        light_object.location = DEFAULT_LIGHT_POSITION

    def render_image(self, params: RenderParams, output_path: Path) -> None:
        """Render the current scene with the given parameters.

        Args:
            params: The rendering parameters.
            output_path: Path where the rendered image will be saved.
        """
        # Set up scene and import glTF.
        self._setup_scene(params)
        self._import_and_organize_gltf(params.scene)

        # Configure camera.
        configure_camera_from_params(params=params)

        # Apply render settings.
        apply_render_settings(params=params)
        apply_image_type_settings(params=params, client_objects=self._client_objects)

        # Set output path and render.
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

    def render_multiview_for_analysis(
        self,
        mesh_path: Path,
        output_dir: Path,
        elevation_degrees: float,
        num_side_views: int = DEFAULT_NUM_SIDE_VIEWS,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        include_vertical_views: bool = True,
        light_energy: float | None = None,
        start_azimuth_degrees: float = 0.0,
        show_coordinate_frame: bool = True,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> list[Path]:
        """Render a mesh from multiple views for VLM physics analysis.

        This creates renders with optional coordinate frame visualization:
        - Image 0: Top view (+Z) [if include_vertical_views=True]
        - Image 1: Bottom view (-Z) [if include_vertical_views=True]
        - Images 2-(1+num_side_views): Equidistant side views with elevation angle
        - Each image shows RGB coordinate axes (+X=red, +Y=green, +Z=blue)
          [if show_coordinate_frame=True]
        - Each image has a numbered label overlay

        Args:
            mesh_path: Path to the mesh file (GLB/GLTF).
            output_dir: Directory where rendered images will be saved.
            elevation_degrees: Elevation angle in degrees for side view cameras.
                Cameras look down at objects from this angle above horizontal.
                Use 0 for ground-level horizontal views, ~20 for slightly elevated.
            num_side_views: Number of equidistant side views to render.
            width: Width of rendered images in pixels (default: 512).
            height: Height of rendered images in pixels (default: 512).
            include_vertical_views: If True, render top/bottom views. If False,
                only render side views (useful for constraining rotation to Z-axis).
            light_energy: Light energy in watts. If None, uses VLM_ANALYSIS_LIGHT_ENERGY.
            start_azimuth_degrees: Starting azimuth angle for side views (default: 0).
                Use 0 for first view at +X, 90 for first view at +Y. Useful for
                wall-mounted objects where front face is at +Y.
            show_coordinate_frame: If True, show RGB coordinate axes overlay.
                Set to False for cleaner validation renders.

        Returns:
            List of paths to rendered PNG images.
        """
        start_time = time.time()

        num_vertical_views = 2 if include_vertical_views else 0
        total_views = num_side_views + num_vertical_views
        console_logger.info(
            f"Rendering {total_views} views for VLM analysis ({width}x{height}px)"
        )

        # Suppress Blender's verbose rendering output.
        with suppress_stdout_stderr():
            # Clear existing scene.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Set up rendering with EEVEE for speed (6x faster than CYCLES).
            scene = bpy.context.scene
            _set_eevee_render_engine(scene)
            scene.eevee.taa_render_samples = taa_samples
            _apply_eevee_speed_settings(scene)
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            # Use transparent background for VLM analysis.
            # This avoids bias toward light/dark objects.
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Add a camera-following point light to ensure visible surfaces
            # are illuminated from all viewing angles.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = (
                light_energy if light_energy is not None else VLM_ANALYSIS_LIGHT_ENERGY
            )
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import mesh.
            bpy.ops.import_scene.gltf(filepath=str(mesh_path))
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            disable_backface_culling(list(bpy.context.selected_objects))

            # Compute bounding box.
            mesh_objs = [
                obj for obj in bpy.context.selected_objects if obj.type == "MESH"
            ]
            bbox_min = Vector((float("inf"),) * 3)
            bbox_max = Vector((float("-inf"),) * 3)
            for obj in mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    bbox_min = Vector(map(min, bbox_min, world_corner))
                    bbox_max = Vector(map(max, bbox_max, world_corner))

            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            max_dim = max(bbox_size)

            # Create coordinate frame with labels (optional).
            if show_coordinate_frame:
                create_coordinate_frame(
                    position=bbox_center,
                    max_dim=max_dim,
                    scale_factor=COORDINATE_FRAME_SCALE_FACTOR,
                    add_labels=True,
                )

            # Setup camera.
            camera = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj
            camera.type = "PERSP"
            camera.lens = DEFAULT_CAMERA_LENS_MM
            camera.sensor_width = DEFAULT_CAMERA_SENSOR_WIDTH_MM
            camera.clip_start = DEFAULT_CAMERA_CLIP_START
            camera.clip_end = DEFAULT_CAMERA_CLIP_END

            # Compute camera distance.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Define views.
            views = []
            if include_vertical_views:
                views.append({"name": "0_top", "direction": Vector((0, 0, 1))})
                views.append({"name": "1_bottom", "direction": Vector((0, 0, -1))})
                side_index_offset = 2
            else:
                side_index_offset = 0

            # Convert elevation to radians for spherical coordinate calculation.
            elevation_rad = math.radians(elevation_degrees)
            cos_elev = math.cos(elevation_rad)
            sin_elev = math.sin(elevation_rad)

            # Convert start azimuth to radians.
            start_azimuth_rad = math.radians(start_azimuth_degrees)

            for i in range(num_side_views):
                azimuth = start_azimuth_rad + 2 * math.pi * i / num_side_views
                # Spherical to Cartesian: camera positioned at elevation, looking at center.
                # x = r * cos(elev) * cos(az)
                # y = r * cos(elev) * sin(az)
                # z = r * sin(elev)
                dir_vec = Vector(
                    (
                        cos_elev * math.cos(azimuth),
                        cos_elev * math.sin(azimuth),
                        sin_elev,
                    )
                )
                views.append(
                    {"name": f"{i + side_index_offset}_side", "direction": dir_vec}
                )

            # Render each view.
            image_paths = []
            output_dir.mkdir(parents=True, exist_ok=True)

            for idx, view in enumerate(views):
                direction = view["direction"].normalized()
                camera_obj.location = bbox_center + direction * camera_distance
                look_at_target(camera_obj, bbox_center)

                # Position light near camera to illuminate the viewed surface.
                light_obj.location = camera_obj.location + direction * (
                    camera_distance * LIGHT_DISTANCE_RATIO
                )

                # Render to file.
                output_path = output_dir / f"{view['name']}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)

                image_paths.append(output_path)

        # Add number overlays (outside suppression - uses PIL, not Blender).
        for idx, output_path in enumerate(image_paths):
            add_number_overlay(output_path, idx)

        console_logger.info(
            f"Rendered {len(image_paths)} views to {output_dir} in "
            f"{time.time()-start_time:.2f}s"
        )
        return image_paths

    def render_multiview_for_clip_embedding(
        self,
        mesh_path: Path,
        output_dir: Path,
        width: int = 224,
        height: int = 224,
        elevation_degrees: float = 30.0,
        light_energy: float | None = None,
    ) -> list[Path]:
        """Render clean multi-view images for CLIP embedding computation.

        Renders 8 views optimized for CLIP image encoding:
        - 4 views at +elevation (upper hemisphere) at 0°, 90°, 180°, 270° azimuth
        - 4 views at -elevation (lower hemisphere) at 0°, 90°, 180°, 270° azimuth

        Unlike render_multiview_for_analysis, this produces clean renders:
        - No coordinate frame overlay
        - No number labels
        - Neutral grey background (works for both light and dark objects)
        - 224x224 default resolution (CLIP's native input size)

        Args:
            mesh_path: Path to the mesh file (GLB/GLTF/OBJ).
            output_dir: Directory where rendered images will be saved.
            width: Image width in pixels (default: 224 for CLIP).
            height: Image height in pixels (default: 224 for CLIP).
            elevation_degrees: Elevation angle in degrees (default: 30).
            light_energy: Light energy in watts. If None, uses VLM_ANALYSIS_LIGHT_ENERGY.

        Returns:
            List of paths to rendered PNG images (8 images).
        """
        start_time = time.time()
        num_views = 8
        console_logger.info(
            f"Rendering {num_views} views for CLIP embedding ({width}x{height}px)"
        )

        with suppress_stdout_stderr():
            # Clear existing scene.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Set up rendering with CYCLES (offline process, higher quality).
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            scene.cycles.samples = CYCLES_CLIP_SAMPLES
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            # Render with transparent background, composite onto neutral grey.
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Black world background (will be transparent due to film_transparent).
            world = bpy.data.worlds.new("ClipWorld")
            scene.world = world
            world.use_nodes = True
            bg_node = world.node_tree.nodes["Background"]
            bg_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

            # Add camera-following point light.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = (
                light_energy if light_energy is not None else VLM_ANALYSIS_LIGHT_ENERGY
            )
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import mesh based on file extension.
            mesh_path_str = str(mesh_path)
            if mesh_path_str.lower().endswith(".obj"):
                bpy.ops.wm.obj_import(filepath=mesh_path_str)
            else:
                bpy.ops.import_scene.gltf(filepath=mesh_path_str)

            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            disable_backface_culling(list(bpy.context.selected_objects))

            # Compute bounding box.
            mesh_objs = [
                obj for obj in bpy.context.selected_objects if obj.type == "MESH"
            ]
            bbox_min = Vector((float("inf"),) * 3)
            bbox_max = Vector((float("-inf"),) * 3)
            for obj in mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    bbox_min = Vector(map(min, bbox_min, world_corner))
                    bbox_max = Vector(map(max, bbox_max, world_corner))

            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            max_dim = max(bbox_size)

            # Setup camera.
            camera = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj
            camera.type = "PERSP"
            camera.lens = DEFAULT_CAMERA_LENS_MM
            camera.sensor_width = DEFAULT_CAMERA_SENSOR_WIDTH_MM
            camera.clip_start = DEFAULT_CAMERA_CLIP_START
            camera.clip_end = DEFAULT_CAMERA_CLIP_END

            # Compute camera distance.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Define 8 views: 4 upper + 4 lower at cardinal directions.
            elevation_rad = math.radians(elevation_degrees)
            azimuth_angles = [0, 90, 180, 270]  # Cardinal directions in degrees.
            elevations = [elevation_rad, -elevation_rad]  # Upper and lower.

            views = []
            for elev_idx, elev in enumerate(elevations):
                elev_name = "upper" if elev > 0 else "lower"
                cos_elev = math.cos(elev)
                sin_elev = math.sin(elev)

                for az_deg in azimuth_angles:
                    az_rad = math.radians(az_deg)
                    # Spherical to Cartesian: camera looks toward origin.
                    # x = r * cos(elev) * cos(az)
                    # y = r * cos(elev) * sin(az)
                    # z = r * sin(elev)
                    dir_vec = Vector(
                        (
                            cos_elev * math.cos(az_rad),
                            cos_elev * math.sin(az_rad),
                            sin_elev,
                        )
                    )
                    views.append(
                        {
                            "name": f"{elev_name}_az{az_deg}",
                            "direction": dir_vec,
                        }
                    )

            # Render each view.
            image_paths = []
            output_dir.mkdir(parents=True, exist_ok=True)

            for view in views:
                direction = view["direction"].normalized()
                camera_obj.location = bbox_center + direction * camera_distance
                look_at_target(camera_obj, bbox_center)

                # Position light near camera.
                light_obj.location = camera_obj.location + direction * (
                    camera_distance * LIGHT_DISTANCE_RATIO
                )

                # Render to file (RGBA with transparent background).
                output_path = output_dir / f"{view['name']}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)

                image_paths.append(output_path)

        # Composite all images onto neutral grey background.
        for img_path in image_paths:
            _composite_onto_grey(img_path)

        console_logger.info(
            f"Rendered {len(image_paths)} CLIP views to {output_dir} in "
            f"{time.time()-start_time:.2f}s"
        )
        return image_paths

    def render_floor_plan(
        self,
        mesh_path: Path,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        light_energy: float | None = None,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> Path:
        """Render a clean top-down view of a floor plan without coordinate frame.

        This produces a clean render suitable for floor plan visualization:
        - Single top-down view (looking down -Z axis)
        - No coordinate frame overlay
        - No number labels
        - Transparent background
        - Bright, even lighting to clearly show materials

        Args:
            mesh_path: Path to the floor plan GLB/GLTF file.
            output_path: Path where the rendered PNG will be saved.
            width: Image width in pixels (default: 1024).
            height: Image height in pixels (default: 1024).
            light_energy: Sun light energy. If None, uses default (5.0).

        Returns:
            Path to the rendered PNG image.
        """
        start_time = time.time()
        console_logger.info(f"Rendering floor plan top view ({width}x{height}px)")

        # Default sun energy for floor plans (brighter than typical scene).
        default_sun_energy = 5.0

        with suppress_stdout_stderr():
            # Clear existing scene.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Set up rendering with EEVEE for speed (6x faster than CYCLES).
            scene = bpy.context.scene
            _set_eevee_render_engine(scene)
            scene.eevee.taa_render_samples = taa_samples
            _apply_eevee_speed_settings(scene)
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            # Transparent background for compositing.
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Set up world with ambient lighting (adds fill light to shadows).
            scene.world = bpy.data.worlds.new("FloorPlanWorld")
            scene.world.use_nodes = True
            bg_node = scene.world.node_tree.nodes.get("Background")
            if bg_node:
                # Light gray ambient fill (0.3) - brightens shadows significantly.
                bg_node.inputs[0].default_value = (0.3, 0.3, 0.3, 1.0)
                bg_node.inputs[1].default_value = 1.0  # Strength.

            # Use SUN light for even illumination (no distance falloff).
            light = bpy.data.lights.new(name="Sun", type="SUN")
            light.energy = (
                light_energy if light_energy is not None else default_sun_energy
            )
            # Slightly warm color for natural look.
            light.color = (1.0, 0.98, 0.95)
            light_obj = bpy.data.objects.new("Sun", light)
            scene.collection.objects.link(light_obj)
            # Angle sun slightly (15 degrees from vertical) for some shadow definition.
            light_obj.rotation_euler = (math.radians(15), math.radians(15), 0)

            # Import mesh.
            bpy.ops.import_scene.gltf(filepath=str(mesh_path))
            bpy.ops.object.select_all(action="SELECT")

            # Apply rotation to counteract glTF import rotation.
            bpy.ops.transform.rotate(
                value=-math.pi / 2,
                orient_axis="X",
                orient_type="GLOBAL",
                center_override=(0, 0, 0),
            )

            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            disable_backface_culling(list(bpy.context.selected_objects))

            # Compute bounding box.
            mesh_objs = [
                obj for obj in bpy.context.selected_objects if obj.type == "MESH"
            ]
            bbox_min = Vector((float("inf"),) * 3)
            bbox_max = Vector((float("-inf"),) * 3)
            for obj in mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    bbox_min = Vector(map(min, bbox_min, world_corner))
                    bbox_max = Vector(map(max, bbox_max, world_corner))

            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            max_dim = max(bbox_size)

            # Setup camera.
            camera = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj
            camera.type = "PERSP"
            camera.lens = DEFAULT_CAMERA_LENS_MM
            camera.sensor_width = DEFAULT_CAMERA_SENSOR_WIDTH_MM
            camera.clip_start = DEFAULT_CAMERA_CLIP_START
            camera.clip_end = DEFAULT_CAMERA_CLIP_END

            # Compute camera distance.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Top-down view (looking down -Z axis).
            direction = Vector((0, 0, 1))  # Camera above, looking down.
            camera_obj.location = bbox_center + direction * camera_distance
            look_at_target(camera_obj, bbox_center)

            # SUN light doesn't need positioning - it's directional.

            # Render to file.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(output_path)
            bpy.ops.render.render(write_still=True)

        # Keep transparent background (RGBA) for floor plan renders.

        console_logger.info(
            f"Rendered floor plan to {output_path} in {time.time()-start_time:.2f}s"
        )
        return output_path

    def render_multiview_from_obj_directory(
        self,
        obj_directory: Path,
        output_dir: Path,
        num_side_views: int = DEFAULT_NUM_SIDE_VIEWS,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        include_vertical_views: bool = True,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> list[Path]:
        """Render multi-view images from a directory of OBJ files.

        This is designed for PartNet-Mobility assets which store each link's
        geometry as separate OBJ files. All OBJ files are loaded and rendered
        together as a single articulated object.

        The rendering setup matches render_multiview_for_analysis:
        - Transparent background
        - Coordinate frame visualization (+X=red, +Y=green, +Z=blue)
        - Numbered view labels
        - Camera-following point light

        Args:
            obj_directory: Path to directory containing OBJ files. All .obj
                files in this directory will be loaded.
            output_dir: Directory where rendered images will be saved.
            num_side_views: Number of equidistant side views to render.
            width: Width of rendered images in pixels (default: 512).
            height: Height of rendered images in pixels (default: 512).
            include_vertical_views: If True, render top/bottom views. If False,
                only render side views.

        Returns:
            List of paths to rendered PNG images.

        Raises:
            FileNotFoundError: If obj_directory does not exist.
            ValueError: If no OBJ files are found in the directory.
        """
        start_time = time.time()

        if not obj_directory.exists():
            raise FileNotFoundError(f"OBJ directory not found: {obj_directory}")

        # Find all OBJ files in the directory.
        obj_files = sorted(obj_directory.glob("*.obj"))
        if not obj_files:
            raise ValueError(f"No OBJ files found in {obj_directory}")

        num_vertical_views = 2 if include_vertical_views else 0
        total_views = num_side_views + num_vertical_views
        console_logger.info(
            f"Rendering {total_views} views from {len(obj_files)} OBJ files "
            f"({width}x{height}px)"
        )

        # Suppress Blender's verbose rendering output.
        with suppress_stdout_stderr():
            # Clear existing scene.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Set up rendering with EEVEE for speed (6x faster than CYCLES).
            scene = bpy.context.scene
            _set_eevee_render_engine(scene)
            scene.eevee.taa_render_samples = taa_samples
            _apply_eevee_speed_settings(scene)
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            # Use transparent background for VLM analysis.
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Add a camera-following point light.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = VLM_ANALYSIS_LIGHT_ENERGY
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import all OBJ files.
            for obj_file in obj_files:
                bpy.ops.wm.obj_import(filepath=str(obj_file))

            # Select all mesh objects and apply transforms.
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            disable_backface_culling(list(bpy.context.scene.objects))

            # Compute combined bounding box.
            mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
            if not mesh_objs:
                raise ValueError(
                    f"No mesh objects after importing from {obj_directory}"
                )

            bbox_min = Vector((float("inf"),) * 3)
            bbox_max = Vector((float("-inf"),) * 3)
            for obj in mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    bbox_min = Vector(map(min, bbox_min, world_corner))
                    bbox_max = Vector(map(max, bbox_max, world_corner))

            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            max_dim = max(bbox_size)

            # Create coordinate frame with labels.
            create_coordinate_frame(
                position=bbox_center,
                max_dim=max_dim,
                scale_factor=COORDINATE_FRAME_SCALE_FACTOR,
                add_labels=True,
            )

            # Setup camera.
            camera = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj
            camera.type = "PERSP"
            camera.lens = DEFAULT_CAMERA_LENS_MM
            camera.sensor_width = DEFAULT_CAMERA_SENSOR_WIDTH_MM
            camera.clip_start = DEFAULT_CAMERA_CLIP_START
            camera.clip_end = DEFAULT_CAMERA_CLIP_END

            # Compute camera distance.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Define views.
            views = []
            if include_vertical_views:
                views.append({"name": "0_top", "direction": Vector((0, 0, 1))})
                views.append({"name": "1_bottom", "direction": Vector((0, 0, -1))})
                side_index_offset = 2
            else:
                side_index_offset = 0

            for i in range(num_side_views):
                angle = 2 * math.pi * i / num_side_views
                dir_vec = Vector((math.cos(angle), math.sin(angle), 0))
                views.append(
                    {"name": f"{i + side_index_offset}_side", "direction": dir_vec}
                )

            # Render each view.
            image_paths = []
            output_dir.mkdir(parents=True, exist_ok=True)

            for view in views:
                direction = view["direction"].normalized()
                camera_obj.location = bbox_center + direction * camera_distance
                look_at_target(camera_obj, bbox_center)

                # Position light near camera to illuminate the viewed surface.
                light_obj.location = camera_obj.location + direction * (
                    camera_distance * LIGHT_DISTANCE_RATIO
                )

                # Render to file.
                output_path = output_dir / f"{view['name']}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)

                image_paths.append(output_path)

        # Add number overlays (outside suppression - uses PIL, not Blender).
        for idx, output_path in enumerate(image_paths):
            add_number_overlay(output_path, idx)

        console_logger.info(
            f"Rendered {len(image_paths)} views to {output_dir} in "
            f"{time.time()-start_time:.2f}s"
        )
        return image_paths

    def render_multiview_articulated(
        self,
        link_meshes: list[LinkMeshInfo],
        output_dir: Path,
        num_combined_side_views: int = DEFAULT_NUM_SIDE_VIEWS,
        num_link_side_views: int = 4,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        light_energy: float | None = None,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> ArticulatedRenderResult:
        """Render multi-view images for an articulated object with per-link views.

        This renders:
        1. Combined views showing all links together (combined_0.png, etc.)
        2. Per-link views showing each link in isolation (link_name_0.png, etc.)

        The combined views use the same setup as render_multiview_from_obj_directory.
        Per-link views use fewer angles since they're supplementary.

        Args:
            link_meshes: List of LinkMeshInfo with link names and OBJ paths.
            output_dir: Directory where rendered images will be saved.
            num_combined_side_views: Number of side views for combined render.
            num_link_side_views: Number of side views for each link (fewer than
                combined since they're supplementary).
            width: Width of rendered images in pixels.
            height: Height of rendered images in pixels.
            light_energy: Light energy in watts. If None, uses VLM_ANALYSIS_LIGHT_ENERGY.

        Returns:
            ArticulatedRenderResult with paths to all images and dimensions.

        Raises:
            ValueError: If no valid meshes are found.
        """
        start_time = time.time()

        # Collect all mesh files across all links.
        all_mesh_files = []
        for link_info in link_meshes:
            for mesh_path in link_info.mesh_paths:
                if mesh_path.exists():
                    all_mesh_files.append(mesh_path)

        if not all_mesh_files:
            raise ValueError("No valid mesh files found in link meshes")

        output_dir.mkdir(parents=True, exist_ok=True)
        combined_image_paths = []
        link_image_paths: dict[str, list[Path]] = {}
        link_dimensions: dict[str, tuple[float, float, float]] = {}

        with suppress_stdout_stderr():
            # Clear existing scene.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Set up rendering with EEVEE for speed (6x faster than CYCLES).
            scene = bpy.context.scene
            _set_eevee_render_engine(scene)
            scene.eevee.taa_render_samples = taa_samples
            _apply_eevee_speed_settings(scene)
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Add a camera-following point light.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = (
                light_energy if light_energy is not None else VLM_ANALYSIS_LIGHT_ENERGY
            )
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import all mesh files and track which objects belong to which link.
            link_to_objects: dict[str, list[bpy.types.Object]] = {}

            for link_info in link_meshes:
                link_to_objects[link_info.link_name] = []

                # Build world transform matrix for this link.
                # Transform order: visual_origin -> link_world_transform.
                world_pos = link_info.world_position
                world_rot = link_info.world_rotation

                # Create world rotation matrix (identity if not provided).
                world_rot_matrix = (
                    Matrix(
                        (
                            (world_rot[0][0], world_rot[0][1], world_rot[0][2], 0),
                            (world_rot[1][0], world_rot[1][1], world_rot[1][2], 0),
                            (world_rot[2][0], world_rot[2][1], world_rot[2][2], 0),
                            (0, 0, 0, 1),
                        )
                    )
                    if world_rot is not None
                    else Matrix.Identity(4)
                )

                for mesh_path, origin in zip(
                    link_info.mesh_paths, link_info.origins, strict=True
                ):
                    if not mesh_path.exists():
                        continue

                    # Record objects before import.
                    objects_before = set(bpy.context.scene.objects)

                    # Import based on file extension.
                    ext = mesh_path.suffix.lower()
                    if ext == ".obj":
                        bpy.ops.wm.obj_import(filepath=str(mesh_path))
                    elif ext in {".gltf", ".glb"}:
                        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
                    else:
                        console_logger.warning(f"Unsupported mesh format: {ext}")
                        continue

                    # Find newly imported objects.
                    objects_after = set(bpy.context.scene.objects)
                    new_objects = objects_after - objects_before

                    for obj in new_objects:
                        if obj.type == "MESH":
                            # Apply visual origin offset first (in link's local frame).
                            obj.location.x += origin[0]
                            obj.location.y += origin[1]
                            obj.location.z += origin[2]

                            # Then apply link's world transform.
                            # First apply rotation around origin, then translate.
                            if world_rot is not None:
                                # Get current location as vector.
                                local_pos = Vector(obj.location)
                                # Rotate position by world rotation.
                                rotated_pos = world_rot_matrix @ Vector(
                                    (local_pos.x, local_pos.y, local_pos.z, 1)
                                )
                                obj.location = Vector(
                                    (rotated_pos.x, rotated_pos.y, rotated_pos.z)
                                )
                                # Apply rotation to object orientation.
                                obj.matrix_world = world_rot_matrix @ obj.matrix_world
                                # Reset location after matrix multiply.
                                obj.location = Vector(
                                    (rotated_pos.x, rotated_pos.y, rotated_pos.z)
                                )

                            # Apply world position offset.
                            obj.location.x += world_pos[0]
                            obj.location.y += world_pos[1]
                            obj.location.z += world_pos[2]

                            link_to_objects[link_info.link_name].append(obj)

            # Apply transforms to all mesh objects.
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            # This ensures meshes render correctly from both sides, fixing issues
            # with single-sided meshes (common in PartNet-Mobility models).
            disable_backface_culling(list(bpy.context.scene.objects))

            # Compute combined bounding box of all mesh objects.
            all_mesh_objs = [
                obj for obj in bpy.context.scene.objects if obj.type == "MESH"
            ]
            if not all_mesh_objs:
                raise ValueError("No mesh objects after importing")

            combined_bbox_min = Vector((float("inf"),) * 3)
            combined_bbox_max = Vector((float("-inf"),) * 3)
            for obj in all_mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    combined_bbox_min = Vector(
                        map(min, combined_bbox_min, world_corner)
                    )
                    combined_bbox_max = Vector(
                        map(max, combined_bbox_max, world_corner)
                    )

            combined_bbox_center = (combined_bbox_min + combined_bbox_max) / 2
            combined_bbox_size = combined_bbox_max - combined_bbox_min
            combined_max_dim = max(combined_bbox_size)
            combined_dimensions = (
                combined_bbox_size.x,
                combined_bbox_size.y,
                combined_bbox_size.z,
            )

            # Compute per-link bounding boxes.
            for link_name, link_objs in link_to_objects.items():
                if not link_objs:
                    link_dimensions[link_name] = (0.0, 0.0, 0.0)
                    continue

                link_bbox_min = Vector((float("inf"),) * 3)
                link_bbox_max = Vector((float("-inf"),) * 3)
                for obj in link_objs:
                    for corner in obj.bound_box:
                        world_corner = obj.matrix_world @ Vector(corner)
                        link_bbox_min = Vector(map(min, link_bbox_min, world_corner))
                        link_bbox_max = Vector(map(max, link_bbox_max, world_corner))

                link_size = link_bbox_max - link_bbox_min
                link_dimensions[link_name] = (link_size.x, link_size.y, link_size.z)

            # Create coordinate frame.
            create_coordinate_frame(
                position=combined_bbox_center,
                max_dim=combined_max_dim,
                scale_factor=COORDINATE_FRAME_SCALE_FACTOR,
                add_labels=True,
            )

            # Setup camera.
            camera = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj
            camera.type = "PERSP"
            camera.lens = DEFAULT_CAMERA_LENS_MM
            camera.sensor_width = DEFAULT_CAMERA_SENSOR_WIDTH_MM
            camera.clip_start = DEFAULT_CAMERA_CLIP_START
            camera.clip_end = DEFAULT_CAMERA_CLIP_END

            # Compute camera distance for combined view.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (combined_max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Define combined views (top, bottom, sides).
            combined_views = [
                {"name": "combined_0_top", "direction": Vector((0, 0, 1))},
                {"name": "combined_1_bottom", "direction": Vector((0, 0, -1))},
            ]
            for i in range(num_combined_side_views):
                angle = 2 * math.pi * i / num_combined_side_views
                dir_vec = Vector((math.cos(angle), math.sin(angle), 0))
                combined_views.append(
                    {"name": f"combined_{i + 2}_side", "direction": dir_vec}
                )

            # Render combined views.
            for view in combined_views:
                direction = view["direction"].normalized()
                camera_obj.location = combined_bbox_center + direction * camera_distance
                look_at_target(camera_obj, combined_bbox_center)
                light_obj.location = camera_obj.location + direction * (
                    camera_distance * LIGHT_DISTANCE_RATIO
                )

                output_path = output_dir / f"{view['name']}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)
                combined_image_paths.append(output_path)

            # Render per-link views.
            for link_name, link_objs in link_to_objects.items():
                if not link_objs:
                    link_image_paths[link_name] = []
                    continue

                # Hide all objects except this link's objects.
                for obj in all_mesh_objs:
                    obj.hide_render = obj not in link_objs

                # Compute link-specific camera distance.
                link_bbox_min = Vector((float("inf"),) * 3)
                link_bbox_max = Vector((float("-inf"),) * 3)
                for obj in link_objs:
                    for corner in obj.bound_box:
                        world_corner = obj.matrix_world @ Vector(corner)
                        link_bbox_min = Vector(map(min, link_bbox_min, world_corner))
                        link_bbox_max = Vector(map(max, link_bbox_max, world_corner))

                link_center = (link_bbox_min + link_bbox_max) / 2
                link_size = link_bbox_max - link_bbox_min
                link_max_dim = max(link_size)
                link_camera_distance = (
                    (link_max_dim / 2) / math.tan(fov / 2)
                ) * CAMERA_DISTANCE_MARGIN_MULTIPLIER

                # Define link views (fewer than combined).
                link_views = []
                for i in range(num_link_side_views):
                    angle = 2 * math.pi * i / num_link_side_views
                    dir_vec = Vector((math.cos(angle), math.sin(angle), 0))
                    link_views.append(
                        {"name": f"{link_name}_{i}_side", "direction": dir_vec}
                    )

                link_image_paths[link_name] = []
                for view in link_views:
                    direction = view["direction"].normalized()
                    camera_obj.location = link_center + direction * link_camera_distance
                    look_at_target(camera_obj, link_center)
                    light_obj.location = camera_obj.location + direction * (
                        link_camera_distance * LIGHT_DISTANCE_RATIO
                    )

                    output_path = output_dir / f"{view['name']}.png"
                    scene.render.filepath = str(output_path)
                    bpy.ops.render.render(write_still=True)
                    link_image_paths[link_name].append(output_path)

            # Restore all objects to visible.
            for obj in all_mesh_objs:
                obj.hide_render = False

        # Add number overlays to combined images.
        for idx, output_path in enumerate(combined_image_paths):
            add_number_overlay(output_path, idx)

        # Add number overlays to per-link images.
        # Note: image filenames already include link names (e.g., link_0_0_side.png).
        for link_name, paths in link_image_paths.items():
            for idx, output_path in enumerate(paths):
                add_number_overlay(output_path, idx)

        total_images = len(combined_image_paths) + sum(
            len(p) for p in link_image_paths.values()
        )
        console_logger.info(
            f"Rendered {total_images} articulated views ({len(combined_image_paths)} "
            f"combined, {len(link_image_paths)} links) to {output_dir} in "
            f"{time.time()-start_time:.2f}s"
        )

        return ArticulatedRenderResult(
            combined_image_paths=combined_image_paths,
            link_image_paths=link_image_paths,
            link_dimensions=link_dimensions,
            combined_dimensions=combined_dimensions,
        )

    def render_agent_observation_views(
        self,
        params: RenderParams,
        output_dir: Path,
        layout: str,
        top_view_width: int,
        top_view_height: int,
        side_view_count: int,
        side_view_width: int,
        side_view_height: int,
        scene_objects: list[dict] | None = None,
        annotations: dict | None = None,
        wall_normals: dict[str, list[float]] | None = None,
        support_surfaces: list[dict] | None = None,
        show_support_surface: bool = False,
        current_furniture_id: str | None = None,
        context_furniture_ids: list[str] | None = None,
        render_single_view: dict | None = None,
        openings: list[ClearanceOpeningData] | None = None,
        wall_surfaces: list[dict] | None = None,
        wall_surfaces_for_labels: list[dict] | None = None,
        room_bounds: tuple[float, float, float, float] | None = None,
        ceiling_height: float | None = None,
        side_view_elevation_degrees: float | None = None,
        side_view_start_azimuth_degrees: float | None = None,
        include_vertical_views: bool = True,
    ) -> list[Path]:
        """Render scene views based on layout configuration.

        Each view is rendered individually at native resolution and saved
        directly to output_dir. Supports multiple layout types for ablations.

        For manipuland mode with multiple support surfaces, generates separate
        top views for each surface with filtered coordinate markers and labels.

        For wall rendering mode, generates context top-down view plus per-wall
        orthographic views for wall-mounted object placement.

        For ceiling perspective mode, generates an elevated corner view
        looking down at the ceiling plane with furniture context below.

        Args:
            params: Rendering parameters with scene path and camera settings.
            output_dir: Directory where rendered images will be saved.
            layout: Layout type - "grid_3x3", "single_top", "top_plus_sides",
                "wall_orthographic", "wall", or "ceiling_perspective".
            top_view_width: Width of top-down view in pixels.
            top_view_height: Height of top-down view in pixels.
            side_view_count: Number of side views to render.
            side_view_width: Width of each side view in pixels.
            side_view_height: Height of each side view in pixels.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Optional annotation config flags.
            wall_normals: Pre-computed room-facing normals for walls.
            support_surfaces: Optional list of support surface data. Each surface dict
                contains: surface_id, corners (8 bbox corners), convex_hull_vertices
                (mesh vertices for marker filtering). For multi-surface furniture,
                generates one top view per surface.
            show_support_surface: If True, render green wireframe bbox showing support
                surface bounds for debugging.
            render_single_view: If provided, renders ONLY this single view instead of
                the full layout. Dict with keys: enabled, name, direction (list[float]).
                Used for per-drawer rendering.
            openings: Optional list of ClearanceOpeningData for door/window/open
                labels. Labels are rendered on top views using camera projection.
            wall_surfaces: List of wall surface dicts for wall rendering modes.
                Each dict contains wall_id, direction, length, height, transform,
                and excluded_regions.
            room_bounds: Room XY bounds (min_x, min_y, max_x, max_y) for ceiling mode.
            ceiling_height: Ceiling height in meters for ceiling mode.

        Returns:
            List of paths to rendered PNG files, ordered (top first, then sides).

        Raises:
            ValueError: If layout type is not recognized.
        """
        start_time = time.time()
        console_logger.info(
            f"Rendering {layout} layout with top ({top_view_width}x{top_view_height})"
        )

        # Reset state from previous renders to prevent leakage.
        self._reset_rendering_state()

        # Store wall normals, support surfaces, and debug flags for use during rendering.
        self._wall_normals = wall_normals or {}
        self._show_support_surface = show_support_surface
        self._wall_surfaces_for_labels = wall_surfaces_for_labels

        # Process support surfaces if provided.
        if support_surfaces is not None and len(support_surfaces) > 0:
            # Validate surface data structure.
            for i, surface in enumerate(support_surfaces):
                if "corners" not in surface:
                    raise ValueError(
                        f"Surface {i} missing required key 'corners'. "
                        f"Available keys: {list(surface.keys())}"
                    )
                if "surface_id" not in surface:
                    raise ValueError(
                        f"Surface {i} missing required key 'surface_id'. "
                        f"Available keys: {list(surface.keys())}"
                    )

            self._support_surfaces = support_surfaces

            # Use first surface for camera alignment (backward compatibility).
            # For multi-surface, all surfaces typically share same orientation.
            first_surface = support_surfaces[0]
            corners = first_surface["corners"]

            (
                self._surface_corners,
                self._surface_bounds_min,
                self._surface_bounds_max,
            ) = _compute_bounds_from_corners(corners)

            # Compute furniture rotation angle for camera alignment.
            # Corners are in Drake Z-up coordinates but the overlay camera
            # alignment uses the legacy glTF overlay frame:
            # Drake (x, y, z) → overlay (x, -z, y)

            def drake_to_blender(point):
                """Transform point from Drake Z-up to overlay camera space."""
                return np.array([point[0], -point[2], point[1]])

            # Transform corners to Blender space.
            corners_blender = [
                drake_to_blender(corner) for corner in self._surface_corners
            ]

            # Extract furniture axes from corner edges in Blender space.
            edge_x = corners_blender[1] - corners_blender[0]
            edge_y = corners_blender[2] - corners_blender[0]

            # In Blender Y-up, for a top-down view (camera looking along -Y),
            # the camera's +X is "right" and +Z is "up" in the image.
            # Project furniture +X onto the XZ plane to find rotation angle.
            grid_axis_x_xz = np.array([edge_x[0], edge_x[2]])  # Project onto XZ plane
            grid_axis_x_xz_norm = grid_axis_x_xz / np.linalg.norm(grid_axis_x_xz)

            # Compute angle in XZ plane (rotation around Y axis in Blender).
            self._furniture_rotation_z = math.atan2(
                grid_axis_x_xz_norm[1], grid_axis_x_xz_norm[0]
            )

            # Position coordinate frame at corner for visual consistency with
            # furniture floor mode, even though (0,0) is at center.
            self._frame_origin = self._surface_corners[0]

            # Compute bbox axes for coordinate frame alignment.
            # These axes define the furniture's local coordinate system.
            # Compute from Drake-space corners (before Blender transformation).
            edge_x_drake = self._surface_corners[1] - self._surface_corners[0]
            edge_y_drake = self._surface_corners[2] - self._surface_corners[0]
            edge_z_drake = self._surface_corners[4] - self._surface_corners[0]

            extent_x_drake = np.linalg.norm(edge_x_drake)
            extent_y_drake = np.linalg.norm(edge_y_drake)
            extent_z_drake = np.linalg.norm(edge_z_drake)

            axis_x = (
                edge_x_drake / extent_x_drake
                if extent_x_drake > 0
                else np.array([1, 0, 0])
            )
            axis_y = (
                edge_y_drake / extent_y_drake
                if extent_y_drake > 0
                else np.array([0, 1, 0])
            )
            axis_z = (
                edge_z_drake / extent_z_drake
                if extent_z_drake > 0
                else np.array([0, 0, 1])
            )

            # Store as numpy arrays (coordinate_frame.py expects .tolist() to work).
            self._bbox_axis_x = axis_x
            self._bbox_axis_y = axis_y
            self._bbox_axis_z = axis_z
        else:
            self._support_surfaces = None
            self._surface_corners = None
            self._surface_bounds_min = None
            self._surface_bounds_max = None
            self._furniture_rotation_z = None
            self._frame_origin = None

        # Convert annotations dict to OmegaConf for consistent attribute access.
        # This handles the HTTP boundary where OmegaConf → JSON → plain dict.
        if annotations and not isinstance(annotations, DictConfig):
            annotations = OmegaConf.create(annotations)

        # Set default values.
        if scene_objects is None:
            scene_objects = []
        if annotations is None:
            annotations = OmegaConf.create({})

        # Store scene objects for filtering in per-surface rendering.
        self._scene_objects = scene_objects

        # Store current furniture ID for per-surface rendering.
        # In manipuland mode, this is the furniture whose surfaces are being rendered.
        self._current_furniture_id = current_furniture_id

        # Store context furniture IDs for per-surface rendering.
        # These nearby furniture objects should remain visible in top-down views
        # to provide spatial context for item placement orientation.
        self._context_furniture_ids = set(context_furniture_ids or [])

        # Generate surface colors for multi-surface mode.
        self._surface_colors: dict[str, tuple[int, int, int]] = {}
        if self._support_surfaces is not None and len(self._support_surfaces) > 1:
            self._surface_colors = generate_surface_colors(
                surface_ids=[str(s["surface_id"]) for s in self._support_surfaces]
            )
            console_logger.info(
                f"Generated {len(self._surface_colors)} unique colors for surfaces"
            )

        # Generate views based on layout.
        # Check for single view mode (per-drawer rendering).
        if render_single_view is not None and render_single_view.get("enabled", False):
            console_logger.info(
                f"Single view mode: rendering only '{render_single_view.get('name', 'drawer')}'"
            )
            direction = render_single_view.get("direction", [0.0, 0.7, 0.7])
            # If we have support surfaces, attach the first one's data to the view.
            surface_data = None
            if self._support_surfaces and len(self._support_surfaces) > 0:
                surface_data = self._support_surfaces[0]
            views = [
                {
                    "name": render_single_view.get("name", "drawer_view"),
                    "direction": Vector(direction),
                    "is_side": False,
                    "surface_data": surface_data,
                    "is_drawer_view": True,
                }
            ]
        # For multi-surface manipuland mode, generate per-surface top views + side views.
        elif self._support_surfaces is not None and len(self._support_surfaces) > 1:
            console_logger.info(
                f"Multi-surface mode: generating {len(self._support_surfaces)} top views "
                f"+ {side_view_count} side views"
            )
            # Generate per-surface top views.
            top_views = generate_multi_surface_views(
                support_surfaces=self._support_surfaces
            )
            # Generate standard side views for overall context.
            furniture_rotation = (
                self._furniture_rotation_z
                if hasattr(self, "_furniture_rotation_z")
                and self._furniture_rotation_z is not None
                else None
            )
            side_views = self._generate_top_plus_sides_views(
                count=side_view_count,
                furniture_rotation_z=furniture_rotation,
                is_multi_surface_mode=True,
                elevation_degrees=side_view_elevation_degrees,
                start_azimuth_degrees=side_view_start_azimuth_degrees,
                include_vertical_views=include_vertical_views,
            )
            # Combine: side views first, then top views.
            # This ensures original full-scene setup_data is computed and saved before
            # per-surface top views modify it with tight surface bounds.
            # Filter to only side views (skip the single top view from top_plus_sides).
            side_views_only = [v for v in side_views if v.get("is_side", False)]
            views = side_views_only + top_views
        elif layout == "grid_3x3":
            views = self._generate_grid_3x3_views()
        elif layout == "single_top":
            views = self._generate_single_top_view()
        elif layout == "top_plus_sides":
            # Pass furniture rotation for manipuland mode to align side views.
            furniture_rotation = (
                self._furniture_rotation_z
                if hasattr(self, "_furniture_rotation_z")
                and self._furniture_rotation_z is not None
                else None
            )
            views = self._generate_top_plus_sides_views(
                count=side_view_count,
                furniture_rotation_z=furniture_rotation,
                is_multi_surface_mode=False,
                elevation_degrees=side_view_elevation_degrees,
                start_azimuth_degrees=side_view_start_azimuth_degrees,
                include_vertical_views=include_vertical_views,
            )
        elif layout == "wall_orthographic":
            # Per-wall orthographic view with grid overlay.
            views = self._generate_wall_orthographic_view(wall_surfaces=wall_surfaces)
        elif layout == "wall":
            # Context-only view for wall mode.
            # Per-wall orthographic views are rendered separately via
            # render_per_wall_ortho_views with filtered furniture per wall.
            views = self._generate_wall_context_views()
        elif layout == "ceiling_perspective":
            # Elevated perspective view for ceiling observation.
            if room_bounds is None:
                raise ValueError("ceiling_perspective layout requires room_bounds")
            if ceiling_height is None:
                raise ValueError("ceiling_perspective layout requires ceiling_height")
            views = self._generate_ceiling_perspective_view(
                room_bounds=room_bounds,
                ceiling_height=ceiling_height,
            )
        else:
            raise ValueError(
                f"Unknown layout '{layout}'. "
                f"Options: grid_3x3, single_top, top_plus_sides, wall_orthographic, "
                f"wall, ceiling_perspective"
            )

        # Inject ceiling_height into all views for coordinate grid rendering.
        # This ensures the grid is drawn at ceiling level instead of floor.
        if ceiling_height is not None:
            for view in views:
                view["ceiling_height"] = ceiling_height

        # Create output directory.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Render each view.
        image_paths = []
        setup_data = None
        original_setup_data = None  # Store full-scene setup for side views.
        with suppress_stdout_stderr():
            for view in views:
                is_top_view = view["name"].endswith("_top") or "_top_" in view["name"]
                width = top_view_width if is_top_view else side_view_width
                height = top_view_height if is_top_view else side_view_height

                # For multi-surface views, update current surface data before rendering.
                is_per_surface_top_view = "surface_data" in view and is_top_view
                is_per_surface_side_view = "surface_data" in view and not is_top_view
                if "surface_data" in view:
                    surface_data = view["surface_data"]
                    # Update _surface_corners and bounds for this specific surface.
                    (
                        self._surface_corners,
                        self._surface_bounds_min,
                        self._surface_bounds_max,
                    ) = _compute_bounds_from_corners(surface_data["corners"])
                    # Store convex hull for coordinate marker filtering.
                    self._current_convex_hull = surface_data.get(
                        "convex_hull_vertices", None
                    )
                    self._current_surface_id = surface_data.get("surface_id", "unknown")

                    # For per-surface top views, set coordinate frame at surface corner.
                    if is_per_surface_top_view:
                        # Compute surface local coordinate frame.
                        # Use corner 0 as origin (min x, min y, min z).
                        corners = np.array(surface_data["corners"])
                        origin = corners[0]  # Corner at (min_x, min_y, min_z).

                        # Compute local axes from edges.
                        edge_x = corners[1] - corners[0]  # X axis direction.
                        edge_y = corners[2] - corners[0]  # Y axis direction.
                        edge_z = corners[4] - corners[0]  # Z axis direction.

                        # Normalize to get unit axes.
                        axis_x = edge_x / np.linalg.norm(edge_x)
                        axis_y = edge_y / np.linalg.norm(edge_y)
                        axis_z = edge_z / np.linalg.norm(edge_z)

                        # Store for coordinate frame rendering (keep as numpy arrays).
                        self._frame_origin = origin
                        self._bbox_axis_x = axis_x
                        self._bbox_axis_y = axis_y
                        self._bbox_axis_z = axis_z

                        # Clean up previous surface mesh first.
                        if (
                            hasattr(self, "_surface_mesh_objects")
                            and self._surface_mesh_objects
                        ):
                            self._cleanup_surface_meshes()
                        self._setup_per_surface_rendering(surface_data)
                else:
                    self._current_convex_hull = None
                    self._current_surface_id = None

                    # Restore object visibility for side views.
                    if (
                        hasattr(self, "_surface_mesh_objects")
                        and self._surface_mesh_objects
                    ):
                        self._cleanup_surface_meshes()
                        self._restore_object_visibility()

                # Setup scene on first iteration only.
                # Resolution is set per-view below, so initial value doesn't matter.
                if setup_data is None:
                    # For side views, use tighter margin (30%) to show full scene.
                    # For top views, use default margin (80%) with extra coordinate space.
                    is_side_view = view.get("is_side", False)
                    margin_scale = 1.40 if is_side_view else 1.8
                    setup_data = self._setup_overlay_rendering(
                        params=params, view_size=None, margin_scale=margin_scale
                    )
                    # Save original full-scene setup for side views ONLY if this is not
                    # a per-surface view. In multi-surface mode, side views come first and
                    # this will save the full-scene bounds with tight 5% margin.
                    if not is_per_surface_top_view:
                        original_setup_data = setup_data

                    # Create overlays AFTER scene and GLTF have been imported as
                    # reset_scene() deletes all objects. Must create overlays after GLTF
                    # import so they persist for rendering.
                    if (
                        self._support_surfaces is not None
                        and len(self._support_surfaces) > 1
                        and self._surface_colors
                        and not self._overlay_mesh_objects
                    ):
                        console_logger.info(
                            f"Creating {len(self._support_surfaces)} overlay meshes "
                            "for multi-surface mode (after scene setup)"
                        )
                        self._overlay_mesh_objects = []
                        for surface_data in self._support_surfaces:
                            surface_id = surface_data.get("surface_id", "unknown")
                            if surface_id in self._surface_colors:
                                overlay_color = self._surface_colors[surface_id]
                                overlay_obj = self._create_surface_overlay_mesh(
                                    surface_data=surface_data, color=overlay_color
                                )
                                if overlay_obj is not None:
                                    self._overlay_mesh_objects.append(overlay_obj)
                        console_logger.info(
                            f"Created {len(self._overlay_mesh_objects)} overlay meshes"
                        )
                else:
                    # Restore walls hidden in previous view.
                    restore_hidden_walls()

                # For per-surface top views, recompute camera setup after hiding objects.
                if is_per_surface_top_view:
                    console_logger.info(
                        "Recomputing camera setup for per-surface view "
                        f"(surface {self._current_surface_id})"
                    )
                    # Compute bounds from surface corners directly (not entire scene).
                    corners_array = np.array(surface_data["corners"])
                    bbox_min = corners_array.min(axis=0)
                    bbox_max = corners_array.max(axis=0)
                    bbox_center = Vector((bbox_min + bbox_max) / 2)
                    bbox_size = bbox_max - bbox_min
                    max_dim = max(bbox_size)

                    # Add margin for manipulands extending beyond surface bounds.
                    max_dim *= 1.1  # 10% margin.

                    camera_distance = calculate_camera_distance(
                        camera_obj=setup_data.camera_obj,
                        max_dim=max_dim,
                        margin_scale=1.1,  # 10% camera distance margin.
                    )
                    # Save original setup before modifying (for side views).
                    if original_setup_data is None:
                        original_setup_data = setup_data

                    # Update setup_data with new bounds.
                    setup_data = OverlayRenderingSetup(
                        camera_obj=setup_data.camera_obj,
                        bbox_center=bbox_center,
                        max_dim=max_dim,
                        camera_distance=camera_distance,
                    )
                elif is_per_surface_side_view:
                    # For side views, restore original full-scene setup.
                    # Side views should show full furniture, not just the surface.
                    if original_setup_data is not None:
                        setup_data = original_setup_data

                # Update render resolution for this view.
                scene = bpy.context.scene
                scene.render.resolution_x = width
                scene.render.resolution_y = height

                # Render view with metric overlays and annotations.
                output_path = output_dir / f"{view['name']}.png"
                self._render_single_view_with_metric_overlay_to_path(
                    view=view,
                    setup_data=setup_data,
                    output_path=output_path,
                    scene_objects=scene_objects,
                    annotations=annotations,
                    openings=openings,
                )
                image_paths.append(output_path)

        # Clean up temporary surface meshes and overlays.
        self._cleanup_surface_meshes()
        self._cleanup_overlay_meshes()

        console_logger.info(
            f"Rendered {len(image_paths)} views to {output_dir} in "
            f"{time.time() - start_time:.2f}s"
        )
        return image_paths

    def _setup_camera_and_coordinate_frame(
        self,
        setup_data: OverlayRenderingSetup,
        view: dict[str, str | Vector | bool],
        annotations: DictConfig | None = None,
    ) -> None:
        """Position camera and add coordinate frame overlay for a view.

        Args:
            setup_data: Metric rendering setup data.
            view: Dictionary containing view information.
            annotations: Optional annotation config with rendering_mode.
        """
        # Reset camera clip_start to default before each view.
        # This ensures lower surface clipping from previous views doesn't persist.
        setup_data.camera_obj.data.clip_start = DEFAULT_CAMERA_CLIP_START

        # Position camera.
        direction = view["direction"].normalized()
        is_side = view.get("is_side", True)
        is_orthographic = view.get("is_orthographic", False)
        is_wall_orthographic = view.get("is_wall_orthographic", False)

        # Get rendering mode from annotations.
        rendering_mode = "furniture"
        if annotations and hasattr(annotations, "rendering_mode"):
            rendering_mode = annotations.rendering_mode

        # Handle orthographic camera for wall views.
        if is_orthographic and is_wall_orthographic:
            wall_surface = view.get("wall_surface", {})
            self._setup_wall_orthographic_camera(
                camera_obj=setup_data.camera_obj,
                wall_surface=wall_surface,
            )

            # Add coordinate frame for wall orthographic view.
            # Extract wall parameters for frame positioning.
            wall_length = wall_surface.get("length", 4.0)
            wall_height = wall_surface.get("height", 2.5)
            wall_direction = wall_surface.get("direction", "north")
            transform = wall_surface.get("transform", [0, 0, 0, 1, 0, 0, 0])

            # Calculate wall center from transform.
            wall_center = _compute_wall_center_from_transform(
                transform=transform, wall_length=wall_length, wall_height=wall_height
            )

            add_coordinate_frame_wall_view(
                wall_center=wall_center,
                wall_length=wall_length,
                wall_height=wall_height,
                wall_direction=wall_direction,
            )

            return  # Skip standard camera positioning for wall orthographic views.

        # Zoom in more for top view to reduce black borders.
        # Use less aggressive zoom for manipuland/ceiling mode to avoid cutting off edges.
        camera_distance = setup_data.camera_distance
        if not is_side:
            if rendering_mode == "manipuland":
                camera_distance *= 0.85  # Less aggressive zoom for manipuland mode.
            elif rendering_mode == "ceiling_perspective":
                camera_distance *= 0.8  # Moderate zoom for ceiling to show full grid.
            else:
                camera_distance *= 0.7  # More aggressive zoom for furniture mode.
        setup_data.camera_obj.location = (
            setup_data.bbox_center + direction * camera_distance
        )
        look_at_target(obj=setup_data.camera_obj, target=setup_data.bbox_center)

        # For manipuland mode top views, align camera with furniture orientation.
        if (
            not is_side
            and rendering_mode == "manipuland"
            and hasattr(self, "_furniture_rotation_z")
        ):
            # Apply roll rotation around viewing direction (Z-axis) to align camera
            # with furniture axes, making rotated furniture appear axis-aligned.
            setup_data.camera_obj.rotation_euler.rotate_axis(
                "Z", self._furniture_rotation_z
            )

        # Apply camera near-plane clipping for lower surfaces in per-surface top views.
        # This clips furniture geometry above the current surface so lower surfaces
        # (e.g., shelves under a table top) are visible from top-down views.
        surface_data = view.get("surface_data")
        is_per_surface_top_view = surface_data is not None and not is_side
        if (
            is_per_surface_top_view
            and rendering_mode == "manipuland"
            and self._support_surfaces is not None
            and len(self._support_surfaces) > 1
        ):
            self._apply_lower_surface_clipping(
                camera_obj=setup_data.camera_obj,
                surface_data=surface_data,
                camera_distance=camera_distance,
            )

        # Add support surface debug volumes for manipuland mode.
        if rendering_mode == "manipuland" and self._show_support_surface:
            if self._support_surfaces is not None and len(self._support_surfaces) > 1:
                # Multi-surface mode: draw colored volume for ALL surfaces.
                for surface in self._support_surfaces:
                    surface_id = surface.get("surface_id", "unknown")
                    corners = np.array(surface["corners"])
                    # Get surface color (RGB 0-255) and convert to Blender (RGBA 0-1).
                    if surface_id in self._surface_colors:
                        rgb = self._surface_colors[surface_id]
                        color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0)
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)  # Fallback green.
                    add_support_surface_debug_volume(corners=corners, color=color)
            elif self._surface_corners is not None:
                # Single-surface mode: draw green volume for the one surface.
                add_support_surface_debug_volume(corners=self._surface_corners)

        # Add coordinate frame overlay.
        from mathutils import Vector

        floor_bounds = get_floor_bounds(self._client_objects)
        frame_origin = self._frame_origin if hasattr(self, "_frame_origin") else None
        # Convert numpy arrays to Vectors for coordinate frame functions.
        bbox_axis_x = (
            Vector(self._bbox_axis_x.tolist())
            if hasattr(self, "_bbox_axis_x") and self._bbox_axis_x is not None
            else None
        )
        bbox_axis_y = (
            Vector(self._bbox_axis_y.tolist())
            if hasattr(self, "_bbox_axis_y") and self._bbox_axis_y is not None
            else None
        )
        bbox_axis_z = (
            Vector(self._bbox_axis_z.tolist())
            if hasattr(self, "_bbox_axis_z") and self._bbox_axis_z is not None
            else None
        )

        # Only add coordinate frame for furniture/manipuland top views.
        # Skip for wall_context, wall_orthographic views, and furniture_selection mode.
        is_wall_context = view.get("is_wall_context", False)
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        show_coord_frame = getattr(annotations, "show_coordinate_frame", True)
        if (
            not is_side
            and not is_wall_context
            and not is_wall_orthographic
            and show_coord_frame
        ):
            add_coordinate_frame_top_view(
                bbox_center=setup_data.bbox_center,
                max_dim=setup_data.max_dim,
                floor_bounds=floor_bounds,
                rendering_mode=rendering_mode,
                frame_origin=frame_origin,
                bbox_axis_x=bbox_axis_x,
                bbox_axis_y=bbox_axis_y,
                bbox_axis_z=bbox_axis_z,
            )

    def _apply_lower_surface_clipping(
        self,
        camera_obj: bpy.types.Object,
        surface_data: dict,
        camera_distance: float,
    ) -> None:
        """Apply camera near-plane clipping for lower support surfaces.

        When rendering a top-down view of a surface that is NOT the highest surface,
        furniture geometry above the surface blocks the view. This method clips that
        geometry by adjusting the camera's near clipping plane.

        For a top-down camera at height camera_z looking down (direction -Z):
        - Near clip at distance d clips everything at z > camera_z - d
        - To clip at surface_z + offset: clip_start = camera_z - (surface_z + offset)

        Args:
            camera_obj: Blender camera object to modify.
            surface_data: Dictionary containing current surface info with 'corners' key.
            camera_distance: Distance from camera to bbox_center along view direction.
        """
        if self._support_surfaces is None or len(self._support_surfaces) <= 1:
            return

        # Get current surface's Z range from corners.
        current_corners = np.array(surface_data.get("corners", []))
        if current_corners.size == 0:
            return
        current_z_max = current_corners[:, 2].max()

        # Find the highest surface's Z max among all surfaces.
        highest_z_max = current_z_max
        for surface in self._support_surfaces:
            corners = np.array(surface.get("corners", []))
            if corners.size > 0:
                highest_z_max = max(highest_z_max, corners[:, 2].max())

        # Only apply clipping if current surface is NOT the highest surface.
        z_tolerance = 0.01  # 1cm tolerance for floating point comparison.
        if current_z_max >= highest_z_max - z_tolerance:
            # Current surface is the highest (or tied for highest), no clipping needed.
            console_logger.debug(
                f"Surface at z={current_z_max:.3f} is highest, no clipping needed"
            )
            return

        # Calculate clip height: just above the current surface.
        clip_z = current_z_max + LOWER_SURFACE_CLIP_OFFSET_M

        # Camera is at bbox_center + direction * camera_distance.
        # For top-down view, direction is (0, 0, 1), so camera_z = bbox_center_z + camera_distance.
        # bbox_center_z is approximately current_z_max (center of surface bounding box).
        # Using surface_z_max as approximation for camera target height.
        camera_z = current_z_max + camera_distance

        # Calculate clip_start to clip at clip_z.
        # Near plane clips objects at z > camera_z - clip_start.
        clip_start = camera_z - clip_z

        # Ensure clip_start is positive and reasonable.
        if clip_start < DEFAULT_CAMERA_CLIP_START:
            console_logger.warning(
                f"Calculated clip_start={clip_start:.3f} too small, using default"
            )
            clip_start = DEFAULT_CAMERA_CLIP_START

        # Apply clipping to camera.
        camera_obj.data.clip_start = clip_start
        console_logger.info(
            f"Applied lower surface clipping: clip_start={clip_start:.3f}m "
            f"(clips at z>{clip_z:.3f}m, surface z_max={current_z_max:.3f}m)"
        )

    def _apply_view_annotations(
        self,
        view: dict[str, str | Vector | bool],
        scene_objects: list[dict] | None,
        annotations: DictConfig,
    ) -> None:
        """Apply wall hiding and Blender 3D annotations for a view.

        Args:
            view: Dictionary containing view information.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Annotation config flags.
        """
        # Apply partial wall hiding if enabled.
        if annotations.enable_partial_walls:
            direction = view["direction"].normalized()
            is_top_view = not view.get("is_side", True)
            all_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
            wall_objects = [obj for obj in all_meshes if looks_like_wall(obj)]

            # Hide walls that should be hidden to not occlude the view.
            # Note: 'direction' is from center to camera, so we need to invert it
            # to get the camera viewing direction (from camera to center).
            camera_viewing_dir = -direction

            walls_hidden = 0
            for obj in wall_objects:
                should_hide = should_hide_wall(
                    obj=obj,
                    camera_direction=camera_viewing_dir,
                    is_top_view=is_top_view,
                    wall_normals=self._wall_normals,
                )

                if should_hide:
                    obj.hide_render = True
                    obj.hide_viewport = True
                    walls_hidden += 1

            # Force view layer update after visibility changes.
            if walls_hidden > 0:
                bpy.context.view_layer.update()

        # Add Blender 3D annotation objects before rendering (only for top views).
        # Skip for wall_orthographic views - they use PIL grid annotations instead.
        is_top_view = not view.get("is_side", True)
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        annotations_enabled = any(
            [
                annotations.enable_set_of_mark_labels,
                annotations.enable_bounding_boxes,
                annotations.enable_direction_arrows,
            ]
        )
        console_logger.info(
            f"Annotation check: is_top_view={is_top_view}, "
            f"is_wall_orthographic={is_wall_orthographic}, "
            f"scene_objects={len(scene_objects) if scene_objects else 0}, "
            f"annotations_enabled={annotations_enabled}, annotations={annotations}"
        )
        # Skip Blender 3D annotations for wall orthographic views.
        if (
            is_top_view
            and scene_objects
            and annotations_enabled
            and not is_wall_orthographic
        ):
            # Filter scene_objects by current surface if in per-surface mode.
            filtered_scene_objects = scene_objects
            if hasattr(self, "_current_surface_id") and self._current_surface_id:
                console_logger.info(
                    f"Filtering scene_objects for surface {self._current_surface_id}"
                )
                filtered_scene_objects = self._filter_objects_by_surface(
                    scene_objects=scene_objects,
                    current_surface_id=self._current_surface_id,
                )
                console_logger.info(
                    f"Filtered from {len(scene_objects)} to "
                    f"{len(filtered_scene_objects)} objects"
                )

            try:
                add_blender_scene_annotations(
                    scene_objects=filtered_scene_objects, annotations=annotations
                )
                console_logger.info("Successfully added Blender annotations")
            except Exception as e:
                console_logger.error(
                    f"Failed to add Blender annotations: {e}", exc_info=True
                )
        else:
            console_logger.info(
                f"Skipping annotations: is_top_view={is_top_view}, "
                f"scene_objects_count={len(scene_objects) if scene_objects else 0}, "
                f"annotations_enabled={annotations_enabled}"
            )

    def _render_and_postprocess_view(
        self,
        view: dict[str, str | Vector | bool],
        output_path: Path,
        setup_data: OverlayRenderingSetup,
        scene_objects: list[dict] | None,
        annotations: DictConfig,
        openings: list[ClearanceOpeningData] | None = None,
    ) -> None:
        """Render view and apply PIL post-processing annotations.

        Args:
            view: Dictionary containing view information.
            output_path: Path where rendered image will be saved.
            setup_data: Metric rendering setup data.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Annotation config flags.
            openings: Optional opening metadata for door/window labels.

        Raises:
            RuntimeError: If rendering or post-processing fails.
        """
        # Render to output path.
        scene = bpy.context.scene
        scene.render.filepath = str(output_path)

        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            raise RuntimeError(f"Blender render failed for {view['name']}: {e}")

        if not output_path.exists():
            raise RuntimeError(f"Render failed to create file: {output_path}")

        # Add coordinate annotations (PIL post-processing for metric markers).
        is_top_view = not view.get("is_side", True)
        is_multi_surface_mode = (
            self._support_surfaces is not None and len(self._support_surfaces) > 1
        )

        # Check if this is a wall orthographic view (has its own grid annotations).
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        is_wall_context = view.get("is_wall_context", False)

        # For multi-surface side views, skip coordinate markers and add surface labels.
        if not is_top_view and is_multi_surface_mode:
            try:
                add_surface_labels_to_side_view(
                    image_path=output_path,
                    camera_obj=setup_data.camera_obj,
                    support_surfaces=self._support_surfaces,
                    surface_colors=self._surface_colors,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add surface labels {view['name']}: {e}"
                ) from e
        elif is_wall_orthographic:
            # Wall orthographic views use their own grid annotation system.
            # Skip regular room coordinate markers.
            pass
        elif is_wall_context:
            # Wall context views show furniture for context.
            # Skip red floor coordinate grid - it's not useful for wall placement.
            pass
        elif not getattr(annotations, "enable_coordinate_grid", True):
            # Skip coordinate grid when disabled (e.g., furniture_selection mode).
            pass
        else:
            # Regular coordinate markers for single-surface or top views.
            try:
                is_drawer_view = view.get("is_drawer_view", False)
                # Pass ceiling_height and room_bounds for stable grid markers.
                view_ceiling_height = view.get("ceiling_height", None)
                view_room_bounds = view.get("room_bounds", None)
                marks = self._get_visual_marks(
                    scene=scene,
                    camera_obj=setup_data.camera_obj,
                    is_top_view=is_top_view,
                    is_drawer_view=is_drawer_view,
                    ceiling_height=view_ceiling_height,
                    room_bounds=view_room_bounds,
                )
                if marks:
                    annotate_image_with_coordinates(image_path=output_path, marks=marks)
                # Debug: Visualize convex hull outline for multi-surface mode.
                if (
                    is_multi_surface_mode
                    and is_top_view
                    and getattr(annotations, "enable_convex_hull_debug", False)
                ):
                    self._debug_visualize_convex_hull(
                        image_path=output_path, camera_obj=setup_data.camera_obj
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to annotate {view['name']}: {e}") from e

        # Add set-of-mark labels (PIL post-processing for guaranteed top layer).
        # Include wall_context views for labels (is_wall_context flag is on view).
        is_wall_context = view.get("is_wall_context", False)
        should_add_labels = (is_top_view or is_wall_context) and scene_objects
        if should_add_labels and annotations.enable_set_of_mark_labels:
            try:
                # Extract rendering mode from annotations.
                rendering_mode = getattr(annotations, "rendering_mode", "furniture")
                # For per-surface views, filter labels by current surface.
                current_surface_id = (
                    self._current_surface_id
                    if hasattr(self, "_current_surface_id")
                    else None
                )
                # Get annotate_object_types filter if specified.
                annotate_object_types = getattr(
                    annotations, "annotate_object_types", None
                )
                # Filter scene_objects by current surface for per-surface top views.
                filtered_scene_objects = scene_objects
                if current_surface_id is not None:
                    filtered_scene_objects = self._filter_objects_by_surface(
                        scene_objects=scene_objects,
                        current_surface_id=current_surface_id,
                    )
                add_set_of_mark_labels_pil(
                    image_path=output_path,
                    scene_objects=filtered_scene_objects,
                    camera_obj=setup_data.camera_obj,
                    rendering_mode=rendering_mode,
                    current_surface_id=current_surface_id,
                    annotate_object_types=annotate_object_types,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add set-of-mark labels {view['name']}: {e}"
                ) from e

        # Add wall labels to wall_context top-down views.
        if is_wall_context and hasattr(self, "_wall_surfaces_for_labels"):
            wall_surfaces = self._wall_surfaces_for_labels
            if wall_surfaces:
                try:
                    add_wall_labels_to_top_view(
                        image_path=output_path,
                        camera_obj=setup_data.camera_obj,
                        wall_surfaces=wall_surfaces,
                    )
                except Exception as e:
                    console_logger.warning(
                        f"Failed to add wall labels for {view['name']}: {e}"
                    )

        # Add opening labels (door/window/open connection) for top views.
        # Skip for wall views - only show wall labels, not openings.
        is_wall_view = is_wall_context or is_wall_orthographic
        if is_top_view and openings and not is_wall_view:
            try:
                add_opening_labels_pil(
                    image_path=output_path,
                    openings=openings,
                    camera_obj=setup_data.camera_obj,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add opening labels {view['name']}: {e}"
                ) from e

        # Add surface ID label for multi-surface top views.
        if (
            is_top_view
            and hasattr(self, "_current_surface_id")
            and self._current_surface_id
        ):
            try:
                add_surface_id_label(
                    image_path=output_path,
                    surface_id=self._current_surface_id,
                    surface_colors=self._surface_colors,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add surface ID label {view['name']}: {e}"
                ) from e

        # Add wall orthographic annotations (grid and excluded regions).
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        if is_wall_orthographic:
            wall_surface = view.get("wall_surface", {})
            if wall_surface:
                try:
                    # Add coordinate grid overlay.
                    if getattr(annotations, "enable_wall_grid", True):
                        add_wall_grid_annotations_pil(
                            image_path=output_path,
                            wall_surface_data=wall_surface,
                            camera_obj=setup_data.camera_obj,
                            num_markers=getattr(annotations, "num_markers", 5),
                        )
                    surface_id = wall_surface.get(
                        "surface_id", wall_surface.get("wall_id", "")
                    )
                    if surface_id:
                        add_wall_surface_id_label(
                            image_path=output_path, wall_surface_id=surface_id
                        )
                except Exception as e:
                    console_logger.warning(
                        f"Failed to add wall annotations for {view['name']}: {e}"
                    )

        # Clean up overlays for next view.
        remove_coordinate_frame()
        remove_wall_coordinate_frame()
        remove_annotation_objects()

    def _render_single_view_with_metric_overlay_to_path(
        self,
        view: dict[str, str | Vector | bool],
        setup_data: OverlayRenderingSetup,
        output_path: Path,
        scene_objects: list[dict] | None = None,
        annotations: dict | None = None,
        openings: list[ClearanceOpeningData] | None = None,
    ) -> None:
        """Render a single view with metric overlays to specified path.

        Args:
            view: Dictionary containing view information.
            setup_data: Metric rendering setup data.
            output_path: Path where rendered image will be saved.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Optional annotation config flags.
            openings: Optional opening metadata for door/window labels.

        Raises:
            RuntimeError: If Blender rendering fails.
        """
        self._setup_camera_and_coordinate_frame(
            setup_data=setup_data, view=view, annotations=annotations
        )
        self._apply_view_annotations(
            view=view,
            scene_objects=scene_objects,
            annotations=annotations,
        )
        self._render_and_postprocess_view(
            view=view,
            output_path=output_path,
            setup_data=setup_data,
            scene_objects=scene_objects,
            annotations=annotations,
            openings=openings,
        )

    def _setup_overlay_rendering(
        self, params: RenderParams, view_size: int | None, margin_scale: float = 1.8
    ) -> OverlayRenderingSetup:
        """Setup scene for metric rendering and return rendering data.

        Args:
            params: Rendering parameters with scene path and camera settings.
            view_size: Optional view size for resolution settings.
            margin_scale: Camera distance margin scale factor. Default 1.8 (80% margin).
                Use 1.30 for 30% margin on side views.

        Returns:
            OverlayRenderingSetup containing camera, bbox, and distance data.
        """
        grid_size = params.width
        console_logger.debug(
            f"Grid size: {grid_size}, Individual view size: {view_size}"
        )

        self._setup_scene(params)
        self._import_and_organize_gltf(params.scene)

        bbox_center, max_dim = compute_scene_bounds(self._client_objects)
        camera_obj = configure_metric_camera(params=params)
        apply_render_settings(params=params, view_size=view_size)
        setup_metric_world()

        # Additional metric-specific settings.
        scene = bpy.context.scene
        _set_eevee_render_engine(scene)
        scene.render.film_transparent = True  # Enable alpha channel.
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.render.resolution_percentage = 100

        # EEVEE performance optimization settings.
        # Note: Some settings may not be available in EEVEE_NEXT (Blender 4.5+).
        # We set them conditionally to handle API changes gracefully.
        # TAA samples can be configured via _taa_samples attribute (default 16).
        taa_samples = getattr(self, "_taa_samples", 16)
        try:
            scene.eevee.taa_render_samples = taa_samples
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_gtao = False  # Disable ambient occlusion.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_bloom = False  # Disable bloom.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_ssr = False  # Disable screen space reflections.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_volumetric_shadows = False  # Disable volumetric shadows.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.shadow_cube_size = "128"  # Reduce from default 1024.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.shadow_cascade_size = "128"
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_shadows = False
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        for light in bpy.data.lights:
            try:
                light.use_shadow = False
            except AttributeError:
                pass  # Not available in EEVEE_NEXT.

        camera_distance = calculate_camera_distance(
            camera_obj=camera_obj, max_dim=max_dim, margin_scale=margin_scale
        )

        return OverlayRenderingSetup(
            camera_obj=camera_obj,
            bbox_center=bbox_center,
            max_dim=max_dim,
            camera_distance=camera_distance,
        )

    def _reset_rendering_state(self) -> None:
        """Reset instance variables to prevent state leakage between renders."""
        self._support_surfaces = None
        self._surface_colors = {}
        self._wall_normals = {}
        self._show_support_surface = False
        self._scene_objects = None
        self._current_convex_hull = None
        self._current_surface_id = None
        self._overlay_mesh_objects = []
        self._surface_mesh_objects = []
        self._hidden_objects = []
        self._debug_camera_objects = []

    def _add_wall_camera_debug_cones(self, wall_surfaces: list[dict]) -> None:
        """Add debug cones showing camera positions and directions for wall views.

        Creates colored cones at each camera position, pointing toward wall centers.
        Colors: North=Red, South=Green, East=Blue, West=Yellow.

        Args:
            wall_surfaces: List of wall surface dicts with direction, length, height,
            transform.
        """
        colors = {
            "north": (1.0, 0.0, 0.0, 1.0),  # Red
            "south": (0.0, 1.0, 0.0, 1.0),  # Green
            "east": (0.0, 0.0, 1.0, 1.0),  # Blue
            "west": (1.0, 1.0, 0.0, 1.0),  # Yellow
        }

        for wall_surface in wall_surfaces:
            wall_direction = wall_surface.get("direction", "north").lower()
            wall_length = wall_surface.get("length", 5.0)
            wall_height = wall_surface.get("height", 2.5)
            transform = wall_surface.get("transform", [0, 0, 0, 1, 0, 0, 0])

            # Parse transform.
            wall_origin = np.array([transform[0], transform[1], transform[2]])
            qw, qx, qy, qz = transform[3], transform[4], transform[5], transform[6]

            # Build rotation matrix from quaternion.
            rotation_matrix = np.array(
                [
                    [
                        1 - 2 * (qy**2 + qz**2),
                        2 * (qx * qy - qw * qz),
                        2 * (qx * qz + qw * qy),
                    ],
                    [
                        2 * (qx * qy + qw * qz),
                        1 - 2 * (qx**2 + qz**2),
                        2 * (qy * qz - qw * qx),
                    ],
                    [
                        2 * (qx * qz - qw * qy),
                        2 * (qy * qz + qw * qx),
                        1 - 2 * (qx**2 + qy**2),
                    ],
                ]
            )

            # Compute wall center.
            wall_center_local = np.array([wall_length / 2, 0, wall_height / 2])
            wall_center_world = wall_origin + rotation_matrix @ wall_center_local

            # Compute camera position.
            camera_offset = 3.0
            if wall_direction == "north":
                camera_pos = np.array(
                    [
                        wall_center_world[0],
                        wall_center_world[1] - camera_offset,
                        wall_center_world[2],
                    ]
                )
            elif wall_direction == "south":
                camera_pos = np.array(
                    [
                        wall_center_world[0],
                        wall_center_world[1] + camera_offset,
                        wall_center_world[2],
                    ]
                )
            elif wall_direction == "east":
                camera_pos = np.array(
                    [
                        wall_center_world[0] - camera_offset,
                        wall_center_world[1],
                        wall_center_world[2],
                    ]
                )
            elif wall_direction == "west":
                camera_pos = np.array(
                    [
                        wall_center_world[0] + camera_offset,
                        wall_center_world[1],
                        wall_center_world[2],
                    ]
                )
            else:
                continue

            # Log positions for debugging.
            console_logger.info(
                f"DEBUG CONE {wall_direction}: wall_origin={wall_origin}, "
                f"wall_center={wall_center_world}, camera_pos={camera_pos}"
            )

            # Create cone mesh.
            bpy.ops.mesh.primitive_cone_add(
                radius1=0.15,
                radius2=0.0,
                depth=0.4,
                location=(camera_pos[0], camera_pos[1], camera_pos[2]),
            )
            cone = bpy.context.active_object
            cone.name = f"debug_camera_{wall_direction}"

            # Point cone toward wall center.
            direction = Vector(wall_center_world.tolist()) - Vector(camera_pos.tolist())
            if direction.length > 0:
                direction.normalize()
                quat = direction.to_track_quat("-Z", "Z")
                cone.rotation_euler = quat.to_euler()

            # Apply color.
            mat = bpy.data.materials.new(name=f"debug_mat_{wall_direction}")
            mat.use_nodes = False
            mat.diffuse_color = colors.get(wall_direction, (1.0, 1.0, 1.0, 1.0))
            cone.data.materials.append(mat)

            self._debug_camera_objects.append(cone)

    def _remove_wall_camera_debug_cones(self) -> None:
        """Remove debug camera cone objects."""
        for obj in self._debug_camera_objects:
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        self._debug_camera_objects = []

    def _setup_wall_orthographic_camera(
        self,
        camera_obj: bpy.types.Object,
        wall_surface: dict,
        margin_factor: float = 1.1,
    ) -> None:
        """Configure orthographic camera facing wall center.

        Sets up an orthographic camera perpendicular to the wall surface,
        positioned inside the room looking at the wall. Uses wall direction
        to compute camera position directly in world coordinates.

        Args:
            camera_obj: Blender camera object to configure.
            wall_surface: Wall surface data dict containing:
                - direction: Wall direction ("north", "south", "east", "west").
                - length: Wall length in meters.
                - height: Wall height in meters.
                - transform: [x, y, z, qw, qx, qy, qz] pose in world frame.
            direction: Camera viewing direction from view generator.
            margin_factor: Scale factor for orthographic view (1.0 = exact fit).
        """
        # Fail fast if required fields are missing (research codebase principle).
        if "direction" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'direction' field. Got: {wall_surface}"
            )
        if "transform" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'transform' field. Got: {wall_surface}"
            )
        if "length" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'length' field. Got: {wall_surface}"
            )
        if "height" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'height' field. Got: {wall_surface}"
            )

        wall_length = wall_surface["length"]
        wall_height = wall_surface["height"]
        wall_direction = wall_surface["direction"]
        transform = wall_surface["transform"]
        wall_id = wall_surface.get("wall_id", "unknown")

        # Debug: log wall data for each wall.
        console_logger.info(
            f"Wall camera setup for {wall_id}:\n"
            f"  direction={wall_direction}\n"
            f"  length={wall_length}, height={wall_height}\n"
            f"  transform={transform}"
        )

        # Set camera to orthographic mode.
        camera_obj.data.type = "ORTHO"

        # Orthographic scale = max dimension * margin to fit entire wall.
        camera_obj.data.ortho_scale = max(wall_length, wall_height) * margin_factor

        # Compute wall center from transform.
        wall_center_world = _compute_wall_center_from_transform(
            transform=transform, wall_length=wall_length, wall_height=wall_height
        )
        wall_origin = np.array(transform[:3])
        wall_dir = wall_direction.lower()

        # Camera offset from wall center (inside room, looking at wall).
        # Use room_depth if available to avoid placing camera outside small rooms.
        room_depth = wall_surface.get("room_depth")
        if room_depth is not None and room_depth > 0:
            # Position camera at 80% of room depth, capped at 3m.
            # This ensures camera stays inside the room with some margin.
            camera_offset = min(room_depth * 0.8, 3.0)
        else:
            camera_offset = 3.0  # Default distance from wall center.

        # Compute camera position and look direction based on wall direction.
        if wall_dir == "north":
            camera_pos = np.array(
                [
                    wall_center_world[0],
                    wall_center_world[1] - camera_offset,
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((0, 1, 0))
        elif wall_dir == "south":
            camera_pos = np.array(
                [
                    wall_center_world[0],
                    wall_center_world[1] + camera_offset,
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((0, -1, 0))
        elif wall_dir == "east":
            camera_pos = np.array(
                [
                    wall_center_world[0] - camera_offset,
                    wall_center_world[1],
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((1, 0, 0))
        elif wall_dir == "west":
            camera_pos = np.array(
                [
                    wall_center_world[0] + camera_offset,
                    wall_center_world[1],
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((-1, 0, 0))
        else:
            camera_pos = np.array(
                [
                    wall_center_world[0],
                    wall_center_world[1] - camera_offset,
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((0, 1, 0))

        # Debug logging.
        console_logger.info(
            f"Wall orthographic camera setup for {wall_dir}:\n"
            f"  wall_origin: {wall_origin}\n"
            f"  wall_center_world: {wall_center_world}\n"
            f"  camera_pos: {camera_pos}\n"
            f"  look_dir: {look_dir}"
        )

        # Set camera position.
        camera_obj.location = Vector(camera_pos.tolist())

        # Set camera rotation explicitly to ensure walls appear horizontal.
        # All walls should appear with width horizontal and height vertical.
        # rotation_euler.x = 90° points camera horizontally (from looking down).
        # rotation_euler.z controls which compass direction camera faces.
        if wall_dir == "north":
            # Looking toward +Y. (pi/2, 0, 0) gives forward = +Y.
            camera_obj.rotation_euler = (math.pi / 2, 0, 0)
        elif wall_dir == "south":
            # Looking toward -Y. (pi/2, 0, pi) gives forward = -Y.
            camera_obj.rotation_euler = (math.pi / 2, 0, math.pi)
        elif wall_dir == "east":
            # Looking toward +X.
            camera_obj.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
        elif wall_dir == "west":
            # Looking toward -X.
            camera_obj.rotation_euler = (math.pi / 2, 0, math.pi / 2)
        else:
            # Fallback to north.
            camera_obj.rotation_euler = (math.pi / 2, 0, math.pi)

    def save_blend_file(self, params: RenderParams, output_path: Path) -> Path:
        """Save the scene as a .blend file.

        Args:
            params: Rendering parameters containing scene path (glTF).
            output_path: Path where .blend file will be saved.

        Returns:
            Path to the saved .blend file.
        """
        # Setup scene and import glTF (reuse existing methods).
        self._setup_scene(params)
        self._import_and_organize_gltf(params.scene)

        # Ensure output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save blend file.
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        console_logger.info(f"Saved .blend file to {output_path}")
        return output_path
