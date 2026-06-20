import atexit
import copy
import logging
import math
import os
import sys
import tempfile
import time

from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

from omegaconf import DictConfig
from pydrake.all import (
    ApplyCameraConfig,
    CameraConfig,
    DiagramBuilder,
    RenderEngineGltfClientParams,
    RenderEngineVtkParams,
    Rgba,
    RigidTransform,
    Transform,
)

from scenesmith.agent_utils.blender import BlenderServer
from scenesmith.agent_utils.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
    create_plant_from_dmd,
    get_all_link_transforms,
    get_closed_position,
    get_joint_limits,
    get_open_position,
    parse_joint_child_links,
    set_articulated_joints_to_max,
    set_joints_to_config,
)
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    extract_base_link_name_from_sdf,
)
from scenesmith.utils.geometry_utils import compute_aabb_corners

console_logger = logging.getLogger(__name__)


# Track virtual display for cleanup on process exit.
_virtual_display = None


def _cleanup_virtual_display() -> None:
    """Clean up virtual display on process exit."""
    global _virtual_display
    if _virtual_display is not None:
        try:
            _virtual_display.stop()
        except Exception:
            pass  # Best effort cleanup.
        _virtual_display = None


def setup_virtual_display_if_needed() -> None:
    """Set up a virtual display for headless rendering if needed.

    On Linux without a DISPLAY, creates a virtual display. The DISPLAY env var
    check prevents duplicate creation within a process. Registers atexit handler
    to clean up Xvfb on process exit.
    """
    global _virtual_display
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        console_logger.info("Setting up virtual display for rendering.")
        from pyvirtualdisplay import Display

        _virtual_display = Display(visible=0, size=(1400, 900))
        _virtual_display.start()
        atexit.register(_cleanup_virtual_display)


def get_drake_model_name(obj: SceneObject) -> str:
    """Compute Drake model name matching to_drake_directive logic in scene.py.

    Drake model names are created by combining the object name (lowercased,
    spaces replaced with underscores) with a suffix derived from the object ID.
    """
    id_suffix = str(obj.object_id).split("_")[-1][:8]
    return f"{obj.name.lower().replace(' ', '_')}_{id_suffix}"


def apply_fk_to_surfaces(
    surfaces: list[SupportSurface],
    rest_transforms: dict[str, RigidTransform],
    open_transforms: dict[str, RigidTransform],
    link_to_joint: dict[str, str],
    open_joints: set[str],
) -> list[SupportSurface]:
    """Apply FK delta transforms to surfaces based on which joints are open.

    For articulated furniture, surfaces are extracted at rest (closed) position.
    When rendering with specific joints open, surfaces on those links need FK
    transforms to match the opened geometry.

    Args:
        surfaces: Support surfaces with link_name populated.
        rest_transforms: Link transforms at closed joint positions.
        open_transforms: Link transforms at current joint positions.
        link_to_joint: Mapping from child link name to controlling joint name.
        open_joints: Set of joint names that are open in this render.

    Returns:
        New list of surfaces with FK transforms applied where applicable.
        Surfaces on closed joints (or base link) keep their original transforms.
    """
    result = []

    for surface in surfaces:
        link_name = surface.link_name

        # No link association = base link = no FK needed.
        if not link_name or link_name not in link_to_joint:
            result.append(surface)
            continue

        joint_name = link_to_joint.get(link_name)

        # Only transform if this joint is in the open set.
        if joint_name and joint_name in open_joints:
            # Check we have transforms for this link.
            if link_name in rest_transforms and link_name in open_transforms:
                # Compute delta: open @ rest.inverse().
                rest_tf = rest_transforms[link_name]
                open_tf = open_transforms[link_name]
                delta = open_tf @ rest_tf.inverse()

                # Create new surface with transformed pose.
                new_surface = copy.copy(surface)
                new_surface.transform = delta @ surface.transform
                result.append(new_surface)
                console_logger.debug(
                    f"Applied FK to surface {surface.surface_id} on link {link_name}"
                )
            else:
                # Missing transform data - keep original.
                console_logger.warning(
                    f"Missing transform for link {link_name}, keeping original surface"
                )
                result.append(surface)
        else:
            # Joint not open - keep at rest position.
            result.append(surface)

    return result


def classify_surfaces_for_rendering(
    surfaces: list[SupportSurface], link_to_joint: dict[str, str]
) -> tuple[list[SupportSurface], dict[str, list[SupportSurface]]]:
    """Classify surfaces for per-joint rendering strategy.

    Surfaces are classified into:
    - Static surfaces: On base link (not controlled by any joint)
    - Per-joint surfaces: On child links (need per-joint renders)

    For articulated furniture like nightstands with drawers:
    - Body surfaces → static (main render)
    - Drawer surfaces → per-joint (each drawer gets its own render)

    Args:
        surfaces: Support surfaces with link_name populated.
        link_to_joint: Mapping from child link name to controlling joint name.

    Returns:
        Tuple of (static_surfaces, joint_surfaces) where joint_surfaces is a
        dict mapping joint_name to list of surfaces controlled by that joint.
    """
    static_surfaces: list[SupportSurface] = []
    joint_surfaces: dict[str, list[SupportSurface]] = defaultdict(list)

    for surface in surfaces:
        link_name = surface.link_name

        # No link association or base link = static surface.
        if not link_name or link_name not in link_to_joint:
            static_surfaces.append(surface)
            continue

        # Surface on a moving link = per-joint render.
        joint_name = link_to_joint[link_name]
        joint_surfaces[joint_name].append(surface)

    console_logger.debug(
        f"Classified {len(surfaces)} surfaces: {len(static_surfaces)} static, "
        f"{len(joint_surfaces)} joints with surfaces"
    )

    return static_surfaces, dict(joint_surfaces)


def compute_drawer_direction(
    rest_transform: RigidTransform, open_transform: RigidTransform
) -> list[float]:
    """Compute the direction a drawer moves when opening.

    Uses FK delta (open vs rest) to determine drawer sliding direction.
    This is used to position the camera to look into the drawer opening.

    Args:
        rest_transform: Link transform at rest (closed) position.
        open_transform: Link transform at open position.

    Returns:
        3D direction vector [x, y, z] of drawer movement (not normalized).
    """
    # FK delta = open @ rest.inverse()
    delta = open_transform @ rest_transform.inverse()
    # Translation component tells us how far and which way the drawer moved.
    translation = delta.translation()
    return translation.tolist()


def _generate_angled_drawer_view(
    surface: dict,
    joint_name: str,
    drawer_direction: list[float] | None = None,
    view_index: int = 0,
) -> dict:
    """Generate a drawer-inspection view without importing Blender-only modules."""
    surface_id = surface.get("surface_id", f"surface_{view_index}")

    if drawer_direction is not None:
        dx, dy, _ = drawer_direction
        horiz_mag = math.sqrt(dx * dx + dy * dy)

        if horiz_mag > 0.01:
            horiz_scale = 0.7
            direction = [
                (dx / horiz_mag) * horiz_scale,
                (dy / horiz_mag) * horiz_scale,
                0.7,
            ]
            norm = math.sqrt(sum(component * component for component in direction))
            direction = [component / norm for component in direction]
        else:
            direction = [0.0, 0.7071067811865476, 0.7071067811865476]
    else:
        direction = [0.0, 0.7071067811865476, 0.7071067811865476]

    return {
        "name": f"drawer_{joint_name}_{surface_id}",
        "direction": direction,
        "is_side": False,
        "surface_data": surface,
        "is_drawer_view": True,
    }


def build_support_surfaces_data(surfaces: list[SupportSurface]) -> list[dict]:
    """Build serializable surface data for Blender overlay config.

    Args:
        surfaces: List of support surfaces (possibly FK-transformed).

    Returns:
        List of dicts with surface_id, corners, convex_hull_vertices, mesh_faces.
    """
    support_surfaces_data = []

    for surface in surfaces:
        # Compute 8 corners of the bounding box in local space.
        corners_local = compute_aabb_corners(
            bbox_min=surface.bounding_box_min,
            bbox_max=surface.bounding_box_max,
        )

        # Transform all corners to world space using Drake Z-up coordinates.
        corners_world = [
            (surface.transform @ corner).tolist() for corner in corners_local
        ]

        # Get convex hull vertices for coordinate marker filtering.
        convex_hull_vertices = None
        mesh_faces = None
        if surface.mesh is not None:
            # Get mesh vertices in world space (mesh-local -> surface -> world).
            mesh_vertices_local = surface.mesh.vertices
            transform_matrix = surface.transform.GetAsMatrix4()
            mesh_vertices_world = []
            for v in mesh_vertices_local:
                v_hom = np.append(v, 1.0)
                v_world_hom = transform_matrix @ v_hom
                mesh_vertices_world.append(v_world_hom[:3].tolist())
            convex_hull_vertices = mesh_vertices_world
            # Include face indices for rendering the surface mesh.
            mesh_faces = surface.mesh.faces.tolist()

        surface_data = {
            "surface_id": str(surface.surface_id),
            "corners": corners_world,
            "convex_hull_vertices": convex_hull_vertices,
            "mesh_faces": mesh_faces,
        }
        support_surfaces_data.append(surface_data)

    return support_surfaces_data


def render_per_drawer_views(
    plant,
    context,
    diagram,
    server: BlenderServer,
    drawer_surfaces: dict[str, list[SupportSurface]],
    all_surfaces: list[SupportSurface],
    scene_objects: list[SceneObject],
    link_to_joint: dict[str, str],
    rest_transforms: dict[str, RigidTransform],
    config_payload: dict,
    output_dir: Path,
    cfg: DictConfig,
) -> list[Path]:
    """Render per-drawer angled views with only one drawer open at a time.

    For each drawer joint, resets to rest position, opens only that drawer,
    computes FK transforms, and renders an angled view looking into the drawer.
    Manipulands placed on drawer surfaces are also FK-transformed to move with
    the opened drawer.

    Args:
        plant: Drake MultibodyPlant (already finalized).
        context: Plant context to modify joint positions.
        diagram: Built Drake diagram for evaluation.
        server: Running Blender server.
        drawer_surfaces: Mapping from joint_name to surfaces controlled by that joint.
        all_surfaces: All support surfaces (for surface_id to link_name lookup).
        scene_objects: All scene objects (to find manipulands for FK transform).
        link_to_joint: Mapping from link name to joint name.
        rest_transforms: Link transforms at rest (closed) position.
        config_payload: Base config payload for Blender (will be modified per drawer).
        output_dir: Directory for output images.
        cfg: Rendering config.

    Returns:
        List of paths to rendered drawer view images.
    """
    if not drawer_surfaces:
        return []

    drawer_images = []
    joint_limits = get_joint_limits(plant)

    # Build reverse mapping: joint_name -> link_name.
    joint_to_link = {v: k for k, v in link_to_joint.items()}

    # Build surface_id -> link_name lookup for manipuland FK transforms.
    surface_to_link = {str(s.surface_id): s.link_name for s in all_surfaces}

    config_url = f"{server.get_url()}/set_overlay_config"

    for joint_name, surfaces in drawer_surfaces.items():
        if not surfaces:
            continue

        console_logger.info(f"Rendering per-drawer view for joint: {joint_name}")

        # 1. Reset all joints to rest (closed) position.
        closed_config = {}
        for jname, (lower, upper) in joint_limits.items():
            closed_config[jname] = get_closed_position(lower, upper)
        set_joints_to_config(plant, context, closed_config)

        # 2. Open only this drawer.
        if joint_name in joint_limits:
            lower, upper = joint_limits[joint_name]
            open_pos = get_open_position(lower, upper)
            set_joints_to_config(plant, context, {joint_name: open_pos})

        # 3. Get transforms at this configuration.
        current_transforms = get_all_link_transforms(plant, context)

        # 4. Compute drawer direction from FK delta.
        link_name = joint_to_link.get(joint_name)
        drawer_direction = None
        if (
            link_name
            and link_name in rest_transforms
            and link_name in current_transforms
        ):
            drawer_direction = compute_drawer_direction(
                rest_transforms[link_name],
                current_transforms[link_name],
            )
            console_logger.debug(f"Drawer {joint_name} direction: {drawer_direction}")

        # 5. Apply FK transform to this drawer's surfaces only.
        transformed_surfaces = apply_fk_to_surfaces(
            surfaces=surfaces,
            rest_transforms=rest_transforms,
            open_transforms=current_transforms,
            link_to_joint=link_to_joint,
            open_joints={joint_name},
        )

        # 6. Build surface data for this drawer.
        surfaces_data = build_support_surfaces_data(transformed_surfaces)

        # 7. Generate angled view configuration.
        if surfaces_data:
            view = _generate_angled_drawer_view(
                surface=surfaces_data[0],
                joint_name=joint_name,
                drawer_direction=drawer_direction,
            )

            # 8. Build drawer-specific config.
            drawer_config = copy.deepcopy(config_payload)
            drawer_config["support_surfaces"] = surfaces_data

            # Remove context furniture for drawer views - we only want to see the drawer
            # interior, not nearby furniture like beds that would affect camera framing.
            drawer_config.pop("context_furniture_ids", None)

            # 8a. Apply FK transform to scene_objects metadata for manipulands
            # on this drawer. This ensures bounding boxes and labels move with
            # the drawer in the rendered overlay.
            if (
                link_name
                and link_name in rest_transforms
                and link_name in current_transforms
            ):
                delta = (
                    current_transforms[link_name] @ rest_transforms[link_name].inverse()
                )
                for obj_meta in drawer_config.get("scene_objects", []):
                    parent_surface_id = obj_meta.get("parent_surface_id")
                    if not parent_surface_id:
                        continue
                    # Check if this object is on this drawer's surface.
                    obj_link = surface_to_link.get(parent_surface_id)
                    if not obj_link or link_to_joint.get(obj_link) != joint_name:
                        continue
                    # Apply FK transform to position.
                    old_pos = np.array(obj_meta["position"])
                    new_pos = delta @ old_pos
                    obj_meta["position"] = new_pos.tolist()
                    # Apply FK transform to bounding_box center.
                    if obj_meta.get("bounding_box"):
                        old_center = np.array(obj_meta["bounding_box"]["center"])
                        new_center = delta @ old_center
                        obj_meta["bounding_box"]["center"] = new_center.tolist()

            drawer_config["render_single_view"] = {
                "enabled": True,
                "name": view["name"],
                "direction": list(view["direction"]),
            }

            # 9. Send config to Blender.
            response = requests.post(config_url, json=drawer_config, timeout=10)
            if response.status_code != 200:
                console_logger.warning(
                    f"Failed to set drawer config for {joint_name}: "
                    f"{response.status_code} {response.text}"
                )
                continue

            # 10. Eval diagram (triggers Drake to send glTF with current joint positions).
            # Must use the root context that contains our modified plant context.
            root_context = diagram.CreateDefaultContext()
            plant_context = plant.GetMyContextFromRoot(root_context)
            # Re-apply joint configuration to the new context.
            set_joints_to_config(
                plant=plant, context=plant_context, joint_config=closed_config
            )
            set_joints_to_config(
                plant=plant, context=plant_context, joint_config={joint_name: open_pos}
            )

            # 10a. Apply FK transforms to manipulands on this drawer.
            # Objects placed on drawer surfaces at REST stay at REST world positions
            # unless we move them. Compute FK delta and set new poses.
            for obj in scene_objects:
                if obj.object_type != ObjectType.MANIPULAND:
                    continue  # Only transform free bodies (manipulands).
                if obj.placement_info is None:
                    continue

                # Get parent surface ID from placement info.
                parent_surface_id = obj.placement_info.parent_surface_id
                if not parent_surface_id:
                    continue

                # Find which link this surface belongs to.
                obj_link_name = surface_to_link.get(str(parent_surface_id))
                if not obj_link_name:
                    continue

                # Check if this object is on THIS drawer's link.
                obj_joint = link_to_joint.get(obj_link_name)
                if obj_joint != joint_name:
                    continue  # Object not on this drawer.

                # Compute FK delta: open_transform @ rest_transform.inverse().
                if (
                    obj_link_name not in rest_transforms
                    or obj_link_name not in current_transforms
                ):
                    continue
                delta = (
                    current_transforms[obj_link_name]
                    @ rest_transforms[obj_link_name].inverse()
                )
                new_pose = delta @ obj.transform

                # Set new pose in Drake using SetFreeBodyPose.
                try:
                    model_name = get_drake_model_name(obj)
                    base_link_name = extract_base_link_name_from_sdf(obj.sdf_path)
                    model_instance = plant.GetModelInstanceByName(model_name)
                    body = plant.GetBodyByName(base_link_name, model_instance)
                    plant.SetFreeBodyPose(plant_context, body, new_pose)
                    console_logger.debug(
                        f"Applied FK to manipuland {obj.name} on {joint_name}"
                    )
                except Exception as e:
                    console_logger.warning(f"Failed to set FK pose for {obj.name}: {e}")

            # 11. Track existing images before render, then find new image via
            # set difference (same pattern as wall rendering).
            existing_images = set(output_dir.glob("*.png"))

            _ = diagram.GetOutputPort("rgba_image").Eval(root_context)

            current_images = set(output_dir.glob("*.png"))
            new_images = current_images - existing_images

            if new_images:
                new_image = next(iter(new_images))
                drawer_image_name = f"drawer_{joint_name}.png"
                drawer_image_path = output_dir / drawer_image_name
                new_image.rename(drawer_image_path)
                drawer_images.append(drawer_image_path)
                console_logger.info(f"Rendered drawer view: {drawer_image_name}")

    return drawer_images


def render_per_wall_ortho_views(
    scene: "RoomScene",
    server: BlenderServer,
    wall_surfaces: list[dict],
    wall_furniture_map: dict[str, list],
    base_config_payload: dict,
    output_dir: Path,
    cfg: DictConfig,
) -> list[Path]:
    """Render per-wall orthographic views with filtered furniture per wall.

    For each wall, creates a new Drake plant with only furniture near that wall
    and renders an orthographic view facing the wall.

    Args:
        scene: RoomScene containing all objects.
        server: Running Blender server.
        wall_surfaces: List of wall surface dicts with surface_id, wall_id, direction, etc.
        wall_furniture_map: Mapping from surface_id to list of furniture UniqueIDs
            to include in that wall's render.
        base_config_payload: Base config payload for Blender (will be modified per wall).
        output_dir: Directory for output images.
        cfg: Rendering config.

    Returns:
        List of paths to rendered wall orthographic images.
    """
    if not wall_surfaces:
        return []

    wall_images = []
    config_url = f"{server.get_url()}/set_overlay_config"

    for wall_surface in wall_surfaces:
        surface_id = wall_surface.get("surface_id", "unknown")
        wall_id = wall_surface.get("wall_id", "unknown")

        # Get furniture IDs for this wall (keyed by surface_id).
        furniture_ids = wall_furniture_map.get(surface_id, [])

        # Get wall object IDs on this wall.
        wall_object_ids = []
        for obj in scene.objects.values():
            if obj.object_type != ObjectType.WALL_MOUNTED:
                continue
            if obj.placement_info is None:
                continue
            # Check if wall object is on this wall surface (match by surface_id).
            parent_surface_id = str(obj.placement_info.parent_surface_id)
            if parent_surface_id == surface_id:
                wall_object_ids.append(obj.object_id)

        include_objects = furniture_ids + wall_object_ids

        # Build scene objects metadata for this wall's objects only.
        scene_objects_metadata = []
        for obj in scene.objects.values():
            if obj.object_id not in include_objects:
                continue
            translation = obj.transform.translation()
            rotation = obj.transform.rotation()
            rotation_matrix = rotation.matrix().tolist()

            bbox = None
            if obj.bbox_min is not None and obj.bbox_max is not None:
                local_center = (obj.bbox_min + obj.bbox_max) / 2.0
                extents = obj.bbox_max - obj.bbox_min
                world_center = obj.transform @ local_center
                bbox = {"center": world_center.tolist(), "extents": extents.tolist()}

            parent_surface_id = None
            if obj.placement_info:
                parent_surface_id = str(obj.placement_info.parent_surface_id)

            scene_objects_metadata.append(
                {
                    "name": obj.name,
                    "object_id": str(obj.object_id),
                    "object_type": obj.object_type.value,
                    "position": translation.tolist(),
                    "rotation_matrix": rotation_matrix,
                    "bounding_box": bbox,
                    "parent_surface_id": parent_surface_id,
                }
            )

        # Create per-wall config payload with single wall in list.
        wall_config = base_config_payload.copy()
        wall_config["layout"] = "wall_orthographic"
        wall_config["wall_surfaces"] = [wall_surface]
        wall_config["wall_surfaces_for_labels"] = [wall_surface]
        wall_config["scene_objects"] = scene_objects_metadata

        # Set config on Blender server.
        response = requests.post(config_url, json=wall_config, timeout=10)
        if response.status_code != 200:
            console_logger.error(
                f"Failed to set wall config for {wall_id}: {response.text}"
            )
            continue

        # Create new Drake plant with only this wall's objects.
        # Wall rendering always includes room geometry (walls are needed).
        builder = DiagramBuilder()
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
            scene=scene,
            builder=builder,
            include_objects=include_objects,
            exclude_room_geometry=False,
        )

        # Placeholder camera (Blender handles actual camera).
        placeholder_pose = RigidTransform.Identity()
        camera_config = CameraConfig(
            X_PB=Transform(placeholder_pose),
            width=4,
            height=4,
            background=Rgba(
                cfg.background_color[0],
                cfg.background_color[1],
                cfg.background_color[2],
                1.0,
            ),
            renderer_class=RenderEngineGltfClientParams(
                base_url=server.get_url(),
                render_endpoint="render_overlay",
            ),
        )

        ApplyCameraConfig(
            config=camera_config,
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
        )

        builder.ExportOutput(
            builder.GetSubsystemByName(
                f"rgbd_sensor_{camera_config.name}"
            ).color_image_output_port(),
            "rgba_image",
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        # Track existing images before rendering.
        existing_images = set(output_dir.glob("*.png"))

        # Trigger render.
        _ = diagram.GetOutputPort("rgba_image").Eval(context)

        # Find the newly created image by diffing with existing.
        current_images = set(output_dir.glob("*.png"))
        new_images = current_images - existing_images

        if new_images:
            # Should be exactly one new image.
            new_image = next(iter(new_images))
            wall_image_name = f"wall_{wall_id}_ortho.png"
            wall_image_path = output_dir / wall_image_name
            new_image.rename(wall_image_path)
            wall_images.append(wall_image_path)
            console_logger.info(f"Rendered wall ortho view: {wall_image_name}")
        else:
            console_logger.error(
                f"No new image found after rendering wall {wall_id}. "
                f"Existing: {len(existing_images)}, Current: {len(current_images)}"
            )

    return wall_images


def render_plant(
    plant,
    scene_graph,
    builder: DiagramBuilder,
    camera_X_WC: RigidTransform,
    camera_width: int = 640,
    camera_height: int = 480,
    background_color: list[float] = [1.0, 1.0, 1.0],
    use_blender_server: bool = False,
    blender_server_url: str = "http://127.0.0.1:8000",
) -> np.ndarray:
    """
    Render a plant and scene graph by adding camera to existing DiagramBuilder.

    Args:
        plant: The MultibodyPlant to render.
        scene_graph: The SceneGraph to render.
        builder: The DiagramBuilder containing the plant and scene_graph.
        camera_X_WC (RigidTransform): The camera pose in the world frame.
        camera_width (int): The camera width.
        camera_height (int): The camera height.
        background_color (list[float]): The background color of the rendered image.
        use_blender_server (bool): Whether to send render requests to a Blender server.
        blender_server_url (str): The URL of the Blender server.

    Returns:
        np.ndarray: The rendered RGBA image. Shape (H, W, 4) where H is the image height
        and W is the image width.
    """
    # Add camera to the existing builder.
    camera_config = CameraConfig(
        X_PB=Transform(camera_X_WC),
        width=camera_width,
        height=camera_height,
        background=Rgba(
            background_color[0], background_color[1], background_color[2], 1.0
        ),
        renderer_class=(
            RenderEngineGltfClientParams(base_url=blender_server_url)
            if use_blender_server
            else RenderEngineVtkParams(backend="GLX")
        ),
    )
    ApplyCameraConfig(
        config=camera_config,
        builder=builder,
        plant=plant,
        scene_graph=scene_graph,
    )
    builder.ExportOutput(
        builder.GetSubsystemByName(
            f"rgbd_sensor_{camera_config.name}"
        ).color_image_output_port(),
        "rgba_image",
    )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    rgba_image = copy.deepcopy(diagram.GetOutputPort("rgba_image").Eval(context).data)
    return rgba_image


def render_scene(
    scene: RoomScene,
    camera_X_WC: RigidTransform,
    camera_width: int = 640,
    camera_height: int = 480,
    background_color: list[float] = [1.0, 1.0, 1.0],
    use_blender_server: bool = False,
    blender_server_url: str = "http://127.0.0.1:8000",
) -> np.ndarray:
    """
    Render a scene from a given camera pose.

    This function creates the plant and scene graph, then calls render_plant
    to do the actual rendering work.

    Args:
        scene (RoomScene): The scene to render.
        camera_X_WC (RigidTransform): The camera pose in the world frame.
        camera_width (int): The camera width.
        camera_height (int): The camera height.
        background_color (list[float]): The background color of the rendered image.
        use_blender_server (bool): Whether to send render requests to a Blender server.
        blender_server_url (str): The URL of the Blender server.

    Returns:
        np.ndarray: The rendered RGBA image. Shape (H, W, 4) where H is the image height
        and W is the image width.
    """
    # Virtual display only needed for VTK/GLX rendering, not Blender.
    # Xvfb causes 6x slowdown for Blender which runs headless natively.
    if not use_blender_server:
        setup_virtual_display_if_needed()

    start_time = time.time()

    # Create a diagram builder for rendering.
    builder = DiagramBuilder()

    # Create the plant and scene graph in the rendering diagram.
    plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
        scene=scene, builder=builder
    )

    # Render using the plant and scene_graph.
    image = render_plant(
        plant=plant,
        scene_graph=scene_graph,
        builder=builder,
        camera_X_WC=camera_X_WC,
        camera_width=camera_width,
        camera_height=camera_height,
        background_color=background_color,
        use_blender_server=use_blender_server,
        blender_server_url=blender_server_url,
    )

    end_time = time.time()
    console_logger.info(f"Rendered scene in {end_time - start_time:.2f} seconds.")

    return image


def render_scene_for_agent_observation(
    scene: RoomScene,
    cfg: DictConfig,
    blender_server: BlenderServer,
    include_objects: list | None = None,
    exclude_room_geometry: bool = False,
    rendering_mode: str = "furniture",
    support_surfaces: list["SupportSurface"] | None = None,
    show_support_surface: bool = False,
    articulated_open: bool = False,
    wall_surfaces: list[dict] | None = None,
    annotate_object_types: list[str] | None = None,
    wall_surfaces_for_labels: list[dict] | None = None,
    wall_furniture_map: dict[str, list] | None = None,
    room_bounds: tuple[float, float, float, float] | None = None,
    ceiling_height: float | None = None,
    taa_samples: int = 16,
    context_furniture_ids: list | None = None,
    side_view_elevation_degrees: float | None = None,
    side_view_start_azimuth_degrees: float | None = None,
    include_vertical_views: bool = True,
    override_side_view_count: int | None = None,
) -> list[Path]:
    """Render scene with config-driven layout for agent observation.

    This function uses Drake's rendering pipeline with a Blender server backend.
    Drake exports the scene to glTF internally and sends it to the /render_overlay
    endpoint, which saves individual view images to a temporary directory.

    For manipuland mode with multiple support surfaces, generates separate renders
    for each surface with appropriate labels and coordinate markers.

    For wall mode ("wall"), renders context top-down view first, then per-wall
    orthographic views with furniture filtered per wall.

    For ceiling_perspective mode, renders an elevated perspective view showing
    the ceiling plane with furniture context below.

    Args:
        scene: The scene to render.
        cfg: Configuration with layout and dimension settings.
        blender_server: BlenderServer instance for rendering. REQUIRED - forked
            workers cannot safely use embedded bpy due to GPU/OpenGL state
            corruption from fork. The caller owns the server lifecycle.
        include_objects: Optional list of UniqueID objects to include in rendering.
            If provided, only these objects will be rendered. Useful for focused
            rendering (e.g., manipuland agent viewing only current furniture).
        exclude_room_geometry: If True, completely exclude the floor plan from rendering.
            Useful for focused rendering of furniture + manipulands only.
        rendering_mode: Rendering mode - "furniture" for room-scale annotations,
            "manipuland" for surface-focused annotations, "wall" for combined
            context top-down view + per-wall orthographic views, "ceiling_perspective"
            for elevated ceiling view.
        support_surfaces: For manipuland mode, list of SupportSurface objects.
            Each surface generates separate rendering views with appropriate labels
            and coordinate markers filtered to surface convex hull.
        show_support_surface: If True, render green wireframe bbox showing support
            surface bounds for debugging.
        articulated_open: If True, render articulated furniture with doors/drawers
            open (joints at max values). Useful for manipuland placement to show
            internal surfaces.
        wall_surfaces: List of wall surface dicts for wall rendering modes.
            Each dict contains wall_id, direction, length, height, transform,
            and excluded_regions.
        annotate_object_types: Optional list of object types to annotate. If provided,
            only objects of these types get annotations (e.g., ["wall_mounted"] for
            wall_context mode). None means annotate all objects.
        wall_surfaces_for_labels: Wall surfaces for top-down wall labels.
        wall_furniture_map: For wall mode, mapping from surface_id to list of furniture
            UniqueIDs to include in that wall's orthographic render. Required when
            rendering_mode="wall".
        room_bounds: For ceiling_perspective mode, room XY bounds
            (min_x, min_y, max_x, max_y) in meters.
        ceiling_height: For ceiling_perspective mode, ceiling height in meters.
        context_furniture_ids: For manipuland mode, list of furniture IDs to keep
            visible in per-surface top-down renders. These provide spatial context
            for item placement orientation (e.g., chairs around a table).
        side_view_elevation_degrees: Optional elevation angle in degrees for side
            view cameras. Overrides default (30 degrees). Useful for context image
            rendering where different angles work better for different furniture.
        side_view_start_azimuth_degrees: Optional starting azimuth angle in degrees
            for side views. 90 degrees positions camera at +Y (front). Overrides
            default (0 degrees with 45° offset for corner views).
        include_vertical_views: Whether to include pure vertical views (top/bottom).
            Defaults to True. Set to False for angled-only context image rendering.
        override_side_view_count: Optional override for number of side views. If
            provided, overrides cfg.side_view_count. Set to 1 for single angled view.

    Returns:
        List of Paths to rendered PNG files.

    Raises:
        RuntimeError: If BlenderServer is not running or rendering fails.
    """
    # NOTE: Virtual display NOT needed for Blender rendering. Blender runs headless
    # natively and Xvfb causes a 6x slowdown. Only VTK/GLX rendering needs Xvfb.

    # Validate BlenderServer is running.
    # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
    # due to GPU/OpenGL state corruption from fork.
    if not blender_server.is_running():
        raise RuntimeError(
            "BlenderServer is not running. Cannot render scene for agent observation. "
            "Forked workers cannot safely use embedded bpy."
        )

    # Create temporary directory for rendered outputs.
    temp_dir = Path(tempfile.mkdtemp(prefix="scene_render_"))
    output_dir = temp_dir / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Configure the server for overlay rendering.
        config_url = f"{blender_server.get_url()}/set_overlay_config"

        # Extract scene object metadata for annotations.
        # Filter by include_objects if provided to avoid rendering clutter.
        scene_objects_metadata = []
        objects_for_metadata = (
            [obj for obj in scene.objects.values() if obj.object_id in include_objects]
            if include_objects is not None
            else scene.objects.values()
        )
        for obj in objects_for_metadata:
            translation = obj.transform.translation()
            rotation = obj.transform.rotation()

            # Get rotation matrix as nested list for JSON serialization.
            rotation_matrix = rotation.matrix().tolist()

            # Get bounding box if available.
            bbox = None
            if obj.bbox_min is not None and obj.bbox_max is not None:
                # Compute center and extents from AABB bounds in local space.
                local_center = (obj.bbox_min + obj.bbox_max) / 2.0
                extents = obj.bbox_max - obj.bbox_min

                # Transform local center to world space.
                world_center = obj.transform @ local_center

                bbox = {
                    "center": world_center.tolist(),
                    "extents": extents.tolist(),
                }

            # Get parent surface ID for manipulands.
            parent_surface_id = None
            if obj.placement_info:
                parent_surface_id = str(obj.placement_info.parent_surface_id)

            scene_objects_metadata.append(
                {
                    "name": obj.name,
                    "object_id": str(obj.object_id),
                    "object_type": obj.object_type.value,
                    "position": translation.tolist(),
                    "rotation_matrix": rotation_matrix,
                    "bounding_box": bbox,
                    "parent_surface_id": parent_surface_id,
                }
            )

        # Get annotation config flags.
        # Direct attribute access to fail fast if any field is missing.
        annotations_cfg = cfg.annotations

        # Extract current furniture ID for manipuland mode.
        # In manipuland mode, include_objects[0] is always the current furniture.
        current_furniture_id = None
        if (
            rendering_mode == "manipuland"
            and include_objects is not None
            and len(include_objects) > 0
        ):
            current_furniture_id = str(include_objects[0])

        # Determine layout based on rendering_mode.
        # Wall modes use their own layout; other modes use config layout.
        if rendering_mode in ("wall_orthographic", "wall"):
            layout = rendering_mode
        else:
            layout = cfg.layout

        # Use override_side_view_count if provided, otherwise use config value.
        effective_side_view_count = (
            override_side_view_count
            if override_side_view_count is not None
            else cfg.side_view_count
        )

        config_payload = {
            "output_dir": str(output_dir.absolute()),
            "layout": layout,
            "top_view_width": cfg.top_view_width,
            "top_view_height": cfg.top_view_height,
            "side_view_count": effective_side_view_count,
            "side_view_width": cfg.side_view_width,
            "side_view_height": cfg.side_view_height,
            "scene_objects": scene_objects_metadata,
            "wall_normals": {
                name: normal.tolist()
                for name, normal in scene.room_geometry.wall_normals.items()
            },
            "annotations": {
                "enable_set_of_mark_labels": annotations_cfg.enable_set_of_mark_labels,
                "enable_bounding_boxes": annotations_cfg.enable_bounding_boxes,
                # Disable direction arrows for furniture_selection mode.
                "enable_direction_arrows": (
                    False
                    if rendering_mode == "furniture_selection"
                    else annotations_cfg.enable_direction_arrows
                ),
                "enable_partial_walls": annotations_cfg.enable_partial_walls,
                "rendering_mode": rendering_mode,
                "enable_support_surface_debug": annotations_cfg.enable_support_surface_debug,
                "enable_convex_hull_debug": annotations_cfg.enable_convex_hull_debug,
                "annotate_object_types": annotate_object_types,
                # Disable coordinate grid and frame for furniture_selection mode.
                "enable_coordinate_grid": rendering_mode != "furniture_selection",
                "show_coordinate_frame": rendering_mode != "furniture_selection",
            },
            "current_furniture_id": current_furniture_id,
            "openings": (
                [o.to_dict() for o in scene.room_geometry.openings]
                if scene.room_geometry
                else []
            ),
        }

        # Add wall surfaces for wall rendering modes.
        if wall_surfaces is not None:
            config_payload["wall_surfaces"] = wall_surfaces

        # Add wall surfaces for top-down wall labels.
        if wall_surfaces_for_labels is not None:
            config_payload["wall_surfaces_for_labels"] = wall_surfaces_for_labels

        # Add ceiling parameters for ceiling_perspective mode.
        if room_bounds is not None:
            config_payload["room_bounds"] = list(room_bounds)
        if ceiling_height is not None:
            config_payload["ceiling_height"] = ceiling_height

        # Add TAA samples for EEVEE render quality/speed control.
        config_payload["taa_samples"] = taa_samples

        # Add context furniture IDs for manipuland mode.
        # These furniture objects should remain visible in per-surface top-down views.
        if context_furniture_ids is not None and len(context_furniture_ids) > 0:
            config_payload["context_furniture_ids"] = [
                str(ctx_id) for ctx_id in context_furniture_ids
            ]

        # Add camera angle parameters for context image rendering.
        if side_view_elevation_degrees is not None:
            config_payload["side_view_elevation_degrees"] = side_view_elevation_degrees
        if side_view_start_azimuth_degrees is not None:
            config_payload["side_view_start_azimuth_degrees"] = (
                side_view_start_azimuth_degrees
            )
        config_payload["include_vertical_views"] = include_vertical_views

        # Create Drake plant and diagram FIRST to enable FK transforms.
        # This allows us to query link transforms before and after opening joints.
        builder = DiagramBuilder()
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
            scene=scene,
            builder=builder,
            include_objects=include_objects,
            exclude_room_geometry=exclude_room_geometry,
        )
        console_logger.info(
            f"Drake plant created with {plant.num_model_instances()} model instances"
        )

        # Placeholder camera pose (actual rendering uses configured views).
        placeholder_pose = RigidTransform.Identity()
        camera_config = CameraConfig(
            X_PB=Transform(placeholder_pose),
            width=4,
            height=4,
            background=Rgba(
                cfg.background_color[0],
                cfg.background_color[1],
                cfg.background_color[2],
                1.0,
            ),
            renderer_class=RenderEngineGltfClientParams(
                base_url=blender_server.get_url(),
                render_endpoint="render_overlay",
            ),
        )

        # Apply camera config.
        ApplyCameraConfig(
            config=camera_config,
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
        )

        # Export the color image output.
        builder.ExportOutput(
            builder.GetSubsystemByName(
                f"rgbd_sensor_{camera_config.name}"
            ).color_image_output_port(),
            "rgba_image",
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        # Apply FK transforms to support surfaces for articulated objects.
        # When articulated_open=True, we transform surface bounding boxes to match
        # the opened joint positions.
        surfaces_for_rendering = support_surfaces
        if articulated_open:
            # Get REST transforms before opening joints.
            rest_transforms = get_all_link_transforms(plant, plant_context)

            # Open joints.
            set_articulated_joints_to_max(plant, plant_context)

            # Apply FK transforms to support surfaces.
            if support_surfaces is not None and len(support_surfaces) > 0:
                # Get OPEN transforms after opening joints.
                open_transforms = get_all_link_transforms(plant, plant_context)

                # Find the furniture object to get SDF path for link-to-joint mapping.
                # In manipuland mode, include_objects[0] is always the current furniture.
                # Otherwise, find the first articulated furniture in the scene.
                furniture_obj = None
                if include_objects is not None and len(include_objects) > 0:
                    furniture_obj = scene.get_object(include_objects[0])
                else:
                    # Find articulated furniture from scene objects.
                    for obj in scene.objects.values():
                        if obj.metadata.get("is_articulated", False):
                            furniture_obj = obj
                            break

                # Apply FK transforms if we have the furniture's SDF.
                if (
                    furniture_obj is not None
                    and furniture_obj.sdf_path is not None
                    and furniture_obj.metadata.get("is_articulated", False)
                ):
                    link_to_joint = parse_joint_child_links(furniture_obj.sdf_path)
                    if link_to_joint:
                        # All joints are open when using set_articulated_joints_to_max.
                        open_joints = set(link_to_joint.values())
                        surfaces_for_rendering = apply_fk_to_surfaces(
                            surfaces=support_surfaces,
                            rest_transforms=rest_transforms,
                            open_transforms=open_transforms,
                            link_to_joint=link_to_joint,
                            open_joints=open_joints,
                        )
                        console_logger.info(
                            f"Applied FK transforms to {len(surfaces_for_rendering)} "
                            f"support surfaces"
                        )

                        # Apply FK transforms to manipulands on articulated surfaces.
                        # Build surface_id -> link_name lookup.
                        surface_to_link = {
                            str(s.surface_id): s.link_name for s in support_surfaces
                        }

                        # Transform manipulands placed on surfaces of open joints.
                        manipuland_fk_count = 0
                        for obj in scene.objects.values():
                            if obj.object_type != ObjectType.MANIPULAND:
                                continue
                            if obj.placement_info is None:
                                continue

                            parent_surface_id = obj.placement_info.parent_surface_id
                            if not parent_surface_id:
                                continue

                            obj_link_name = surface_to_link.get(str(parent_surface_id))
                            if not obj_link_name:
                                continue

                            # Check if this link has an open joint.
                            obj_joint = link_to_joint.get(obj_link_name)
                            if obj_joint not in open_joints:
                                continue

                            # Compute FK delta and new pose.
                            if (
                                obj_link_name not in rest_transforms
                                or obj_link_name not in open_transforms
                            ):
                                continue
                            delta = (
                                open_transforms[obj_link_name]
                                @ rest_transforms[obj_link_name].inverse()
                            )
                            new_pose = delta @ obj.transform

                            # Set new pose in Drake.
                            try:
                                model_name = get_drake_model_name(obj)
                                base_link_name = extract_base_link_name_from_sdf(
                                    obj.sdf_path
                                )
                                model_instance = plant.GetModelInstanceByName(
                                    model_name
                                )
                                body = plant.GetBodyByName(
                                    base_link_name, model_instance
                                )
                                plant.SetFreeBodyPose(plant_context, body, new_pose)
                                manipuland_fk_count += 1
                            except Exception as e:
                                console_logger.warning(
                                    f"Failed to set FK pose for {obj.name}: {e}"
                                )

                        if manipuland_fk_count > 0:
                            console_logger.info(
                                f"Applied FK transforms to {manipuland_fk_count} "
                                f"manipulands"
                            )

        # Add support surfaces for manipuland mode using FK-transformed surfaces.
        if surfaces_for_rendering is not None and len(surfaces_for_rendering) > 0:
            config_payload["support_surfaces"] = build_support_surfaces_data(
                surfaces_for_rendering
            )

        # Add debug visualization flag.
        config_payload["show_support_surface"] = show_support_surface

        # Send overlay config to Blender server.
        response = requests.post(config_url, json=config_payload, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to set overlay config: {response.status_code} "
                f"{response.text}"
            )
        console_logger.info("Overlay config set on Blender server")

        # Evaluate diagram (triggers Drake to send glTF to /render_overlay).
        # Joints are already opened above if articulated_open=True.
        _ = diagram.GetOutputPort("rgba_image").Eval(context)

        # Collect rendered image paths from output directory.
        image_paths = sorted(output_dir.glob("*.png"))
        if not image_paths:
            raise RuntimeError(f"No images found in {output_dir}")

        console_logger.info(f"Rendered {len(image_paths)} main views successfully")

        # Per-drawer rendering for articulated furniture.
        # After main render, render each drawer separately with only that drawer open.
        if (
            articulated_open
            and support_surfaces is not None
            and len(support_surfaces) > 0
            and furniture_obj is not None
            and link_to_joint
        ):
            # Classify surfaces into static vs per-joint (drawer).
            _, drawer_surfaces = classify_surfaces_for_rendering(
                surfaces=support_surfaces, link_to_joint=link_to_joint
            )

            if drawer_surfaces:
                console_logger.info(
                    f"Rendering {len(drawer_surfaces)} per-drawer views"
                )
                drawer_images = render_per_drawer_views(
                    plant=plant,
                    context=plant_context,
                    diagram=diagram,
                    server=blender_server,
                    drawer_surfaces=drawer_surfaces,
                    all_surfaces=support_surfaces,
                    scene_objects=list(scene.objects.values()),
                    link_to_joint=link_to_joint,
                    rest_transforms=rest_transforms,
                    config_payload=config_payload,
                    output_dir=output_dir,
                    cfg=cfg,
                )
                image_paths.extend(drawer_images)
                console_logger.info(
                    f"Total rendered: {len(image_paths)} views "
                    f"({len(image_paths) - len(drawer_images)} main + "
                    f"{len(drawer_images)} drawer)"
                )

        # Per-wall orthographic rendering for combined wall mode.
        # After context render, render each wall with filtered furniture.
        if (
            rendering_mode == "wall"
            and wall_surfaces is not None
            and len(wall_surfaces) > 0
            and wall_furniture_map is not None
        ):
            console_logger.info(
                f"Rendering {len(wall_surfaces)} per-wall orthographic views"
            )
            wall_images = render_per_wall_ortho_views(
                scene=scene,
                server=blender_server,
                wall_surfaces=wall_surfaces,
                wall_furniture_map=wall_furniture_map,
                base_config_payload=config_payload,
                output_dir=output_dir,
                cfg=cfg,
            )
            image_paths.extend(wall_images)
            console_logger.info(
                f"Total rendered: {len(image_paths)} views "
                f"(1 context + {len(wall_images)} wall ortho)"
            )

        return image_paths

    except Exception as e:
        console_logger.error(f"Failed to render scene: {e}")
        raise


def save_scene_as_blend(
    scene: RoomScene,
    output_path: Path,
    blender_server_host: str = "127.0.0.1",
    blender_server_port_range: tuple[int, int] = (8000, 8050),
    server_startup_delay: float = 0.1,
    port_cleanup_delay: float = 0.1,
) -> Path:
    """Export scene to a .blend file.

    Uses Drake to export scene to glTF, then Blender server imports and saves as .blend.

    Args:
        scene: The scene to export.
        output_path: Path where .blend file will be saved.
        blender_server_host: Host address for the Blender server.
        blender_server_port_range: Port range for the Blender server.
        server_startup_delay: Delay after starting server subprocess.
        port_cleanup_delay: Delay after stopping server.

    Returns:
        Path to the saved .blend file.

    Raises:
        RuntimeError: If Blender server fails or export fails.
    """
    # NOTE: Virtual display NOT needed for Blender. Blender runs headless natively.

    console_logger.info(f"Exporting scene to .blend file: {output_path}")

    server = BlenderServer(
        host=blender_server_host,
        port_range=blender_server_port_range,
        server_startup_delay=server_startup_delay,
        port_cleanup_delay=port_cleanup_delay,
    )
    server.start()
    server.wait_until_ready()

    try:
        # Configure server for blend export.
        config_url = f"{server.get_url()}/set_blend_config"
        config_payload = {"output_path": str(output_path.absolute())}

        response = requests.post(config_url, json=config_payload, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to set blend config: {response.text}")

        # Create Drake diagram to export glTF.
        builder = DiagramBuilder()
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
            scene=scene,
            builder=builder,
            include_objects=None,
            exclude_room_geometry=False,
        )

        # Use minimal camera config (just to trigger glTF export).
        placeholder_pose = RigidTransform.Identity()
        camera_config = CameraConfig(
            X_PB=Transform(placeholder_pose),
            width=4,
            height=4,
            background=Rgba(1.0, 1.0, 1.0, 1.0),
            renderer_class=RenderEngineGltfClientParams(
                base_url=server.get_url(),
                render_endpoint="save_blend",
            ),
        )

        ApplyCameraConfig(
            config=camera_config,
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
        )

        builder.ExportOutput(
            builder.GetSubsystemByName(
                f"rgbd_sensor_{camera_config.name}"
            ).color_image_output_port(),
            "rgba_image",
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        # Trigger glTF export to /save_blend endpoint.
        _ = diagram.GetOutputPort("rgba_image").Eval(context)

        if not output_path.exists():
            raise RuntimeError(f"Blend file was not created at {output_path}")

        console_logger.info(f"Successfully saved .blend file to {output_path}")
        return output_path

    finally:
        if server.is_running():
            server.stop()


def save_directive_as_blend(
    directive_path: Path,
    output_path: Path,
    blender_server_host: str = "127.0.0.1",
    blender_server_port_range: tuple[int, int] = (8000, 8050),
    server_startup_delay: float = 0.1,
    port_cleanup_delay: float = 0.1,
    scene_dir: Path | None = None,
    max_retries: int = 3,
) -> Path:
    """Export a Drake model directive to a .blend file.

    Loads a Drake model directive YAML file and exports all models to a .blend file.
    Automatically retries with a fresh Blender server if the export fails.

    Args:
        directive_path: Path to the Drake model directive YAML file.
        output_path: Path where .blend file will be saved.
        blender_server_host: Host address for the Blender server.
        blender_server_port_range: Port range for the Blender server.
        server_startup_delay: Delay after starting server subprocess.
        port_cleanup_delay: Delay after stopping server.
        scene_dir: Optional scene root directory for package:// URI resolution.
            If not provided, searches parent directories for package.xml.
        max_retries: Maximum number of retry attempts if export fails.

    Returns:
        Path to the saved .blend file.

    Raises:
        RuntimeError: If Blender server fails or export fails after all retries.
        FileNotFoundError: If directive_path does not exist.
    """
    # NOTE: Virtual display NOT needed for Blender. Blender runs headless natively.

    console_logger.info(f"Exporting directive to .blend file: {output_path}")

    for attempt in range(max_retries):
        server = BlenderServer(
            host=blender_server_host,
            port_range=blender_server_port_range,
            server_startup_delay=server_startup_delay,
            port_cleanup_delay=port_cleanup_delay,
        )
        server.start()
        server.wait_until_ready()

        try:
            # Configure server for blend export.
            config_url = f"{server.get_url()}/set_blend_config"
            config_payload = {"output_path": str(output_path.absolute())}

            response = requests.post(config_url, json=config_payload, timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to set blend config: {response.text}")

            # Create Drake plant from directive.
            builder, plant, scene_graph = create_plant_from_dmd(
                directive_path, scene_dir=scene_dir
            )

            # Use minimal camera config (just to trigger glTF export).
            placeholder_pose = RigidTransform.Identity()
            camera_config = CameraConfig(
                X_PB=Transform(placeholder_pose),
                width=4,
                height=4,
                background=Rgba(1.0, 1.0, 1.0, 1.0),
                renderer_class=RenderEngineGltfClientParams(
                    base_url=server.get_url(),
                    render_endpoint="save_blend",
                ),
            )

            ApplyCameraConfig(
                config=camera_config,
                builder=builder,
                plant=plant,
                scene_graph=scene_graph,
            )

            builder.ExportOutput(
                builder.GetSubsystemByName(
                    f"rgbd_sensor_{camera_config.name}"
                ).color_image_output_port(),
                "rgba_image",
            )

            diagram = builder.Build()
            context = diagram.CreateDefaultContext()

            # Trigger glTF export to /save_blend endpoint.
            _ = diagram.GetOutputPort("rgba_image").Eval(context)

            if output_path.exists():
                console_logger.info(f"Successfully saved .blend file to {output_path}")
                return output_path

            # File not created - server likely crashed during export.
            if attempt < max_retries - 1:
                console_logger.warning(
                    f"Blend export failed (file not created), "
                    f"retrying ({attempt + 1}/{max_retries})"
                )
                continue

            raise RuntimeError(f"Blend file was not created at {output_path}")

        finally:
            if server.is_running():
                server.stop()
