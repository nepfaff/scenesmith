"""Floor plan vision tools for rendering and visualization.

These tools allow the floor plan agent to visualize the floor plan design
using ASCII rendering, Blender perspective rendering, and material preview images.
Images are returned via ToolOutputImage so they persist in the session.
"""

import logging

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from agents import ToolOutputImage, ToolOutputText, function_tool
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import ApplyCameraConfig, CameraConfig, RenderEngineGltfClientParams
from pydrake.common.schema import Transform
from pydrake.math import RigidTransform as DrakeRigidTransform

from scenesmith.agent_utils.blender.font_utils import load_annotation_font
from scenesmith.agent_utils.blender.server_manager import BlenderServer
from scenesmith.agent_utils.drake_utils import create_plant_from_dmd
from scenesmith.agent_utils.house import HouseLayout
from scenesmith.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenesmith.utils.material import Material
from scenesmith.utils.openai import encode_image_to_base64

console_logger = logging.getLogger(__name__)


@dataclass
class MaterialUsage:
    """Tracks usage of a material in the floor plan."""

    material: Material
    material_type: str  # "floor" or "wall"
    room_ids: list[str]


@dataclass
class RenderResult:
    """Result from render_floor_plan tool."""

    success: bool
    message: str
    image_path: str = ""
    material_image_paths: list[str] | None = None  # Additional material preview images.


class FloorPlanVisionTools:
    """Rendering tools for floor plan visualization.

    Provides tools for visualizing the floor plan using Blender perspective
    rendering and ASCII text representation. Images are returned via
    ToolOutputImage so they persist in the session.
    """

    def __init__(
        self,
        layout: HouseLayout,
        output_dir: Path,
        blender_server: BlenderServer,
        wall_thickness: float = 0.05,
        floor_thickness: float = 0.1,
        render_size: int = 1024,
        generate_geometries_callback: Callable[[], None] | None = None,
    ):
        """Initialize floor plan vision tools.

        Args:
            layout: The HouseLayout to visualize.
            output_dir: Directory for rendering output.
            blender_server: BlenderServer instance for rendering. Lifecycle is
                managed by the agent, not this class.
            wall_thickness: Wall thickness in meters (default 5cm).
            floor_thickness: Floor thickness in meters (default 10cm).
            render_size: Render image size in pixels (width=height, default 1024).
            generate_geometries_callback: Callback to generate room geometries
                when they are missing. Required for DMD rendering during design loop.
        """
        self.layout = layout
        self.output_dir = output_dir
        self.blender_server = blender_server
        self.wall_thickness = wall_thickness
        self.floor_thickness = floor_thickness
        self.render_size = render_size
        self._generate_geometries_callback = generate_geometries_callback

        # Render counter for unique output directories.
        self._render_counter = 0

        # Track last render directory (for saving scores after critique).
        self._last_render_dir: Path | None = None

        # Render cache (layout_hash -> (render_dir, material_usage)).
        self._render_cache: dict[str, tuple[Path, dict[str, MaterialUsage]]] = {}

        # Build tools dictionary using closure pattern.
        self.tools = self._create_tool_closures()

    @property
    def last_render_dir(self) -> Path | None:
        """Get the last render directory path.

        Returns:
            Path to the last render directory, or None if no renders yet.
        """
        return self._last_render_dir

    def clear_cache(self) -> None:
        """Clear all caches to force re-rendering.

        Called when layout is reset to a previous checkpoint to ensure
        renders reflect the restored state.
        """
        self._render_cache.clear()
        self._last_render_dir = None
        console_logger.debug("Cleared vision tools caches")

    def update_layout(self, layout: "HouseLayout") -> None:
        """Update the layout reference after checkpoint restore.

        This allows reusing the same vision tools instance (preserving render
        counter) when the layout object is replaced during checkpoint reset.
        """
        self.layout = layout

    def _create_tool_closures(self) -> dict:
        """Create tool closures with access to instance data.

        Uses the same pattern as furniture/manipuland VisionTools to avoid
        including 'self' in the function schema.

        Returns:
            Dictionary mapping tool names to tool functions.
        """

        @function_tool
        def observe_scene() -> list[ToolOutputImage | ToolOutputText]:
            """Observe the current floor plan visually.

            Shows rooms with wall/floor materials, door openings, and windows from
            an elevated angle. Use to verify visual appearance and material choices.

            Returns:
                Images of the floor plan plus material previews and ASCII reference.
                These images persist in your conversation history.
            """
            return self._observe_scene_impl()

        @function_tool
        def render_ascii() -> str:
            """Generate text representation of floor plan.

            Shows room boundaries, room names, and wall segment labels (A, B, C...).
            Use for quick layout overview or when planning door/window placement.

            Returns:
                ASCII floor plan string with legend.
            """
            return self._render_ascii_impl()

        return {
            "observe_scene": observe_scene,
            "render_ascii": render_ascii,
        }

    def _add_room_labels(self, image_path: Path) -> None:
        """Add room ID labels to the rendered floor plan image.

        Uses PIL to overlay room labels at room centers. Labels use the same
        style as the set-of-mark labels in furniture rendering (blue background
        with white text).

        Args:
            image_path: Path to the rendered image to modify in place.
        """
        # Open image.
        pil_image = Image.open(str(image_path))
        draw = ImageDraw.Draw(pil_image)

        # Load font with fallback logic.
        font = load_annotation_font(
            image_width=pil_image.size[0], base_font_size_divisor=40
        )

        # Calculate scene bounds for coordinate mapping.
        all_rooms = self.layout.placed_rooms
        if not all_rooms:
            pil_image.save(str(image_path))
            return

        # Find bounding box of all rooms (in Z-up world coordinates).
        min_x = min(r.position[0] for r in all_rooms)
        max_x = max(r.position[0] + r.width for r in all_rooms)
        min_y = min(r.position[1] for r in all_rooms)
        max_y = max(r.position[1] + r.depth for r in all_rooms)

        scene_width = max_x - min_x
        scene_height = max_y - min_y
        scene_center_x = (min_x + max_x) / 2
        scene_center_y = (min_y + max_y) / 2

        # Image dimensions.
        img_width, img_height = pil_image.size

        # Scale factor to map world coords to pixels (must match renderer margin).
        margin_factor = 0.8  # Scene occupies ~80% of image (10% margin per side).
        scale = (
            min(img_width, img_height) * margin_factor / max(scene_width, scene_height)
        )

        for room in all_rooms:
            # Room center in Z-up world coordinates.
            room_center_x = room.position[0] + room.width / 2
            room_center_y = room.position[1] + room.depth / 2

            # Convert to image coordinates.
            # Image origin is top-left, Y increases downward.
            # Camera is looking down, so Y-up world maps to -Y image.
            rel_x = room_center_x - scene_center_x
            rel_y = room_center_y - scene_center_y

            pixel_x = img_width / 2 + rel_x * scale
            pixel_y = img_height / 2 - rel_y * scale  # Flip Y for image coords.

            # Draw label.
            label = room.room_id

            # Measure text size.
            text_bbox = draw.textbbox((0, 0), label, font=font)
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
                fill=(77, 153, 255),  # Blue background.
            )

            # Draw white text centered on background.
            text_x = int(pixel_x - text_width // 2)
            text_y = int(pixel_y - text_height // 2)
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

        pil_image.save(str(image_path))
        console_logger.info(f"Added {len(all_rooms)} room labels to {image_path}")

    def _format_room_list_for_preview(
        self,
        room_ids: list[str],
        font: ImageFont.ImageFont,
        max_width: int,
        draw: ImageDraw.Draw,
    ) -> list[str]:
        """Format room list into lines that fit within max_width.

        Word wraps to show all rooms without truncation.

        Args:
            room_ids: List of room IDs to format.
            font: Font to use for text measurement.
            max_width: Maximum width in pixels for each line.
            draw: ImageDraw object for text measurement.

        Returns:
            List of formatted lines showing all rooms.
        """
        if not room_ids:
            return ["Rooms: (none)"]

        # Start with "Rooms: " prefix.
        prefix = "Rooms: "
        prefix_width = draw.textlength(prefix, font=font)
        available_width = max_width - prefix_width

        lines: list[str] = []
        current_line_rooms: list[str] = []

        for room in room_ids:
            # Build test string with current rooms + new room.
            test_rooms = current_line_rooms + [room]
            test_text = ", ".join(test_rooms)

            # For first line, account for "Rooms: " prefix.
            if len(lines) == 0:
                test_width = draw.textlength(test_text, font=font)
                fits = test_width <= available_width
            else:
                test_width = draw.textlength(test_text, font=font)
                fits = test_width <= max_width

            if fits:
                current_line_rooms.append(room)
            else:
                # Current room doesn't fit - save current line and start new one.
                if current_line_rooms:
                    if len(lines) == 0:
                        lines.append(prefix + ", ".join(current_line_rooms))
                    else:
                        lines.append(", ".join(current_line_rooms))
                    current_line_rooms = [room]
                else:
                    # Single room on its own line (even if long).
                    if len(lines) == 0:
                        lines.append(prefix + room)
                    else:
                        lines.append(room)

        # Add final line.
        if current_line_rooms:
            if len(lines) == 0:
                lines.append(prefix + ", ".join(current_line_rooms))
            else:
                lines.append(", ".join(current_line_rooms))

        return lines if lines else ["Rooms: (none)"]

    def _generate_material_preview(
        self,
        material_usage: MaterialUsage,
        output_path: Path,
        preview_size: int = 256,
    ) -> Path:
        """Generate a material preview image with informative header.

        Creates a square preview of the material's color texture with a header
        showing the material type and which rooms use it. Room list wraps to
        multiple lines if needed, with "+N more" truncation for very long lists.

        Args:
            material_usage: MaterialUsage object with material path and room info.
            output_path: Path where the preview image will be saved.
            preview_size: Size of the texture preview in pixels (default: 256).

        Returns:
            Path to the generated preview image.
        """
        material = material_usage.material

        # Find the Color texture file.
        color_texture = material.get_texture("Color")

        # Load fonts (scale with preview size, targeting ~14 and ~11 for 256px).
        font = load_annotation_font(
            image_width=preview_size, base_font_size_divisor=18.3, min_font_size=11
        )
        font_small = load_annotation_font(
            image_width=preview_size, base_font_size_divisor=23.3, min_font_size=11
        )

        # Calculate header content to determine height.
        # Create temporary draw object for text measurement.
        temp_image = Image.new("RGBA", (1, 1))
        temp_draw = ImageDraw.Draw(temp_image)

        # Format room list with smart wrapping.
        padding = 8
        max_text_width = preview_size - (2 * padding)
        room_lines = self._format_room_list_for_preview(
            room_ids=material_usage.room_ids,
            font=font_small,
            max_width=max_text_width,
            draw=temp_draw,
        )

        # Calculate header height based on content.
        # Line 1 (material name): ~20px, Room lines: ~16px each, padding: 8px top/bottom.
        line1_height = 22
        room_line_height = 16
        header_height = (
            padding + line1_height + (len(room_lines) * room_line_height) + padding
        )

        # Create preview image with calculated header height.
        total_height = preview_size + header_height
        preview_image = Image.new(
            "RGBA", (preview_size, total_height), (255, 255, 255, 255)
        )
        draw = ImageDraw.Draw(preview_image)

        # Load and resize texture.
        if color_texture and color_texture.exists():
            texture = Image.open(color_texture)
            texture = texture.resize(
                (preview_size, preview_size), Image.Resampling.LANCZOS
            )
            preview_image.paste(texture, (0, header_height))
        else:
            # Gray placeholder if texture not found.
            draw.rectangle(
                [0, header_height, preview_size, total_height],
                fill=(128, 128, 128, 255),
            )

        # Material name from folder name (clean up for display).
        material_name = (
            material.name.replace("_", " ").replace("-JPG", "").replace("1K", "")
        )

        # Header line 1: Material type and name.
        surface_type = material_usage.material_type.capitalize()
        header_line1 = f"{surface_type}: {material_name}"

        # Draw header background.
        draw.rectangle([0, 0, preview_size, header_height], fill=(50, 50, 50, 255))

        # Draw header text.
        draw.text((padding, padding), header_line1, fill=(255, 255, 255), font=font)

        # Draw room lines.
        y_offset = padding + line1_height
        for line in room_lines:
            draw.text((padding, y_offset), line, fill=(200, 200, 200), font=font_small)
            y_offset += room_line_height

        # Save preview.
        preview_image.save(str(output_path))
        return output_path

    def _compute_material_usage_from_layout(self) -> dict[str, MaterialUsage]:
        """Compute material usage directly from HouseLayout.

        Returns:
            Dictionary mapping material keys to MaterialUsage objects.
        """
        # Default materials.
        materials_dir = Path(__file__).parent.parent.parent.parent / "materials"
        default_floor_material = Material.from_path(materials_dir / "Wood094_1K-JPG")
        default_wall_material = Material.from_path(materials_dir / "Plaster001_1K-JPG")

        material_usage: dict[str, MaterialUsage] = {}

        for room in self.layout.placed_rooms:
            # Get room materials (or use defaults).
            room_materials = self.layout.room_materials.get(room.room_id)
            floor_material = default_floor_material
            wall_material = default_wall_material
            if room_materials:
                if room_materials.floor_material:
                    floor_material = room_materials.floor_material
                if room_materials.wall_material:
                    wall_material = room_materials.wall_material

            # Track floor material usage.
            floor_key = f"{floor_material.path}:floor"
            if floor_key not in material_usage:
                material_usage[floor_key] = MaterialUsage(
                    material=floor_material,
                    material_type="floor",
                    room_ids=[room.room_id],
                )
            else:
                material_usage[floor_key].room_ids.append(room.room_id)

            # Track wall material usage.
            wall_key = f"{wall_material.path}:wall"
            if wall_key not in material_usage:
                material_usage[wall_key] = MaterialUsage(
                    material=wall_material,
                    material_type="wall",
                    room_ids=[room.room_id],
                )
            else:
                material_usage[wall_key].room_ids.append(room.room_id)

        # Track exterior material usage if set.
        if self.layout.exterior_material:
            exterior_key = f"{self.layout.exterior_material.path}:exterior_wall"
            material_usage[exterior_key] = MaterialUsage(
                material=self.layout.exterior_material,
                material_type="exterior_wall",
                room_ids=["exterior"],
            )

        return material_usage

    def _observe_scene_impl(self) -> list[ToolOutputImage | ToolOutputText]:
        """Implementation of observe_scene tool.

        Uses DMD-based Drake pipeline to render the floor plan. Returns images
        directly via ToolOutputImage so they persist in the session.
        """
        console_logger.info("Tool called: observe_scene")
        if not self.layout.placed_rooms:
            return [
                ToolOutputText(
                    text="No rooms to render. Call generate_room_specs first."
                )
            ]

        # Ensure room geometries exist for ALL rooms.
        # Check if any room is missing geometry (may have been invalidated).
        missing_rooms = [
            r.room_id
            for r in self.layout.placed_rooms
            if r.room_id not in self.layout.room_geometries
        ]
        if missing_rooms:
            if self._generate_geometries_callback is None:
                raise RuntimeError(
                    "Room geometries missing and no generator configured. "
                    f"Missing rooms: {missing_rooms}. "
                    "This indicates a configuration error in FloorPlanVisionTools."
                )
            console_logger.info(
                f"Generating room geometries for {len(missing_rooms)} rooms: "
                f"{missing_rooms}"
            )
            self._generate_geometries_callback()

        # Check render cache first (Level 2 cache).
        layout_hash = self.layout.content_hash()
        if layout_hash in self._render_cache:
            cached_render_dir, _ = self._render_cache[layout_hash]
            if cached_render_dir.exists():
                console_logger.info(f"RENDER CACHE HIT: {layout_hash}")
                # Update last render dir for scores saving.
                self._last_render_dir = cached_render_dir

                # Collect cached images.
                outputs: list[ToolOutputImage | ToolOutputText] = []
                for img_path in sorted(cached_render_dir.glob("*.png")):
                    img_base64 = encode_image_to_base64(img_path)
                    outputs.append(
                        ToolOutputImage(image_url=f"data:image/png;base64,{img_base64}")
                    )

                # Read ASCII for result message.
                ascii_result = generate_ascii_floor_plan(self.layout.placed_rooms)

                outputs.append(
                    ToolOutputText(
                        text=f"Floor plan rendered (cached).\n\n"
                        f"ASCII Reference:\n{ascii_result.ascii_art}\n\n"
                        f"{ascii_result.legend}"
                    )
                )

                console_logger.info(
                    f"Returning {len(outputs) - 1} cached images via ToolOutputImage"
                )
                return outputs

        console_logger.info("Rendering floor plan")

        # Create output directory FIRST (before any operations that might fail).
        # This ensures _last_render_dir is always current for score saving.
        self._render_counter += 1
        render_dir = (
            self.output_dir / f"floor_plan_renders/renders_{self._render_counter:03d}"
        )
        render_dir.mkdir(parents=True, exist_ok=True)
        self._last_render_dir = render_dir

        # Always generate and save ASCII (even if visual render fails).
        ascii_result = generate_ascii_floor_plan(self.layout.placed_rooms)
        ascii_path = render_dir / "ascii.txt"
        with open(ascii_path, "w") as f:
            f.write(ascii_result.ascii_art)
            f.write("\n\n")
            f.write(ascii_result.legend)

        try:
            # Generate DMD file for Drake with package://scene/ URIs.
            # Use house_dir as base to generate portable relative paths.
            dmd_path = render_dir / "floor_plan.dmd.yaml"
            house_dir = self.layout.house_dir
            directive_content = self.layout.to_drake_directive(base_dir=house_dir)
            with open(dmd_path, "w") as f:
                f.write(directive_content)
            console_logger.info(f"Generated DMD: {dmd_path}")

            # Get Blender server and configure floor plan rendering.
            blender = self.blender_server
            output_path = render_dir / "floor_plan.png"

            # Set floor plan render config on Blender server.
            config_response = requests.post(
                f"{blender.get_url()}/set_floor_plan_config",
                json={
                    "output_path": str(output_path),
                    "width": self.render_size,
                    "height": self.render_size,
                },
                timeout=10,
            )
            config_response.raise_for_status()
            console_logger.info("Floor plan config set on Blender server")

            # Create Drake plant from DMD.
            # Explicitly pass house_dir as scene_dir for package://scene/ resolution.
            builder, plant, scene_graph = create_plant_from_dmd(
                dmd_path, scene_dir=house_dir
            )

            # Configure camera for Drake → Blender GLTF export.
            # The actual camera position is computed by Blender from the geometry.
            camera_config = CameraConfig(
                name="floor_plan_camera",
                X_PB=Transform(DrakeRigidTransform()),  # Placeholder, Blender computes.
                width=4,  # Minimal, Blender uses config dimensions.
                height=4,
                renderer_class=RenderEngineGltfClientParams(
                    base_url=blender.get_url(),
                    render_endpoint="render_floor_plan",
                ),
            )

            # Apply camera config (creates camera sensor in scene graph).
            ApplyCameraConfig(
                config=camera_config,
                builder=builder,
                plant=plant,
                scene_graph=scene_graph,
            )

            # Export color output port.
            sensor_name = f"rgbd_sensor_{camera_config.name}"
            builder.ExportOutput(
                builder.GetSubsystemByName(sensor_name).color_image_output_port(),
                "rgba_image",
            )

            # Build diagram and create context.
            diagram = builder.Build()
            context = diagram.CreateDefaultContext()

            # Evaluate → Drake sends GLTF → Blender renders to output_path.
            console_logger.info("Triggering Drake → Blender render")
            _ = diagram.GetOutputPort("rgba_image").Eval(context)

            # Verify render succeeded.
            if not output_path.exists():
                raise RuntimeError(f"Render output not found: {output_path}")

            console_logger.info(f"Floor plan rendered: {output_path}")

            # Add room labels to the rendered image.
            self._add_room_labels(output_path)

            # Compute material usage from layout (no GLB generation needed).
            material_usage = self._compute_material_usage_from_layout()

            # Generate material preview images.
            for i, (key, usage) in enumerate(material_usage.items()):
                material_preview_path = (
                    render_dir / f"material_{i:02d}_{usage.material_type}.png"
                )
                self._generate_material_preview(
                    material_usage=usage, output_path=material_preview_path
                )
            console_logger.info(
                f"Generated {len(material_usage)} material preview images"
            )

            # Update render cache (Level 2).
            self._render_cache[layout_hash] = (render_dir, material_usage)
            console_logger.info(f"Cached render with key: {layout_hash}")

            # Collect images and return them directly.
            outputs: list[ToolOutputImage | ToolOutputText] = []
            for img_path in sorted(render_dir.glob("*.png")):
                img_base64 = encode_image_to_base64(img_path)
                outputs.append(
                    ToolOutputImage(image_url=f"data:image/png;base64,{img_base64}")
                )

            num_images = len(outputs)
            outputs.append(
                ToolOutputText(
                    text=f"Floor plan rendered with {num_images} images.\n\n"
                    f"ASCII Reference:\n{ascii_result.ascii_art}\n\n"
                    f"{ascii_result.legend}"
                )
            )

            console_logger.info(f"Returning {num_images} images via ToolOutputImage")
            return outputs

        except Exception as e:
            console_logger.error(f"Floor plan rendering failed: {e}")
            # ASCII already saved to render_dir - return failure with ASCII reference.
            return [
                ToolOutputText(
                    text=f"Blender rendering failed ({e}). ASCII saved to: {ascii_path}\n\n"
                    f"{ascii_result.ascii_art}\n\n"
                    f"{ascii_result.legend}"
                )
            ]

    def _render_ascii_impl(self) -> str:
        """Implementation of render_ascii tool."""
        console_logger.info("Tool called: render_ascii")
        if not self.layout.placed_rooms:
            return "No rooms to render. Call generate_room_specs first."

        result = generate_ascii_floor_plan(self.layout.placed_rooms)
        ascii_output = f"{result.ascii_art}\n\n{result.legend}"

        # Log ASCII for visibility during runs.
        console_logger.info("Floor plan layout:\n%s", ascii_output)

        return ascii_output
