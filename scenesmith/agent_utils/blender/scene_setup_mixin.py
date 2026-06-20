"""Mixin class for scene setup functionality in BlenderRenderer.

This mixin encapsulates all scene setup and initialization methods for Blender
rendering, including scene reset, GLTF import, and scene verification.
"""

import hashlib
import math

from pathlib import Path

import bpy

from scenesmith.agent_utils.blender.params import RenderParams
from scenesmith.agent_utils.blender.render_settings import setup_regular_world
from scenesmith.agent_utils.blender.scene_utils import disable_backface_culling


class SceneSetupMixin:
    """Mixin providing scene setup methods for BlenderRenderer.

    This mixin contains methods for initializing and configuring the Blender
    scene, including reset, GLTF import, and scene verification.

    Attributes:
        _blend_file: Optional path to a .blend file to use as base scene.
        _bpy_settings_file: Optional path to a .py file with Blender settings.
        _client_objects: Blender collection containing imported scene objects.
    """

    def reset_scene(self) -> None:
        """Reset the scene in Blender by loading the factory defaults,
        and then remove the default cube object.
        """
        # Load factory defaults (matches reference implementation).
        bpy.ops.wm.read_factory_settings()

        # Remove default objects using the same method as reference.
        for item in bpy.data.objects:
            item.select_set(True)
        bpy.ops.object.delete()

    def _verify_scene_checksum(self, params: RenderParams) -> None:
        """Verify scene file integrity.

        Args:
            params: Rendering parameters containing scene path and expected checksum.

        Raises:
            ValueError: If computed checksum doesn't match expected checksum.
        """
        scene_data = params.scene.read_bytes()
        computed_sha256 = hashlib.sha256(scene_data).hexdigest()
        if computed_sha256 != params.scene_sha256:
            raise ValueError(
                f"Scene checksum mismatch. Expected {params.scene_sha256}, "
                f"got {computed_sha256}"
            )

    def _setup_scene(self, params: RenderParams) -> None:
        """Common scene setup for both regular and metric rendering.

        Args:
            params: Rendering parameters containing scene configuration.
        """
        self._verify_scene_checksum(params)

        # Load blend file or reset scene.
        if self._blend_file is not None:
            bpy.ops.wm.open_mainfile(filepath=str(self._blend_file))
        else:
            self.reset_scene()
            self.add_default_light_source()
            setup_regular_world()

        # Apply custom settings.
        if self._bpy_settings_file:
            with open(self._bpy_settings_file) as f:
                code = compile(f.read(), self._bpy_settings_file, "exec")
                exec(code, {"bpy": bpy}, dict())

    def _import_and_organize_gltf(self, scene_path: Path) -> None:
        """Import glTF and organize objects into collections.

        Args:
            scene_path: Path to the glTF file to import.
        """
        old_count = len(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(scene_path))
        new_count = len(bpy.data.objects)
        assert new_count - old_count == len(bpy.context.selected_objects)

        # Blender 5 imports Drake's glTF output with the opposite X-axis
        # correction needed by Blender 4.
        bpy.ops.transform.rotate(
            value=-math.pi / 2,
            orient_axis="X",
            orient_type="GLOBAL",
            center_override=(0, 0, 0),
        )

        # Create collection for imported objects.
        self._client_objects = bpy.data.collections.new("ClientObjects")
        bpy.context.scene.collection.children.link(self._client_objects)

        # Move imported objects to our collection.
        for obj in bpy.context.selected_objects:
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            self._client_objects.objects.link(obj)

        # Disable backface culling for all imported materials.
        # This ensures meshes render correctly from both sides, fixing issues
        # with single-sided meshes (common in PartNet-Mobility models).
        disable_backface_culling(list(self._client_objects.objects))
