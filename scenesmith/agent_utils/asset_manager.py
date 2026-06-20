import logging
import re
import shutil
import time
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from omegaconf import DictConfig
from pydrake.all import RigidTransform

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalClient,
)
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.asset_router import AssetRouter
from scenesmith.agent_utils.asset_router.dataclasses import (
    ArticulatedGeometry,
    AssetItem,
    GeneratedGeometry,
    ModificationInfo,
)
from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionClient
from scenesmith.agent_utils.geometry_generation_server.client import (
    GeometryGenerationClient,
)
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationError,
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalClient
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.image_generation import (
    AssetOperationType,
    create_image_generator,
)
from scenesmith.agent_utils.materials_retrieval_server import MaterialsRetrievalClient
from scenesmith.agent_utils.mesh_canonicalization import canonicalize_mesh
from scenesmith.agent_utils.mesh_physics_analyzer import (
    MeshPhysicsAnalysis,
    analyze_mesh_orientation_and_material,
)
from scenesmith.agent_utils.mesh_utils import (
    load_mesh_as_trimesh,
    remove_mesh_floaters,
    scale_mesh_uniformly_to_dimensions,
)
from scenesmith.agent_utils.objaverse_retrieval_server import ObjaverseRetrievalClient
from scenesmith.agent_utils.objaverse_retrieval_server.dataclasses import (
    ObjaverseRetrievalServerRequest,
)
from scenesmith.agent_utils.room import AgentType, ObjectType, SceneObject, UniqueID
from scenesmith.agent_utils.sdf_generator import (
    add_self_collision_filter,
    generate_drake_sdf,
)
from scenesmith.agent_utils.sdf_mesh_utils import combine_sdf_meshes_at_joint_angles
from scenesmith.agent_utils.thin_covering_generator import (
    generate_thin_covering_sdf,
    infer_thin_covering_shape,
)
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.utils.logging import BaseLogger

if TYPE_CHECKING:
    from scenesmith.agent_utils.asset_router import AssetRouter
    from scenesmith.agent_utils.blender import BlenderServer

console_logger = logging.getLogger(__name__)


@dataclass
class AssetPathConfig:
    """Configuration for asset file paths and metadata."""

    description: str
    """Description of the object."""

    short_name: str
    """Short name for the object."""

    image_path: Path | None
    """Path to the generated image."""

    geometry_path: Path
    """Path to the generated 3D geometry."""

    sdf_dir: Path
    """Directory containing the generated SDF file."""


@dataclass
class AssetGenerationRequest:
    """Request for generating scene assets (furniture, manipulands, etc.)."""

    object_descriptions: list[str]
    """List of object descriptions to generate."""

    short_names: list[str]
    """List of short names for filesystem-safe file naming."""

    object_type: ObjectType
    """Type of objects to generate (FURNITURE, MANIPULAND, etc.)."""

    desired_dimensions: list[list[float]]
    """Desired dimensions (width, depth, height) in meters for each object.
    Agent must predict dimensions considering scene context.
    Must match the length of object_descriptions.
    """

    style_context: str | None = None
    """Style context for consistency (e.g., 'modern minimalist kitchen')."""

    operation_type: AssetOperationType = AssetOperationType.INITIAL
    """Type of generation operation."""

    scene_id: str | None = None
    """Optional scene identifier for fair round-robin scheduling on servers.

    When multiple scenes generate assets concurrently, passing scene_id ensures
    fair GPU time allocation across scenes in the geometry and HSSD servers.
    """


@dataclass
class FailedAsset:
    """Information about a failed asset generation."""

    index: int
    """Index of the failed asset in the original request."""

    description: str
    """Description of the object that failed to generate."""

    error_message: str
    """Error message describing why generation failed."""


@dataclass
class AssetGenerationResult:
    """Result of asset generation with potential partial success."""

    successful_assets: list[SceneObject]
    """List of successfully generated scene objects."""

    failed_assets: list[FailedAsset]
    """List of assets that failed during generation."""

    modification_info: ModificationInfo | None = None
    """Set when router modified the original request (split composites or filtered
    items). Contains original description, resulting items, and any discarded
    manipulands (furniture agent only). None when router is disabled or request
    was not modified.
    """

    @property
    def has_failures(self) -> bool:
        """Check if any assets failed to generate."""
        return len(self.failed_assets) > 0

    @property
    def all_succeeded(self) -> bool:
        """Check if all assets were generated successfully."""
        return len(self.failed_assets) == 0


class AssetManager:
    """Manages 3D asset acquisition for scene generation.

    Supports two acquisition strategies configured via `general_asset_source`:
    - "generated": Text-to-3D generation (text → image → 3D mesh)
    - "hssd": Retrieval from HSSD library

    Has two operating modes based on `router.enabled` config:

    **Router path** (router.enabled=True):
    - LLM analyzes requests to split composites and select strategies
    - Parallel HTTP calls for generation/retrieval (thread-safe)
    - Sequential bpy operations for mesh processing (main thread)
    - VLM validation with retry loop for quality control

    **Non-router path** (router.enabled=False):
    - Direct dispatch to generation or retrieval based on config
    - Batch processing without LLM analysis
    - Simpler but less flexible

    Both paths produce simulation-ready Drake SDF files with:
    - Canonical orientation (Z-up, Y-forward)
    - Convex decomposition collision geometry (CoACD or V-HACD)
    - VLM-estimated physics properties (material, mass)

    Maintains style consistency through conversational context and includes
    an asset registry to track generated assets for reuse.
    """

    def __init__(
        self,
        logger: BaseLogger,
        vlm_service: VLMService,
        blender_server: "BlenderServer | None",
        collision_client: ConvexDecompositionClient | None,
        cfg: DictConfig,
        agent_type: AgentType,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        objaverse_server_host: str = "127.0.0.1",
        objaverse_server_port: int = 7009,
    ) -> None:
        """Initialize the asset manager.

        Args:
            logger: Logger instance for tracking operations.
            vlm_service: VLM service instance for mesh physics analysis.
            blender_server: Blender server instance for multi-view rendering.
            collision_client: Client for collision geometry generation via convex
                decomposition. Can be None for checkpoint loading (no collision
                generation needed).
            cfg: Configuration with asset_manager settings.
            agent_type: Agent type for directory organization. Assets will be
                stored in generated_assets/{agent_type.value}/.
            geometry_server_host: Host for geometry generation server.
            geometry_server_port: Port for geometry generation server.
            hssd_server_host: Host for HSSD retrieval server.
            hssd_server_port: Port for HSSD retrieval server.
            articulated_server_host: Host for articulated retrieval server.
            articulated_server_port: Port for articulated retrieval server.
            materials_server_host: Host for materials retrieval server.
            materials_server_port: Port for materials retrieval server.
            objaverse_server_host: Host for Objaverse retrieval server.
            objaverse_server_port: Port for Objaverse retrieval server.
        """
        self.output_dir = logger.output_dir
        self.logger = logger
        self.cfg = cfg
        self.agent_type = agent_type

        # Extract config values.
        self.num_side_views_for_physics_analysis = (
            cfg.asset_manager.num_side_views_for_physics_analysis
        )
        self.side_view_elevation_degrees = cfg.asset_manager.side_view_elevation_degrees
        self.min_mesh_dimension_meters = cfg.asset_manager.min_mesh_dimension_meters
        self.mesh_relative_dimension_threshold = (
            cfg.asset_manager.mesh_relative_dimension_threshold
        )
        # Store collision geometry configuration.
        self.collision_method = cfg.collision_geometry.method
        self.collision_coacd_cfg = cfg.collision_geometry.coacd
        self.collision_vhacd_cfg = cfg.collision_geometry.vhacd

        self.vlm_service = vlm_service
        self.blender_server = blender_server
        self.collision_client = collision_client
        self.image_generator = create_image_generator(
            backend=cfg.asset_manager.image_generation.backend,
            config=cfg.asset_manager.image_generation,
        )

        # Create agent-specific subdirectories for organization.
        generated_assets_dir = self.output_dir / "generated_assets" / agent_type.value
        self.images_dir = generated_assets_dir / "images"
        self.geometry_dir = generated_assets_dir / "geometry"
        self.sdf_dir = generated_assets_dir / "sdf"
        self.debug_dir = generated_assets_dir / "debug"

        for dir_path in [
            self.images_dir,
            self.geometry_dir,
            self.sdf_dir,
            self.debug_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize registry with auto-save to enable incremental persistence.
        registry_path = generated_assets_dir / "asset_registry.json"
        self.registry = AssetRegistry(auto_save_path=registry_path)

        # Initialize strategy-specific clients.
        self.general_asset_source = cfg.asset_manager.general_asset_source
        if self.general_asset_source not in ["generated", "hssd", "objaverse"]:
            raise ValueError(f"Unknown asset source: {self.general_asset_source}")

        # Initialize geometry generation client if source is "generated".
        self.geometry_client: GeometryGenerationClient | None = None
        if self.general_asset_source == "generated":
            console_logger.info("Initializing geometry generation client")
            self.geometry_client = GeometryGenerationClient(
                host=geometry_server_host, port=geometry_server_port
            )

        # Initialize HSSD client if source is "hssd".
        self.hssd_client: HssdRetrievalClient | None = None
        if self.general_asset_source == "hssd":
            console_logger.info("Initializing HSSD retrieval client")
            self.hssd_client = HssdRetrievalClient(
                host=hssd_server_host, port=hssd_server_port
            )

        # Initialize Objaverse client if source is "objaverse".
        self.objaverse_client: ObjaverseRetrievalClient | None = None
        if self.general_asset_source == "objaverse":
            console_logger.info("Initializing Objaverse retrieval client")
            self.objaverse_client = ObjaverseRetrievalClient(
                host=objaverse_server_host, port=objaverse_server_port
            )

        # Initialize articulated retrieval client if articulated strategy is enabled.
        self.articulated_client: ArticulatedRetrievalClient | None = None
        articulated_enabled = cfg.asset_manager.router.strategies.articulated.enabled
        if articulated_enabled:
            console_logger.info("Initializing articulated retrieval client")
            self.articulated_client = ArticulatedRetrievalClient(
                host=articulated_server_host, port=articulated_server_port
            )

        # Initialize materials retrieval client if thin_covering strategy is enabled.
        self.materials_client: MaterialsRetrievalClient | None = None
        thin_covering_enabled = (
            cfg.asset_manager.router.strategies.thin_covering.enabled
        )
        if thin_covering_enabled:
            console_logger.info("Initializing materials retrieval client")
            self.materials_client = MaterialsRetrievalClient(
                host=materials_server_host, port=materials_server_port
            )

        # Initialize asset router if enabled in config.
        self.router: "AssetRouter | None" = None
        if cfg.asset_manager.router.enabled:
            console_logger.info("Initializing asset router for LLM-advised generation")
            self.router = AssetRouter(
                agent_type=agent_type,
                vlm_service=vlm_service,
                cfg=cfg,
                blender_server=blender_server,
            )

        # Track duplicate requests from the last generate_assets call.
        self.last_duplicate_info: dict[str, list[int]] | None = None

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = 50) -> str:
        """Sanitize a name for use as a filename.

        Args:
            name: Name to sanitize.
            max_length: Maximum length for the filename.

        Returns:
            Filesystem-safe filename string.
        """
        # Replace problematic characters with underscores.
        sanitized = re.sub(r"[^\w\-_.]", "_", name)
        # Remove consecutive underscores.
        sanitized = re.sub(r"_+", "_", sanitized)
        # Trim to max length.
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip("_")
        return sanitized

    @staticmethod
    def _unique_asset_basename(safe_name: str) -> str:
        """Return a filesystem basename that remains unique under parallel generation."""
        return f"{safe_name}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"

    def _generate_collision_geometry(self, mesh_path: Path) -> list[trimesh.Trimesh]:
        """Generate collision geometry using the configured convex decomposition method.

        Args:
            mesh_path: Path to the mesh file (GLTF/GLB/OBJ).

        Returns:
            List of convex trimesh objects from the decomposition.

        Raises:
            RuntimeError: If collision client is not available.
        """
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        # Build parameter dict based on method.
        if self.collision_method == "coacd":
            return self.collision_client.generate_collision_geometry(
                mesh_path=mesh_path,
                method="coacd",
                threshold=self.collision_coacd_cfg.threshold,
                max_convex_hull=self.collision_coacd_cfg.max_convex_hull,
                preprocess_mode=self.collision_coacd_cfg.preprocess_mode,
                preprocess_resolution=self.collision_coacd_cfg.preprocess_resolution,
                resolution=self.collision_coacd_cfg.resolution,
                mcts_nodes=self.collision_coacd_cfg.mcts_nodes,
                mcts_iterations=self.collision_coacd_cfg.mcts_iterations,
                mcts_max_depth=self.collision_coacd_cfg.mcts_max_depth,
                pca=self.collision_coacd_cfg.pca,
                merge=self.collision_coacd_cfg.merge,
                decimate=self.collision_coacd_cfg.decimate,
                max_ch_vertex=self.collision_coacd_cfg.max_ch_vertex,
                extrude=self.collision_coacd_cfg.extrude,
                extrude_margin=self.collision_coacd_cfg.extrude_margin,
                apx_mode=self.collision_coacd_cfg.apx_mode,
                seed=self.collision_coacd_cfg.seed,
            )
        else:
            # V-HACD method.
            return self.collision_client.generate_collision_geometry(
                mesh_path=mesh_path,
                method="vhacd",
                max_convex_hulls=self.collision_vhacd_cfg.max_convex_hulls,
                vhacd_resolution=self.collision_vhacd_cfg.resolution,
                max_recursion_depth=self.collision_vhacd_cfg.max_recursion_depth,
                max_num_vertices_per_ch=self.collision_vhacd_cfg.max_num_vertices_per_ch,
                min_volume_percent_error=self.collision_vhacd_cfg.min_volume_percent_error,
                shrink_wrap=self.collision_vhacd_cfg.shrink_wrap,
                fill_mode=self.collision_vhacd_cfg.fill_mode,
                min_edge_length=self.collision_vhacd_cfg.min_edge_length,
                find_best_plane=self.collision_vhacd_cfg.find_best_plane,
            )

    def _validate_sam3d_config(self) -> None:
        """Validate SAM3D configuration at startup.

        Raises:
            ValueError: If SAM3D configuration is invalid or missing required fields.
            FileNotFoundError: If checkpoint files do not exist.
        """
        if "sam3d" not in self.cfg.asset_manager:
            raise ValueError(
                "SAM3D backend selected but 'sam3d' configuration is missing. "
                "Add 'sam3d' section to asset_manager config."
            )

        sam3d_cfg = self.cfg.asset_manager.sam3d

        # Validate required checkpoint fields.
        required_fields = ["sam3_checkpoint", "sam3d_checkpoint"]
        for field in required_fields:
            if field not in sam3d_cfg:
                raise ValueError(f"SAM3D configuration missing required field: {field}")

        # Validate checkpoint files exist.
        sam3_checkpoint = Path(sam3d_cfg.sam3_checkpoint)
        sam3d_checkpoint = Path(sam3d_cfg.sam3d_checkpoint)

        if not sam3_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {sam3_checkpoint}. "
                f"Run 'bash scripts/install_sam3d.sh' to download checkpoints."
            )

        if not sam3d_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM 3D Objects checkpoint not found: {sam3d_checkpoint}. "
                f"Run 'bash scripts/install_sam3d.sh' to download checkpoints."
            )

        # Validate mode field.
        mode = sam3d_cfg.mode
        if mode not in ["foreground", "object_description"]:
            raise ValueError(
                f"Invalid SAM3D mode: {mode}. "
                "Must be 'foreground' or 'object_description'."
            )

        # Validate threshold.
        threshold = sam3d_cfg.threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Invalid SAM3D threshold: {threshold}. Must be between 0.0 and 1.0."
            )

        console_logger.info(
            f"SAM3D configuration validated successfully (mode={mode}, "
            f"threshold={threshold})"
        )

    def _retrieve_hssd_assets(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Retrieve assets from HSSD library using server client.

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with retrieved assets.
        """
        if self.hssd_client is None:
            raise RuntimeError("HSSD retrieval client not initialized")
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Retrieving {len(request.object_descriptions)} assets from HSSD server"
        )

        # Create asset path configurations for output directories.
        asset_path_configs = self._create_asset_paths(
            object_descriptions=request.object_descriptions,
            short_names=request.short_names,
        )

        # Ensure output directories exist.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Create batch requests for HSSD server with client-specified output dirs.
        retrieval_requests = [
            HssdRetrievalServerRequest(
                object_description=desc,
                object_type=request.object_type.value,
                desired_dimensions=tuple(dims) if dims else None,
                output_dir=str(config.sdf_dir),
                scene_id=request.scene_id,
            )
            for desc, dims, config in zip(
                request.object_descriptions,
                request.desired_dimensions,
                asset_path_configs,
            )
        ]

        successful_objects: list[SceneObject] = []
        failed_assets: list[FailedAsset] = []

        # Submit batch to server and process streaming responses.
        for index, response in self.hssd_client.retrieve_objects(retrieval_requests):
            desc = request.object_descriptions[index]
            short_name = request.short_names[index]
            config = asset_path_configs[index]

            try:
                console_logger.info(
                    "Processing HSSD response "
                    f"{index+1}/{len(request.object_descriptions)}: '{desc}'"
                )

                # Server returns mesh path (already exported to our output_dir).
                if not response.results:
                    raise ValueError("No results returned from HSSD server")

                result = response.results[0]  # Get top result.
                server_mesh_path = Path(result.mesh_path)
                mesh_id = result.hssd_id

                # Server exported to our specified output_dir, convert GLB to GLTF if
                # needed. Uses BlenderServer for crash isolation.
                if server_mesh_path.suffix.lower() == ".glb":
                    # Server exported GLB, convert to GLTF with Y-up coordinates.
                    gltf_path = server_mesh_path.with_suffix(".gltf")
                    self.blender_server.convert_glb_to_gltf(
                        input_path=server_mesh_path,
                        output_path=gltf_path,
                        export_yup=True,
                    )
                    server_mesh_path.unlink()  # Remove GLB after conversion.
                else:
                    # Already GLTF, use as-is.
                    gltf_path = server_mesh_path

                # Run VLM analysis for material and mass estimation.
                # Use HSSD-specific prompts and only side views to constrain
                # rotation to Z-axis. Orientation (Z-up) is correct from HSSD
                # transformation pipeline.
                # Create debug directory for saving multi-view physics analysis images.
                debug_dir = self.debug_dir / short_name

                console_logger.info(
                    f"Running VLM analysis for HSSD material/mass: {short_name}"
                )
                vlm_physics = analyze_mesh_orientation_and_material(
                    mesh_path=gltf_path,
                    vlm_service=self.vlm_service,
                    cfg=self.cfg,
                    elevation_degrees=self.side_view_elevation_degrees,
                    blender_server=self.blender_server,
                    num_side_views=self.num_side_views_for_physics_analysis,
                    prompt_type="hssd",
                    include_vertical_views=False,
                    debug_output_dir=debug_dir,
                )
                console_logger.info(
                    f"VLM analysis complete: material={vlm_physics.material}, "
                    f"mass={vlm_physics.mass_kg}kg, front={vlm_physics.front_axis}"
                )

                # Use VLM's material, mass, and front axis determination.
                # up_axis is always Z for HSSD (validated by VLM).
                physics_analysis = MeshPhysicsAnalysis(
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    material=vlm_physics.material,
                    mass_kg=vlm_physics.mass_kg,
                    mass_range_kg=vlm_physics.mass_range_kg,
                )

                # Canonicalize mesh orientation to align with scenesmith canonical
                # (Z-up, Y-forward). For HSSD objects already with front=+Y, this is
                # a no-op (fast return). Otherwise, applies Z-rotation to align front.
                console_logger.info(
                    f"Canonicalizing HSSD mesh: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis} → +Y"
                )
                final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
                canonicalize_mesh(
                    gltf_path=gltf_path,
                    output_path=final_gltf_path,
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    blender_server=self.blender_server,
                    object_type=request.object_type,
                )

                # Generate collision geometry via convex decomposition server.
                collision_pieces = self._generate_collision_geometry(final_gltf_path)

                # Load mesh for bounding box calculation.
                mesh = load_mesh_as_trimesh(final_gltf_path, force_merge=True)

                sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
                generate_drake_sdf(
                    visual_mesh_path=final_gltf_path,
                    collision_pieces=collision_pieces,
                    physics_analysis=physics_analysis,
                    output_path=sdf_path,
                    asset_name=config.short_name,
                )

                # Extract bounding box from Y-up GLTF.
                bounds = mesh.bounds  # In Y-up coordinates (GLTF native format).

                # Transform from Y-up (GLTF) to Z-up (Drake) coordinate system.
                # Y-up → Z-up transformation: (x, y, z) → (x, -z, y)
                # Maps: X→X (right), Y→Z (up), Z→-Y (forward with sign flip).
                bbox_min_yup = bounds[0]
                bbox_max_yup = bounds[1]

                # Apply coordinate transformation.
                bbox_min = np.array(
                    [bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]]
                )
                bbox_max = np.array(
                    [bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]]
                )

                # Ensure min < max after transformation (negation can swap order).
                bbox_min, bbox_max = (
                    np.minimum(bbox_min, bbox_max),
                    np.maximum(bbox_min, bbox_max),
                )

                # Create SceneObject using shared helper.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={
                        "asset_source": "hssd",
                        "hssd_mesh_id": mesh_id,
                    },
                )

                successful_objects.append(scene_obj)

                console_logger.info(
                    f"HSSD asset retrieved successfully: {config.short_name}"
                )

            except Exception as e:
                console_logger.error(
                    f"Failed to process HSSD asset '{desc}': {e}", exc_info=True
                )
                failed_assets.append(
                    FailedAsset(index=index, description=desc, error_message=str(e))
                )

        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def _retrieve_objaverse_assets(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Retrieve assets from Objaverse (ObjectThor) library using server client.

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with retrieved assets.
        """
        if self.objaverse_client is None:
            raise RuntimeError("Objaverse retrieval client not initialized")
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Retrieving {len(request.object_descriptions)} assets from Objaverse server"
        )

        # Create asset path configurations for output directories.
        asset_path_configs = self._create_asset_paths(
            object_descriptions=request.object_descriptions,
            short_names=request.short_names,
        )

        # Ensure output directories exist.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Create batch requests for Objaverse server with client-specified output dirs.
        retrieval_requests = [
            ObjaverseRetrievalServerRequest(
                object_description=desc,
                object_type=request.object_type.value,
                desired_dimensions=tuple(dims) if dims else None,
                output_dir=str(config.sdf_dir),
                scene_id=request.scene_id,
            )
            for desc, dims, config in zip(
                request.object_descriptions,
                request.desired_dimensions,
                asset_path_configs,
            )
        ]

        successful_objects: list[SceneObject] = []
        failed_assets: list[FailedAsset] = []

        # Submit batch to server and process streaming responses.
        for index, response in self.objaverse_client.retrieve_objects(
            retrieval_requests
        ):
            desc = request.object_descriptions[index]
            short_name = request.short_names[index]
            config = asset_path_configs[index]

            try:
                console_logger.info(
                    "Processing Objaverse response "
                    f"{index+1}/{len(request.object_descriptions)}: '{desc}'"
                )

                # Server returns mesh path (already exported to our output_dir).
                if not response.results:
                    raise ValueError("No results returned from Objaverse server")

                result = response.results[0]  # Get top result.
                server_mesh_path = Path(result.mesh_path)
                mesh_id = result.objaverse_uid

                # Server exported to our specified output_dir, convert GLB to GLTF if
                # needed. Uses BlenderServer for crash isolation.
                if server_mesh_path.suffix.lower() == ".glb":
                    # Server exported GLB, convert to GLTF with Y-up coordinates.
                    gltf_path = server_mesh_path.with_suffix(".gltf")
                    self.blender_server.convert_glb_to_gltf(
                        input_path=server_mesh_path,
                        output_path=gltf_path,
                        export_yup=True,
                    )
                    server_mesh_path.unlink()  # Remove GLB after conversion.
                else:
                    # Already GLTF, use as-is.
                    gltf_path = server_mesh_path

                # Run VLM analysis for orientation, material and mass estimation.
                console_logger.info(
                    f"Running VLM analysis for Objaverse orientation/material/mass: "
                    f"{short_name}"
                )
                vlm_physics = analyze_mesh_orientation_and_material(
                    mesh_path=gltf_path,
                    vlm_service=self.vlm_service,
                    cfg=self.cfg,
                    elevation_degrees=self.side_view_elevation_degrees,
                    blender_server=self.blender_server,
                    num_side_views=self.num_side_views_for_physics_analysis,
                    prompt_type="generated",  # Full VLM analysis (not pre-canonicalized).
                    include_vertical_views=True,
                    debug_output_dir=self.debug_dir / short_name,
                )
                console_logger.info(
                    f"VLM analysis complete: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis}, material={vlm_physics.material}, "
                    f"mass={vlm_physics.mass_kg}kg"
                )

                # Use VLM's orientation, material, and mass determination.
                physics_analysis = MeshPhysicsAnalysis(
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    material=vlm_physics.material,
                    mass_kg=vlm_physics.mass_kg,
                    mass_range_kg=vlm_physics.mass_range_kg,
                )

                # Canonicalize mesh orientation to align with scenesmith canonical
                # (Z-up, Y-forward).
                console_logger.info(
                    f"Canonicalizing Objaverse mesh: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis} → +Y"
                )
                final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
                canonicalize_mesh(
                    gltf_path=gltf_path,
                    output_path=final_gltf_path,
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    blender_server=self.blender_server,
                    object_type=request.object_type,
                )

                # Generate collision geometry via collision server.
                collision_pieces = self._generate_collision_geometry(final_gltf_path)

                # Load mesh for bounding box calculation.
                mesh = load_mesh_as_trimesh(final_gltf_path, force_merge=True)

                sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
                generate_drake_sdf(
                    visual_mesh_path=final_gltf_path,
                    collision_pieces=collision_pieces,
                    physics_analysis=physics_analysis,
                    output_path=sdf_path,
                    asset_name=config.short_name,
                )

                # Extract bounding box from Y-up GLTF.
                bounds = mesh.bounds  # In Y-up coordinates (GLTF native format).

                # Transform from Y-up (GLTF) to Z-up (Drake) coordinate system.
                # Y-up → Z-up transformation: (x, y, z) → (x, -z, y)
                bbox_min_yup = bounds[0]
                bbox_max_yup = bounds[1]

                # Apply coordinate transformation.
                bbox_min = np.array(
                    [bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]]
                )
                bbox_max = np.array(
                    [bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]]
                )

                # Ensure min < max after transformation (negation can swap order).
                bbox_min, bbox_max = (
                    np.minimum(bbox_min, bbox_max),
                    np.maximum(bbox_min, bbox_max),
                )

                # Create SceneObject using shared helper.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={
                        "asset_source": "objaverse",
                        "objaverse_mesh_id": mesh_id,
                    },
                )

                successful_objects.append(scene_obj)

                console_logger.info(
                    f"Objaverse asset retrieved successfully: {config.short_name}"
                )

            except Exception as e:
                console_logger.error(
                    f"Failed to process Objaverse asset '{desc}': {e}", exc_info=True
                )
                failed_assets.append(
                    FailedAsset(index=index, description=desc, error_message=str(e))
                )

        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def _generate_assets_with_model(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate assets using text-to-3D model (Hunyuan3D).

        This method handles the complete generation pipeline:
        - Style change detection and registry reset
        - Request validation (descriptions vs short names, dimensions)
        - Duplicate detection and deduplication
        - Asset path creation
        - Image generation via VLM
        - Mesh generation via geometry server
        - Asset processing and conversion

        Args:
            request: Asset generation request with descriptions and parameters.

        Returns:
            AssetGenerationResult with generated scene objects and metadata.
        """
        # Validate request.
        if len(request.object_descriptions) != len(request.short_names):
            raise ValueError(
                f"Mismatch between descriptions ({len(request.object_descriptions)}) "
                f"and short names ({len(request.short_names)})"
            )

        # Validate desired_dimensions.
        if len(request.desired_dimensions) != len(request.object_descriptions):
            raise ValueError(
                f"Mismatch between desired_dimensions ({len(request.desired_dimensions)}) "
                f"and object_descriptions ({len(request.object_descriptions)})"
            )

        # Detect duplicates based on (description, desired_dimensions).
        unique_items: dict[tuple[str, tuple[float, ...]], int] = {}
        duplicate_indices: dict[str, list[int]] = {}

        for i, (desc, dims) in enumerate(
            zip(request.object_descriptions, request.desired_dimensions)
        ):
            key = (desc, tuple(dims))
            if key in unique_items:
                # This is a duplicate.
                original_idx = unique_items[key]
                if desc not in duplicate_indices:
                    duplicate_indices[desc] = []
                duplicate_indices[desc].append(i)
                console_logger.warning(
                    f"Duplicate detected at index {i}: '{desc}' with dimensions "
                    f"{dims} (same as index {original_idx})"
                )
            else:
                # This is unique.
                unique_items[key] = i

        # Store duplicate info for tool feedback.
        self.last_duplicate_info = duplicate_indices if duplicate_indices else None

        # Log summary if duplicates found.
        if duplicate_indices:
            total_duplicates = sum(
                len(indices) for indices in duplicate_indices.values()
            )
            console_logger.warning(
                f"Found {total_duplicates} duplicate request(s) across "
                f"{len(duplicate_indices)} description(s). Generating only unique items."
            )

        # Build unique request lists.
        unique_indices = sorted(unique_items.values())
        unique_descriptions = [request.object_descriptions[i] for i in unique_indices]
        unique_short_names = [request.short_names[i] for i in unique_indices]
        unique_dimensions = [request.desired_dimensions[i] for i in unique_indices]

        # Create reduced request with only unique items.
        unique_request = AssetGenerationRequest(
            object_descriptions=unique_descriptions,
            short_names=unique_short_names,
            object_type=request.object_type,
            desired_dimensions=unique_dimensions,
            style_context=request.style_context,
            operation_type=request.operation_type,
            scene_id=request.scene_id,
        )

        # Create asset path configurations.
        asset_paths_configs = self._create_asset_paths(
            object_descriptions=unique_request.object_descriptions,
            short_names=unique_request.short_names,
        )

        # Generate images for all assets.
        self._generate_images(
            request=unique_request, asset_paths_configs=asset_paths_configs
        )

        # Convert images to 3D assets and create SceneObjects.
        successful_objects, failed_assets = self._process_assets_to_scene_objects(
            request=unique_request, asset_path_configs=asset_paths_configs
        )

        console_logger.info(
            f"Asset generation completed: {len(successful_objects)} unique objects "
            f"created, {len(failed_assets)} failed"
        )
        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def generate_assets(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        """Generate scene assets using configured source (generated or hssd).

        If router is enabled, analyzes requests to split composites and filter
        items before dispatching to the configured asset source.

        Args:
            request: Asset generation request with descriptions and context.

        Returns:
            AssetGenerationResult with successful assets and failure information.
        """
        console_logger.info(
            f"Starting {request.object_type.value} asset acquisition for "
            f"{len(request.object_descriptions)} items using "
            f"'{self.general_asset_source}' source. Router is "
            f"{'enabled' if self.router is not None else 'disabled'}."
        )

        # If router is enabled, analyze and potentially modify the request.
        if self.router is not None:
            return self._generate_assets_with_router(request)

        # Dispatch based on asset source (router disabled).
        if self.general_asset_source == "hssd":
            return self._retrieve_hssd_assets(request)
        elif self.general_asset_source == "objaverse":
            return self._retrieve_objaverse_assets(request)
        elif self.general_asset_source == "generated":
            return self._generate_assets_with_model(request)
        else:
            # This should never happen due to __init__ validation.
            raise ValueError(f"Unknown asset source: {self.general_asset_source}")

    def _generate_assets_with_router(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate assets using router for LLM-advised analysis and validation.

        Two-phase processing for thread safety:

        **Phase 1 - Parallel (thread-safe HTTP calls):**
        1. Validate request and check for style changes
        2. Deduplicate by (description, dimensions) to save LLM calls
        3. LLM analysis per unique item (split composites, select strategies)
        4. Parallel generation/retrieval via geometry or HSSD server
        5. VLM validation with retry loop (configured max_retries per strategy)

        **Phase 2 - Sequential (main thread, uses bpy):**
        6. GLB→GLTF conversion, floater removal, mesh canonicalization
        7. CoACD collision geometry, SDF generation
        8. Build SceneObjects and modification_info

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with modification_info if request was modified.
        """
        # Validate request lengths.
        if len(request.object_descriptions) != len(request.short_names):
            raise ValueError(
                f"Mismatch between descriptions ({len(request.object_descriptions)}) "
                f"and short names ({len(request.short_names)})"
            )

        if len(request.desired_dimensions) != len(request.object_descriptions):
            raise ValueError(
                f"Mismatch between desired_dimensions ({len(request.desired_dimensions)}) "
                f"and object_descriptions ({len(request.object_descriptions)})"
            )

        all_items: list[AssetItem] = []
        all_discarded_manipulands: list[str] = []
        original_descriptions: list[str] = []
        had_modifications = False
        failed_assets: list[FailedAsset] = []

        # Pre-analysis deduplication: group by (description, dimensions) to save LLM calls.
        # Track duplicates for tool feedback (same format as _generate_assets_with_model).
        unique_requests: dict[tuple[str, tuple[float, ...]], int] = {}
        duplicate_indices: dict[str, list[int]] = {}

        for idx, (desc, dims) in enumerate(
            zip(request.object_descriptions, request.desired_dimensions)
        ):
            key = (desc, tuple(dims))
            if key in unique_requests:
                # Track duplicate.
                if desc not in duplicate_indices:
                    duplicate_indices[desc] = []
                duplicate_indices[desc].append(idx)
            else:
                unique_requests[key] = idx

        # Store duplicate info for tool feedback.
        self.last_duplicate_info = duplicate_indices if duplicate_indices else None

        if len(unique_requests) < len(request.object_descriptions):
            console_logger.info(
                f"Pre-analysis deduplication: {len(request.object_descriptions)} requests "
                f"-> {len(unique_requests)} unique"
            )

        # Parallel analysis: LLM API calls are thread-safe.
        configured_workers = self.cfg.asset_manager.router.parallel_workers
        max_workers = min(configured_workers, len(unique_requests))

        console_logger.info(
            f"Analyzing {len(unique_requests)} requests in parallel "
            f"with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.router.analyze_request,
                    description=desc,
                    dimensions=list(dims),
                ): (idx, desc)
                for (desc, dims), idx in unique_requests.items()
            }

            for future in as_completed(futures):
                idx, desc = futures[future]
                try:
                    analysis = future.result()

                    if analysis.error:
                        console_logger.warning(
                            f"Router rejected '{desc}': {analysis.error}"
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx,
                                description=desc,
                                error_message=analysis.error,
                            )
                        )
                        continue

                    # Validate item types match this agent.
                    type_error = self.router.validate_item_types(analysis.items)
                    if type_error:
                        console_logger.warning(
                            f"Router type validation failed: {type_error}"
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx, description=desc, error_message=type_error
                            )
                        )
                        continue

                    # Collect items and track modifications.
                    all_items.extend(analysis.items)

                    if analysis.was_modified:
                        had_modifications = True
                        original_descriptions.append(
                            analysis.original_description or desc
                        )
                        if analysis.discarded_manipulands:
                            all_discarded_manipulands.extend(
                                analysis.discarded_manipulands
                            )

                except Exception as e:
                    console_logger.error(
                        f"Analysis failed for '{desc}': {e}", exc_info=True
                    )
                    failed_assets.append(
                        FailedAsset(index=idx, description=desc, error_message=str(e))
                    )

        if not all_items:
            console_logger.warning("Router returned no items to generate")
            return AssetGenerationResult(
                successful_assets=[],
                failed_assets=failed_assets,
                modification_info=None,
            )

        # Deduplicate items by description (same description = generate once).
        unique_items: dict[str, AssetItem] = {}
        for item in all_items:
            if item.description not in unique_items:
                unique_items[item.description] = item
        console_logger.info(
            f"Router produced {len(unique_items)} unique items from "
            f"{len(request.object_descriptions)} requests"
        )

        # Generate/retrieve using router. Handles multiple asset sources internally.
        result = self._generate_items_with_validation(
            unique_items=unique_items, request=request
        )

        # Build modification_info if request was modified.
        modification_info = None
        if had_modifications:
            modification_info = ModificationInfo(
                original_description=", ".join(original_descriptions),
                resulting_descriptions=[
                    item.description for item in unique_items.values()
                ],
                discarded_manipulands=(
                    all_discarded_manipulands if all_discarded_manipulands else None
                ),
            )

        # Combine failed assets from analysis phase with those from generation phase.
        all_failed = failed_assets + result.failed_assets

        return AssetGenerationResult(
            successful_assets=result.successful_assets,
            failed_assets=all_failed,
            modification_info=modification_info,
        )

    def _generate_items_with_validation(
        self, unique_items: dict[str, "AssetItem"], request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate items with overlapped generation and conversion.

        Generates geometry via parallel HTTP calls (thread-safe) and converts each
        mesh to a simulation asset immediately as it completes. This overlaps
        GPU-bound generation with CPU-bound conversion for better resource utilization.

        The main thread runs the as_completed loop and handles conversion (bpy),
        while worker threads continue fetching geometry in parallel.

        Args:
            unique_items: Dict of description -> AssetItem to generate.
            request: Original request (for style_context, object_type).

        Returns:
            AssetGenerationResult with successful assets and failures.
        """
        failed_assets: list[FailedAsset] = []
        successful_assets: list[SceneObject] = []

        configured_workers = self.cfg.asset_manager.router.parallel_workers
        items_list = list(unique_items.items())
        max_workers = min(configured_workers, len(items_list))

        console_logger.info(
            f"Generating {len(items_list)} items with {max_workers} parallel workers "
            "(overlapping generation with conversion)"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_geometry_with_validation,
                    item=item,
                    request=request,
                ): (idx, desc, item)
                for idx, (desc, item) in enumerate(items_list)
            }

            for future in as_completed(futures):
                idx, desc, item = futures[future]
                try:
                    generated = future.result()
                    if generated is None:
                        console_logger.warning(f"All attempts exhausted for '{desc}'")
                        failed_assets.append(
                            FailedAsset(
                                index=idx,
                                description=desc,
                                error_message="All generation/retrieval attempts exhausted",
                            )
                        )
                        continue

                    console_logger.info(
                        f"Geometry acquired for '{desc}', converting..."
                    )

                    # Convert immediately while other geometries are still generating.
                    # This runs on main thread (bpy) while workers fetch next geometry.
                    try:
                        # Handle ArticulatedGeometry (SDF assets) vs GeneratedGeometry.
                        if isinstance(generated, ArticulatedGeometry):
                            scene_obj = self._convert_articulated_to_scene_object(
                                articulated=generated, request=request
                            )
                        else:
                            scene_obj = self._convert_generated_to_scene_object(
                                item=item, generated=generated, request=request
                            )
                        successful_assets.append(scene_obj)
                        console_logger.info(f"Successfully converted asset: '{desc}'")
                    except Exception as e:
                        console_logger.error(
                            f"Mesh conversion failed for '{desc}': {e}", exc_info=True
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx, description=desc, error_message=str(e)
                            )
                        )

                except Exception as e:
                    console_logger.error(
                        f"Geometry generation failed for '{desc}': {e}", exc_info=True
                    )
                    failed_assets.append(
                        FailedAsset(index=idx, description=desc, error_message=str(e))
                    )

        console_logger.info(
            f"Router generation completed: {len(successful_assets)} success, "
            f"{len(failed_assets)} failed"
        )

        return AssetGenerationResult(
            successful_assets=successful_assets, failed_assets=failed_assets
        )

    def _generate_geometry_with_validation(
        self, item: AssetItem, request: AssetGenerationRequest
    ) -> GeneratedGeometry | ArticulatedGeometry | None:
        """Generate/retrieve validated geometry for a single item. Thread-safe.

        This method only performs HTTP-based operations (geometry server, HSSD server,
        BlenderServer for validation rendering) and is safe to call from worker threads.

        Args:
            item: The asset item to generate/retrieve.
            request: Original request (for style_context).

        Returns:
            GeneratedGeometry or ArticulatedGeometry if successful,
            None if all strategies/candidates exhausted.
        """
        return self.router.generate_with_validation(
            item=item,
            geometry_client=self.geometry_client,
            image_generator=self.image_generator,
            images_dir=self.images_dir,
            geometry_dir=self.geometry_dir,
            debug_dir=self.debug_dir,
            style_context=request.style_context,
            hssd_client=self.hssd_client,
            objaverse_client=self.objaverse_client,
            articulated_client=self.articulated_client,
            materials_client=self.materials_client,
            scene_id=request.scene_id,
        )

    def _convert_generated_to_scene_object(
        self,
        item: "AssetItem",
        generated: "GeneratedGeometry",
        request: AssetGenerationRequest,
    ) -> SceneObject:
        """Convert validated geometry to SceneObject. Must run on main thread.

        This method uses bpy for GLB→GLTF conversion and must be called from the
        main thread, not from ThreadPoolExecutor workers.

        Args:
            item: The asset item that was generated.
            generated: The validated geometry from router.
            request: Original request (for object_type).

        Returns:
            SceneObject ready for scene placement.

        Raises:
            Exception: If mesh conversion or SDF generation fails.
        """
        # Derive base_name from geometry path (already has unique timestamp or HSSD ID).
        base_name = generated.geometry_path.stem

        config = AssetPathConfig(
            description=item.description,
            short_name=item.short_name,
            image_path=generated.image_path,
            geometry_path=generated.geometry_path,
            sdf_dir=self.sdf_dir / base_name,
        )
        config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Thin coverings use simplified conversion: no VLM analysis.
        # Wall thin coverings (paintings, posters) get collision geometry.
        if generated.asset_source == "thin_covering":
            is_wall_covering = request.object_type == ObjectType.WALL_MOUNTED

            # Only add collision for wall coverings (paintings, posters).
            collision_dims = None
            collision_shape = "rectangular"
            if is_wall_covering and item.dimensions:
                # Wall covering dims: (width, depth, height) where depth is thickness.
                thickness = (
                    self.cfg.asset_manager.router.strategies.thin_covering.thickness_m
                )
                collision_dims = (item.dimensions[0], thickness, item.dimensions[2])
                collision_shape = infer_thin_covering_shape(item.description)

            sdf_path, final_gltf_path, bbox_min, bbox_max = (
                self._convert_thin_covering_to_simulation_asset(
                    geometry_path=generated.geometry_path,
                    config=config,
                    collision_dims=collision_dims,
                    collision_shape=collision_shape,
                )
            )
            initial_scale = 1.0  # Thin coverings don't scale the mesh.
        else:
            # Convert validated geometry to simulation asset (physics analysis, SDF).
            sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale = (
                self._convert_mesh_to_simulation_asset(
                    geometry_path=generated.geometry_path,
                    config=config,
                    object_type=request.object_type,
                    desired_dimensions=item.dimensions,
                    asset_source=generated.asset_source,
                )
            )

        # Build additional metadata using explicit asset_source from GeneratedGeometry.
        additional_metadata = {"asset_source": generated.asset_source}
        if generated.hssd_id is not None:
            additional_metadata["hssd_mesh_id"] = generated.hssd_id

        # Add thin_covering-specific metadata for physics validation.
        if generated.asset_source == "thin_covering":
            additional_metadata["width_m"] = item.dimensions[0]
            additional_metadata["depth_m"] = item.dimensions[1]
            additional_metadata["shape"] = infer_thin_covering_shape(item.description)
            # Wall coverings use Drake collision; floor/manipuland use 2D OBB overlap.
            additional_metadata["is_wall_covering"] = (
                request.object_type == ObjectType.WALL_MOUNTED
            )

        # Keep original object_type - thin coverings are identified via asset_source
        # metadata, not object_type. This preserves semantic category (FURNITURE,
        # WALL_MOUNTED, MANIPULAND) for stage-based filtering in snapshots.
        object_type = request.object_type

        # Create SceneObject.
        return self._create_scene_object(
            config=config,
            object_type=object_type,
            sdf_path=sdf_path,
            final_gltf_path=final_gltf_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            additional_metadata=additional_metadata,
            scale_factor=initial_scale,
        )

    def _convert_articulated_to_scene_object(
        self, articulated: ArticulatedGeometry, request: AssetGenerationRequest
    ) -> SceneObject:
        """Convert articulated retrieval result to SceneObject.

        Unlike generated assets, articulated objects already have:
        - Pre-processed SDF with links and joints
        - Bounding box at default pose (joints=0)
        - No need for VLM analysis or mesh canonicalization

        We combine the visual meshes at default pose for geometry_path (needed
        for collision checks, support surface extraction, snapping).

        Args:
            articulated: The articulated geometry from router.
            request: Original request (for object_type).

        Returns:
            SceneObject ready for scene placement.
        """
        item = articulated.item
        safe_name = self._sanitize_filename(item.short_name)
        base_name = self._unique_asset_basename(safe_name)

        # Create output directory for combined geometry.
        output_dir = self.geometry_dir / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy articulated SDF directory to output for replay and export.
        # The SDF references meshes via relative paths, so we copy the entire directory.
        source_sdf_dir = articulated.sdf_path.parent
        dest_sdf_dir = self.sdf_dir / base_name
        console_logger.info(
            f"Copying articulated SDF directory from {source_sdf_dir} to {dest_sdf_dir}"
        )
        shutil.copytree(source_sdf_dir, dest_sdf_dir)
        copied_sdf_path = dest_sdf_dir / articulated.sdf_path.name

        # Add self-collision filtering if enabled.
        if self.cfg.asset_manager.articulated.enable_self_collision_filtering:
            add_self_collision_filter(copied_sdf_path)

        # Fix ArtVIP texture paths: GLTF files reference textures with relative paths,
        # but textures are in *_meshes/ subdirectories. Copy textures to parent dir.
        for meshes_subdir in dest_sdf_dir.glob("*_meshes"):
            for texture_file in meshes_subdir.glob("*.png"):
                dest_texture = dest_sdf_dir / texture_file.name
                if not dest_texture.exists():
                    shutil.copy2(texture_file, dest_texture)

        # Combine SDF visual meshes at default pose (joints=0) for geometry operations.
        console_logger.info(
            f"Combining articulated meshes at default pose for '{item.description}'"
        )
        combined_mesh = combine_sdf_meshes_at_joint_angles(
            copied_sdf_path, use_max_angles=False
        )

        # Save combined mesh as GLTF for collision checks, snapping, etc.
        combined_gltf_path = output_dir / f"{safe_name}_combined.gltf"
        combined_mesh.export(combined_gltf_path)

        console_logger.info(
            f"Articulated asset combined mesh saved to {combined_gltf_path}"
        )

        # Build metadata for provenance tracking.
        metadata = {
            "asset_source": "articulated",
            "articulated_source": articulated.source,
            "articulated_id": articulated.object_id,
            "is_articulated": True,
            "generation_timestamp": time.time(),
        }

        # Create SceneObject with copied SDF path and combined geometry.
        scene_obj = SceneObject(
            object_id=self.registry.generate_unique_id(item.short_name),
            object_type=request.object_type,
            name=item.short_name,
            description=item.description,
            transform=RigidTransform(),  # Will be set during placement.
            geometry_path=combined_gltf_path,
            sdf_path=copied_sdf_path,
            image_path=None,  # No generated image for articulated assets.
            bbox_min=np.array(articulated.bounding_box_min),
            bbox_max=np.array(articulated.bounding_box_max),
            metadata=metadata,
        )

        # Register the asset for reuse.
        self.registry.register(scene_obj)

        console_logger.info(
            f"Articulated asset registered: {item.short_name} "
            f"(source={articulated.source}, id={articulated.object_id})"
        )

        return scene_obj

    def _create_asset_paths(
        self, object_descriptions: list[str], short_names: list[str]
    ) -> list[AssetPathConfig]:
        """Create file paths and identifiers for each asset to be generated.

        Args:
            object_descriptions: List of object descriptions to generate.
            short_names: List of short names for filesystem-safe file naming.

        Returns:
            List of AssetPathConfig objects containing asset paths and metadata.
        """
        asset_paths = []
        for desc, short_name in zip(object_descriptions, short_names):
            # Use sanitized short name for file naming.
            safe_name = self._sanitize_filename(short_name)
            base_name = self._unique_asset_basename(safe_name)

            asset_paths.append(
                AssetPathConfig(
                    description=desc,
                    short_name=short_name,
                    image_path=self.images_dir / f"{base_name}.png",
                    geometry_path=self.geometry_dir / f"{base_name}.glb",
                    sdf_dir=self.sdf_dir / base_name,
                )
            )
        return asset_paths

    def _generate_images(
        self,
        request: AssetGenerationRequest,
        asset_paths_configs: list[AssetPathConfig],
    ) -> None:
        """Generate images for all assets using the image generator.

        Args:
            request: Asset generation request with style and operation details.
            asset_paths_configs: List of asset path configurations.
        """
        style_prompt = request.style_context or "Modern style"
        console_logger.info(f"Generating {len(request.object_descriptions)} images")
        console_logger.debug(f"Style prompt: {style_prompt}")

        output_paths = [config.image_path for config in asset_paths_configs]

        start_time = time.time()
        self.image_generator.generate_images(
            style_prompt=style_prompt,
            object_descriptions=request.object_descriptions,
            output_paths=output_paths,
        )

        elapsed = time.time() - start_time
        console_logger.info(
            f"Generated {len(request.object_descriptions)} images in "
            f"{elapsed:.2f} seconds"
        )

    def _process_assets_to_scene_objects(
        self, request: AssetGenerationRequest, asset_path_configs: list[AssetPathConfig]
    ) -> tuple[list[SceneObject], list[FailedAsset]]:
        """Convert generated images to 3D assets and create SceneObjects.

        Uses batch processing to optimize GPU utilization by pipelining geometry
        generation and Drake SDF conversion. Handles failures gracefully by
        collecting failed assets instead of raising exceptions, allowing all
        generated geometries to be processed.

        Args:
            request: Asset generation request.
            asset_path_configs: List of asset path configurations.

        Returns:
            Tuple of (successful_objects, failed_assets). The successful_objects
            list contains SceneObject instances ready for placement. The failed_assets
            list contains FailedAsset instances with error details.
        """
        if not asset_path_configs:
            return [], []

        # Create Drake asset directories for all configs.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Prepare batch geometry generation requests.
        geometry_requests = []
        for config in asset_path_configs:
            expected_filename = config.geometry_path.name

            # Extract backend configuration.
            backend = self.cfg.asset_manager.backend

            # Prepare SAM3D config if backend is sam3d.
            sam3d_config = None
            if backend == "sam3d":
                sam3d_cfg = self.cfg.asset_manager.sam3d
                mode = sam3d_cfg.mode
                sam3d_config = {
                    "sam3_checkpoint": str(sam3d_cfg.sam3_checkpoint),
                    "sam3d_checkpoint": str(sam3d_cfg.sam3d_checkpoint),
                    "mode": mode,
                    "text_prompt": getattr(sam3d_cfg, "text_prompt", None),
                    "threshold": sam3d_cfg.threshold,
                }
                # Pass object description for "object_description" mode.
                # Uses the same description that generated the image for
                # semantic-aware segmentation.
                if mode == "object_description":
                    sam3d_config["object_description"] = config.description

            geometry_request = GeometryGenerationServerRequest(
                image_path=str(config.image_path),
                output_dir=str(self.geometry_dir),
                prompt=config.description,
                debug_folder=str(self.debug_dir),
                output_filename=expected_filename,
                backend=backend,
                sam3d_config=sam3d_config,
                scene_id=request.scene_id,
            )
            geometry_requests.append(geometry_request)

        console_logger.info(
            f"Submitting batch geometry generation for {len(geometry_requests)} assets"
        )

        # Initialize result tracking.
        scene_objects = []
        failed_assets = []

        # Process batch results as they stream back.
        # This enables pipelining: Drake conversion for asset N while GPU processes
        # asset N+1.
        for index, result in self.geometry_client.generate_geometries(
            geometry_requests
        ):
            # Handle geometry generation failures.
            if isinstance(result, GeometryGenerationError):
                console_logger.error(
                    f"Geometry generation failed for asset {index + 1}/"
                    f"{len(asset_path_configs)} ({asset_path_configs[index].description}): "
                    f"{result.error_message}"
                )
                failed_assets.append(
                    FailedAsset(
                        index=index,
                        description=asset_path_configs[index].description,
                        error_message=result.error_message,
                    )
                )
                continue

            try:
                config = asset_path_configs[index]
                server_geometry_path = Path(result.geometry_path)

                console_logger.info(
                    f"Converting asset {index + 1}/{len(asset_path_configs)} to Drake "
                    f"format: {config.description}"
                )

                # Process mesh: VLM → canonicalize → scale → collision → SDF.
                sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale = (
                    self._convert_mesh_to_simulation_asset(
                        geometry_path=server_geometry_path,
                        config=config,
                        object_type=request.object_type,
                        desired_dimensions=request.desired_dimensions[index],
                    )
                )

                # Create the SceneObject.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={"asset_source": "generated"},
                    scale_factor=initial_scale,
                )

                scene_objects.append(scene_obj)
                console_logger.info(
                    f"Successfully generated asset {index + 1}/{len(asset_path_configs)}: "
                    f"{config.description}"
                )

            except Exception as e:
                # Log failure but continue processing remaining assets.
                console_logger.error(
                    f"Failed to process asset {index + 1}/{len(asset_path_configs)} "
                    f"({asset_path_configs[index].description}): {e}",
                    exc_info=True,
                )
                failed_assets.append(
                    FailedAsset(
                        index=index,
                        description=asset_path_configs[index].description,
                        error_message=str(e),
                    )
                )

        # Log summary.
        if failed_assets:
            console_logger.warning(
                f"Asset generation completed with {len(failed_assets)} failure(s) "
                f"and {len(scene_objects)} success(es)"
            )
        else:
            console_logger.info(
                f"Successfully processed all {len(scene_objects)} assets"
            )

        return scene_objects, failed_assets

    def _convert_mesh_to_simulation_asset(
        self,
        geometry_path: Path,
        config: AssetPathConfig,
        object_type: ObjectType,
        desired_dimensions: list[float] | None = None,
        asset_source: str = "generated",
    ) -> tuple[Path, Path, np.ndarray, np.ndarray, float]:
        """Convert mesh to a simulatable Drake SDF.

        Pipeline:
        - Convert GLB → Y-up GLTF (enables VLM analysis in Blender's Z-up space)
        - Remove mesh floaters (disconnected components below volume threshold)
        - VLM analysis → orientation + material + mass (in Blender coords)
        - Canonicalize in Blender → rotate to canonical orientation + placement
          (Y-up GLTF input → Z-up GLTF output for Drake)
        - Scale to desired dimensions (if provided)
        - Collision → CoACD decomposition
        - SDF → Drake format with physics properties

        Multi-view images used for VLM physics analysis are saved to
        generated_assets/debug/<base_name>/ where <base_name> follows the pattern
        {sanitized_short_name}_{timestamp} (e.g., "office_desk_A_1759997032").

        Args:
            geometry_path: Path to raw GLB mesh from Hunyuan3D or HSSD.
            config: Asset path configuration.
            object_type: Type of object (determines placement strategy).
            desired_dimensions: Optional dimensions (width, depth, height) from agent.
            asset_source: Source of the asset ("generated" or "hssd"). HSSD assets
                use specialized VLM prompts and skip vertical views since they're
                already upright.

        Returns:
            Tuple of (sdf_path, final_gltf_path, bbox_min, bbox_max, scale_factor).
            The scale_factor is the uniform scaling applied during mesh scaling
            (1.0 if no scaling was applied). This is needed to correctly scale
            HSSD pre-computed support surfaces.
        """
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Processing mesh ({geometry_path}) to simulation asset "
            f"(object_type={object_type.value})"
        )

        # Convert GLB to Y-up GLTF (enables VLM analysis in Blender's Z-up space).
        # Uses BlenderServer for crash isolation.
        gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
        self.blender_server.convert_glb_to_gltf(
            input_path=geometry_path,
            output_path=gltf_path,
            export_yup=True,
        )

        # Remove floaters from mesh before VLM analysis.
        console_logger.info("Removing disconnected mesh floaters")
        remove_mesh_floaters(
            mesh_path=gltf_path,
            output_path=gltf_path,
            distance_threshold=self.cfg.asset_manager.floater_distance_threshold,
        )

        # VLM analysis for orientation, material, mass.
        # Create debug directory for saving multi-view physics analysis images.
        # Use geometry_path stem to match asset naming pattern (e.g., "desk_A_1234567890").
        debug_dir = self.debug_dir / config.geometry_path.stem

        # HSSD assets use specialized prompts and skip vertical views since they're
        # already upright (Z-up). Generated assets need full orientation analysis.
        is_hssd = asset_source == "hssd"
        prompt_type = "hssd" if is_hssd else "generated"
        include_vertical_views = not is_hssd

        console_logger.info(
            f"Running VLM analysis for mesh physics "
            f"(asset_source={asset_source}, prompt_type={prompt_type})"
        )
        physics_analysis = analyze_mesh_orientation_and_material(
            mesh_path=gltf_path,
            vlm_service=self.vlm_service,
            cfg=self.cfg,
            elevation_degrees=self.side_view_elevation_degrees,
            blender_server=self.blender_server,
            num_side_views=self.num_side_views_for_physics_analysis,
            debug_output_dir=debug_dir,
            prompt_type=prompt_type,
            include_vertical_views=include_vertical_views,
        )

        console_logger.info(
            f"VLM analysis complete: up={physics_analysis.up_axis}, "
            f"front={physics_analysis.front_axis}, material={physics_analysis.material}, "
            f"mass={physics_analysis.mass_kg}kg"
        )

        # Canonicalize mesh in Blender (rotate to canonical orientation + placement).
        # Input: Y-up GLTF, Output: Z-up GLTF for Drake.
        canonical_path = config.sdf_dir / f"{config.short_name}_canonical.gltf"
        canonicalize_mesh(
            gltf_path=gltf_path,
            output_path=canonical_path,
            up_axis=physics_analysis.up_axis,
            front_axis=physics_analysis.front_axis,
            blender_server=self.blender_server,
            object_type=object_type,
        )

        # Scale mesh to desired dimensions (if provided).
        # For generated assets: scale_factor=1.0 because support surface extraction runs
        # on the already-scaled mesh, so surfaces are at correct dimensions.
        # For HSSD assets: scale_factor=applied_scale because pre-computed surfaces
        # are at original HSSD dimensions and need scaling.
        final_gltf_path = canonical_path
        initial_scale = 1.0
        if desired_dimensions is not None:
            console_logger.info(
                f"Scaling mesh to desired dimensions: {desired_dimensions}"
            )
            final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
            preserve_aspect_ratio = is_hssd
            desired_dimensions_frame = "mesh" if is_hssd else "drake_yup_to_zup"
            final_gltf_path, applied_scale = scale_mesh_uniformly_to_dimensions(
                mesh_path=canonical_path,
                desired_dimensions=desired_dimensions,
                output_path=final_gltf_path,
                min_dimension_meters=self.min_mesh_dimension_meters,
                relative_threshold=self.mesh_relative_dimension_threshold,
                preserve_aspect_ratio=preserve_aspect_ratio,
                desired_dimensions_frame=desired_dimensions_frame,
            )
            # HSSD pre-computed surfaces are at original mesh dimensions.
            # They need scale_factor to match the physical scaling applied above.
            if is_hssd:
                initial_scale = applied_scale
        else:
            # Rename canonical to final name if no scaling needed.
            final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
            canonical_path.rename(final_gltf_path)

        # Generate collision geometry via convex decomposition server.
        collision_pieces = self._generate_collision_geometry(final_gltf_path)

        # Load mesh for bounding box calculation.
        mesh = load_mesh_as_trimesh(final_gltf_path, force_merge=True)

        # Generate Drake SDF.
        sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
        generate_drake_sdf(
            visual_mesh_path=final_gltf_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics_analysis,
            output_path=sdf_path,
            asset_name=config.short_name,
        )

        # Extract bounding box from scaled mesh.
        bounds = mesh.bounds  # In Y-up coordinates (GLTF native format).

        # Transform from Y-up (GLTF) to Z-up (Drake) coordinate system.
        # Y-up → Z-up transformation: (x, y, z) → (x, -z, y)
        # Maps: X→X (right), Y→Z (up), Z→-Y (forward with sign flip).
        bbox_min_yup = bounds[0]
        bbox_max_yup = bounds[1]

        # Apply coordinate transformation.
        bbox_min = np.array([bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]])
        bbox_max = np.array([bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]])

        # Ensure min < max after transformation (negation can swap order).
        bbox_min, bbox_max = (
            np.minimum(bbox_min, bbox_max),
            np.maximum(bbox_min, bbox_max),
        )

        console_logger.info(
            f"Drake SDF complete: SDF at {sdf_path}, bounds: {bbox_min} to {bbox_max}"
        )

        return sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale

    def _convert_thin_covering_to_simulation_asset(
        self,
        geometry_path: Path,
        config: AssetPathConfig,
        collision_dims: tuple[float, float, float] | None = None,
        collision_shape: str = "rectangular",
    ) -> tuple[Path, Path, np.ndarray, np.ndarray]:
        """Convert thin covering mesh to Drake SDF (simplified pipeline).

        Thin coverings are static decorative objects that don't require:
        - VLM orientation analysis (already correctly oriented)
        - Canonicalization (already in correct pose)
        - Collision geometry for floor/manipuland coverings (decorative only)

        Wall thin coverings (paintings, posters) DO get collision geometry so
        Drake can detect furniture collisions.

        Pipeline:
        - Convert GLB → GLTF with separate textures (for Drake)
        - Generate static SDF (with optional collision for wall coverings)
        - Compute bounding box from mesh

        Args:
            geometry_path: Path to thin covering GLB file.
            config: Asset path configuration.
            collision_dims: Optional (width, depth, height) for collision geometry.
                Used for wall thin coverings.
            collision_shape: Shape of collision ("rectangular" or "circular").

        Returns:
            Tuple of (sdf_path, final_gltf_path, bbox_min, bbox_max).
        """
        console_logger.info(f"Processing thin covering ({geometry_path}) to static SDF")

        # Convert GLB to GLTF with separate textures for Drake.
        # Uses BlenderServer for crash isolation.
        gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
        self.blender_server.convert_glb_to_gltf(
            input_path=geometry_path,
            output_path=gltf_path,
            export_yup=True,
        )

        # Generate static SDF (with optional collision geometry for wall coverings).
        sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
        generate_thin_covering_sdf(
            visual_mesh_path=gltf_path,
            output_path=sdf_path,
            model_name=config.short_name,
            collision_dims=collision_dims,
            collision_shape=collision_shape,
        )

        # Load mesh for bounding box calculation.
        mesh = load_mesh_as_trimesh(gltf_path, force_merge=True)
        bounds = mesh.bounds  # In Y-up coordinates (GLTF native format).

        # Transform from Y-up (GLTF) to Z-up (Drake) coordinate system.
        bbox_min_yup = bounds[0]
        bbox_max_yup = bounds[1]

        # Apply coordinate transformation: (x, y, z)_Yup → (x, -z, y)_Zup
        bbox_min = np.array([bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]])
        bbox_max = np.array([bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]])

        # Ensure min < max after transformation.
        bbox_min, bbox_max = (
            np.minimum(bbox_min, bbox_max),
            np.maximum(bbox_min, bbox_max),
        )

        console_logger.info(
            f"Thin covering SDF complete: {sdf_path}, bounds: {bbox_min} to {bbox_max}"
        )

        return sdf_path, gltf_path, bbox_min, bbox_max

    def _find_sdf_file(self, sdf_dir: Path) -> Path:
        """Find the generated SDF file in the asset directory.

        Args:
            sdf_dir: Directory containing the generated SDF file.

        Returns:
            Path to the SDF file.

        Raises:
            RuntimeError: If no SDF file or multiple SDF files are found.
        """
        # First try direct search in the directory.
        sdf_files = list(sdf_dir.glob("*.sdf"))

        # If not found, search recursively (create_drake_asset_from_geometry creates
        # nested dirs).
        if not sdf_files:
            sdf_files = list(sdf_dir.glob("**/*.sdf"))

        if not sdf_files:
            raise RuntimeError(f"No SDF file generated in {sdf_dir}")
        if len(sdf_files) > 1:
            raise RuntimeError(f"Multiple SDF files generated in {sdf_dir}")
        return sdf_files[0].absolute()

    def _create_scene_object(
        self,
        config: AssetPathConfig,
        object_type: ObjectType,
        sdf_path: Path,
        final_gltf_path: Path,
        bbox_min: np.ndarray | None = None,
        bbox_max: np.ndarray | None = None,
        additional_metadata: dict | None = None,
        scale_factor: float = 1.0,
    ) -> SceneObject:
        """Convert assets to SceneObject (supports both generated and HSSD).

        Args:
            config: Asset path configuration containing metadata and paths.
            object_type: Type of object.
            sdf_path: Path to the generated SDF file.
            final_gltf_path: Path to the final scaled GLTF mesh file.
            bbox_min: Minimum corner of object-frame bounding box.
            bbox_max: Maximum corner of object-frame bounding box.
            additional_metadata: Optional metadata to merge into the object's
                metadata dict. Useful for HSSD assets to add {"asset_source": "hssd"}.
            scale_factor: Initial uniform scale factor applied during mesh scaling.
                This is needed to correctly scale HSSD pre-computed support surfaces.

        Returns:
            Complete SceneObject ready for scene placement.
        """
        # Base metadata common to all assets.
        metadata = {"generation_timestamp": time.time()}

        # Merge additional metadata (for HSSD: {"asset_source": "hssd"}).
        if additional_metadata:
            metadata.update(additional_metadata)

        scene_obj = SceneObject(
            object_id=self.registry.generate_unique_id(config.short_name),
            object_type=object_type,
            name=config.short_name,
            description=config.description,
            transform=RigidTransform(),  # Will be set during placement.
            geometry_path=final_gltf_path,
            sdf_path=sdf_path,
            image_path=config.image_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            metadata=metadata,
            scale_factor=scale_factor,
        )

        # Register the asset for reuse.
        self.registry.register(scene_obj)

        return scene_obj

    def get_asset_by_id(self, asset_id: UniqueID) -> SceneObject | None:
        """Get a registered asset by ID.

        Args:
            asset_id: Unique identifier of the asset.

        Returns:
            SceneObject if found, None otherwise.
        """
        return self.registry.get(asset_id)

    def list_available_assets(self) -> list[SceneObject]:
        """List all assets available for reuse.

        Returns:
            List of all registered SceneObjects.
        """
        return self.registry.list_all()

    def _extract_bounds_from_visual_mesh(
        self, sdf_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract AABB from the visual GLTF mesh after conversion.

        Args:
            sdf_path: Path to the SDF file.

        Returns:
            Tuple of (bbox_min, bbox_max) arrays.

        Raises:
            FileNotFoundError: If GLTF file is not found.
            ValueError: If mesh cannot be loaded or is invalid.
        """
        # Pattern: {sdf_dir}/{asset_name}/{asset_name}.gltf
        gltf_path = sdf_path.with_suffix(".gltf")

        if not gltf_path.exists():
            raise FileNotFoundError(
                f"Visual GLTF not found at expected path: {gltf_path}"
            )

        # Load mesh using trimesh.
        mesh = trimesh.load(gltf_path, force="mesh")

        # Handle Scene objects (multiple meshes).
        if isinstance(mesh, trimesh.Scene):
            combined_mesh = trimesh.Trimesh()
            for geom in mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
            mesh = combined_mesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load valid mesh from {gltf_path}")

        # Extract bounds.
        bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        bbox_min = bounds[0]
        bbox_max = bounds[1]

        console_logger.debug(
            f"Extracted bounds from {gltf_path}: min={bbox_min}, max={bbox_max}"
        )

        return bbox_min, bbox_max

    def clear_asset_registry(self) -> None:
        """Clear the asset registry."""
        self.registry.clear()
