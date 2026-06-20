"""Asset router for LLM-advised asset generation."""

import json
import logging
import tempfile
import time
import uuid

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalClient,
    ArticulatedRetrievalServerRequest,
)
from scenesmith.agent_utils.articulated_retrieval_server.dataclasses import (
    ArticulatedRetrievalResult,
)
from scenesmith.agent_utils.asset_router.dataclasses import (
    AnalysisResult,
    ArticulatedGeometry,
    AssetItem,
    GeneratedGeometry,
    ValidationResult,
)
from scenesmith.agent_utils.blender.constants import MATERIAL_VALIDATION_LIGHT_ENERGY
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationError,
)
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.material_generator import (
    MaterialGenerator,
    MaterialGeneratorConfig,
)
from scenesmith.agent_utils.materials_retrieval_server import (
    MaterialsRetrievalServerRequest,
)
from scenesmith.agent_utils.objaverse_retrieval_server.dataclasses import (
    ObjaverseRetrievalServerRequest,
)
from scenesmith.agent_utils.room import AgentType, ObjectType
from scenesmith.agent_utils.thin_covering_generator import (
    create_circular_thin_covering_glb,
    create_rectangular_thin_covering_glb,
    infer_thin_covering_shape,
)
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.prompts import AssetRouterPrompts, prompt_manager
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.blender import BlenderServer
    from scenesmith.agent_utils.geometry_generation_server.client import (
        GeometryGenerationClient,
    )
    from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalClient
    from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
        HssdRetrievalResult,
    )
    from scenesmith.agent_utils.image_generation import BaseImageGenerator
    from scenesmith.agent_utils.materials_retrieval_server import (
        MaterialsRetrievalClient,
    )
    from scenesmith.agent_utils.objaverse_retrieval_server import (
        ObjaverseRetrievalClient,
    )
    from scenesmith.agent_utils.objaverse_retrieval_server.dataclasses import (
        ObjaverseRetrievalResult,
    )

console_logger = logging.getLogger(__name__)


def _unique_asset_basename(short_name: str) -> str:
    """Return a filesystem basename that remains unique under parallel generation."""
    return f"{short_name}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"


class AssetRouter:
    """Routes asset generation requests through LLM analysis and validation.

    This implements a hybrid "LLM-Advised Deterministic Loop" architecture:
    - LLM handles semantic decisions (routing, composite detection, validation)
    - Deterministic Python handles execution (retry logic, fallback chains)
    """

    def __init__(
        self,
        agent_type: AgentType,
        vlm_service: VLMService,
        cfg: DictConfig,
        blender_server: "BlenderServer | None" = None,
    ) -> None:
        """Initialize the asset router.

        Args:
            agent_type: Type of placement agent.
            vlm_service: VLM service for analysis and validation.
            cfg: Configuration with OpenAI and router settings.
            blender_server: Optional BlenderServer for thread-safe validation rendering.
                When provided and running, validation uses HTTP requests to the server
                instead of direct BlenderRenderer calls. This is required for parallel
                generation since bpy (Blender Python API) requires main thread execution.

        Raises:
            ValueError: If agent_type is not a placement agent.
        """
        if not agent_type.is_placement_agent:
            raise ValueError(
                f"AssetRouter requires a placement agent, got {agent_type.value}"
            )
        self.agent_type = agent_type
        self.vlm_service = vlm_service
        self.cfg = cfg
        self.blender_server = blender_server
        self.side_view_elevation_degrees = cfg.asset_manager.side_view_elevation_degrees
        self.validation_taa_samples = cfg.asset_manager.validation_taa_samples

    def analyze_request(
        self, description: str, dimensions: list[float]
    ) -> AnalysisResult:
        """Analyze an asset request using VLM.

        Calls the appropriate analysis prompt (furniture or manipuland) to:
        - Extract valid items for this agent type
        - Filter out items for other agents (e.g., furniture agent discards manipulands)
        - Split composite requests into individual items
        - Select appropriate generation strategies

        Retries if parsing fails for all items (e.g., invalid object_type values).

        Args:
            description: Object description from the designer.
            dimensions: Desired dimensions [width, depth, height] in meters.

        Returns:
            AnalysisResult with extracted items and any modifications.
        """
        # Select prompt based on agent type.
        if self.agent_type == AgentType.FURNITURE:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_FURNITURE
        elif self.agent_type == AgentType.WALL_MOUNTED:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_WALL
        elif self.agent_type == AgentType.CEILING_MOUNTED:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_CEILING
        else:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_MANIPULAND

        # Render prompt with template variables.
        prompt = prompt_manager.get_prompt(
            prompt_name=prompt_enum, description=description, dimensions=dimensions
        )

        # Call VLM for analysis.
        messages = [{"role": "user", "content": prompt}]

        openai_config = self.cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.asset_analysis
        verbosity = openai_config.verbosity.asset_analysis

        # Retry loop for parsing failures (e.g., LLM returns invalid object_type).
        max_retries = self.cfg.asset_manager.router.analysis_max_retries
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response_text = self.vlm_service.create_completion(
                    model=model,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    response_format={"type": "json_object"},
                )
                elapsed = time.time() - start_time
                response_json = json.loads(response_text)
                console_logger.info(
                    f"Router analysis completed in {elapsed:.1f}s:\n{response_json}"
                )
            except Exception as e:
                console_logger.error(f"VLM analysis failed: {e}")
                return AnalysisResult(
                    items=[],
                    original_description=None,
                    discarded_manipulands=None,
                    error=f"Analysis failed: {e}",
                )

            # Parse response into AnalysisResult.
            result = self._parse_analysis_response(response_json)

            # Check if parsing succeeded or if there's nothing to parse.
            raw_items = response_json.get("items", [])
            if result.items or not raw_items or result.error:
                # Success: got valid items, no items to parse, or explicit error.
                return result

            # All items failed to parse - retry VLM call.
            console_logger.warning(
                f"All {len(raw_items)} items failed to parse "
                f"(attempt {attempt + 1}/{max_retries}), retrying VLM call..."
            )

        # All retries exhausted.
        return AnalysisResult(
            items=[],
            original_description=description,
            discarded_manipulands=None,
            error=f"Failed to parse valid items after {max_retries} attempts",
        )

    def _parse_analysis_response(self, response: dict) -> AnalysisResult:
        """Parse VLM analysis response into AnalysisResult.

        Args:
            response: JSON response from VLM.

        Returns:
            Parsed AnalysisResult.
        """
        # Check for error in response.
        if "error" in response and response["error"]:
            return AnalysisResult(
                items=[],
                original_description=response.get("original_description"),
                discarded_manipulands=response.get("discarded_manipulands"),
                error=response["error"],
            )

        # Parse items.
        items = []
        for item_data in response.get("items", []):
            try:
                object_type = ObjectType(item_data["object_type"].lower())
                strategies = list(item_data["strategies"])
                if (
                    self.agent_type == AgentType.CEILING_MOUNTED
                    and "articulated" in strategies
                ):
                    strategies = [s for s in strategies if s != "articulated"]
                    if not strategies:
                        strategies = ["generated"]
                    console_logger.info(
                        "Removed unsupported articulated strategy from ceiling "
                        f"asset '{item_data.get('description', '<unknown>')}'"
                    )

                items.append(
                    AssetItem(
                        description=item_data["description"],
                        short_name=item_data["short_name"],
                        dimensions=item_data["dimensions"],
                        object_type=object_type,
                        strategies=strategies,
                        thin_covering_type=item_data.get("thin_covering_type"),
                    )
                )
            except (KeyError, ValueError) as e:
                console_logger.warning(f"Failed to parse item: {item_data}, error: {e}")
                continue

        return AnalysisResult(
            items=items,
            original_description=response.get("original_description"),
            discarded_manipulands=response.get("discarded_manipulands"),
            error=None,
        )

    def validate_asset(
        self,
        mesh_path: Path,
        description: str,
        output_dir: Path | None = None,
        use_lenient: bool = False,
    ) -> ValidationResult:
        """Validate a generated asset using VLM.

        Renders multi-view images of the mesh and asks VLM to verify:
        - Correct object type (matches description)
        - Style matches (if specified in description)
        - Single object (not multiple objects)
        - Completeness (no missing parts)
        - Reasonable proportions

        Args:
            mesh_path: Path to the generated mesh file.
            description: Original description to validate against.
            output_dir: Optional directory to save rendered images.
            use_lenient: If True, use lenient validation prompt. Lenient validation
                accepts minor imperfections common in library assets.

        Returns:
            ValidationResult with acceptance decision and reasoning.
        """
        # Determine output directory for rendered images.
        if output_dir is not None:
            render_dir = output_dir
            render_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp(prefix="asset_validation_")
            render_dir = Path(temp_dir)

        # Render multi-view images via BlenderServer.
        # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
        # due to GPU/OpenGL state corruption from fork.
        # Disable coordinate frame for cleaner validation renders.
        try:
            if self.blender_server is None or not self.blender_server.is_running():
                raise RuntimeError(
                    "BlenderServer required for asset validation. "
                    "Forked workers cannot safely use embedded bpy."
                )
            image_paths = self.blender_server.render_multiview_for_analysis(
                mesh_path=mesh_path,
                output_dir=render_dir,
                elevation_degrees=self.side_view_elevation_degrees,
                num_side_views=4,
                include_vertical_views=True,
                show_coordinate_frame=False,
                taa_samples=self.validation_taa_samples,
            )
            console_logger.debug(f"Rendered {len(image_paths)} images for validation")
        except Exception as e:
            console_logger.error(f"Failed to render mesh for validation: {e}")
            return ValidationResult(
                is_acceptable=False,
                reason=f"Rendering failed: {e}",
                suggestions=["Check mesh file validity"],
            )

        # Encode images for VLM.
        encoded_images = [encode_image_to_base64(img) for img in image_paths]

        # Build prompt with template variables.
        prompt_name = (
            AssetRouterPrompts.ASSET_VALIDATION_LENIENT
            if use_lenient
            else AssetRouterPrompts.ASSET_VALIDATION
        )
        prompt = prompt_manager.get_prompt(
            prompt_name=prompt_name,
            description=description,
            num_images=len(image_paths),
        )

        # Build message with images.
        user_content = [{"type": "text", "text": prompt}]
        for img_base64 in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                }
            )

        messages = [{"role": "user", "content": user_content}]

        # Call VLM for validation.
        openai_config = self.cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.asset_validation
        verbosity = openai_config.verbosity.asset_validation
        vision_detail = openai_config.vision_detail

        try:
            start_time = time.time()
            response_text = self.vlm_service.create_completion(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                response_format={"type": "json_object"},
                vision_detail=vision_detail,
            )
            elapsed = time.time() - start_time
            response_json = json.loads(response_text)
            console_logger.info(
                f"Router validation completed in {elapsed:.1f}s for "
                f"'{description}':\n{response_json}"
            )
        except Exception as e:
            console_logger.error(f"VLM validation failed: {e}")
            return ValidationResult(
                is_acceptable=False,
                reason=f"Validation call failed: {e}",
                suggestions=["Retry validation"],
            )

        # Parse response.
        return ValidationResult(
            is_acceptable=response_json.get("is_acceptable", False),
            reason=response_json.get("reason", "Unknown"),
            suggestions=response_json.get("suggestions", []),
        )

    def validate_item_types(self, items: list[AssetItem]) -> str | None:
        """Validate that all items match this agent's type.

        This is a safety check for LLM errors - the analysis prompt should
        already filter correctly, but this catches unexpected responses.

        Args:
            items: List of items from analysis.

        Returns:
            Error message if validation fails, None if all items are valid.
        """
        for item in items:
            # EITHER type is allowed in both agents.
            if item.object_type == ObjectType.EITHER:
                continue

            # Check if item type matches agent type.
            if item.object_type != self.agent_type.to_object_type():
                console_logger.warning(
                    f"LLM analysis returned wrong type: {item.object_type} "
                    f"for {self.agent_type.value} agent"
                )
                return (
                    f"Item '{item.description}' has type {item.object_type.value}, "
                    f"but this is the {self.agent_type} agent."
                )

        return None

    def generate_with_validation(
        self,
        item: AssetItem,
        geometry_client: "GeometryGenerationClient | None",
        image_generator: "BaseImageGenerator | None",
        images_dir: Path | None,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        hssd_client: "HssdRetrievalClient | None" = None,
        objaverse_client: "ObjaverseRetrievalClient | None" = None,
        articulated_client: "ArticulatedRetrievalClient | None" = None,
        materials_client: "MaterialsRetrievalClient | None" = None,
        scene_id: str | None = None,
    ) -> GeneratedGeometry | ArticulatedGeometry | None:
        """Generate or retrieve geometry for item with validation and retry.

        For generated assets: Tries each strategy in item.strategies. For each
        strategy, validation is controlled by the config's max_retries.

        For HSSD/Objaverse assets: Retrieves top-k candidates and validates each
        until one passes or all fail.

        Args:
            item: The asset item to generate/retrieve.
            geometry_client: Client for geometry generation server (for generated).
            image_generator: Image generator for creating reference images (for generated).
            images_dir: Directory to save generated images (for generated).
            geometry_dir: Directory to save generated/retrieved geometry.
            debug_dir: Directory to save debug outputs (validation renders).
            style_context: Optional style context for image generation.
            hssd_client: Client for HSSD retrieval server (for HSSD).
            objaverse_client: Client for Objaverse retrieval server (for Objaverse).
            articulated_client: Client for articulated retrieval server.
            materials_client: Client for materials retrieval server (for thin coverings).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry if successful, None if all strategies/candidates exhausted.
        """
        for strategy in item.strategies:
            # Get strategy config.
            strategies_cfg = self.cfg.asset_manager.router.strategies
            if not hasattr(strategies_cfg, strategy):
                console_logger.warning(f"Strategy '{strategy}' not in config, skipping")
                continue

            strategy_cfg = getattr(strategies_cfg, strategy)
            if hasattr(strategy_cfg, "enabled") and not strategy_cfg.enabled:
                console_logger.warning(f"Strategy '{strategy}' disabled, skipping")
                continue

            max_retries = strategy_cfg.max_retries
            console_logger.info(
                f"Trying strategy '{strategy}' for '{item.description}' "
                f"(max_retries={max_retries})"
            )

            # Dispatch to strategy-specific helper.
            if strategy == "generated":
                result = self._try_generated_strategy(
                    item=item,
                    max_retries=max_retries,
                    geometry_client=geometry_client,
                    hssd_client=hssd_client,
                    objaverse_client=objaverse_client,
                    image_generator=image_generator,
                    images_dir=images_dir,
                    geometry_dir=geometry_dir,
                    debug_dir=debug_dir,
                    style_context=style_context,
                    scene_id=scene_id,
                )
            elif strategy == "articulated":
                result = self._try_articulated_strategy(
                    item=item,
                    max_retries=max_retries,
                    debug_dir=debug_dir,
                    articulated_client=articulated_client,
                )
            elif strategy == "thin_covering":
                # Thin covering: textured flat surface for floors (e.g, rugs),
                # manipulands (e.g., tablecloths), and walls (e.g., posters).
                # Strategy auto-detects orientation based on agent_type.
                result = self._try_thin_covering_strategy(
                    item=item,
                    max_retries=max_retries,
                    materials_client=materials_client,
                    image_generator=image_generator,
                    geometry_dir=geometry_dir,
                    debug_dir=debug_dir,
                    scene_id=scene_id,
                )
            else:
                console_logger.warning(
                    f"Unknown strategy '{strategy}' for '{item.description}'"
                )
                continue

            if result is not None:
                return result

        console_logger.warning(f"All strategies exhausted for '{item.description}'")
        return None

    def _try_articulated_strategy(
        self,
        item: AssetItem,
        max_retries: int,
        debug_dir: Path,
        articulated_client: "ArticulatedRetrievalClient | None",
    ) -> ArticulatedGeometry | None:
        """Try the articulated strategy for objects with doors/drawers/etc.

        Retrieves pre-processed SDF assets from articulated object libraries
        (PartNet-Mobility, ArtVIP) using CLIP semantic matching and bounding
        box ranking via the articulated retrieval server.

        Args:
            item: The asset item to retrieve.
            max_retries: Number of candidates to try (from router config).
            debug_dir: Directory to save debug outputs (validation renders).
            articulated_client: Client for articulated retrieval server.

        Returns:
            ArticulatedGeometry if successful, None if no suitable candidate found.
        """
        if articulated_client is None:
            console_logger.warning(
                f"Articulated client not available for '{item.description}'"
            )
            return None

        # Map EITHER to concrete type based on which agent is calling.
        object_type = item.object_type.value.upper()
        if object_type == "EITHER":
            if self.agent_type == AgentType.FURNITURE:
                object_type = "FURNITURE"
            elif self.agent_type == AgentType.WALL_MOUNTED:
                object_type = "WALL_MOUNTED"
            else:
                object_type = "MANIPULAND"
            console_logger.debug(
                f"Mapped 'EITHER' to '{object_type}' for {self.agent_type} agent"
            )

        # Create output directory for retrieved meshes.
        articulated_output_dir = debug_dir / "articulated_meshes"
        articulated_output_dir.mkdir(parents=True, exist_ok=True)

        # Request enough candidates for validation retries.
        num_candidates = max(1, max_retries)
        request = ArticulatedRetrievalServerRequest(
            object_description=item.description,
            object_type=object_type,
            output_dir=str(articulated_output_dir),
            desired_dimensions=tuple(item.dimensions) if item.dimensions else None,
            num_candidates=num_candidates,
        )

        # Fetch candidates via server.
        try:
            responses = list(articulated_client.retrieve_objects([request]))
            if not responses:
                console_logger.warning(
                    f"No articulated response for '{item.description}'"
                )
                return None

            _, response = responses[0]
            candidates = response.results

        except Exception as e:
            console_logger.error(
                f"Articulated retrieval failed for '{item.description}': {e}"
            )
            return None

        if not candidates:
            console_logger.warning(
                f"No articulated candidates found for '{item.description}'"
            )
            return None

        console_logger.info(
            f"Got {len(candidates)} articulated candidates for '{item.description}'"
        )

        # If max_retries=0, return first candidate without validation.
        if max_retries == 0:
            candidate = candidates[0]
            console_logger.info(
                f"Returning first articulated candidate without validation: "
                f"{candidate.object_id}"
            )
            return self._result_to_articulated_geometry(result=candidate, item=item)

        # Validation loop: try each candidate until one passes.
        for i, candidate in enumerate(candidates):
            console_logger.info(
                f"Validating articulated candidate {i + 1}/{len(candidates)}: "
                f"{candidate.object_id} (clip={candidate.clip_score:.3f}, "
                f"bbox={candidate.bbox_score:.3f})"
            )

            # Validate with VLM.
            validation = self._validate_articulated_result(
                result=candidate, description=item.description, debug_dir=debug_dir
            )
            if validation.is_acceptable:
                console_logger.info(
                    f"Articulated validation passed for '{item.description}': "
                    f"{validation.reason}"
                )
                return self._result_to_articulated_geometry(candidate, item)

            console_logger.info(
                f"Articulated validation failed for '{item.description}': "
                f"{validation.reason}. Suggestions: {validation.suggestions}"
            )

        console_logger.warning(
            f"All {len(candidates)} articulated candidates failed validation "
            f"for '{item.description}'"
        )
        return None

    def _result_to_articulated_geometry(
        self, result: ArticulatedRetrievalResult, item: AssetItem
    ) -> ArticulatedGeometry:
        """Convert a server retrieval result to ArticulatedGeometry.

        Args:
            result: The retrieval result from the server.
            item: The original asset item.

        Returns:
            ArticulatedGeometry with result data.
        """
        return ArticulatedGeometry(
            sdf_path=Path(result.sdf_path),
            item=item,
            source=result.source,
            object_id=result.object_id,
            bounding_box_min=result.bounding_box_min,
            bounding_box_max=result.bounding_box_max,
        )

    def _validate_articulated_result(
        self, result: ArticulatedRetrievalResult, description: str, debug_dir: Path
    ) -> ValidationResult:
        """Validate an articulated result using VLM.

        The server has already exported a combined mesh, so we use that directly
        for validation rendering.

        Args:
            result: The retrieval result from the server.
            description: Original description to validate against.
            debug_dir: Directory to save rendered images.

        Returns:
            ValidationResult with acceptance decision and reasoning.
        """
        # Create validation directory for this result.
        validation_dir = debug_dir / f"articulated_{result.object_id}_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        # The server has already exported the combined mesh.
        mesh_path = Path(result.mesh_path)
        if not mesh_path.exists():
            console_logger.error(f"Combined mesh not found: {mesh_path}")
            return ValidationResult(
                is_acceptable=False,
                reason="Combined mesh file not found",
                suggestions=["Check server mesh export"],
            )

        # Render the mesh for validation.
        try:
            image_paths = self._render_mesh_for_validation(
                mesh_path=mesh_path, output_dir=validation_dir
            )
        except Exception as e:
            console_logger.error(
                f"Failed to render articulated object for validation: {e}"
            )
            return ValidationResult(
                is_acceptable=False,
                reason=f"Rendering failed: {e}",
                suggestions=["Check mesh file validity"],
            )

        if not image_paths:
            console_logger.error(
                f"No images rendered for articulated result {result.object_id}"
            )
            return ValidationResult(
                is_acceptable=False,
                reason="No images rendered",
                suggestions=["Check mesh visual geometry"],
            )

        # Encode images for VLM.
        encoded_images = [encode_image_to_base64(img) for img in image_paths]

        # Select prompt based on lenient validation flag.
        use_lenient = (
            self.cfg.asset_manager.router.strategies.articulated.use_lenient_validation
        )
        if use_lenient:
            prompt_name = AssetRouterPrompts.ASSET_VALIDATION_LENIENT
        else:
            prompt_name = AssetRouterPrompts.ASSET_VALIDATION

        # Build prompt with template variables.
        prompt = prompt_manager.get_prompt(
            prompt_name=prompt_name,
            description=description,
            num_images=len(image_paths),
        )

        # Build message with images.
        user_content = [{"type": "text", "text": prompt}]
        for img_base64 in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                }
            )

        messages = [{"role": "user", "content": user_content}]

        # Call VLM for validation.
        openai_config = self.cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.asset_validation
        verbosity = openai_config.verbosity.asset_validation
        vision_detail = openai_config.vision_detail

        try:
            start_time = time.time()
            response_text = self.vlm_service.create_completion(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                response_format={"type": "json_object"},
                vision_detail=vision_detail,
            )
            elapsed = time.time() - start_time
            response_json = json.loads(response_text)
            console_logger.info(
                f"Articulated validation completed in {elapsed:.1f}s for "
                f"'{description}':\n{response_json}"
            )
        except Exception as e:
            console_logger.error(f"VLM validation failed: {e}")
            return ValidationResult(
                is_acceptable=False,
                reason=f"Validation call failed: {e}",
                suggestions=["Retry validation"],
            )

        # Parse response.
        return ValidationResult(
            is_acceptable=response_json.get("is_acceptable", False),
            reason=response_json.get("reason", "Unknown"),
            suggestions=response_json.get("suggestions", []),
        )

    def _render_mesh_for_validation(
        self, mesh_path: Path, output_dir: Path
    ) -> list[Path]:
        """Render a mesh for VLM validation.

        Args:
            mesh_path: Path to the mesh file (GLB format).
            output_dir: Directory to save rendered images.

        Returns:
            List of paths to rendered images.
        """
        # Use lower light energy for articulated objects (more reflective materials).
        from scenesmith.agent_utils.blender.constants import ARTICULATED_LIGHT_ENERGY

        # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
        # due to GPU/OpenGL state corruption from fork.
        # Disable coordinate frame for cleaner validation renders.
        if self.blender_server is None or not self.blender_server.is_running():
            raise RuntimeError(
                "BlenderServer required for articulated asset validation. "
                "Forked workers cannot safely use embedded bpy."
            )
        image_paths = self.blender_server.render_multiview_for_analysis(
            mesh_path=mesh_path,
            output_dir=output_dir,
            elevation_degrees=self.side_view_elevation_degrees,
            num_side_views=4,
            include_vertical_views=True,
            light_energy=ARTICULATED_LIGHT_ENERGY,
            show_coordinate_frame=False,
            taa_samples=self.validation_taa_samples,
        )

        return image_paths

    def _try_thin_covering_strategy(
        self,
        item: AssetItem,
        max_retries: int,
        materials_client: "MaterialsRetrievalClient | None",
        image_generator: "BaseImageGenerator | None",
        geometry_dir: Path,
        debug_dir: Path,
        scene_id: str | None = None,
    ) -> GeneratedGeometry | None:
        """Try the thin covering strategy for procedural textured surface generation.

        Generates thin textured meshes with PBR materials from the materials
        retrieval server. For floor agents, creates horizontal textured surfaces
        (rugs). For wall agents, creates vertical wall-mounted surfaces (paintings).

        Falls back to AI texture generation if all retrieval candidates fail
        validation and the generator config is enabled.

        Args:
            item: The asset item to generate.
            max_retries: Number of material candidates to try with VLM validation.
                0 = no validation (use first material).
            materials_client: Client for materials retrieval server.
            image_generator: Image generator for fallback texture generation.
            geometry_dir: Directory to save generated GLTF.
            debug_dir: Directory to save debug outputs (validation renders).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry if successful, None if retrieval/validation fails.
        """
        if materials_client is None:
            console_logger.warning(
                f"Materials client not available for '{item.description}'"
            )
            return None

        # Wall agent uses vertical geometry (wall-mounted).
        is_wall_mode = self.agent_type == AgentType.WALL_MOUNTED

        # Infer shape from description (circular vs rectangular).
        shape = infer_thin_covering_shape(item.description)

        # Extract dimensions - wall mode uses width/height, floor mode uses width/depth.
        if is_wall_mode:
            if not item.dimensions or len(item.dimensions) < 3:
                console_logger.warning(
                    f"Wall thin covering '{item.description}' missing dimensions "
                    f"(need width, depth, height)"
                )
                return None
            width = item.dimensions[0]
            height = item.dimensions[2]
            console_logger.info(
                f"Generating {shape} wall thin covering '{item.description}' "
                f"({width:.2f}m x {height:.2f}m)"
            )
        else:
            if not item.dimensions or len(item.dimensions) < 2:
                console_logger.warning(
                    f"Floor thin covering '{item.description}' missing dimensions "
                    f"(need width, depth)"
                )
                return None
            width = item.dimensions[0]
            depth = item.dimensions[1]
            console_logger.info(
                f"Generating {shape} floor thin covering '{item.description}' "
                f"({width:.2f}m x {depth:.2f}m)"
            )

        # Request materials from server.
        request = MaterialsRetrievalServerRequest(
            material_description=item.description,
            output_dir=str(geometry_dir),
            scene_id=scene_id,
            num_candidates=max(1, max_retries),
        )

        try:
            # Fetch materials (single request returns single response).
            responses = list(materials_client.retrieve_materials([request]))
            if not responses:
                console_logger.warning(
                    f"No material response for thin covering '{item.description}'"
                )
                return None
            _, response = responses[0]
            materials = response.results
        except Exception as e:
            console_logger.error(
                f"Materials retrieval failed for thin covering '{item.description}': {e}"
            )
            return None

        if not materials:
            console_logger.warning(
                f"No materials found for thin covering '{item.description}'"
            )
            return None
        console_logger.info(
            f"Got {len(materials)} material candidates for '{item.description}'"
        )

        thin_covering_cfg = self.cfg.asset_manager.router.strategies.thin_covering
        thickness = thin_covering_cfg.thickness_m
        # single_image (artwork) uses cover mode, tileable uses configured scale.
        is_single_image = item.thin_covering_type == "single_image"
        texture_scale = None if is_single_image else thin_covering_cfg.texture_scale

        # If max_retries=0, use first material without validation.
        if max_retries == 0:
            material = materials[0]
            console_logger.info(
                f"Using first material without validation: {material.material_id}"
            )
            return self._generate_thin_covering_geometry(
                item=item,
                material_path=Path(material.material_path),
                width=width,
                second_dim=height if is_wall_mode else depth,
                thickness=thickness,
                geometry_dir=geometry_dir,
                is_wall_mode=is_wall_mode,
                shape=shape,
                texture_scale=texture_scale,
            )

        # Validation loop: try each material until one passes.
        for i, material in enumerate(materials):
            console_logger.info(
                f"Validating material {i + 1}/{len(materials)}: "
                f"{material.material_id} (score={material.similarity_score:.3f})"
            )

            # Generate geometry with this material.
            result = self._generate_thin_covering_geometry(
                item=item,
                material_path=Path(material.material_path),
                width=width,
                second_dim=height if is_wall_mode else depth,
                thickness=thickness,
                geometry_dir=geometry_dir,
                is_wall_mode=is_wall_mode,
                shape=shape,
                texture_scale=texture_scale,
            )

            if result is None:
                continue

            # Validate with VLM.
            validation = self._validate_thin_covering(
                mesh_path=result.geometry_path,
                description=item.description,
                debug_dir=debug_dir,
                is_wall_mode=is_wall_mode,
            )
            if validation.is_acceptable:
                console_logger.info(
                    f"Validation passed for '{item.description}': "
                    f"{validation.reason}"
                )
                return result

            console_logger.info(
                f"Validation failed for '{item.description}': "
                f"{validation.reason}. Suggestions: {validation.suggestions}"
            )

        console_logger.warning(
            f"All {len(materials)} materials failed validation for "
            f"thin covering '{item.description}'"
        )

        # Fallback to AI texture generation if enabled and image_generator available.
        return self._try_generated_thin_covering(
            item=item,
            image_generator=image_generator,
            width=width,
            second_dim=height if is_wall_mode else depth,
            thickness=thickness,
            geometry_dir=geometry_dir,
            debug_dir=debug_dir,
            is_wall_mode=is_wall_mode,
            shape=shape,
        )

    def _try_generated_thin_covering(
        self,
        item: AssetItem,
        image_generator: "BaseImageGenerator | None",
        width: float,
        second_dim: float,
        thickness: float,
        geometry_dir: Path,
        debug_dir: Path,
        is_wall_mode: bool,
        shape: str,
    ) -> GeneratedGeometry | None:
        """Try generating thin covering texture with AI when retrieval fails.

        Creates MaterialGenerator locally (lightweight, no GPU/server resources).

        Args:
            item: The asset item to generate.
            image_generator: Image generator for texture generation.
            width: Width in meters.
            second_dim: Height (wall) or depth (floor) in meters.
            thickness: Thickness in meters.
            geometry_dir: Directory to save generated GLTF.
            debug_dir: Directory for debug outputs.
            is_wall_mode: True for wall coverings, False for floor.
            shape: "rectangular" or "circular".

        Returns:
            GeneratedGeometry if successful, None otherwise.
        """
        # Check if generator is enabled in config.
        gen_cfg = self.cfg.asset_manager.router.strategies.thin_covering.generator
        if not gen_cfg.enabled:
            console_logger.info("Thin covering generator disabled in config")
            return None

        if image_generator is None:
            console_logger.warning(
                "Image generator not available for thin covering fallback"
            )
            return None

        console_logger.info(f"Trying AI texture generation for '{item.description}'")

        # Create output directory for generated materials.
        generated_materials_dir = geometry_dir / "generated_materials"
        generated_materials_dir.mkdir(exist_ok=True)

        # Create MaterialGenerator locally.
        material_generator = MaterialGenerator(
            config=MaterialGeneratorConfig(
                enabled=True,
                backend=gen_cfg.backend,
                max_retries=gen_cfg.max_retries,
                default_roughness=gen_cfg.default_roughness,
                texture_scale=gen_cfg.texture_scale,
            ),
            output_dir=generated_materials_dir,
            image_generator=image_generator,
        )

        # Determine if single image (artwork) or tileable texture.
        is_single_image = item.thin_covering_type == "single_image"

        for retry in range(material_generator.config.max_retries):
            console_logger.info(
                f"Generation attempt {retry + 1}/{material_generator.config.max_retries}"
            )

            if is_single_image:
                # For circular shapes, always use square (1:1) since circular UV mapping
                # expects square textures. For rectangular, use actual dimensions.
                if shape == "circular":
                    generated = material_generator.generate_artwork(
                        description=item.description, width=1.0, height=1.0
                    )
                else:
                    generated = material_generator.generate_artwork(
                        description=item.description, width=width, height=second_dim
                    )
            else:
                # Tileable textures always use square - tiling handles non-square surfaces.
                generated = material_generator.generate_material(
                    description=item.description
                )

            if generated is None:
                continue

            result = self._generate_thin_covering_geometry(
                item=item,
                material_path=generated.path,
                width=width,
                second_dim=second_dim,
                thickness=thickness,
                geometry_dir=geometry_dir,
                is_wall_mode=is_wall_mode,
                shape=shape,
                texture_scale=generated.texture_scale,
            )

            if result is None:
                continue

            # Validate with VLM.
            validation = self._validate_thin_covering(
                mesh_path=result.geometry_path,
                description=item.description,
                debug_dir=debug_dir,
                is_wall_mode=is_wall_mode,
            )

            if validation.is_acceptable:
                console_logger.info(
                    f"Generated material passed validation: {validation.reason}"
                )
                return result

            console_logger.info(
                f"Generated material failed validation: {validation.reason}"
            )

        console_logger.warning(
            f"All {material_generator.config.max_retries} generation attempts "
            f"failed for '{item.description}'"
        )
        return None

    def _generate_thin_covering_geometry(
        self,
        item: AssetItem,
        material_path: Path,
        width: float,
        second_dim: float,
        thickness: float,
        geometry_dir: Path,
        is_wall_mode: bool,
        texture_scale: float | None,
        shape: str = "rectangular",
    ) -> GeneratedGeometry | None:
        """Generate thin covering GLTF mesh with given material.

        Creates either a horizontal floor covering (rug) or vertical wall
        covering (painting/mirror) depending on is_wall_mode.

        Args:
            item: The asset item.
            material_path: Path to material folder with PBR textures.
            width: Width in meters (X dimension).
            second_dim: Height for wall mode, depth for floor mode.
            thickness: Thickness in meters.
            geometry_dir: Directory to save the GLB.
            is_wall_mode: True for vertical wall surfaces, False for floor.
            texture_scale: Meters per texture tile (None = cover mode, no tiling).
            shape: "rectangular" or "circular" (applies to both floor and wall).

        Returns:
            GeneratedGeometry if successful, None on error.
        """
        base_name = _unique_asset_basename(item.short_name)
        glb_path = geometry_dir / f"{base_name}.glb"

        try:
            if shape == "circular":
                # Circular covering (floor or wall).
                radius = min(width, second_dim) / 2.0
                create_circular_thin_covering_glb(
                    radius=radius,
                    thickness=thickness,
                    material_folder=material_path,
                    output_path=glb_path,
                    texture_scale=texture_scale,
                    is_wall=is_wall_mode,
                )
            else:
                # Rectangular covering (floor or wall).
                create_rectangular_thin_covering_glb(
                    width=width,
                    second_dim=second_dim,
                    thickness=thickness,
                    material_folder=material_path,
                    output_path=glb_path,
                    texture_scale=texture_scale,
                    is_wall=is_wall_mode,
                )

            console_logger.info(f"Generated thin covering mesh: {glb_path}")

            return GeneratedGeometry(
                geometry_path=glb_path, item=item, asset_source="thin_covering"
            )

        except Exception as e:
            console_logger.error(f"Failed to generate thin covering mesh: {e}")
            return None

    def _validate_thin_covering(
        self, mesh_path: Path, description: str, debug_dir: Path, is_wall_mode: bool
    ) -> ValidationResult:
        """Validate thin covering material matches description using VLM.

        Renders appropriate view (top-down for floor, frontal for wall) and
        validates that the texture pattern matches the description.

        Args:
            mesh_path: Path to the thin covering GLB mesh.
            description: Original description for validation.
            debug_dir: Directory to save validation renders.
            is_wall_mode: True for wall coverings, False for floor coverings.

        Returns:
            ValidationResult with is_acceptable, reason, and suggestions.
        """
        validation_dir = debug_dir / f"{mesh_path.stem}_thin_covering_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        # Render appropriate view based on orientation.
        image_path = self._render_thin_covering_for_validation(
            mesh_path=mesh_path, output_dir=validation_dir, is_wall_mode=is_wall_mode
        )

        if image_path is None:
            return ValidationResult(
                is_acceptable=False,
                reason="Failed to render thin covering for validation",
                suggestions=["Check Blender server availability"],
            )

        image_base64 = encode_image_to_base64(image_path)

        system_prompt = prompt_manager.get_prompt(
            AssetRouterPrompts.THIN_COVERING_VALIDATION_PROMPT
        )
        user_prompt = (
            f"Validate this {'wall' if is_wall_mode else 'floor'} "
            f"covering texture for the description: {description}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            },
        ]

        model = self.cfg.openai.model
        reasoning_effort = self.cfg.openai.reasoning_effort.asset_validation
        verbosity = self.cfg.openai.verbosity.asset_validation
        vision_detail = self.cfg.openai.vision_detail

        try:
            start_time = time.time()
            response_text = self.vlm_service.create_completion(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                response_format={"type": "json_object"},
                vision_detail=vision_detail,
            )
            elapsed = time.time() - start_time
            response_json = json.loads(response_text)
            console_logger.info(
                f"Thin covering validation completed in {elapsed:.1f}s for "
                f"'{description}':\n{response_json}"
            )
        except Exception as e:
            console_logger.error(f"Thin covering VLM validation failed: {e}")
            return ValidationResult(
                is_acceptable=False,
                reason=f"Validation call failed: {e}",
                suggestions=["Retry validation"],
            )

        return ValidationResult(
            is_acceptable=response_json.get("is_acceptable", False),
            reason=response_json.get("reason", "Unknown"),
            suggestions=response_json.get("suggestions", []),
        )

    def _render_thin_covering_for_validation(
        self, mesh_path: Path, output_dir: Path, is_wall_mode: bool
    ) -> Path | None:
        """Render thin covering for VLM validation.

        For floor coverings, renders top-down view. For wall coverings,
        renders frontal view.

        Args:
            mesh_path: Path to the thin covering GLB mesh.
            output_dir: Directory to save rendered images.
            is_wall_mode: True for wall coverings (frontal), False for floor (top-down).

        Returns:
            Path to rendered image, or None if rendering failed.
        """
        # Elevation from config.
        elevation = self.side_view_elevation_degrees

        if is_wall_mode:
            # Wall covering: frontal view from +Y (object front face is at +Y).
            num_side_views = 1
            include_vertical = False
            # Start azimuth at 90° to position camera at +Y.
            start_azimuth = 90.0
        else:
            # Floor covering: top-down view.
            num_side_views = 0
            include_vertical = True
            start_azimuth = 0.0

        # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
        # due to GPU/OpenGL state corruption from fork.
        # Disable coordinate frame for cleaner validation renders.
        # Use lower light energy to avoid washing out material colors.
        if self.blender_server is None or not self.blender_server.is_running():
            raise RuntimeError(
                "BlenderServer required for thin covering validation. "
                "Forked workers cannot safely use embedded bpy."
            )
        image_paths = self.blender_server.render_multiview_for_analysis(
            mesh_path=mesh_path,
            output_dir=output_dir,
            elevation_degrees=elevation,
            num_side_views=num_side_views,
            include_vertical_views=include_vertical,
            start_azimuth_degrees=start_azimuth,
            show_coordinate_frame=False,
            light_energy=MATERIAL_VALIDATION_LIGHT_ENERGY,
            taa_samples=self.validation_taa_samples,
        )

        return image_paths[0] if image_paths else None

    def _try_generated_strategy(
        self,
        item: AssetItem,
        max_retries: int,
        geometry_client: "GeometryGenerationClient | None",
        hssd_client: "HssdRetrievalClient | None",
        objaverse_client: "ObjaverseRetrievalClient | None",
        image_generator: "BaseImageGenerator | None",
        images_dir: Path | None,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        scene_id: str | None = None,
    ) -> GeneratedGeometry | None:
        """Try the generated strategy with text-to-3D or library retrieval.

        For "generated" strategy, asset_source config (general_asset_source)
        determines whether to use text-to-3D generation or library retrieval
        (HSSD or Objaverse).

        Args:
            item: The asset item to generate.
            max_retries: Number of retries. 0 means single attempt without validation.
            geometry_client: Client for geometry generation server (for text-to-3D).
            hssd_client: Client for HSSD retrieval server (for HSSD retrieval).
            objaverse_client: Client for Objaverse retrieval server (for Objaverse).
            image_generator: Image generator for creating reference images (for text-to-3D).
            images_dir: Directory to save generated images (for text-to-3D).
            geometry_dir: Directory to save generated geometry.
            debug_dir: Directory to save debug outputs (validation renders).
            style_context: Optional style context for image generation.
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry if successful, None if all retries exhausted.
        """
        # Determine asset source (text-to-3D vs library retrieval).
        asset_source = self.cfg.asset_manager.general_asset_source

        # For library retrieval, pre-fetch candidates (single server call).
        hssd_candidates: list | None = None
        objaverse_candidates: list | None = None
        if asset_source == "hssd":
            hssd_candidates = self._fetch_hssd_candidates(
                item=item,
                hssd_client=hssd_client,
                geometry_dir=geometry_dir,
                max_retries=max_retries,
                scene_id=scene_id,
            )
            if not hssd_candidates:
                console_logger.warning(f"No HSSD candidates for '{item.description}'")
                return None
        elif asset_source == "objaverse":
            objaverse_candidates = self._fetch_objaverse_candidates(
                item=item,
                objaverse_client=objaverse_client,
                geometry_dir=geometry_dir,
                max_retries=max_retries,
                scene_id=scene_id,
            )
            if not objaverse_candidates:
                console_logger.warning(
                    f"No Objaverse candidates for '{item.description}'"
                )
                return None

        if max_retries == 0:
            # Single attempt, no validation.
            console_logger.info(
                f"Acquiring '{item.description}' with generated (no validation)"
            )
            return self._acquire_generated_candidate(
                item=item,
                asset_source=asset_source,
                attempt=0,
                geometry_client=geometry_client,
                image_generator=image_generator,
                images_dir=images_dir,
                geometry_dir=geometry_dir,
                debug_dir=debug_dir,
                style_context=style_context,
                scene_id=scene_id,
                hssd_candidates=hssd_candidates,
                objaverse_candidates=objaverse_candidates,
            )

        # Validation + retry loop.
        for attempt in range(max_retries):
            console_logger.info(
                f"Attempt {attempt + 1}/{max_retries} for '{item.description}'"
            )

            result = self._acquire_generated_candidate(
                item=item,
                asset_source=asset_source,
                attempt=attempt,
                geometry_client=geometry_client,
                image_generator=image_generator,
                images_dir=images_dir,
                geometry_dir=geometry_dir,
                debug_dir=debug_dir,
                style_context=style_context,
                scene_id=scene_id,
                hssd_candidates=hssd_candidates,
                objaverse_candidates=objaverse_candidates,
            )

            if result is None:
                console_logger.warning(
                    f"Candidate acquisition failed for '{item.description}'"
                )
                continue

            # Validate with VLM.
            # Use lenient validation for retrieved library assets (HSSD, Objaverse) based
            # on config. Generated assets always use strict validation.
            use_lenient = False
            if asset_source == "hssd":
                use_lenient = self.cfg.asset_manager.hssd.use_lenient_validation
            elif asset_source == "objaverse":
                use_lenient = self.cfg.asset_manager.objaverse.use_lenient_validation

            validation_dir = debug_dir / f"{item.short_name}_validation"
            validation = self.validate_asset(
                mesh_path=result.geometry_path,
                description=item.description,
                output_dir=validation_dir,
                use_lenient=use_lenient,
            )

            if validation.is_acceptable:
                console_logger.info(
                    f"Validation passed for '{item.description}': {validation.reason}"
                )
                return result

            console_logger.info(
                f"Validation failed for '{item.description}': {validation.reason}. "
                f"Suggestions: {validation.suggestions}"
            )

        return None

    def _fetch_hssd_candidates(
        self,
        item: AssetItem,
        hssd_client: "HssdRetrievalClient | None",
        geometry_dir: Path,
        max_retries: int,
        scene_id: str | None = None,
    ) -> list["HssdRetrievalResult"] | None:
        """Fetch HSSD candidates in a single server call.

        Args:
            item: The asset item to retrieve candidates for.
            hssd_client: Client for HSSD retrieval server.
            geometry_dir: Directory to save retrieved geometry.
            max_retries: Number of validation retries (determines num_candidates).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            List of HssdRetrievalResult candidates, or None if fetch failed.
        """
        if hssd_client is None:
            console_logger.error(
                f"HSSD client not provided for '{item.description}', "
                "but asset_source is 'hssd'"
            )
            return None

        console_logger.info(f"Fetching HSSD candidates for '{item.description}'")

        # Request enough candidates for all retry attempts.
        # max_retries=0 means single attempt, so we need at least 1.
        num_candidates = max(1, max_retries)

        # Map EITHER to concrete type based on which agent is calling.
        object_type = item.object_type.value
        if object_type == "either":
            object_type = (
                "furniture" if self.agent_type == AgentType.FURNITURE else "manipuland"
            )
            console_logger.debug(
                f"Mapped 'either' to '{object_type}' for {self.agent_type} agent"
            )

        request = HssdRetrievalServerRequest(
            object_description=item.description,
            object_type=object_type,
            desired_dimensions=tuple(item.dimensions) if item.dimensions else None,
            output_dir=str(geometry_dir),
            scene_id=scene_id,
            num_candidates=num_candidates,
        )

        try:
            responses = list(hssd_client.retrieve_objects([request]))
            if not responses:
                console_logger.error(f"No HSSD response for '{item.description}'")
                return None

            _, response = responses[0]

            if not response.results:
                console_logger.error(f"No HSSD results for '{item.description}'")
                return None

            console_logger.info(
                f"Got {len(response.results)} HSSD candidates for '{item.description}'"
            )
            return response.results

        except Exception as e:
            console_logger.error(f"HSSD fetch failed for '{item.description}': {e}")
            return None

    def _fetch_objaverse_candidates(
        self,
        item: AssetItem,
        objaverse_client: "ObjaverseRetrievalClient | None",
        geometry_dir: Path,
        max_retries: int,
        scene_id: str | None = None,
    ) -> list["ObjaverseRetrievalResult"] | None:
        """Fetch Objaverse candidates in a single server call.

        Args:
            item: The asset item to retrieve candidates for.
            objaverse_client: Client for Objaverse retrieval server.
            geometry_dir: Directory to save retrieved geometry.
            max_retries: Number of validation retries (determines num_candidates).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            List of ObjaverseRetrievalResult candidates, or None if fetch failed.
        """
        if objaverse_client is None:
            console_logger.error(
                f"Objaverse client not provided for '{item.description}', "
                "but asset_source is 'objaverse'"
            )
            return None

        console_logger.info(f"Fetching Objaverse candidates for '{item.description}'")

        # Request enough candidates for all retry attempts.
        # max_retries=0 means single attempt, so we need at least 1.
        num_candidates = max(1, max_retries)

        # Map EITHER to concrete type based on which agent is calling.
        object_type = item.object_type.value
        if object_type == "either":
            object_type = (
                "furniture" if self.agent_type == AgentType.FURNITURE else "manipuland"
            )
            console_logger.debug(
                f"Mapped 'either' to '{object_type}' for {self.agent_type} agent"
            )

        request = ObjaverseRetrievalServerRequest(
            object_description=item.description,
            object_type=object_type,
            desired_dimensions=tuple(item.dimensions) if item.dimensions else None,
            output_dir=str(geometry_dir),
            scene_id=scene_id,
            num_candidates=num_candidates,
        )

        try:
            responses = list(objaverse_client.retrieve_objects([request]))
            if not responses:
                console_logger.error(f"No Objaverse response for '{item.description}'")
                return None

            _, response = responses[0]

            if not response.results:
                console_logger.error(f"No Objaverse results for '{item.description}'")
                return None

            console_logger.info(
                f"Got {len(response.results)} Objaverse candidates for "
                f"'{item.description}'"
            )
            return response.results

        except Exception as e:
            console_logger.error(
                f"Objaverse fetch failed for '{item.description}': {e}"
            )
            return None

    def _acquire_generated_candidate(
        self,
        item: AssetItem,
        asset_source: str,
        attempt: int,
        geometry_client: "GeometryGenerationClient | None",
        image_generator: "BaseImageGenerator | None",
        images_dir: Path | None,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        scene_id: str | None = None,
        hssd_candidates: list["HssdRetrievalResult"] | None = None,
        objaverse_candidates: list["ObjaverseRetrievalResult"] | None = None,
    ) -> GeneratedGeometry | None:
        """Acquire a single candidate based on asset source.

        For text-to-3D: generates a new mesh (attempt number ignored, randomness varies).
        For HSSD/Objaverse: returns candidate at index `attempt` from pre-fetched list.

        Args:
            item: The asset item to acquire.
            asset_source: "generated", "hssd", or "objaverse".
            attempt: Attempt number (used as index for library candidates).
            geometry_client: Client for geometry generation server.
            image_generator: Image generator for reference images.
            images_dir: Directory for generated images.
            geometry_dir: Directory for geometry files.
            debug_dir: Directory for debug outputs.
            style_context: Style context for image generation.
            scene_id: Scene identifier for scheduling.
            hssd_candidates: Pre-fetched HSSD candidates (required if asset_source="hssd").
            objaverse_candidates: Pre-fetched Objaverse candidates (if asset_source="objaverse").

        Returns:
            GeneratedGeometry if successful, None if failed or no more candidates.
        """
        if asset_source == "hssd":
            if hssd_candidates is None or attempt >= len(hssd_candidates):
                console_logger.warning(
                    f"No more HSSD candidates for '{item.description}' "
                    f"(attempt {attempt}, available {len(hssd_candidates or [])})"
                )
                return None

            candidate = hssd_candidates[attempt]
            console_logger.info(
                f"Using HSSD candidate {attempt + 1}/{len(hssd_candidates)} "
                f"for '{item.description}': {candidate.hssd_id}"
            )

            return GeneratedGeometry(
                geometry_path=Path(candidate.mesh_path),
                item=item,
                asset_source="hssd",
                hssd_id=candidate.hssd_id,
            )

        if asset_source == "objaverse":
            if objaverse_candidates is None or attempt >= len(objaverse_candidates):
                console_logger.warning(
                    f"No more Objaverse candidates for '{item.description}' "
                    f"(attempt {attempt}, available {len(objaverse_candidates or [])})"
                )
                return None

            candidate = objaverse_candidates[attempt]
            console_logger.info(
                f"Using Objaverse candidate {attempt + 1}/{len(objaverse_candidates)} "
                f"for '{item.description}': {candidate.objaverse_uid}"
            )

            return GeneratedGeometry(
                geometry_path=Path(candidate.mesh_path),
                item=item,
                asset_source="objaverse",
                objaverse_uid=candidate.objaverse_uid,
            )

        # Text-to-3D generation.
        return self._generate_geometry(
            item=item,
            geometry_client=geometry_client,
            image_generator=image_generator,
            images_dir=images_dir,
            geometry_dir=geometry_dir,
            debug_dir=debug_dir,
            style_context=style_context,
            scene_id=scene_id,
        )

    def _generate_geometry(
        self,
        item: AssetItem,
        geometry_client: "GeometryGenerationClient",
        image_generator: "BaseImageGenerator",
        images_dir: Path,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        scene_id: str | None = None,
    ) -> GeneratedGeometry | None:
        """Generate geometry for a single item.

        Args:
            item: The asset item to generate.
            geometry_client: Client for geometry generation server.
            image_generator: Image generator for creating reference images.
            images_dir: Directory to save generated images.
            geometry_dir: Directory to save generated geometry.
            debug_dir: Directory to save debug outputs (segmentation masks, etc.).
            style_context: Optional style context for image generation.
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry with paths, or None if generation failed.
        """
        from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
            GeometryGenerationServerRequest,
        )

        base_name = _unique_asset_basename(item.short_name)
        image_path = images_dir / f"{base_name}.png"
        geometry_path = geometry_dir / f"{base_name}.glb"

        # Generate reference image.
        try:
            style_prompt = style_context or "Modern style"
            image_generator.generate_images(
                style_prompt=style_prompt,
                object_descriptions=[item.description],
                output_paths=[image_path],
            )
        except Exception as e:
            console_logger.error(
                f"Image generation failed for '{item.description}': {e}"
            )
            return None

        # Generate geometry from image.
        try:
            # Extract backend configuration from config.
            backend = self.cfg.asset_manager.get("backend", "hunyuan3d")

            # Prepare SAM3D config if backend is sam3d.
            sam3d_config = None
            if backend == "sam3d":
                sam3d_cfg = self.cfg.asset_manager.sam3d
                mode = sam3d_cfg.get("mode", "foreground")
                sam3d_config = {
                    "sam3_checkpoint": str(sam3d_cfg.sam3_checkpoint),
                    "sam3d_checkpoint": str(sam3d_cfg.sam3d_checkpoint),
                    "mode": mode,
                    "text_prompt": sam3d_cfg.get("text_prompt"),
                    "threshold": sam3d_cfg.get("threshold", 0.5),
                }
                # Pass object description for "object_description" mode.
                if mode == "object_description":
                    sam3d_config["object_description"] = item.description

            request = GeometryGenerationServerRequest(
                image_path=str(image_path),
                output_dir=str(geometry_dir),
                prompt=item.description,
                output_filename=geometry_path.name,
                debug_folder=str(debug_dir),
                backend=backend,
                sam3d_config=sam3d_config,
                scene_id=scene_id,
            )

            # Use synchronous single-item generation.
            responses = list(geometry_client.generate_geometries([request]))
            if not responses:
                console_logger.error(f"No geometry response for '{item.description}'")
                return None

            _, response = responses[0]

            # Check for generation error.
            if isinstance(response, GeometryGenerationError):
                console_logger.error(
                    f"Geometry generation error for '{item.description}': "
                    f"{response.error_message}"
                )
                return None

            actual_geometry_path = Path(response.geometry_path)

        except Exception as e:
            console_logger.error(
                f"Geometry generation failed for '{item.description}': {e}"
            )
            return None

        return GeneratedGeometry(
            geometry_path=actual_geometry_path,
            item=item,
            asset_source="generated",
            image_path=image_path,
        )
