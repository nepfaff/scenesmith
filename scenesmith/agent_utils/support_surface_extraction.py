"""Support surface extraction using HSM face clustering algorithm.

This module implements the support surface identification algorithm from the HSM
paper (https://arxiv.org/abs/2503.16848v2).

The algorithm clusters mesh faces by normal similarity, fits planes to clusters,
classifies surfaces as horizontal/vertical, and extracts horizontal support
surfaces for manipuland placement.

We slightly modified the algorithm to make it more robust for our lower-quality
generated rather than artist designed furniture meshes.
"""

from __future__ import annotations

import logging
import time

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from omegaconf import DictConfig
from pydrake.all import RigidTransform, RotationMatrix

from scenesmith.utils.geometry_utils import safe_convex_hull_2d

console_logger = logging.getLogger(__name__)


@dataclass
class SupportSurfaceExtractionConfig:
    """HSM algorithm parameters from paper Section A.2.

    Input meshes in Y-up (GLTF) are converted to Z-up immediately.
    All surface detection happens in Z-up coordinates (Drake convention).
    """

    normal_cluster_threshold: float = 0.9
    """HSM t_norm: Minimum dot product for face normal similarity in cluster."""

    normal_adjacent_threshold: float = 0.95
    """HSM t_adj: Minimum dot product for adjacent face similarity."""

    horizontal_normal_z_min: float = 0.95
    """HSM t_hzn (adapted to Z-up): Minimum Z component for horizontal normals."""

    vertical_normal_z_max: float = 0.05
    """HSM t_vert (adapted to Z-up): Maximum Z component for vertical normals."""

    min_surface_area_m2: float = 0.003
    """HSM MIN_AREA: 30 cm² minimum area (filters tiny surfaces)."""

    min_area_ratio: float = 0.20
    """Minimum mesh_area/bbox_area ratio (filters mesh artifacts)."""

    min_clearance_m: float = 0.05
    """5 cm minimum clearance above surface (filters internal surfaces)."""

    min_inscribed_radius_m: float = 0.10
    """10 cm minimum inscribed radius (filters thin slivers)."""

    height_tolerance_m: float = 0.05
    """5 cm height tolerance for grouping surfaces at same level."""

    self_intersection_threshold_m: float = 0.001
    """1mm threshold for filtering ray-casting self-hits."""

    max_measured_clearance_m: float = 5.0
    """Maximum clearance to measure via ray-casting (cap for efficiency)."""

    top_surface_clearance_m: float = 0.5
    """HSM h_top: 50 cm default clearance for top surfaces."""

    surface_offset_m: float = 0.01
    """Offset above mesh surface for gravity settling."""

    use_max_z_for_surface_height: bool = True
    """Use maximum Z in cluster instead of mean for surface height."""

    max_z_percentile: float = 98.0
    """Percentile for maximum Z (98 filters top 2% outliers)."""

    clearance_percentile: float = 10.0
    """Percentile for clearance calculation. Edge rays often hit nearby vertical
    walls (shelf dividers) at very short distances, while center rays measure the
    actual usable clearance. The 10th percentile filters these edge outliers while
    remaining conservative (not using median/50th)."""

    recompute_hssd_surfaces: bool = False
    """Recompute HSSD surfaces using HSM instead of loading from JSON."""

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SupportSurfaceExtractionConfig":
        """Create config from Hydra/OmegaConf nested structure.

        Args:
            cfg: Support surface extraction config subtree.

        Returns:
            SupportSurfaceExtractionConfig instance.
        """
        return cls(
            # Face clustering parameters.
            normal_cluster_threshold=cfg.face_clustering.normal_cluster_threshold,
            normal_adjacent_threshold=cfg.face_clustering.normal_adjacent_threshold,
            horizontal_normal_z_min=cfg.face_clustering.horizontal_normal_z_min,
            vertical_normal_z_max=cfg.face_clustering.vertical_normal_z_max,
            # Filtering parameters.
            min_surface_area_m2=cfg.filtering.min_surface_area_m2,
            min_area_ratio=cfg.filtering.min_area_ratio,
            min_inscribed_radius_m=cfg.filtering.min_inscribed_radius_m,
            # Clearance parameters.
            min_clearance_m=cfg.clearance.min_clearance_m,
            max_measured_clearance_m=cfg.clearance.max_measured_clearance_m,
            top_surface_clearance_m=cfg.clearance.top_surface_clearance_m,
            self_intersection_threshold_m=cfg.clearance.self_intersection_threshold_m,
            clearance_percentile=cfg.clearance.clearance_percentile,
            # Height parameters.
            surface_offset_m=cfg.height.surface_offset_m,
            use_max_z_for_surface_height=cfg.height.use_max_z_for_surface_height,
            max_z_percentile=cfg.height.max_z_percentile,
            height_tolerance_m=cfg.height.height_tolerance_m,
            # HSSD surface handling.
            recompute_hssd_surfaces=cfg.hssd.recompute_surfaces,
        )


@dataclass
class FaceCluster:
    """Group of mesh faces with similar normals."""

    face_indices: np.ndarray
    """Mesh face indices in this cluster."""

    mean_normal: np.ndarray
    """Average normal vector (3,)."""

    total_area: float
    """Sum of face areas in cluster."""


@dataclass
class ExtractedPlane:
    """Fitted plane from face cluster."""

    normal: np.ndarray
    """Unit normal vector (3,)."""

    centroid: np.ndarray
    """Plane centroid position (3,)."""

    face_indices: np.ndarray
    """Source mesh face indices."""

    area: float
    """Total surface area."""

    is_horizontal: bool
    """Surface classification result."""

    is_upward_facing: bool
    """Whether surface was originally upward-facing (before normal flip)."""


def _cluster_faces_by_normal(
    mesh: trimesh.Trimesh, config: SupportSurfaceExtractionConfig
) -> list[FaceCluster]:
    """Cluster faces by normal similarity using HSM algorithm.

    Algorithm from HSM paper Section A.2:
    1. Sort faces by area (largest first)
    2. While unclustered faces remain:
       a. Select largest unclustered face as seed
       b. Grow cluster via breadth-first search:
          - Adjacent face: dot(normal, seed_normal) >= t_adj
          - Cluster membership: dot(normal, cluster_normal) >= t_norm
       c. Use seed normal as cluster normal (HSM Algorithm 1)
    3. Return clusters

    Args:
        mesh: Input triangle mesh.
        config: Algorithm configuration parameters.

    Returns:
        List of face clusters, ordered by total area (largest first).
    """
    num_faces = len(mesh.faces)
    unclustered = set(range(num_faces))
    clusters = []

    # Precompute face properties.
    face_normals = mesh.face_normals  # (num_faces, 3).
    face_areas = mesh.area_faces  # (num_faces,).

    # Build adjacency graph.
    adjacency = {}
    for face_a, face_b in mesh.face_adjacency:
        adjacency.setdefault(face_a, []).append(face_b)
        adjacency.setdefault(face_b, []).append(face_a)

    # Sort faces by area for processing (largest first). Keep this as an indexed
    # array scan; removing from the front of a Python list is quadratic on large
    # generated meshes.
    sorted_faces = np.argsort(-face_areas)
    sorted_face_index = 0

    while unclustered:
        # Select largest unclustered face as seed.
        seed_face = None
        while sorted_face_index < len(sorted_faces):
            candidate = int(sorted_faces[sorted_face_index])
            sorted_face_index += 1
            if candidate in unclustered:
                seed_face = candidate
                break

        if seed_face is None:
            break

        # Initialize cluster with seed.
        cluster_faces = {seed_face}
        unclustered.remove(seed_face)

        seed_normal = face_normals[seed_face]

        # Grow cluster via breadth-first search.
        queue = deque([seed_face])

        while queue:
            current_face = queue.popleft()

            # Check adjacent faces.
            for neighbor_face in adjacency.get(current_face, []):
                if neighbor_face not in unclustered:
                    continue

                neighbor_normal = face_normals[neighbor_face]

                # Check adjacent threshold (similarity to current face).
                dot_with_current = np.dot(neighbor_normal, face_normals[current_face])
                if dot_with_current < config.normal_adjacent_threshold:
                    continue

                # Check cluster threshold (similarity to seed - HSM Algorithm 1 line 17).
                dot_with_seed = np.dot(neighbor_normal, seed_normal)
                if dot_with_seed < config.normal_cluster_threshold:
                    continue

                # Add to cluster.
                cluster_faces.add(neighbor_face)
                unclustered.remove(neighbor_face)
                queue.append(neighbor_face)

        # Compute total area from clustered faces.
        cluster_area = np.sum(face_areas[list(cluster_faces)])

        # Create cluster.
        cluster = FaceCluster(
            face_indices=np.array(list(cluster_faces)),
            mean_normal=seed_normal,  # Use seed normal (HSM Algorithm 1).
            total_area=cluster_area,
        )
        clusters.append(cluster)

    # Sort clusters by area (largest first).
    clusters.sort(key=lambda c: c.total_area, reverse=True)

    console_logger.debug(f"Clustered {num_faces} faces into {len(clusters)} clusters")

    return clusters


def _split_clusters_by_height(
    clusters: list[FaceCluster],
    mesh: trimesh.Trimesh,
    config: SupportSurfaceExtractionConfig,
) -> list[FaceCluster]:
    """Split face clusters that span multiple height levels.

    Multi-level furniture (desks with shelves, bookcases) can have topologically
    connected surfaces at different heights that get merged into a single cluster
    by normal-based clustering. This function splits such clusters by Z-height.

    Args:
        clusters: Face clusters from normal-based clustering.
        mesh: Source triangle mesh.
        config: Algorithm configuration with height_tolerance_m.

    Returns:
        Split clusters, one per height level. Clusters within height_tolerance_m
        are kept together.
    """
    split_clusters = []
    num_split = 0

    for cluster in clusters:
        # Get Z-positions of face centroids in this cluster.
        face_centroids_z = []
        for face_idx in cluster.face_indices:
            face_verts = mesh.vertices[mesh.faces[face_idx]]
            centroid_z = face_verts[:, 2].mean()  # Z component in Z-up coords.
            face_centroids_z.append(centroid_z)

        face_centroids_z = np.array(face_centroids_z)

        # Check if cluster spans multiple height levels.
        z_range = face_centroids_z.max() - face_centroids_z.min()

        # If cluster has small vertical extent, keep as-is.
        if z_range <= config.height_tolerance_m:
            split_clusters.append(cluster)
            continue

        # Split cluster by Z-layers using binning approach.
        # Group faces whose centroids are within height_tolerance_m.
        layers = {}  # {representative_z: [(face_idx, centroid_z), ...]}.

        for i, face_idx in enumerate(cluster.face_indices):
            z = face_centroids_z[i]

            # Find existing layer within height_tolerance_m.
            assigned = False
            for layer_z in layers.keys():
                if abs(z - layer_z) <= config.height_tolerance_m:
                    layers[layer_z].append((face_idx, z))
                    assigned = True
                    break

            # Create new layer if no match found.
            if not assigned:
                layers[z] = [(face_idx, z)]

        # Create sub-clusters for each height layer.
        if len(layers) > 1:
            num_split += 1

        for layer_faces in layers.values():
            face_indices = np.array([f[0] for f in layer_faces])

            # Recompute area and mean normal for sub-cluster.
            layer_area = mesh.area_faces[face_indices].sum()
            layer_normals = mesh.face_normals[face_indices]
            # Area-weighted mean normal.
            face_areas = mesh.area_faces[face_indices]
            mean_normal = np.average(layer_normals, weights=face_areas, axis=0)

            sub_cluster = FaceCluster(
                face_indices=face_indices,
                mean_normal=mean_normal,
                total_area=layer_area,
            )
            split_clusters.append(sub_cluster)

    if num_split > 0:
        console_logger.debug(
            f"Split {num_split} clusters by height into {len(split_clusters)} total "
            f"(was {len(clusters)})"
        )

    return split_clusters


def _fit_plane_to_cluster(
    mesh: trimesh.Trimesh, cluster: FaceCluster, config: SupportSurfaceExtractionConfig
) -> ExtractedPlane:
    """Fit plane to face cluster using OBB (oriented bounding box).

    Matches HSM paper methodology (Section A.2).

    Algorithm:
    1. Extract face centroids weighted by area
    2. Compute weighted mean centroid
    3. Optionally compute height offset to max Z percentile
    4. Fit OBB to cluster submesh
    5. Normal = OBB axis with largest Z component (upward-facing)
    6. Apply height offset to move centroid to surface top

    Args:
        mesh: Input triangle mesh.
        cluster: Face cluster to fit plane to.
        config: Algorithm configuration.

    Returns:
        Extracted plane with normal, centroid, and planarity score.
    """
    # Extract centroids of faces in cluster.
    face_indices = cluster.face_indices
    face_areas = mesh.area_faces[face_indices]

    # Compute centroids: mean of triangle vertices.
    centroids = np.mean(mesh.vertices[mesh.faces[face_indices]], axis=1)  # (N, 3).

    # Weighted mean centroid (by area).
    # Handle zero-area faces (degenerate triangles).
    total_area = np.sum(face_areas)
    if total_area < 1e-10:
        # All faces are degenerate, use unweighted mean.
        mean_centroid = np.mean(centroids, axis=0)
    else:
        mean_centroid = np.average(centroids, weights=face_areas, axis=0)

    # Compute offset to max Z in cluster if enabled.
    # We'll use this offset to adjust the plane height later.
    height_offset = 0.0
    if config.use_max_z_for_surface_height:
        # Use percentile instead of max to filter outliers.
        max_z = np.percentile(centroids[:, 2], config.max_z_percentile)
        height_offset = max_z - mean_centroid[2]

    # Check for degenerate cluster (zero normal).
    if np.linalg.norm(cluster.mean_normal) < 1e-10:
        raise ValueError(
            f"Cluster has degenerate normal (magnitude < 1e-10). "
            f"Cluster size: {len(face_indices)}"
        )

    # Fit OBB (oriented bounding box) to cluster.
    # For small clusters or when OBB fails, fall back to cluster mean normal.
    if len(face_indices) >= 3:
        try:
            # Create submesh from cluster faces for OBB fitting.
            submesh = mesh.submesh([face_indices], append=True)

            # Get OBB transform.
            obb = submesh.bounding_box_oriented
            obb_transform = obb.primitive.transform

            # Extract rotation matrix (top-left 3x3).
            # The columns are the principal axes of the OBB.
            rotation = obb_transform[:3, :3]

            # Find axis with largest Z component (most vertical).
            # This is the surface normal direction.
            z_components = np.abs(rotation[2, :])
            normal_axis_idx = np.argmax(z_components)
            normal = rotation[:, normal_axis_idx].copy()

            # Validate OBB produced finite values.
            if not np.all(np.isfinite(normal)):
                raise ValueError("OBB produced non-finite normal.")
        except Exception:
            # OBB failed (degenerate geometry), use cluster mean normal.
            normal = cluster.mean_normal.copy()
    else:
        # Too few faces for reliable OBB, use cluster mean normal.
        normal = cluster.mean_normal.copy()

    # Normalize.
    normal /= np.linalg.norm(normal)

    # Check for NaN/inf from degenerate faces (zero normals).
    if not np.all(np.isfinite(normal)):
        raise ValueError(
            f"Cluster has degenerate normal after normalization (NaN/inf detected). "
            f"Cluster size: {len(face_indices)}, original normal norm: "
            f"{np.linalg.norm(cluster.mean_normal):.6f}"
        )

    # Classify as horizontal based on absolute Z component.
    # Uses abs() to detect both upward and downward horizontal surfaces.
    # Downward-facing surfaces will be filtered out later for gravity-based placement.
    is_horizontal = abs(normal[2]) >= config.horizontal_normal_z_min

    # Track original normal direction using cluster mean normal (from mesh faces).
    # Cannot use OBB normal because OBB orientation is arbitrary and may be flipped.
    # Upward-facing surfaces have positive Z component in Drake Z-up coordinates.
    # This filters out bottom surfaces (e.g., underside of shelf planks).
    is_upward_facing = cluster.mean_normal[2] > 0

    # Ensure upward-pointing for downstream processing (transform, bounds).
    # This is only for geometry, not for classification.
    if normal[2] < 0:
        normal = -normal

    # Apply height offset to move plane to max Z if enabled.
    adjusted_centroid = mean_centroid.copy()
    adjusted_centroid[2] += height_offset

    plane = ExtractedPlane(
        normal=normal,
        centroid=adjusted_centroid,
        face_indices=face_indices,
        area=cluster.total_area,
        is_horizontal=is_horizontal,
        is_upward_facing=is_upward_facing,
    )

    return plane


def _create_surface_transform(
    centroid: np.ndarray, normal: np.ndarray
) -> RigidTransform:
    """Create RigidTransform for surface frame in Drake Z-up coordinates.

    Input is in Z-up coordinates, output is in Z-up coordinates.

    Surface frame (Z-up):
    - Origin: centroid
    - Z-axis: normal (upward in Z-up)
    - Y-axis: Z × world_x (perpendicular to normal and world X)
    - X-axis: Y × Z (recomputed for orthogonality, right-hand rule)

    Args:
        centroid: Surface center position in Z-up coordinates (3,).
        normal: Surface normal in Z-up coordinates (3,), points upward (Z+).

    Returns:
        RigidTransform from surface frame to world frame in Drake Z-up coordinates.
    """
    # Z-axis = normal (should be upward in Z-up coordinate system).
    z_axis = normal / np.linalg.norm(normal)

    # Build right-handed coordinate frame with Z pointing up.
    # Choose initial X direction in XY plane, then compute Y = Z × X, then X = Y × Z.
    world_x = np.array([1.0, 0.0, 0.0])

    # Compute Y-axis: Y = Z × world_x (perpendicular to both).
    y_axis = np.cross(z_axis, world_x)

    # Handle degenerate case where world_x is parallel to z_axis.
    if np.linalg.norm(y_axis) < 1e-6:
        # Use world Y instead.
        world_y = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, world_y)

    y_axis /= np.linalg.norm(y_axis)

    # Recompute X-axis: X = Y × Z (ensures right-handed orthogonal system).
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Create rotation matrix in Z-up coordinates [X Y Z] (column vectors).
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # Create RigidTransform in Z-up coordinates.
    transform = RigidTransform(
        R=RotationMatrix(rotation_matrix),
        p=centroid,
    )

    return transform


def _compute_convex_hull_min_width(hull_vertices: np.ndarray) -> float:
    """Compute minimum width of 2D convex hull.

    The minimum width is the smallest perpendicular distance across the hull,
    found by checking the "height" when the hull rests on each edge. This
    correctly handles diagonal surfaces that axis-aligned bounding boxes miss.

    Args:
        hull_vertices: 2D vertices of convex hull, shape (N, 2).

    Returns:
        Minimum width in meters, or inf if degenerate.
    """
    min_width = float("inf")
    n = len(hull_vertices)

    for i in range(n):
        # Edge from vertex i to vertex i+1.
        p1 = hull_vertices[i]
        p2 = hull_vertices[(i + 1) % n]

        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue

        # Unit normal perpendicular to edge.
        normal = np.array([-edge[1], edge[0]]) / edge_len

        # Max distance from edge to any vertex (the "height" on this edge).
        max_dist = 0.0
        for j in range(n):
            dist = abs(np.dot(hull_vertices[j] - p1, normal))
            max_dist = max(max_dist, dist)

        min_width = min(min_width, max_dist)

    return min_width


def _compute_clearance_via_raycasting(
    surface_mesh: trimesh.Trimesh,
    full_mesh: trimesh.Trimesh,
    config: SupportSurfaceExtractionConfig,
    default_clearance: float,
) -> float:
    """Compute clearance by ray-casting upward from surface vertices.

    Casts vertical rays (+Z direction) from each vertex in the surface mesh
    to find intersections with the full mesh geometry. Uses percentile-based
    clearance (default 10th percentile) to filter edge effects where rays hit
    nearby vertical walls (shelf dividers) at very short distances.

    Args:
        surface_mesh: Flattened support surface mesh (vertices at surface height).
        full_mesh: Complete furniture mesh to ray-cast against.
        config: Algorithm configuration parameters.
        default_clearance: Clearance to use when no obstacles found above.

    Returns:
        Clearance distance in meters based on config.clearance_percentile.
    """
    # Get surface vertices as ray origins, with Z offset to avoid self-intersection.
    # The surface mesh is flattened to a single Z height (the surface plane).
    # Without offset, rays immediately hit the surface faces they originate from.
    ray_offset_z = config.self_intersection_threshold_m
    ray_origins = surface_mesh.vertices.copy()  # Shape: (N, 3).
    ray_origins[:, 2] += ray_offset_z

    # Ray directions: straight up along Z-axis.
    ray_directions = np.tile([0, 0, 1], (len(ray_origins), 1))  # Shape: (N, 3).

    # Cast rays against full mesh.
    # Returns locations of intersections and indices of hit triangles.
    locations, index_ray, index_tri = full_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,  # Only need first hit above each vertex.
    )

    console_logger.debug(
        f"Ray-casting: {len(ray_origins)} rays, {len(locations)} hits, "
        f"ray Z: [{np.min(ray_origins[:, 2]):.3f}, {np.max(ray_origins[:, 2]):.3f}], "
        f"mesh Z: [{np.min(full_mesh.vertices[:, 2]):.3f}, "
        f"{np.max(full_mesh.vertices[:, 2]):.3f}]"
        + (
            f", hit Z: [{np.min(locations[:, 2]):.3f}, {np.max(locations[:, 2]):.3f}]"
            if len(locations) > 0
            else ""
        )
    )

    if len(locations) == 0:
        # No intersections - use default clearance for top surfaces.
        console_logger.debug(f"No hits, using default clearance={default_clearance}m")
        return default_clearance

    # Compute distances from ray origins to intersection points.
    # index_ray tells us which ray (vertex) each intersection belongs to.
    # Add ray_offset_z to get true clearance from original surface (not offset origin).
    distances = (
        np.linalg.norm(locations - ray_origins[index_ray], axis=1) + ray_offset_z
    )

    # Filter out tiny distances (self-intersections or numerical noise).
    # With the offset, self-intersections should no longer occur, but filter anyway.
    valid_distances = distances[distances > config.self_intersection_threshold_m]

    # Include non-hit rays with default clearance for correct percentile semantics.
    # The percentile should represent "X% of rays have clearance this low or lower",
    # not "Xth percentile of only the rays that hit something". Non-hit rays have
    # infinite clearance; we use default_clearance as a practical upper bound.
    num_non_hits = len(ray_origins) - len(valid_distances)
    all_distances = np.concatenate(
        [
            valid_distances,
            np.full(num_non_hits, default_clearance),
        ]
    )

    # Use percentile over all rays (hits + non-hits) to compute clearance.
    percentile_clearance = float(
        np.percentile(all_distances, config.clearance_percentile)
    )
    capped_clearance = min(percentile_clearance, config.max_measured_clearance_m)

    console_logger.debug(
        f"Clearance: {len(valid_distances)} hits + {num_non_hits} non-hits = "
        f"{len(all_distances)} rays, "
        f"p{config.clearance_percentile:.0f}={percentile_clearance:.3f}m, "
        f"capped={capped_clearance:.3f}m"
    )
    return capped_clearance


def _compute_surface_bounds(
    mesh: trimesh.Trimesh,
    plane: ExtractedPlane,
    transform: RigidTransform,
    simplified_mesh: trimesh.Trimesh,
    config: SupportSurfaceExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute 3D AABB for surface in surface-local frame.

    Algorithm:
    1. Extract vertices of faces in cluster
    2. Transform vertices to surface frame
    3. Compute 2D AABB in XY plane (surface coordinates)
    4. Ray-cast upward from surface to find clearance height
    5. Center bounding box around surface frame origin in XY

    Clearance is computed by ray-casting vertically upward from each vertex
    in the simplified surface mesh to find intersections with the full mesh.
    This handles thick meshes, tilted obstacles, and complex geometry.

    Args:
        mesh: Input triangle mesh (source geometry for surface vertices).
        plane: Extracted plane.
        transform: Surface frame transform.
        simplified_mesh: Flattened surface mesh for ray-casting.
        config: Algorithm configuration.

    Returns:
        Tuple of (bounds_min, bounds_max, clearance) in surface-local frame.
        bounds_min and bounds_max are (3,) arrays, clearance is float in meters.
    """
    # Extract vertices of faces in this surface.
    face_vertices = mesh.vertices[mesh.faces[plane.face_indices]]  # (N, 3, 3).
    vertices = face_vertices.reshape(-1, 3)  # Flatten to (N*3, 3).

    # Transform to surface frame.
    transform_matrix = transform.GetAsMatrix4()  # 4x4 homogeneous.
    transform_inv = np.linalg.inv(transform_matrix)

    # Convert to homogeneous coordinates.
    vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])

    # Transform to surface frame.
    vertices_surface = (transform_inv @ vertices_hom.T).T[:, :3]

    # Compute 2D bounds in XY plane.
    xy_min = np.min(vertices_surface[:, :2], axis=0)
    xy_max = np.max(vertices_surface[:, :2], axis=0)

    # Center bounding box around origin in XY.
    xy_half_extents = (xy_max - xy_min) / 2

    xy_min_centered = -xy_half_extents
    xy_max_centered = xy_half_extents

    # Z bounds: surface offset to ray-cast clearance height.
    z_min = config.surface_offset_m

    # Ray-cast to find actual clearance above this surface.
    clearance = _compute_clearance_via_raycasting(
        surface_mesh=simplified_mesh,
        full_mesh=mesh,
        config=config,
        default_clearance=config.top_surface_clearance_m,
    )

    z_max = z_min + clearance

    # Combine into 3D bounds.
    bounds_min = np.array([xy_min_centered[0], xy_min_centered[1], z_min])
    bounds_max = np.array([xy_max_centered[0], xy_max_centered[1], z_max])

    return bounds_min, bounds_max, clearance


def _create_flattened_surface_mesh(
    mesh: trimesh.Trimesh, face_indices: np.ndarray
) -> trimesh.Trimesh:
    """Create flattened support surface mesh in Z-up coordinates.

    Extracts submesh and flattens to target height for visualization.
    Output mesh is ready for direct rendering without additional transforms.

    Args:
        mesh: Source mesh in Z-up coordinates (already transformed).
        face_indices: Indices of faces that comprise the support surface.

    Returns:
        Flattened trimesh.Trimesh in Z-up coordinates at horizontal plane.
    """
    # Extract sub-mesh using face indices.
    submesh = mesh.submesh([face_indices], append=True)

    # Get target Z height as maximum Z of surface vertices.
    # Using centroid Z would cause ray-casting self-intersection because some
    # faces lie above the average. Max Z ensures surface is at the top.
    face_vertices = mesh.vertices[mesh.faces[face_indices]]
    target_z = np.max(face_vertices[:, :, 2])

    # Flatten to horizontal plane: Set all Z coordinates to target height.
    # In Z-up coordinates, horizontal surfaces have constant Z.
    # The mesh stores vertices in global coordinates for visualization.
    vertices = submesh.vertices.copy()
    vertices[:, 2] = target_z
    submesh.vertices = vertices

    return submesh


def _load_and_prepare_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load mesh and prepare it for surface extraction.

    Handles:
    - File validation and loading.
    - Scene concatenation (multiple geometries).
    - Y-up to Z-up coordinate conversion (GLTF → Drake).
    - Vertex merging for proper adjacency detection.

    Args:
        mesh_path: Path to mesh file (GLTF, OBJ, etc.).

    Returns:
        Prepared mesh in Z-up coordinates with merged vertices.

    Raises:
        FileNotFoundError: If mesh_path does not exist.
        ValueError: If mesh cannot be loaded or has no faces.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    console_logger.info(f"Extracting support surfaces from {mesh_path.name}")

    mesh = trimesh.load(str(mesh_path), force="mesh")

    # Handle Scene objects (multiple geometries).
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [
                geom
                for geom in mesh.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
            ]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as Trimesh: {mesh_path}")

    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    console_logger.debug(
        f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
    )

    # Convert from Y-up (GLTF) to Z-up (Drake/Blender) immediately.
    # All subsequent processing happens in Z-up coordinates.
    # Transformation: (X, Y, Z) → (X, -Z, Y).
    # This ensures positive Y (up in GLTF) becomes positive Z (up in Drake).
    transform_y_to_z = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    mesh.apply_transform(transform_y_to_z)

    # Merge duplicate vertices for proper adjacency detection.
    # GLTF format often stores duplicate vertices per triangle.
    # Use digits_vertex=4 for ~0.0001m precision (matching Blender defaults).
    # Done after Z-up transform to ensure all processing happens in Z-up.
    vertices_before = len(mesh.vertices)
    mesh.merge_vertices(digits_vertex=4)
    console_logger.debug(
        f"Merged vertices: {vertices_before} → {len(mesh.vertices)} "
        f"({100 * (1 - len(mesh.vertices) / vertices_before):.1f}% reduction)"
    )

    return mesh


def _create_support_surface_from_plane(
    plane: ExtractedPlane,
    mesh: trimesh.Trimesh,
    surface_index: int,
    config: SupportSurfaceExtractionConfig,
) -> "SupportSurface" | None:
    """Create SupportSurface from plane with bounds, clearance, and filtering.

    Computes surface transform, bounds, and clearance via ray-casting.
    Filters out surfaces that are too small, have insufficient clearance,
    or are thin slivers.

    Args:
        plane: Extracted plane to convert to support surface.
        mesh: Source mesh containing the surface geometry.
        surface_index: Index for unique surface ID.
        config: Configuration with filtering thresholds.

    Returns:
        SupportSurface if plane passes all filters, None otherwise.
    """
    from scenesmith.agent_utils.room import SupportSurface, UniqueID

    # Apply surface offset for gravity settling.
    # This lifts the surface origin above the mesh by config.surface_offset_m.
    # Objects placed at Z=0 in surface-local frame will be at this offset height.
    offset_centroid = plane.centroid.copy()
    offset_centroid[2] += config.surface_offset_m

    # Validate plane geometry before creating transform.
    # Check centroid for NaN/Inf.
    if not np.all(np.isfinite(offset_centroid)):
        console_logger.debug(
            f"  ✗ Rejected: centroid contains NaN/Inf: {offset_centroid}"
        )
        return None

    # Check normal for NaN/Inf.
    if not np.all(np.isfinite(plane.normal)):
        console_logger.debug(f"  ✗ Rejected: normal contains NaN/Inf: {plane.normal}")
        return None

    # Check normal magnitude to prevent division by zero.
    normal_magnitude = np.linalg.norm(plane.normal)
    if normal_magnitude < 1e-9:
        console_logger.debug(
            f"  ✗ Rejected: normal magnitude too small: {normal_magnitude:.2e}"
        )
        return None

    # Create surface transform with offset centroid.
    transform = _create_surface_transform(centroid=offset_centroid, normal=plane.normal)

    # Create flattened mesh for this surface (needed for ray-casting).
    flattened_mesh = _create_flattened_surface_mesh(
        mesh=mesh,
        face_indices=plane.face_indices,
    )

    # Compute surface bounds (uses ray-casting with flattened mesh).
    bounds_min, bounds_max, clearance = _compute_surface_bounds(
        mesh=mesh,
        plane=plane,
        transform=transform,
        simplified_mesh=flattened_mesh,
        config=config,
    )

    # Adjust clearance to account for surface offset.
    # The clearance is computed from the original plane, but the surface origin
    # has been moved up by surface_offset_m. We need to reduce the clearance
    # by this amount to reflect the actual available space above the offset surface.
    clearance_adjusted = clearance - config.surface_offset_m

    # Recompute bounds_max with adjusted clearance.
    # bounds_max[2] was computed as z_min + clearance, but should be z_min +
    # clearance_adjusted.
    bounds_max = bounds_max.copy()
    bounds_max[2] = config.surface_offset_m + clearance_adjusted

    # Skip surfaces with insufficient clearance (internal surfaces).
    if clearance_adjusted < config.min_clearance_m:
        console_logger.debug(
            f"  ✗ Rejected: insufficient clearance {clearance_adjusted:.3f}m "
            f"(< {config.min_clearance_m}m threshold)"
        )
        return None

    # Skip thin slivers by checking inscribed radius and aspect ratio.
    # Compute 2D convex hull of surface vertices in XY plane.
    xy_vertices = flattened_mesh.vertices[:, :2]

    # Handle degenerate geometry gracefully using safe wrapper.
    hull, processed_vertices = safe_convex_hull_2d(xy_vertices)
    if hull is None:
        console_logger.debug(
            f"  ✗ Rejected: degenerate geometry "
            f"(ConvexHull failed - likely collinear/duplicate vertices)"
        )
        return None
    hull_vertices = processed_vertices[hull.vertices]

    # Compute minimum width of convex hull (handles diagonal surfaces correctly).
    # The inscribed radius approximation is half the minimum width.
    hull_min_width = _compute_convex_hull_min_width(hull_vertices)
    min_inscribed_radius = hull_min_width / 2.0

    if min_inscribed_radius < config.min_inscribed_radius_m:
        console_logger.debug(
            f"  ✗ Rejected: inscribed radius {min_inscribed_radius:.3f}m "
            f"(< {config.min_inscribed_radius_m}m threshold)"
        )
        return None

    # Transform mesh vertices to surface-local coordinates.
    # The mesh vertices are currently in mesh-local frame (furniture geometry).
    # We need them in surface-local frame for convex hull validation in contains_point_2d.
    # Use the inverse of the surface transform (which maps surface → mesh).
    transform_matrix_inv = transform.inverse().GetAsMatrix4()
    vertices_local = []
    for v in flattened_mesh.vertices:
        v_hom = np.append(v, 1.0)
        v_local_hom = transform_matrix_inv @ v_hom
        vertices_local.append(v_local_hom[:3])

    # Create new mesh with surface-local vertices.
    mesh_local = trimesh.Trimesh(
        vertices=np.array(vertices_local),
        faces=flattened_mesh.faces,
    )

    # Create surface.
    # Note: surface_id is temporary - replaced by scene.generate_surface_id() when added.
    surface = SupportSurface(
        surface_id=UniqueID(f"surface_{surface_index}"),
        bounding_box_min=bounds_min,
        bounding_box_max=bounds_max,
        transform=transform,
        mesh=mesh_local,
    )

    return surface


def extract_support_surfaces_from_mesh(
    mesh_path: Path,
    config: SupportSurfaceExtractionConfig | None = None,
) -> list["SupportSurface"]:
    """Extract all horizontal support surfaces from furniture mesh.

    Implements HSM face clustering algorithm (https://arxiv.org/abs/2503.16848v2).

    Args:
        mesh_path: Path to visual mesh (.gltf or .glb).
        config: Algorithm parameters (defaults to HSM values).

    Returns:
        List of SupportSurface objects, sorted by area (largest first).

    Raises:
        FileNotFoundError: If mesh file doesn't exist.
        ValueError: If mesh loading fails or mesh has no faces.

    Algorithm:
        1. Load mesh and convert Y-up (GLTF) to Z-up (Drake)
        2. Cluster faces by normal similarity
        3. Fit plane to each cluster
        4. Classify surfaces as horizontal or vertical
        5. Filter by minimum area threshold
        6. Create SupportSurface with bounds, transform, and clearance
        7. Filter by minimum clearance and inscribed radius
        8. Sort by area descending
    """

    start_time = time.time()
    if config is None:
        config = SupportSurfaceExtractionConfig()

    mesh = _load_and_prepare_mesh(mesh_path=mesh_path)

    # Cluster faces by normal similarity.
    clusters = _cluster_faces_by_normal(mesh=mesh, config=config)

    # A cluster with less total mesh area than the minimum surface area can only
    # split into still-smaller clusters, so it cannot produce a valid support
    # surface. Filtering here avoids fitting tens of thousands of tiny decorative
    # clusters on high-poly generated furniture.
    num_clusters_before_area_filter = len(clusters)
    clusters = [
        cluster
        for cluster in clusters
        if cluster.total_area >= config.min_surface_area_m2
    ]
    if len(clusters) != num_clusters_before_area_filter:
        console_logger.debug(
            f"After cluster area pre-filter (>= {config.min_surface_area_m2}m²): "
            f"{len(clusters)}/{num_clusters_before_area_filter} clusters remain"
        )

    # Split clusters by height to separate multi-level surfaces.
    clusters = _split_clusters_by_height(clusters=clusters, mesh=mesh, config=config)

    # Fit planes to clusters.
    planes = []
    for cluster in clusters:
        try:
            plane = _fit_plane_to_cluster(mesh=mesh, cluster=cluster, config=config)
            planes.append(plane)
        except ValueError as e:
            # Skip degenerate clusters (e.g., faces with zero normals).
            console_logger.debug(f"Skipping degenerate cluster: {e}")
            continue

    # Separate horizontal and vertical surfaces.
    # Only keep upward-facing horizontal surfaces for gravity-based placement.
    horizontal_planes_all = [plane for plane in planes if plane.is_horizontal]
    horizontal_planes_downward = [
        plane for plane in horizontal_planes_all if not plane.is_upward_facing
    ]
    horizontal_planes = [
        plane for plane in planes if plane.is_horizontal and plane.is_upward_facing
    ]
    vertical_planes = [plane for plane in planes if not plane.is_horizontal]

    # Log filtering details.
    if horizontal_planes_downward:
        normals = [
            f"[{p.normal[0]:.2f}, {p.normal[1]:.2f}, {p.normal[2]:.2f}]"
            for p in horizontal_planes_downward
        ]
        console_logger.debug(
            f"Filtered out {len(horizontal_planes_downward)} downward-facing surfaces "
            f"(normals: {normals})"
        )

    console_logger.debug(
        f"Clustering created {len(planes)} planes: {len(horizontal_planes)} horizontal "
        f"(upward), {len(horizontal_planes_downward)} horizontal (downward), "
        f"{len(vertical_planes)} vertical"
    )

    # Pre-filter by plane mesh area to remove tiny surfaces early.
    # This catches thin dividers/slivers that would otherwise pass bbox area filters
    # because their convex hulls span the full width of the furniture.
    large_planes = [
        plane for plane in horizontal_planes if plane.area >= config.min_surface_area_m2
    ]

    console_logger.debug(
        f"After mesh area filter (>= {config.min_surface_area_m2}m²): "
        f"{len(large_planes)}/{len(horizontal_planes)} planes remain"
    )

    # Convert planes to SupportSurface objects.
    # Also filter by bbox area after creation to handle convex hull edge cases.
    surfaces = []
    for i, plane in enumerate(large_planes):
        console_logger.debug(
            f"Processing plane {i}: area={plane.area:.4f}m², "
            f"centroid_z={plane.centroid[2]:.3f}m, is_horizontal={plane.is_horizontal}"
        )
        surface = _create_support_surface_from_plane(
            plane=plane, mesh=mesh, surface_index=i, config=config
        )
        if surface is not None:
            # Filter by bounding box area (surface.area), not plane area.
            if surface.area >= config.min_surface_area_m2:
                surfaces.append(surface)
                console_logger.debug(
                    f"  → Created surface {i} (bbox area={surface.area:.4f}m²)"
                )
            else:
                console_logger.debug(
                    f"  → Rejected: bbox area {surface.area:.4f}m² < "
                    f"{config.min_surface_area_m2}m² threshold"
                )
        else:
            console_logger.debug(f"  → Rejected by _create_support_surface_from_plane")

    # Sort by bounding box area (largest first).
    surfaces.sort(key=lambda s: s.area, reverse=True)

    console_logger.info(
        f"Extracted {len(surfaces)} support surfaces for {mesh_path.name} in "
        f"{time.time() - start_time:.2f} seconds"
    )

    return surfaces


def _parse_sdf_mesh_to_link(sdf_path: Path) -> dict[str, str]:
    """Parse SDF to build mesh filename -> link name mapping.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        Dict mapping mesh filename (e.g., 'P_7b614f8bbcce8e3f.gltf') to
        link name (e.g., 'E_drawer_1').
    """
    import xml.etree.ElementTree as ET

    mesh_to_link: dict[str, str] = {}
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    for link in root.iter("link"):
        link_name = link.get("name")
        if not link_name:
            continue

        visual_count = 0
        for visual in link.iter("visual"):
            for uri in visual.iter("uri"):
                if uri.text and uri.text.endswith(".gltf"):
                    visual_count += 1
                    # Extract filename from URI (may include subdir path).
                    mesh_filename = Path(uri.text).name
                    mesh_to_link[mesh_filename] = link_name

        if visual_count > 1:
            console_logger.warning(
                f"Link '{link_name}' has {visual_count} visual meshes; "
                f"surfaces from all meshes will be assigned to this link"
            )

    return mesh_to_link


def extract_support_surfaces_articulated(
    sdf_dir: Path,
    config: SupportSurfaceExtractionConfig | None = None,
    sdf_path: Path | None = None,
) -> list["SupportSurface"]:
    """Extract support surfaces from articulated object with per-link association.

    For articulated objects (e.g., furniture with drawers/doors), this function:
    1. Extracts surfaces from each link mesh (for correct link association)
    2. Re-computes clearance against combined mesh (for accurate filtering)
    3. Filters surfaces that don't meet clearance threshold

    Args:
        sdf_dir: Directory containing articulated object files.
        config: Surface extraction configuration. If None, uses defaults.
        sdf_path: Path to the SDF file for mesh-to-link mapping. Required for
            accurate link name resolution with ArtVIP assets.

    Returns:
        List of SupportSurface objects with link_name populated.

    Raises:
        FileNotFoundError: If no link meshes found.
    """
    from scenesmith.agent_utils.room import SupportSurface

    config = config or SupportSurfaceExtractionConfig()
    start_time = time.time()

    # Build mesh-to-link mapping from SDF for accurate link name resolution.
    mesh_to_link: dict[str, str] = {}
    if sdf_path and sdf_path.exists():
        mesh_to_link = _parse_sdf_mesh_to_link(sdf_path)
        console_logger.debug(
            f"Built mesh-to-link mapping with {len(mesh_to_link)} entries"
        )

    # Find per-link mesh files - check multiple locations/patterns.
    link_gltfs: list[Path] = []

    # Pattern 1: *_combined.gltf at top level (PartNet-Mobility converted).
    link_gltfs.extend(
        f for f in sdf_dir.glob("*_combined.gltf") if f.name != "combined_scene.gltf"
    )

    # Pattern 2: visual/ subdirectory (PartNet-Mobility raw).
    visual_dir = sdf_dir / "visual"
    if visual_dir.exists():
        link_gltfs.extend(visual_dir.glob("*_visual.gltf"))

    # Pattern 3: *_meshes/ subdirectory (ArtVIP).
    for meshes_subdir in sdf_dir.glob("*_meshes"):
        link_gltfs.extend(
            f for f in meshes_subdir.glob("*.gltf") if f.name != "combined_scene.gltf"
        )

    link_gltfs = sorted(set(link_gltfs))  # Dedupe and sort.

    if not link_gltfs:
        # Fallback to combined mesh extraction.
        combined_path = sdf_dir / "combined_scene.gltf"
        # Also check *_meshes subdirectory for combined mesh (ArtVIP).
        if not combined_path.exists():
            for meshes_subdir in sdf_dir.glob("*_meshes"):
                alt_combined = meshes_subdir / "combined_scene.gltf"
                if alt_combined.exists():
                    combined_path = alt_combined
                    break
        if combined_path.exists():
            console_logger.warning(f"No per-link meshes in {sdf_dir}, using combined")
            return extract_support_surfaces_from_mesh(combined_path, config)
        raise FileNotFoundError(f"No link meshes found in {sdf_dir}")

    # Load combined mesh for clearance re-computation.
    combined_path = sdf_dir / "combined_scene.gltf"
    # Also check *_meshes subdirectory (ArtVIP).
    if not combined_path.exists():
        for meshes_subdir in sdf_dir.glob("*_meshes"):
            alt_combined = meshes_subdir / "combined_scene.gltf"
            if alt_combined.exists():
                combined_path = alt_combined
                break
    combined_mesh = None
    if combined_path.exists():
        combined_mesh = _load_and_prepare_mesh(combined_path)
        console_logger.debug("Loaded combined mesh for clearance computation")

    console_logger.info(f"Extracting surfaces from {len(link_gltfs)} link meshes")

    all_surfaces: list[SupportSurface] = []

    for link_gltf in link_gltfs:
        # Derive link_name from SDF mapping, fallback to filename.
        mesh_filename = link_gltf.name
        if mesh_filename in mesh_to_link:
            link_name = mesh_to_link[mesh_filename]
        else:
            # Fallback: strip _combined/_visual suffix.
            stem = link_gltf.stem
            link_name = stem.replace("_combined", "").replace("_visual", "")

        console_logger.debug(f"Processing mesh '{mesh_filename}' -> link '{link_name}'")

        # Extract surfaces from this link mesh using the standard algorithm.
        try:
            link_surfaces = extract_support_surfaces_from_mesh(
                mesh_path=link_gltf, config=config
            )
        except (FileNotFoundError, ValueError) as e:
            console_logger.warning(f"Failed to extract from {link_gltf.name}: {e}")
            continue

        # Tag each surface with its source link.
        for surface in link_surfaces:
            surface.link_name = link_name

        console_logger.info(
            f"Link '{link_name}': {len(link_surfaces)} surfaces extracted"
        )
        all_surfaces.extend(link_surfaces)

    # Log per-link surface counts.
    link_counts = {}
    for surface in all_surfaces:
        link_counts[surface.link_name] = link_counts.get(surface.link_name, 0) + 1
    console_logger.info(
        f"Per-link extraction: {len(all_surfaces)} surfaces from "
        f"{len(link_counts)} links: {link_counts}"
    )

    # Re-compute clearance against combined mesh and filter.
    if combined_mesh is not None:
        filtered_surfaces = []
        for surface in all_surfaces:
            # Create flattened mesh at surface height for ray-casting.
            surface_z = surface.transform.translation()[2]
            # Create a simple grid of points covering the surface bbox.
            bounds_min = surface.bounding_box_min
            bounds_max = surface.bounding_box_max
            # Transform bbox corners to world frame.
            transform_matrix = surface.transform.GetAsMatrix4()
            local_corners = np.array(
                [
                    [bounds_min[0], bounds_min[1], 0],
                    [bounds_max[0], bounds_min[1], 0],
                    [bounds_max[0], bounds_max[1], 0],
                    [bounds_min[0], bounds_max[1], 0],
                ]
            )
            corners_hom = np.column_stack([local_corners, np.ones(4)])
            world_corners = (transform_matrix @ corners_hom.T).T[:, :3]

            # Sample points on surface for ray-casting.
            n_samples = 25  # 5x5 grid.
            u = np.linspace(0, 1, 5)
            v = np.linspace(0, 1, 5)
            ray_origins = []
            for ui in u:
                for vi in v:
                    pt = (
                        (1 - ui) * (1 - vi) * world_corners[0]
                        + ui * (1 - vi) * world_corners[1]
                        + ui * vi * world_corners[2]
                        + (1 - ui) * vi * world_corners[3]
                    )
                    ray_origins.append(pt)
            ray_origins = np.array(ray_origins)
            ray_origins[:, 2] = surface_z + config.self_intersection_threshold_m

            # Cast rays upward.
            ray_directions = np.tile([0, 0, 1], (n_samples, 1))
            locations, index_ray, _ = combined_mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False,
            )

            if len(locations) > 0:
                distances = (
                    np.linalg.norm(locations - ray_origins[index_ray], axis=1)
                    + config.self_intersection_threshold_m
                )
                clearance = float(np.percentile(distances, config.clearance_percentile))
            else:
                clearance = config.top_surface_clearance_m

            # Apply clearance filter.
            if clearance >= config.min_clearance_m:
                filtered_surfaces.append(surface)
                console_logger.debug(
                    f"Surface {surface.surface_id} (link={surface.link_name}): "
                    f"clearance={clearance:.3f}m >= {config.min_clearance_m}m ✓"
                )
            else:
                console_logger.debug(
                    f"Surface {surface.surface_id} (link={surface.link_name}): "
                    f"clearance={clearance:.3f}m < {config.min_clearance_m}m ✗ filtered"
                )

        all_surfaces = filtered_surfaces
        console_logger.info(
            f"After clearance re-computation: {len(all_surfaces)} surfaces remain"
        )

    # Sort by bounding box area (largest first).
    all_surfaces.sort(key=lambda s: s.area, reverse=True)

    console_logger.info(
        f"Extracted {len(all_surfaces)} surfaces from articulated object in "
        f"{time.time() - start_time:.2f}s"
    )

    return all_surfaces


def load_link_meshes(sdf_dir: Path) -> dict[str, trimesh.Trimesh]:
    """Load per-link meshes from articulated object directory.

    Articulated objects (from PartNet-Mobility) have per-link mesh files named
    `{link_name}_combined.gltf`. This function loads them for surface-link
    association.

    Args:
        sdf_dir: Directory containing the articulated object files.

    Returns:
        Mapping from link name to loaded mesh. Empty dict if no link meshes found.
    """
    link_meshes: dict[str, trimesh.Trimesh] = {}

    for gltf_file in sdf_dir.glob("*_combined.gltf"):
        # Skip the merged mesh used for extraction.
        if gltf_file.name == "combined_scene.gltf":
            continue

        link_name = gltf_file.stem.replace("_combined", "")
        try:
            mesh = trimesh.load(gltf_file)

            # Handle Scene objects (multi-mesh gltf files).
            if isinstance(mesh, trimesh.Scene):
                # Concatenate all geometries into single mesh.
                meshes = [
                    g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    console_logger.warning(
                        f"No trimesh geometries in {gltf_file.name}, skipping"
                    )
                    continue

            # Convert Y-up (GLTF) to Z-up (Drake) to match combined mesh.
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            )

            link_meshes[link_name] = mesh
            console_logger.debug(
                f"Loaded link mesh '{link_name}' with {len(mesh.vertices)} vertices"
            )
        except Exception as e:
            console_logger.warning(f"Failed to load link mesh {gltf_file.name}: {e}")

    if link_meshes:
        console_logger.debug(f"Loaded {len(link_meshes)} link meshes from {sdf_dir}")

    return link_meshes
