"""Blender rendering constants that are safe to import without bpy."""

# Lower light energy for articulated objects (more reflective materials).
ARTICULATED_LIGHT_ENERGY = 500

# Lower light energy for material/texture validation (avoid washing out colors).
MATERIAL_VALIDATION_LIGHT_ENERGY = 300
