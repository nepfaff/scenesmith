import tempfile

from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from scenesmith.agent_utils.blender.annotations import add_wall_labels_to_top_view


def test_wall_labels_top_view_uses_annotation_font_width_keyword():
    """Wall top-view labels should render without font keyword regressions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "wall_top.png"
        Image.new("RGB", (128, 128), "white").save(image_path)

        wall_surfaces = [
            {
                "surface_id": "bedroom_north",
                "direction": "north",
                "length": 4.0,
                "height": 2.5,
                "transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }
        ]

        with patch(
            "scenesmith.agent_utils.blender.annotations.get_pixel_coordinates",
            return_value=(64, 64),
        ):
            add_wall_labels_to_top_view(
                image_path=image_path,
                camera_obj=Mock(),
                wall_surfaces=wall_surfaces,
            )

        assert image_path.exists()
