import logging

from PIL import ImageFont

console_logger = logging.getLogger(__name__)


def load_annotation_font(
    image_width: int, base_font_size_divisor: float, min_font_size: int = 16
) -> ImageFont.ImageFont:
    """Load font for coordinate annotations with fallback logic."""
    base_font_size = int(image_width / base_font_size_divisor)
    font_size = max(min_font_size, base_font_size)
    console_logger.debug(f"Base font size: {base_font_size}, Font size: {font_size}")

    font_paths = [
        "arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()
