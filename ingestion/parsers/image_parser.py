import logging
from pathlib import Path

import pytesseract
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ----------------image parsing function------------------------
def parse_image(path: str) -> str:
    """
    Extract text from an image file using OCR.

    Args:
        path: Path to the image file.

    Returns:
        Extracted text as a string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file type is unsupported or unreadable.
        RuntimeError: If OCR returns no text.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: '{image_path.suffix}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    try:
        image = Image.open(image_path)
    except UnidentifiedImageError as e:
        raise ValueError(f"Could not open image (possibly corrupt): {path}") from e

    image = _preprocess(image)

    text = pytesseract.image_to_string(image, lang="eng").strip()

    if not text:
        raise RuntimeError(
            f"OCR returned no text for '{path}'. "
            "Image may be too low resolution or mostly non-text."
        )

    logger.info("OCR extracted %d characters from '%s'", len(text), image_path.name)
    return text

# -----------------helper functions------------------------
def _preprocess(image: Image.Image) -> Image.Image:
    """
    Basic preprocessing to improve OCR accuracy.
    Converts to grayscale — sufficient for most document scans.
    """
    return image.convert("L")  # L = grayscale
