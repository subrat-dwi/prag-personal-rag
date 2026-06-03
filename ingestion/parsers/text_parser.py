import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".text"}

#-----------------text parsing function------------------------
def parse_text(path: str) -> str:
    """
    Extract text from a plain text file.

    Args:
        path: Path to the .txt file.

    Returns:
        Cleaned text string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If unsupported extension.
        RuntimeError: If file is empty.
    """
    text_path = Path(path)

    if not text_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if text_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Not a text file: {path}")

    text = text_path.read_text(encoding="utf-8").strip()

    if not text:
        raise RuntimeError(f"Text file is empty: {path}")

    # normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    logger.info("Extracted %d characters from '%s'", len(text), text_path.name)
    return text