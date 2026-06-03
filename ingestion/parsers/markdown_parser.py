import logging
import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown"}

#-----------------markdown parsing function------------------------
def parse_markdown(path: str) -> str:
    """
    Extract plain text from a Markdown file, stripping all syntax.

    Args:
        path: Path to the .md or .markdown file.

    Returns:
        Clean plain text string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If unsupported extension.
        RuntimeError: If no text could be extracted.
    """
    md_path = Path(path)

    if not md_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if md_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Not a markdown file: {path}")

    raw = md_path.read_text(encoding="utf-8")

    if not raw.strip():
        raise RuntimeError(f"Markdown file is empty: {path}")

    # convert markdown → html → strip tags → plain text
    html = markdown.markdown(raw)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # clean up excess whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise RuntimeError(f"No text could be extracted from '{path}'.")

    logger.info("Extracted %d characters from '%s'", len(text), md_path.name)
    return text