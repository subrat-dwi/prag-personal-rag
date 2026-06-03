# ingestion/parsers/__init__.py

from pathlib import Path
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.image_parser import parse_image
from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.markdown_parser import parse_markdown
from ingestion.parsers.text_parser import parse_text

PARSER_MAP = {
    ".pdf":      parse_pdf,
    ".jpg":      parse_image,
    ".jpeg":     parse_image,
    ".png":      parse_image,
    ".docx":     parse_docx,
    ".md":       parse_markdown,
    ".markdown": parse_markdown,
    ".txt":      parse_text,
    ".text":     parse_text,
}


def parse_file(path: str) -> str:
    """
    Route a file to the correct parser based on extension.

    Raises:
        ValueError: If file type is not supported.
    """
    ext = Path(path).suffix.lower()
    parser = PARSER_MAP.get(ext)

    if not parser:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {list(PARSER_MAP.keys())}"
        )

    return parser(path)