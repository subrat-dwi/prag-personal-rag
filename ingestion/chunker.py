import logging
from pathlib import Path
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

logger = logging.getLogger(__name__)

# This module focuses solely on chunking text and creating Chunk objects.
@dataclass
class Chunk:
    """
    A single text chunk with metadata.
    Metadata travels with the chunk into the vector store
    so you know where a retrieved chunk came from.
    """
    text: str
    source_file: str      # original filename
    chunk_index: int      # position in document
    total_chunks: int     # total chunks from this document
    file_hash: str        # hash of entire file for deduplication
    file_url: str         # drive url of the file


def chunk_text(text: str, source_file: str) -> list[Chunk]:
    """
    Split extracted text into overlapping chunks for embedding.

    Args:
        text: Cleaned text from any parser.
        source_file: Original filename (for metadata).

    Returns:
        List of Chunk objects.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError(f"Cannot chunk empty text from '{source_file}'")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
        # tries to split on double newline first, then single newline,
        # then sentence, then word — falls back to character only if needed
    )

    # Split text into raw chunks (strings) using the splitter.
    raw_chunks = splitter.split_text(text)

    # Create Chunk objects with metadata for each raw chunk.
    chunks = [
        Chunk(
            text=chunk,
            source_file=Path(source_file).name,
            chunk_index=i,
            total_chunks=len(raw_chunks),
            file_hash="",
            file_url="",
        )
        for i, chunk in enumerate(raw_chunks)
    ]

    logger.info(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d)",
        Path(source_file).name,
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )

    return chunks