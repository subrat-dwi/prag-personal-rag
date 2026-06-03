# ingestion/pipeline.py

import logging
from pathlib import Path

from ingestion.parsers import parse_file
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_batch
from vectorstore.qdrant_client import upsert_chunks, delete_file_chunks, list_indexed_files

logger = logging.getLogger(__name__)


def ingest_file(path: str) -> None:
    """
    Full ingestion pipeline for a single file.
    parse → chunk → embed → store

    Args:
        path: Absolute or relative path to the file.

    Raises:
        ValueError: If file type is unsupported.
        RuntimeError: If any stage of the pipeline fails.
    """
    file_path = Path(path)
    filename = file_path.name

    logger.info("Starting ingestion for '%s'", filename)

    # step 1 — parse
    try:
        text = parse_file(str(file_path))
        logger.info("Parsed '%s' — %d characters", filename, len(text))
    except Exception as e:
        raise RuntimeError(f"Parsing failed for '{filename}': {e}") from e

    # step 2 — chunk
    try:
        chunks = chunk_text(text, str(file_path))
        logger.info("Chunked '%s' into %d chunks", filename, len(chunks))
    except Exception as e:
        raise RuntimeError(f"Chunking failed for '{filename}': {e}") from e

    # step 3 — embed
    try:
        texts = [chunk.text for chunk in chunks]
        vectors = embed_batch(texts)
        logger.info("Embedded %d chunks from '%s'", len(vectors), filename)
    except Exception as e:
        raise RuntimeError(f"Embedding failed for '{filename}': {e}") from e

    # step 4 — store
    try:
        upsert_chunks(chunks, vectors)
        logger.info("Stored %d chunks from '%s' in Qdrant", len(chunks), filename)
    except Exception as e:
        raise RuntimeError(f"Storing failed for '{filename}': {e}") from e

    logger.info("Ingestion complete for '%s'", filename)


def ingest_file_safe(path: str) -> tuple[bool, str]:
    """
    Wrapper around ingest_file that catches all exceptions.
    Used by the watcher and ingest_all script so one bad file
    doesn't stop the entire batch.

    Returns:
        (True, "") on success
        (False, error_message) on failure
    """
    try:
        ingest_file(path)
        return True, ""
    except Exception as e:
        logger.error("Failed to ingest '%s': %s", Path(path).name, e)
        return False, str(e)


def remove_file(filename: str) -> None:
    """
    Remove all chunks for a file from Qdrant.
    Called by the watcher when a file is deleted from the watch folder.

    Args:
        filename: Just the filename, not the full path.
                  Must match what was stored during ingestion.
    """
    logger.info("Removing chunks for '%s' from Qdrant", filename)
    try:
        delete_file_chunks(filename)
        logger.info("Removed chunks for '%s'", filename)
    except Exception as e:
        raise RuntimeError(f"Failed to remove chunks for '{filename}': {e}") from e


def reingest_file(path: str) -> None:
    """
    Delete existing chunks for a file then re-ingest it.
    Called by the watcher when a file is modified.

    Args:
        path: Full path to the modified file.
    """
    filename = Path(path).name
    logger.info("Re-ingesting modified file '%s'", filename)

    remove_file(filename)
    ingest_file(path)