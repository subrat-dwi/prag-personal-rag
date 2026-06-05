import hashlib
import logging
from pathlib import Path

from embeddings.embedder import embed_batch
from ingestion.chunker import chunk_text
from ingestion.parsers import parse_file
from vectorstore.qdrant_client import delete_file_chunks, upsert_chunks

logger = logging.getLogger(__name__)


def compute_file_hash(path: str) -> str:
    """
    Compute MD5 hash of a file's contents.
    Used when ingesting from local disk (not Drive).
    Drive provides md5Checksum directly so this isn't needed there.

    Args:
        path: Path to the file.

    Returns:
        MD5 hex string.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_file(path: str, file_hash: str = "", source_filename: str = "", file_url: str = "") -> None:
    """
    Full ingestion pipeline for a single file.
    parse → chunk → embed → store

    Args:
        path: Absolute or relative path to the file.
        file_hash: Hash of the file contents for deduplication.
                   If not provided, computed locally from disk.
                   Pass Drive's md5Checksum directly when syncing from Drive.
        source_filename: The original filename for tracking purposes.

    Raises:
        ValueError: If file type is unsupported.
        RuntimeError: If any stage of the pipeline fails.
    """
    file_path = Path(path)
    filename = file_path.name

    # compute hash if not provided (local ingestion)
    if not file_hash:
        file_hash = compute_file_hash(path)

    logger.info("Starting ingestion for '%s' (hash: %s)", filename, file_hash[:8])

    #-------------------- step 1 — parse --------------------
    try:
        text = parse_file(str(file_path))
        logger.info("Parsed '%s' — %d characters", filename, len(text))
    except Exception as e:
        raise RuntimeError(f"Parsing failed for '{filename}': {e}") from e

    #-------------------- step 2 — chunk ---------------------
    try:
        chunks = chunk_text(text, source_filename)
        logger.info("Chunked '%s' into %d chunks", filename, len(chunks))
    except Exception as e:
        raise RuntimeError(f"Chunking failed for '{filename}': {e}") from e

    #-------------------- step 3 — assign hash and file url to all chunks ---------------------
    for chunk in chunks:
        chunk.file_hash = file_hash
        chunk.file_url = file_url

    #-------------------- step 4 — embed ---------------------
    try:
        texts = [chunk.text for chunk in chunks]
        vectors = embed_batch(texts)
        logger.info("Embedded %d chunks from '%s'", len(vectors), filename)
    except Exception as e:
        raise RuntimeError(f"Embedding failed for '{filename}': {e}") from e

    #-------------------- step 5 — store ---------------------
    try:
        upsert_chunks(chunks, vectors)
        logger.info("Stored %d chunks from '%s' in Qdrant", len(chunks), filename)
    except Exception as e:
        raise RuntimeError(f"Storing failed for '{filename}': {e}") from e

    logger.info("Ingestion complete for '%s'", filename)


def ingest_file_safe(path: str, file_hash: str = "", source_filename: str = "", file_url: str = "") -> tuple[bool, str]:
    """
    Wrapper around ingest_file that catches all exceptions.
    One bad file won't stop the entire batch.

    Returns:
        (True, "") on success
        (False, error_message) on failure
    """
    try:
        ingest_file(path, file_hash=file_hash, source_filename=source_filename, file_url=file_url)
        return True, ""
    except Exception as e:
        logger.error("Failed to ingest '%s': %s", Path(path).name, e)
        return False, str(e)


def remove_file(filename: str) -> None:
    """
    Remove all chunks for a file from Qdrant.

    Args:
        filename: Just the filename, not the full path.
    """
    logger.info("Removing chunks for '%s' from Qdrant", filename)
    try:
        delete_file_chunks(filename)
        logger.info("Removed chunks for '%s'", filename)
    except Exception as e:
        raise RuntimeError(f"Failed to remove chunks for '{filename}': {e}") from e



def reingest_file(path: str, file_hash: str = "") -> None:
    """
    Delete existing chunks for a file then re-ingest it.
    Called when a file is modified.

    Args:
        path: Full path to the modified file.
        file_hash: Pass Drive's md5Checksum if syncing from Drive.
    """
    filename = Path(path).name
    logger.info("Re-ingesting modified file '%s'", filename)
    remove_file(filename)
    ingest_file(path, file_hash=file_hash)