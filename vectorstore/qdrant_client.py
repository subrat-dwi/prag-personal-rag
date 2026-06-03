import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from config.settings import load_settings
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)
settings = load_settings()

# singleton client
_client = None


def get_client() -> QdrantClient:
    """
    Returns a shared Qdrant client instance.
    Connects to local or cloud Qdrant based on settings.
    """
    global _client
    if _client is None:
        if settings.qdrant_api_key:
            # cloud
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            # local
            _client = QdrantClient(url=settings.qdrant_url)

        logger.info("Qdrant client connected to '%s'", settings.qdrant_url)
        _ensure_collection(_client)

    return _client


def _ensure_collection(client: QdrantClient) -> None:
    """
    Creates the collection if it doesn't already exist.
    Safe to call on every startup.
    """
    existing = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=Distance.COSINE,  # cosine similarity for text embeddings
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.qdrant_collection)
    else:
        logger.info("Collection '%s' already exists", settings.qdrant_collection)


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """
    Store chunks and their vectors in Qdrant.
    Uses upsert — safe to call again on the same file, won't duplicate.

    Args:
        chunks: List of Chunk objects from chunker.
        vectors: Corresponding embedding vectors (same order).

    Raises:
        ValueError: If chunks and vectors lengths don't match.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunks ({len(chunks)}) and vectors ({len(vectors)}) must be same length."
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),     # unique ID for each chunk
            vector=vector,
            payload={                  # metadata stored alongside vector
                "text": chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    get_client().upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    logger.info(
        "Upserted %d chunks from '%s'",
        len(chunks),
        chunks[0].source_file if chunks else "unknown",
    )


def query_chunks(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the most similar chunks to a query vector.
    """
    client = get_client()

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    chunks = [
        {
            "text": r.payload["text"],
            "source_file": r.payload["source_file"],
            "score": round(r.score, 4),
        }
        for r in results.points if r.payload      # ← .points to get the list
    ]

    logger.info(
        "Query returned %d chunks, top score: %s",
        len(chunks),
        chunks[0]["score"] if chunks else "n/a",
    )

    return chunks


def delete_file_chunks(source_file: str) -> None:
    """
    Delete all chunks belonging to a specific file.
    Called by the watcher when a file is removed from the watch folder.

    Args:
        source_file: Filename (not full path) to delete chunks for.
    """
    get_client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file),
                )
            ]
        ),
    )

    logger.info("Deleted all chunks for '%s'", source_file)


def list_indexed_files() -> list[str]:
    """
    Returns a list of unique filenames currently indexed in Qdrant.
    Useful for debugging and the ingest_all script to avoid re-ingesting.
    """
    # scroll through all points and collect unique source_file values
    results, _ = get_client().scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        limit=1000,
    )

    files = list({r.payload["source_file"] for r in results if r.payload})
    return sorted(files)