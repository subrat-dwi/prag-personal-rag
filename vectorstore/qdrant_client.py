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
    FilterSelector,
    PayloadSchemaType
)
from config.settings import settings        # ← import singleton directly
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Singleton Qdrant client instance
_client = None

# --------------- Qdrant interaction functions ---------------
def get_client() -> QdrantClient:
    """
    Returns a singleton QdrantClient instance, creating it if it doesn't exist.
    """
    global _client
    if _client is None:
        if settings.qdrant_api_key:
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            _client = QdrantClient(url=settings.qdrant_url)

        logger.info("Qdrant client connected to '%s'", settings.qdrant_url)
        _ensure_collection(_client)

    return _client


def _ensure_collection(client: QdrantClient) -> None:
    """
    Ensures the Qdrant collection exists with the correct configuration.

    Args:
        client: An instance of QdrantClient to use for collection management.
    """
    existing = [c.name for c in client.get_collections().collections]

    # Only create collection if it doesn't exist — avoids errors and preserves existing data
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.qdrant_collection)
    else:
        logger.info("Collection '%s' already exists", settings.qdrant_collection)

    # create payload index on source_file — required by Qdrant Cloud for filtered queries
    # safe to call even if index already exists
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="source_file",
        field_schema=PayloadSchemaType.KEYWORD,     # keyword = exact string match
    )
    logger.info("Payload index ensured for 'source_file'")


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """
    Store chunks and their vectors in Qdrant.

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

    client = get_client()       # ← assign first, consistent with other functions

    # Create PointStructs including all relevant metadata in payload
    points = [
        PointStruct(
            id=uuid.uuid4(),    # ← UUID object as ID, Qdrant client will handle conversion to string
            vector=vector,
            payload={
                "text": chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "file_hash": chunk.file_hash,
                "file_url": chunk.file_url,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    # Upsert points into Qdrant, waiting for completion to ensure data is available for immediate queries
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,              # ← wait for write to complete before returning
    )

    logger.info(
        "Upserted %d chunks from '%s'",
        len(chunks),
        chunks[0].source_file if chunks else "unknown",
    )


def query_chunks(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the most similar chunks to a query vector.
    Args:
        query_vector: The embedding vector to query with.
        top_k: The number of top results to return.
    Returns:
        A list of dictionaries containing 'text', 'source_file', and 'score' for each matching chunk.
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
            "file_url": r.payload.get("file_url", ""),
            "score": round(r.score, 4),
        }
        for r in results.points if r.payload
    ]

    logger.info(
        "Query returned %d chunks, top score: %s",
        len(chunks),
        chunks[0]["score"] if chunks else "n/a",
    )

    return chunks


def query_chunks_by_text(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Convenience wrapper — embeds query text then retrieves matching chunks.
    Use this when you have raw text (e.g. from chat_llm).

    Args:
        query_text: Raw user query string.
        top_k: Number of chunks to return.

    Returns:
        List of dicts with 'text', 'source_file', 'file_url', 'score'.
    """
    from embeddings.embedder import embed_text   # ← import here to avoid circular dependency
    
    query_vector = embed_text(query_text)
    return query_chunks(query_vector, top_k=top_k)


def delete_file_chunks(source_file: str) -> None:
    """
    Delete all chunks belonging to a specific file.

    Args:
        source_file: Filename (not full path) to delete chunks for.
    """
    client = get_client()

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )
                ]
            )
        ),
    )

    logger.info("Deleted all chunks for '%s'", source_file)


def get_file_hash_from_store(filename: str) -> str | None:
    """
    Retrieves the file hash for a given filename from Qdrant, if it exists.
    Args:
        filename: The name of the file to look up (not full path).
    Returns:
        The file hash as a string if found, or None if not found.
    """
    client = get_client()

    response = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=filename),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    # scroll returns a tuple — (list[ScoredPoint], next_page_offset)
    points = response[0]      # explicitly index instead of unpacking

    logger.debug(
        "scroll for '%s' returned %d points", filename, len(points)
    )

    if not points:
        return None

    return points[0].payload.get("file_hash") if points[0].payload else None

def list_indexed_files() -> list[str]:
    """
    Returns sorted list of unique filenames currently indexed in Qdrant.
    """
    client = get_client()

    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        with_vectors=False,                 # ← no need to fetch vectors here
        limit=1000,
    )

    files = list({r.payload["source_file"] for r in results if r.payload})
    return sorted(files)