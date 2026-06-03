from vectorstore.qdrant_client import get_client
from config.settings import load_settings
from embeddings.embedder import embed_text
from logging import Logger

logger = Logger(__name__)
settings = load_settings()
client = get_client()

def query_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Find the most similar chunks to a query vector.
    """
    query_vector = embed_text(query)

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