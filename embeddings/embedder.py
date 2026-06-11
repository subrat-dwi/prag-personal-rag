import logging
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel
from fastembed import SparseEmbedding, SparseTextEmbedding
from config.settings import settings

logger = logging.getLogger(__name__)

# Data models for embedded queries and documents
class EmbeddedDocs(BaseModel):
    dense_vectors: list[list[float]]
    sparse_vectors: list[dict[str, list]]

class EmbeddedQuery(BaseModel):
    dense_vectors: list[float]
    sparse_vectors: list[dict[str, list]]


# module-level singleton
_embedder = None
_sparse_embedder = None

# ----------- embedding functions --------------
def get_embedder() -> tuple[OllamaEmbeddings, SparseTextEmbedding]:
    """
    Returns a shared embedder instance.
    Singleton pattern avoids reloading the model on every call.
    """
    global _embedder
    global _sparse_embedder

    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=settings.embed_model,
            # base_url=settings.ollama_base_url,
        )
        logger.info("Embedder initialized with model '%s'", settings.embed_model)
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(
            model_name=settings.sparse_embed_model,
        )
        logger.info("Sparse Embedder initialized with model '%s'", settings.sparse_embed_model)
    
    return _embedder, _sparse_embedder


def embed_text(text: str) -> EmbeddedQuery:
    """
    Embed a single string into a vector.
    Used during ingestion for chunks.

    Args:
        text: Any non-empty string.

    Returns:
        Vector as list of floats.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If embedding fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    try:
        dense_embedder, sparse_embedder = get_embedder()
        dense_vectors = dense_embedder.embed_query(text)
        sparse_vectors = list(sparse_embedder.query_embed(text))
        return EmbeddedQuery(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_as_dict(sparse_vectors)
        )
    
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise RuntimeError(f"Embedding failed: {e}") from e


def embed_batch(texts: list[str]) -> EmbeddedDocs:
    """
    Embed multiple strings in one call — much faster than looping embed_text.
    Used during ingestion to embed all chunks from a document at once.

    Args:
        texts: List of non-empty strings.

    Returns:
        List of vectors in same order as input.

    Raises:
        ValueError: If list is empty or contains empty strings.
        RuntimeError: If embedding fails.
    """
    if not texts:
        raise ValueError("Cannot embed empty list.")

    empty = [i for i, t in enumerate(texts) if not t.strip()]
    if empty:
        raise ValueError(f"Empty strings at indices: {empty}")

    try:
        dense_embedder, sparse_embedder = get_embedder()
        dense_vectors = dense_embedder.embed_documents(texts)
        sparse_vectors = list(sparse_embedder.embed(texts))
        logger.info("Embedded batch of %d texts", len(texts))
        return EmbeddedDocs(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_as_dict(sparse_vectors)
        )
    except Exception as e:
        logger.error("Batch embedding failed: %s", e)
        raise RuntimeError(f"Batch embedding failed: {e}") from e

# ----------- utility functions --------------
def sparse_as_dict(sparse_vectors: list[SparseEmbedding]):
    """
    Convert list of SparseEmbedding objects to list of dicts with 'indices' and 'values' keys.
    This format is easier to serialize and store in Qdrant.
    """
    processed_sparse = []
    for vector in sparse_vectors:
        processed_sparse.append({
            "indices": [int(i) for i in vector.indices.tolist()],
            "values": [float(v) for v in vector.values.tolist()],
        })

    return processed_sparse
