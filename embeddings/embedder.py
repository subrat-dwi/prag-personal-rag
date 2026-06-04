import logging
from langchain_ollama import OllamaEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

# module-level singleton
_embedder = None


def get_embedder() -> OllamaEmbeddings:
    """
    Returns a shared embedder instance.
    Singleton pattern avoids reloading the model on every call.
    """
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=settings.embed_model,
            # base_url=settings.ollama_base_url,
        )
        logger.info("Embedder initialized with model '%s'", settings.embed_model)
    return _embedder


def embed_text(text: str) -> list[float]:
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
        vector = get_embedder().embed_query(text)
        return vector
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise RuntimeError(f"Embedding failed: {e}") from e


def embed_batch(texts: list[str]) -> list[list[float]]:
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
        vectors = get_embedder().embed_documents(texts)
        logger.info("Embedded batch of %d texts", len(texts))
        return vectors
    except Exception as e:
        logger.error("Batch embedding failed: %s", e)
        raise RuntimeError(f"Batch embedding failed: {e}") from e
