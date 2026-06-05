# core/rag.py

import logging
from dataclasses import dataclass, field
from vectorstore.qdrant_client import query_chunks_by_text
from llm.chat_llm import call_llm

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.38


@dataclass
class RAGResponse:
    answer: str
    query: str
    sources: list[dict] = field(default_factory=list)
    # sources shape:
    # [{"filename": "resume.pdf", "relevance_score": 0.91, "file_url": "https://..."}]


def answer_query(query: str, top_k: int = 5) -> RAGResponse:
    """
    Full RAG pipeline. Single entry point for CLI, API, and any future interface.

    Steps:
        1. Retrieve top-k chunks from Qdrant
        2. Filter by confidence score
        3. Deduplicate sources
        4. Call LLM with query + context
        5. Return structured response

    Args:
        query: Natural language question.
        top_k: Number of chunks to retrieve from vector store.

    Returns:
        RAGResponse with answer, sources, and original query.
    """
    logger.info("Processing query: '%s'", query)

    # step 1 — retrieve
    chunks = query_chunks_by_text(query, top_k=top_k)
    logger.info("Retrieved %d chunks", len(chunks))

    # step 2 — filter low confidence
    chunks = [c for c in chunks if c["score"] >= SCORE_THRESHOLD]
    logger.info("%d chunks passed score threshold (%.2f)", len(chunks), SCORE_THRESHOLD)

    # step 3 — handle no results
    if not chunks:
        logger.warning("No chunks passed threshold for query: '%s'", query)
        return RAGResponse(
            answer="I couldn't find anything relevant in your documents.",
            sources=[],
            query=query,
        )

    # step 4 — call LLM
    response = call_llm(query, chunks)
    answer = response.answer
    used_chunk_indices = response.used_chunk_indices

    # step 5 — deduplicate sources (same file can appear in multiple chunks)
    seen = set()
    sources = []
    for i, chunk in enumerate(chunks, 1):
        if i in used_chunk_indices:
            filename = chunk["source_file"]
            if filename not in seen:
                seen.add(filename)
                sources.append({
                    "filename": filename,
                    "relevance_score": chunk["score"],
                    "file_url": chunk.get("file_url", ""),
                })

    logger.info(
        "Query complete — %d sources, answer length: %d chars",
        len(sources),
        len(answer),
    )

    return RAGResponse(answer=answer, sources=sources, query=query)