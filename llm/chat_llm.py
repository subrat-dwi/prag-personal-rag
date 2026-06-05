# llm/chat_llm.py

import logging
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from config.settings import settings
from typing import cast

logger = logging.getLogger(__name__)

_llm = None


class LLMResponse(BaseModel):
    answer: str
    used_chunk_indices: list[int]


def get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        _llm = init_chat_model(
            model=settings.chat_model,
            model_provider=settings.model_provider,
            api_key=settings.groq_api_key if settings.model_provider == "groq" else None,
        )
    return _llm


SYSTEM_PROMPT = """You are Prag, a personal assistant that answers questions from document context.
- Answer only from the context. Be direct and concise.
- In used_chunk_indices, list ONLY the chunk numbers [1,2,3...] that directly contributed to your answer.
- If not found: answer with "I couldn't find that in your documents." and set used_chunk_indices to empty list.
- Never guess or infer."""


def format_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a string for LLM input. Each chunk includes its source filename and relevance score."""
    if not chunks:
        return "No relevant document chunks found."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(
            f"[Chunk {i} — {chunk['source_file']} (relevance: {chunk['score']})]"
            f"\n{chunk['text'].strip()}"
        )
    return "\n\n---\n\n".join(formatted)


def call_llm(query: str, chunks: list[dict]) -> LLMResponse:
    """
    Builds prompt, calls LLM with structured output.
    Returns LLMResponse with answer and used chunk indices.
    """
    context = format_context(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context from my documents:\n\n{context}\n\nQuestion: {query}"
        ),
    ]

    structured_llm = get_llm().with_structured_output(LLMResponse)
    response: LLMResponse = cast(LLMResponse, structured_llm.invoke(messages))

    logger.info(
        "LLM used chunk indices: %s for query: '%s'",
        response.used_chunk_indices,
        query,
    )

    return response