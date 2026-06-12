import logging
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator
from config.settings import settings
from typing import cast
import json

logger = logging.getLogger(__name__)

_llm = None


class LLMResponse(BaseModel):
    answer: str
    used_chunk_indices: list[int] = Field(
        default=[],
        description="List of integer chunk numbers. Must be a JSON array of integers like [1, 3, 5]. Never a string."
    )

    @field_validator("used_chunk_indices", mode="before")
    @classmethod
    def parse_indices(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            # handle "1,3,5" without brackets
            try:
                return [int(x.strip()) for x in v.strip("'{[]}'\"").split(",") if x.strip()]
            except Exception:
                return []
        if isinstance(v, int):
            return [v]      # model returned single int instead of list
        return []


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

FACTUAL_PROMPT = """You are Prag, a personal assistant that extracts precise information from document context.

Populate answer with one or two sentences maximum — state the exact fact as it appears in the chunks.
Populate used_chunk_indices with indices of chunks that directly contained the answer.
If the answer is not present in any chunk, set answer to exactly: "I couldn't find that in your documents." and used_chunk_indices to empty list.
Ensure that used_chunk_indices is a JSON array of integers like [1, 3, 5], never a string or single integer."""

SYNTHESIS_PROMPT = """You are Prag, a personal assistant that generates rich, well-structured content about a person using their document context.

Populate answer with comprehensive, impressive content — do not be brief. Use all relevant information across all chunks. Write in first person. Use short paragraphs or bullets where appropriate. Do not add filler phrases like "Based on the context..." — write the content directly. Only use information present in the chunks.
Populate used_chunk_indices with indices of ALL chunks you drew information from.
Ensure that used_chunk_indices is a JSON array of integers like [1, 3, 5], never a string or single integer. If no relevant information is found in any chunk, set answer to exactly: "I couldn't find that in your documents." and used_chunk_indices to empty list.
"""


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


def call_llm(query: str, chunks: list[dict], query_type: str) -> LLMResponse:
    """
    Builds prompt, calls LLM with structured output.
    Returns LLMResponse with answer and used chunk indices.
    """
    context = format_context(chunks)

    messages = [
        SystemMessage(content=SYNTHESIS_PROMPT if query_type == "synthesis" else FACTUAL_PROMPT),
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