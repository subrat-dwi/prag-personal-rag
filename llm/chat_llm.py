from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from llm.utils import extract_main_content
from vectorstore.qdrant_client import query_chunks_by_text
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# singleton for chat model instance
_llm = None

def get_llm() -> BaseChatModel:
    """Returns a singleton chat model instance, initializing it if it doesn't exist."""
    global _llm
    if _llm is None:
        _llm = init_chat_model(
            model=settings.chat_model,
            model_provider=settings.model_provider,
        )
    return _llm

#------------------ Main chat function ------------------
SYSTEM_PROMPT = """You are Prag, a personal assistant that answers questions strictly from provided document context.

- Answer only from the context. Be direct and concise.
- If found: answer confidently, end with [source: filename]
- If partially found: answer what's available, note what's missing, end with [source: filename]  
- If not found: respond only with "I couldn't find that in your documents."
- Never guess or infer. No filler."""

def format_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a string for LLM context. Each chunk includes text, source filename, and relevance score."""
    if not chunks:
        return "No relevant document chunks found."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(
            f"[Chunk {i} — {chunk['source_file']} (relevance: {chunk['score']})]"
            f"\n{chunk['text'].strip()}"
        )

    return "\n\n---\n\n".join(formatted)


def chat_with_llm(query: str) -> str:
    """Handles a user query by retrieving relevant document chunks and invoking the chat model with context."""
    chunks = query_chunks_by_text(query)

    # filter out low relevance chunks
    chunks = [c for c in chunks if c["score"] >= 0.38]

    context = format_context(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context from my documents:\n\n{context}\n\nQuestion: {query}"),
    ]

    response = get_llm().invoke(messages)
    return extract_main_content(response.content)