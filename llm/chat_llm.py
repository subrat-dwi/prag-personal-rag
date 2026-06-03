from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from llm.utils import extract_main_content
from retrieval.retriever import query_chunks
from config.settings import load_settings

settings = load_settings()

def get_llm():
    return init_chat_model(
        model= settings.chat_model,
        model_provider=settings.model_provider
    )

def chat_with_llm(query: str) -> str:
    chunks = query_chunks(query)
    system_prompt = SystemMessage(
        f"""You are Prag, a personal assistant who answers queries about myself based on the context I provide.
You will be given chunks of text extracted from my documents, which may include my resume, certificates, and other files.

Your job is to answer questions about me using only the information in those chunks.
- If the answer is not in the chunks, say "Sorry, I don't know."
- If the answer is partially in the chunks, use only that information and say "Based on the provided information, ..."
- Be concise and factual, avoid speculation.

CONTEXT CHUNKS:
{chunks}

QUESTION:
{query}
"""
    )

    llm = get_llm()
    response = llm.invoke([system_prompt])
    return extract_main_content(response)