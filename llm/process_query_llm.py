from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from typing import cast
from llm.chat_llm import get_llm

class ProcessedQuery(BaseModel):
    improved_query: str
    query_type: str = "factual"

#--------------------------------------------

QUERY_PROCESSOR_PROMPT = """You are a query processor for a personal document assistant.

Fix spelling mistakes, clarify vague language, and rewrite the query to maximise semantic search accuracy against personal documents like resumes, certificates, and transcripts.

Classify as:
- factual: wants a specific piece of information — name, number, date, email, CGPA, phone, address, skill, technology
- synthesis: wants something generated or reasoned across multiple facts — intro, bio, summary, cover letter, strengths, interview prep, achievements overview

Populate improved_query with the rewritten query.
Populate query_type with either "factual" or "synthesis"."""


def process_query(raw_query: str) -> ProcessedQuery:
    """
    Goal:
        Improves the raw user query
        Classifies the query into  Factual or Synthesis
    Params:
        raw_query: The original user query as inputted by the user
    Returns:        
        ProcessedQuery object containing:
            improved_query: The rewritten query optimized for retrieval
            query_type: "factual" or "synthesis" classification of the query
    """
    messages = [
        SystemMessage(QUERY_PROCESSOR_PROMPT),
        HumanMessage(content= f"Raw Query: {raw_query}")
    ]

    structured_llm = get_llm().with_structured_output(ProcessedQuery)
    response: ProcessedQuery = cast(ProcessedQuery, structured_llm.invoke(messages))

    return response