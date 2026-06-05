# api/routes/query.py

import logging
from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
from api.schemas.query import QueryRequest, QueryResponse, SourceReference
from core.rag import answer_query
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify the API key."""
    if not settings.prag_api_key:
        raise HTTPException(
                status_code=500,
                detail="Server misconfiguration: API key not set.",
            )
        return "no-auth"
    if api_key != settings.prag_api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key.",
        )
    return api_key


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    _: str = Security(verify_api_key),
):
    try:
        result = answer_query(request.query, top_k=request.top_k)
    except Exception as e:
        logger.error("RAG pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate answer.")

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceReference(**s) for s in result.sources
        ],
        query=result.query,
    )