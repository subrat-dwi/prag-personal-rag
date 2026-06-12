import logging
from fastapi import APIRouter, HTTPException, Security
from api.schemas.query import QueryRequest, QueryResponse, SourceReference
from core.rag import answer_query
from api.middleware import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    _: str = Security(verify_api_key),
):
    try:
        result = answer_query(request.query, top_k=request.top_k)
        logger.info("Response: '%s'", result.answer)
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