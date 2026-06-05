import httpx
from fastapi import APIRouter
from api.schemas.query import HealthResponse
from vectorstore.qdrant_client import get_client
from config.settings import settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check system health",
)
async def health_check():
    """
    Returns status of API, Qdrant connection, and Ollama availability.
    Use this to verify all components are running before querying.
    """
    qdrant_status = "ok"
    ollama_status = "ok"

    # check Qdrant
    try:
        get_client().get_collections()
    except Exception as e:
        qdrant_status = f"error: {e}"

    # check Ollama (only relevant locally)
    if settings.model_provider == "ollama":
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.ollama_base_url}/api/tags",
                    timeout=3.0,
                )
            if response.status_code != 200:
                ollama_status = f"error: status {response.status_code}"
        except Exception as e:
            ollama_status = f"error: {e}"
    else:
        ollama_status = "not used (cloud provider)"

    overall = "ok" if qdrant_status == "ok" and (settings.model_provider != "ollama" or ollama_status == "ok") else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        ollama=ollama_status,
    )