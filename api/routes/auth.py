from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
from config.settings import settings

router = APIRouter()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@router.get("/auth")
def verify_auth(api_key: str = Security(api_key_header)):
    """
    Validates the API key.
    Frontend hits this on login — only proceeds to chat if 200 is returned.
    """
    if not settings.prag_api_key:
        # no key configured — allow (local dev)
        return {"status": "authenticated"}

    if not api_key or api_key != settings.prag_api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )

    return {"status": "authenticated"}