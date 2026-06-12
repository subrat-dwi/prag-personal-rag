from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from config.settings import settings

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

