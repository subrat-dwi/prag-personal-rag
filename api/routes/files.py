from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from vectorstore.qdrant_client import list_indexed_files_with_urls
from config.settings import settings

router = APIRouter()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class FileEntry(BaseModel):
    filename: str
    file_url: str


class FilesResponse(BaseModel):
    total: int
    files: list[FileEntry]


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not settings.prag_api_key:
        return "no-auth"
    if not api_key or api_key != settings.prag_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return api_key


@router.get("/files", response_model=FilesResponse)
def get_indexed_files(_: str = Security(verify_api_key)):
    """
    Returns total count and list of all indexed files with Drive URLs.
    Frontend fetches this once on chat load and caches in memory.
    """
    try:
        files = list_indexed_files_with_urls()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch files: {e}"
        )

    return FilesResponse(
        total=len(files),
        files=[FileEntry(**f) for f in files],
    )