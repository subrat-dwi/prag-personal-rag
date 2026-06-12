from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from vectorstore.qdrant_client import list_indexed_files_with_urls
from api.middleware import verify_api_key

router = APIRouter()

class FileEntry(BaseModel):
    filename: str
    file_url: str

class FilesResponse(BaseModel):
    total: int
    files: list[FileEntry]


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