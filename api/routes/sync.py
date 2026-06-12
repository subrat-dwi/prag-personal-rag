from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Security
import logging
from api.routes.files import verify_api_key
from ingestion.drive_sync import sync_drive
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

class SyncResponse(BaseModel):
    message: str

@router.post("/sync", response_model=SyncResponse)
async def sync_documents(
    _: str = Security(verify_api_key),
):
    try:
        await asyncio.to_thread(sync_drive)
        logger.info("Drive sync completed successfully.")
    except Exception as e:
        logger.error("Drive sync failed: %s", e)
        raise HTTPException(status_code=500, detail="Drive sync failed.")
    return SyncResponse(message="Sync completed successfully.")