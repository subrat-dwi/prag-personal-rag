import io
import logging
import tempfile
from pathlib import Path
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config.settings import settings
from ingestion.pipeline import ingest_file_safe, remove_file
from vectorstore.qdrant_client import get_file_hash_from_store, list_indexed_files

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

SUPPORTED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png",
    ".docx", ".md", ".txt"
}

# ── Google Drive Sync Logic ──────────────────────────────────────
def get_drive_service():
    """
    Authenticate and return a Drive API service instance.
    Uses credentials file locally, JSON env var when deployed.
    """
    creds_path = Path(settings.google_credentials_path)

    if creds_path.exists():
        # local — use credentials.json file
        creds = service_account.Credentials.from_service_account_file(
            str(creds_path),
            scopes=SCOPES,
        )
    elif settings.google_credentials_json:
        # deployed — parse JSON string from env var
        creds_info = json.loads(settings.google_credentials_json)
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=SCOPES,
        )
    else:
        raise RuntimeError(
            "No Google credentials found. "
            "Set GOOGLE_CREDENTIALS_PATH (local) or GOOGLE_CREDENTIALS_JSON (deployed)."
        )

    return build("drive", "v3", credentials=creds)

def list_drive_files(service) -> list[dict]:
    """
    List all supported files in the configured Drive folder.
    Returns list of dicts with id, name, modifiedTime, md5Checksum.
    """
    results = service.files().list(
        q=f"'{settings.drive_folder_id}' in parents and trashed=false",
        fields="files(id, name, modifiedTime, md5Checksum)",
    ).execute()

    files = results.get("files", [])

    # filter to supported extensions only
    return [
        f for f in files
        if Path(f["name"]).suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def download_file(service, file_id: str, filename: str) -> str:
    """
    Download a Drive file to /tmp.
    Returns the temp path.
    """
    request = service.files().get_media(fileId=file_id)
    tmp_path = Path(tempfile.mkstemp(suffix=Path(filename).suffix)[1])

    # download to temp file
    with io.FileIO(str(tmp_path), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return str(tmp_path)


def sync_drive() -> None:
    """
    Main sync function:
    1. List files in Drive folder.
    2. For each file:
        - Check if it's new, changed, or unchanged vs Qdrant.
        - Ingest new/changed files, remove deleted files.
    """
    logger.info("Starting Drive sync...")
    service = get_drive_service()

    # List files in Drive folder
    drive_files = list_drive_files(service)
    logger.info("Found %d supported files in Drive", len(drive_files))  # add this

    # Create a set of filenames currently in Drive for quick lookup
    drive_filenames = {f["name"] for f in drive_files}

    # ── handle deletions ──────────────────────────────────────
    indexed_filenames = set(list_indexed_files())   # what's in Qdrant
    removed = indexed_filenames - drive_filenames   # in Qdrant but not in Drive

    for filename in removed:
        logger.info("File removed from Drive, deleting chunks: '%s'", filename)
        remove_file(filename)

    for file in drive_files:
        filename = file["name"]
        drive_hash = file.get("md5Checksum")
        logger.info("Processing '%s' — drive_hash: %s", filename, drive_hash)  # add this

        try:
            stored_hash = get_file_hash_from_store(filename)        # get stored hash from Qdrant
            logger.info("Stored hash for '%s': %s", filename, stored_hash)
        except Exception as e:
            logger.error("get_file_hash_from_store failed for '%s': %s", filename, e)
            raise

        if stored_hash is None:
            logger.info("New file in Drive, ingesting: '%s'", filename)
            _download_and_ingest(service, file)                     # new file, ingest
        elif stored_hash != drive_hash:
            logger.info("File changed in Drive, re-ingesting: '%s'", filename)
            remove_file(filename)                                   # remove old chunks first
            _download_and_ingest(service, file)                     # changed file, re-ingest
        else:
            logger.info("Unchanged, skipping: '%s'", filename)

    logger.info("Drive sync complete.")

def _download_and_ingest(service, file: dict) -> None:
    """Download to /tmp, ingest, delete temp file."""
    tmp_path = None
    try:
        tmp_path = download_file(service, file["id"], file["name"])

        # construct Drive view URL for viewing through Citation
        file_url = f"https://drive.google.com/file/d/{file['id']}/view"

        success, error = ingest_file_safe(
            tmp_path,
            file_hash=file.get("md5Checksum", ""),
            source_filename=file["name"],
            file_url = file_url)
        if not success:
            logger.error("Ingestion failed for '%s': %s", file["name"], error)
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()                                 # always clean up
            logger.debug("Deleted temp file '%s'", tmp_path)