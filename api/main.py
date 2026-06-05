import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import query, health
from ingestion.drive_sync import sync_drive

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Startup: sync Drive before accepting any requests.
    This is the FastAPI-recommended way — replaces deprecated @app.on_event.
    """
    logger.info("Starting up — syncing Drive...")
    try:
        sync_drive()
        logger.info("Drive sync complete.")
    except Exception as e:
        logger.error("Drive sync failed on startup: %s", e)
        # don't crash — serve with existing index

    yield   # app runs here

    # shutdown logic goes here if needed
    logger.info("Shutting down.")


app = FastAPI(
    title="PRAG — Personal RAG API",
    description="Query your personal documents using natural language.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for now
# tighten this when you know your frontend/bot URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["System"])
app.include_router(query.router, tags=["RAG"])