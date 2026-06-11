import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import query, health, files, auth
from ingestion.drive_sync import sync_drive
from config.settings import settings

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

# CORS middleware configuration — only allow origins specified in settings
def get_allowed_origins() -> list[str]:
    if settings.environment == "development":
        return ["http://localhost:5173", "http://localhost:3000"]  # vite + cra defaults

    origins = settings.allowed_origins  # comma-separated string from env
    if not origins:
        raise RuntimeError(
            "ALLOWED_ORIGINS must be set in production. "
            "Example: https://prag.vercel.app"
        )

    return [o.strip() for o in origins.split(",")]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["X-API-Key", "Content-Type"],  # only what you actually need
    allow_credentials=False,                       # you're using API key not cookies
    max_age=3600,                                  # cache preflight for 1 hour
)

app.include_router(health.router, tags=["System"])
app.include_router(auth.router, tags=["Auth"])
app.include_router(files.router, tags=["Files"])
app.include_router(query.router, tags=["RAG"])