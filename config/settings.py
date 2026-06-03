from dotenv import load_dotenv
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    embed_model: str
    qdrant_collection: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    qdrant_api_key: str | None = None
    qdrant_url: str | None = None
    qdrant_vector_size: int = 768

    @classmethod
    def from_env(cls) -> "Settings":
        # Load .env for local development; safe to call multiple times.
        load_dotenv()
        return cls(
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "prag-personal-rag"),
            qdrant_vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "768")),
        )


def load_settings() -> Settings:
    return Settings.from_env()