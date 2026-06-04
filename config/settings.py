from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
from functools import lru_cache


class Settings(BaseSettings):
    #── Logging ───────────────────────────────────────────────
    log_level: str = "WARNING"

    # ── LLM ───────────────────────────────────────────────────
    chat_model: str = "qwen2.5:3b-instruct"
    model_provider: str = "ollama"          # "ollama" or "groq"
    groq_api_key: str = ""                  # required when model_provider=groq

    # ── LangSmith (LangChain tracing) ─────────────────────────
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "personal-rag"

    # ── Embeddings ────────────────────────────────────────────
    embed_model: str = "nomic-embed-text"

    # ── Ollama ────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "personal-rag"
    qdrant_vector_size: int = 768

    # ── Chunking ──────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Google Drive ──────────────────────────────────────────
    google_credentials_path: str = "credentials.json"
    google_credentials_json: str = ""       # for deployed env (Render)
    drive_folder_id: str = ""

    # ── Cross-field validation ────────────────────────────────
    @model_validator(mode="after")
    def validate_provider_config(self) -> "Settings":
        if self.model_provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY must be set when MODEL_PROVIDER=groq"
            )
        if self.model_provider not in {"ollama", "groq"}:
            raise ValueError(
                f"MODEL_PROVIDER must be 'ollama' or 'groq', got '{self.model_provider}'"
            )
        return self

    @model_validator(mode="after")
    def validate_drive_config(self) -> "Settings":
        # warn if neither credentials source is set
        from pathlib import Path
        has_file = Path(self.google_credentials_path).exists()
        has_env = bool(self.google_credentials_json)
        if not has_file and not has_env:
            import logging
            logging.getLogger(__name__).warning(
                "No Google credentials found. "
                "Set GOOGLE_CREDENTIALS_PATH or GOOGLE_CREDENTIALS_JSON."
            )
        return self

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        # can't validate against chunk_size here easily in pydantic v2
        # so just enforce a reasonable max
        if v >= 200:
            raise ValueError("chunk_overlap should be less than 200")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False      # QDRANT_URL and qdrant_url both work


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    lru_cache ensures .env is only read once.
    """
    return Settings()


# module-level singleton for direct import
settings = get_settings()