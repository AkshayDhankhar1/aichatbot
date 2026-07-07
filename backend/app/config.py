"""Central configuration, loaded from environment (.env) via pydantic-settings.

Nothing here is hardcoded with real secrets — every value comes from the
environment. See `.env.example` for the full list.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute paths so scripts work regardless of the current working directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
APP_DIR = Path(__file__).resolve().parent                     # backend/app/
DATA_DIR = APP_DIR / "data"                                   # backend/app/data/
DOCS_DIR = BACKEND_DIR.parent / "docs"                        # <repo>/docs/  (READ-ONLY)
BM25_PARAMS_PATH = DATA_DIR / "bm25_params.json"


class Settings(BaseSettings):
    # LLM (Groq) — secrets come from .env only, never hardcoded.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Pinecone — secret comes from .env only, never hardcoded.
    pinecone_api_key: str = ""
    pinecone_index_name: str = "chatbott"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embeddings (local). all-MiniLM-L6-v2 -> 384 dims (matches index dimension).
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Hybrid retrieval tuning
    hybrid_alpha: float = 0.5      # 1.0=dense only, 0.0=sparse only
    top_k: int = 8                 # chunks fetched from Pinecone
    top_n: int = 4                 # chunks passed to the LLM
    min_score_threshold: float = 0.15  # below this -> "not in documents"

    # LLM generation
    temperature: float = 0.15

    # CORS
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
