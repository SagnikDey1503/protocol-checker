"""
AI Research Protocol Assistant — Application Configuration

Uses pydantic-settings for type-safe environment variable management.
All configuration flows through this single Settings class.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────
    app_name: str = "Research Protocol Assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # ── API Keys ──────────────────────────────────────────────
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic Claude API key")
    google_api_key: Optional[str] = Field(None, description="Google AI Studio Gemini API key")
    pinecone_api_key: str = Field(..., description="Pinecone vector DB API key")
    tavily_api_key: Optional[str] = Field(None, description="Tavily web search key")

    # ── Database (PostgreSQL + asyncpg) ───────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://protocol_user:change_me_in_production@localhost:5432/protocol_assistant",
        description="Async PostgreSQL connection string",
    )

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Pinecone ──────────────────────────────────────────────
    pinecone_index_name: str = "research-protocol-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ── LLM ───────────────────────────────────────────────────
    llm_provider: str = Field(
        "gemini",
        description="LLM backend provider: 'gemini' (default) or 'groq' (high RPM free alternative)",
    )
    fast_llm_provider: str = Field(
        "groq",
        description="Fast LLM provider for metadata extraction/chunking: 'groq' (default) or 'gemini'",
    )

    claude_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-2.5-pro"
    gemini_fast_model: str = "gemini-2.5-flash"

    # Groq (high RPM free alternative)
    groq_api_key: Optional[str] = Field(None, description="Groq API key")
    groq_requests_per_minute: int = Field(
        30,
        description="Client-side request-per-minute limiter for the Groq provider",
    )
    groq_model: str = Field(
        "llama-3.3-70b-versatile",
        description="Groq model for main reasoning",
    )
    groq_fast_model: str = Field(
        "llama-3.1-8b-instant",
        description="Groq model for fast/routing/metadata tasks",
    )
    # Gemini API quotas (free tier can be as low as 5 RPM per model)
    gemini_requests_per_minute: int = Field(
        60,
        description="Client-side request-per-minute limiter for the primary Gemini model",
    )
    gemini_fast_requests_per_minute: int = Field(
        5,
        description="Client-side request-per-minute limiter for the fast Gemini model",
    )
    gemini_max_concurrency: int = Field(
        2,
        description="Max concurrent Gemini requests (prevents bursty quota exhaustion)",
    )
    gemini_retry_max_attempts: int = Field(
        6,
        description="Max attempts for retrying transient Gemini errors like 429 rate limits",
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── RAG ───────────────────────────────────────────────────
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # ── Memory ────────────────────────────────────────────────
    working_memory_ttl: int = 7200  # 2 hours
    experiment_state_ttl: int = 86400  # 24 hours
    conversation_buffer_size: int = 20

    # ── Auth / JWT ────────────────────────────────────────────
    jwt_secret_key: str = "change-this-to-a-random-secret-key-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440  # 24 hours

    # ── Derived helpers ───────────────────────────────────────
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton – reused across the entire application."""
    return Settings()
