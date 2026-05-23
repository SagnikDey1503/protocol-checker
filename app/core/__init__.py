"""Core package — LLM, embeddings, and exceptions."""

from app.core.embeddings import cosine_similarity, embed_query, embed_texts, get_embedding_model
from app.core.exceptions import (
    AgentExecutionError,
    AgentRoutingError,
    AuthenticationError,
    AuthorizationError,
    ChunkingError,
    EmbeddingError,
    ExperimentNotFoundError,
    PDFParsingError,
    ProtocolAssistantError,
    ProtocolNotFoundError,
    RetrievalError,
)
from app.core.llm import get_fast_llm, get_llm

__all__ = [
    "get_llm",
    "get_fast_llm",
    "get_embedding_model",
    "embed_texts",
    "embed_query",
    "cosine_similarity",
    "ProtocolAssistantError",
    "PDFParsingError",
    "ChunkingError",
    "EmbeddingError",
    "RetrievalError",
    "AgentRoutingError",
    "AgentExecutionError",
    "AuthenticationError",
    "AuthorizationError",
    "ProtocolNotFoundError",
    "ExperimentNotFoundError",
]
