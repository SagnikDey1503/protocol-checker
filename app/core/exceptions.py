"""
Custom exception hierarchy for the Research Protocol Assistant.

All domain errors inherit from ProtocolAssistantError so they can be
caught uniformly in the FastAPI error-handler middleware.
"""

from __future__ import annotations


class ProtocolAssistantError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str = "An unexpected error occurred", code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


# ── Ingestion Errors ─────────────────────────────────────────

class PDFParsingError(ProtocolAssistantError):
    """Raised when a PDF cannot be parsed."""

    def __init__(self, message: str = "Failed to parse PDF document"):
        super().__init__(message, code="PDF_PARSING_ERROR")


class ChunkingError(ProtocolAssistantError):
    """Raised when document chunking fails."""

    def __init__(self, message: str = "Failed to chunk document"):
        super().__init__(message, code="CHUNKING_ERROR")


class EmbeddingError(ProtocolAssistantError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str = "Failed to generate embeddings"):
        super().__init__(message, code="EMBEDDING_ERROR")


# ── Retrieval Errors ─────────────────────────────────────────

class RetrievalError(ProtocolAssistantError):
    """Raised when RAG retrieval fails."""

    def __init__(self, message: str = "Failed to retrieve relevant context"):
        super().__init__(message, code="RETRIEVAL_ERROR")


class RerankingError(ProtocolAssistantError):
    """Raised when cross-encoder reranking fails."""

    def __init__(self, message: str = "Failed to rerank documents"):
        super().__init__(message, code="RERANKING_ERROR")


# ── Agent Errors ─────────────────────────────────────────────

class AgentRoutingError(ProtocolAssistantError):
    """Raised when agent orchestrator cannot route a query."""

    def __init__(self, message: str = "Failed to route query to appropriate agent"):
        super().__init__(message, code="AGENT_ROUTING_ERROR")


class AgentExecutionError(ProtocolAssistantError):
    """Raised when an agent node fails during execution."""

    def __init__(self, message: str = "Agent execution failed"):
        super().__init__(message, code="AGENT_EXECUTION_ERROR")


# ── Memory Errors ────────────────────────────────────────────

class MemoryError(ProtocolAssistantError):
    """Raised when memory storage or retrieval fails."""

    def __init__(self, message: str = "Memory operation failed"):
        super().__init__(message, code="MEMORY_ERROR")


# ── Protocol Errors ──────────────────────────────────────────

class ProtocolNotFoundError(ProtocolAssistantError):
    """Raised when a requested protocol does not exist."""

    def __init__(self, protocol_id: str):
        super().__init__(
            message=f"Protocol not found: {protocol_id}",
            code="PROTOCOL_NOT_FOUND",
        )


class ExperimentNotFoundError(ProtocolAssistantError):
    """Raised when a requested experiment session does not exist."""

    def __init__(self, experiment_id: str):
        super().__init__(
            message=f"Experiment session not found: {experiment_id}",
            code="EXPERIMENT_NOT_FOUND",
        )


# ── Auth Errors ──────────────────────────────────────────────

class AuthenticationError(ProtocolAssistantError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code="AUTHENTICATION_ERROR")


class AuthorizationError(ProtocolAssistantError):
    """Raised when a user lacks permission for an action."""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, code="AUTHORIZATION_ERROR")
