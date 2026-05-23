"""
Global error handler middleware.

Catches ProtocolAssistantError (and subclasses) as well as generic
exceptions, returning a structured JSON error body and logging full
tracebacks for server errors.
"""

from __future__ import annotations

import logging
import traceback

from starlette.responses import JSONResponse

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ExperimentNotFoundError,
    ProtocolAssistantError,
    ProtocolNotFoundError,
)

logger = logging.getLogger(__name__)

# Map exception codes to HTTP status codes
_CODE_TO_STATUS: dict[str, int] = {
    "AUTHENTICATION_ERROR": 401,
    "AUTHORIZATION_ERROR": 403,
    "PROTOCOL_NOT_FOUND": 404,
    "EXPERIMENT_NOT_FOUND": 404,
    "PDF_PARSING_ERROR": 422,
    "CHUNKING_ERROR": 422,
    "EMBEDDING_ERROR": 502,
    "RETRIEVAL_ERROR": 502,
    "RERANKING_ERROR": 502,
    "AGENT_ROUTING_ERROR": 500,
    "AGENT_EXECUTION_ERROR": 500,
    "MEMORY_ERROR": 500,
    "INTERNAL_ERROR": 500,
}


class ErrorHandlerMiddleware:
    """Catch all HTTP exceptions and return consistent JSON error responses."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        # Skip error handling for everything except HTTP requests (WebSockets are handled internally)
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        except ProtocolAssistantError as exc:
            status_code = _CODE_TO_STATUS.get(exc.code, 500)
            logger.error(
                "Application error [%s]: %s (path=%s)",
                exc.code,
                exc.message,
                scope.get("path", ""),
            )
            if status_code >= 500:
                logger.error(traceback.format_exc())

            response = JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "details": None,
                    }
                },
            )
            await response(scope, receive, send)
        except Exception as exc:
            logger.critical(
                "Unhandled exception on %s %s: %s",
                scope.get("method", ""),
                scope.get("path", ""),
                str(exc),
            )
            logger.critical(traceback.format_exc())

            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred. Please try again later.",
                        "details": None,
                    }
                },
            )
            await response(scope, receive, send)
