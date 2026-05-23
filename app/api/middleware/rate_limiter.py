"""
Simple in-memory rate limiter middleware.

Tracks requests per client IP using a sliding-window counter backed by
a dict of timestamps.  Defaults to 60 requests per minute.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimiterMiddleware:
    """ASGI middleware that enforces per-IP request rate limits.

    Args:
        app: The ASGI application.
        max_requests: Maximum number of allowed requests per window.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # ip -> list of timestamps (epoch floats)
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, scope) -> str:
        """Extract client IP from ASGI scope, respecting common proxy headers."""
        headers = dict(scope.get("headers", []))
        # HTTP headers in ASGI are lowercased bytes
        forwarded = headers.get(b"x-forwarded-for")
        if forwarded:
            return forwarded.decode("utf-8").split(",")[0].strip()
        
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"

    def _cleanup_window(self, ip: str, now: float) -> None:
        """Remove timestamps that have fallen outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff
        ]

    async def __call__(self, scope, receive, send) -> None:
        """Check rate limit before forwarding the request."""
        # Only rate limit HTTP requests (skip WebSockets, lifecycles, etc.)
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Skip rate-limiting for health checks
        if path.rstrip("/").endswith("/health"):
            await self.app(scope, receive, send)
            return

        client_ip = self._get_client_ip(scope)
        now = time.time()

        self._cleanup_window(client_ip, now)

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded for IP %s (%d requests in %ds)",
                client_ip,
                len(self._requests[client_ip]),
                self.window_seconds,
            )
            retry_after = int(
                self.window_seconds
                - (now - self._requests[client_ip][0])
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": (
                            f"Too many requests. Maximum {self.max_requests} "
                            f"requests per {self.window_seconds}s. "
                            f"Retry after {max(retry_after, 1)}s."
                        ),
                        "details": None,
                    }
                },
                headers={"Retry-After": str(max(retry_after, 1))},
            )
            await response(scope, receive, send)
            return

        self._requests[client_ip].append(now)
        await self.app(scope, receive, send)
