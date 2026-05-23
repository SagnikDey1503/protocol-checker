"""
Core LLM and embedding model initialization.

Provides singleton instances of ChatGoogleGenerativeAI and SentenceTransformer
configured from application settings.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import threading
import time
from collections import deque
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_groq import ChatGroq
except ImportError:
    class ChatGroq:  # type: ignore[no-redef]
        pass
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


_RETRY_DELAY_RE = re.compile(r"(?:retry(?:ing)?|try\s+again)\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", re.IGNORECASE)


def _extract_retry_delay_seconds(exc: BaseException) -> float | None:
    """Best-effort extraction of server-suggested retry delay from exceptions."""

    message = str(exc)
    match = _RETRY_DELAY_RE.search(message)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    # google.rpc.RetryInfo often shows up as "retryDelay': '5s'" in exception repr
    if "retryDelay" in message and "s" in message:
        seconds_match = re.search(r"retryDelay[^0-9]*([0-9]+(?:\.[0-9]+)?)s", message)
        if seconds_match:
            try:
                return float(seconds_match.group(1))
            except ValueError:
                return None

    return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in message
        or "Quota exceeded" in message
        or "rate limit" in message.lower()
        or "429" in message
    )


class _SlidingWindowRateLimiter:
    """Simple sliding-window limiter for both sync and async call sites."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0) -> None:
        self._max_calls = max(1, int(max_calls))
        self._period = float(period_seconds)
        self._timestamps: deque[float] = deque()

        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self._period
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def _compute_wait(self, now: float) -> float:
        self._prune(now)
        if len(self._timestamps) < self._max_calls:
            return 0.0
        oldest = self._timestamps[0]
        return max(0.0, (oldest + self._period) - now)

    def _record(self, now: float) -> None:
        self._timestamps.append(now)

    async def acquire(self) -> None:
        async with self._async_lock:
            while True:
                now = time.monotonic()
                wait_s = self._compute_wait(now)
                if wait_s <= 0:
                    self._record(now)
                    return
                # jitter reduces stampedes when multiple coroutines wake at once
                jitter = random.uniform(0.0, min(0.25, wait_s * 0.1))
                await asyncio.sleep(wait_s + jitter)

    def acquire_sync(self) -> None:
        with self._sync_lock:
            while True:
                now = time.monotonic()
                wait_s = self._compute_wait(now)
                if wait_s <= 0:
                    self._record(now)
                    return
                jitter = random.uniform(0.0, min(0.25, wait_s * 0.1))
                time.sleep(wait_s + jitter)


class _GeminiThrottle:
    """Shared limiter/semaphore for a given model name."""

    def __init__(self, requests_per_minute: int, max_concurrency: int) -> None:
        self.limiter = _SlidingWindowRateLimiter(requests_per_minute, 60.0)
        self.async_semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.sync_semaphore = threading.BoundedSemaphore(max(1, int(max_concurrency)))


_THROTTLES: dict[str, _GeminiThrottle] = {}


def _get_throttle(model: str, rpm: int, max_concurrency: int) -> _GeminiThrottle:
    key = f"{model}:{rpm}:{max_concurrency}"
    throttle = _THROTTLES.get(key)
    if throttle is None:
        throttle = _GeminiThrottle(rpm, max_concurrency)
        _THROTTLES[key] = throttle
    return throttle


class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with client-side rate limiting and 429 backoff."""

    def __init__(
        self,
        *args,
        requests_per_minute: int,
        max_concurrency: int,
        retry_max_attempts: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._requests_per_minute = int(requests_per_minute)
        self._max_concurrency = int(max_concurrency)
        self._retry_max_attempts = int(retry_max_attempts)
        self._throttle = _get_throttle(self.model, self._requests_per_minute, self._max_concurrency)

    async def ainvoke(self, *args, **kwargs):  # type: ignore[override]
        backoff_s = 1.0
        for attempt in range(1, self._retry_max_attempts + 1):
            await self._throttle.limiter.acquire()
            async with self._throttle.async_semaphore:
                try:
                    return await super().ainvoke(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not _is_rate_limit_error(exc) or attempt >= self._retry_max_attempts:
                        raise

                    retry_s = _extract_retry_delay_seconds(exc)
                    sleep_s = max(retry_s or 0.0, backoff_s)
                    sleep_s = min(60.0, sleep_s) + random.uniform(0.0, 0.5)
                    logger.warning(
                        "Gemini rate-limited (attempt %s/%s). Sleeping %.2fs before retry.",
                        attempt,
                        self._retry_max_attempts,
                        sleep_s,
                    )
                    await asyncio.sleep(sleep_s)
                    backoff_s = min(60.0, backoff_s * 2)

        raise RuntimeError("Gemini request failed after retries")

    def invoke(self, *args, **kwargs):  # type: ignore[override]
        backoff_s = 1.0
        for attempt in range(1, self._retry_max_attempts + 1):
            self._throttle.limiter.acquire_sync()
            with self._throttle.sync_semaphore:
                try:
                    return super().invoke(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not _is_rate_limit_error(exc) or attempt >= self._retry_max_attempts:
                        raise

                    retry_s = _extract_retry_delay_seconds(exc)
                    sleep_s = max(retry_s or 0.0, backoff_s)
                    sleep_s = min(60.0, sleep_s) + random.uniform(0.0, 0.5)
                    logger.warning(
                        "Gemini rate-limited (attempt %s/%s). Sleeping %.2fs before retry.",
                        attempt,
                        self._retry_max_attempts,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    backoff_s = min(60.0, backoff_s * 2)

        raise RuntimeError("Gemini request failed after retries")


class RateLimitedChatGroq(ChatGroq):
    """ChatGroq with client-side rate limiting and 429 backoff."""

    def __init__(
        self,
        *args,
        requests_per_minute: int = 20,
        max_concurrency: int = 1,
        retry_max_attempts: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._requests_per_minute = int(requests_per_minute)
        self._max_concurrency = int(max_concurrency)
        self._retry_max_attempts = int(retry_max_attempts)
        self._throttle = _get_throttle(self.model_name, self._requests_per_minute, self._max_concurrency)

    async def ainvoke(self, *args, **kwargs):  # type: ignore[override]
        backoff_s = 1.0
        for attempt in range(1, self._retry_max_attempts + 1):
            await self._throttle.limiter.acquire()
            async with self._throttle.async_semaphore:
                try:
                    return await super().ainvoke(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not _is_rate_limit_error(exc) or attempt >= self._retry_max_attempts:
                        raise

                    retry_s = _extract_retry_delay_seconds(exc)
                    sleep_s = max(retry_s or 0.0, backoff_s)
                    sleep_s = min(60.0, sleep_s) + random.uniform(0.0, 0.5)
                    logger.warning(
                        "Groq rate-limited (attempt %s/%s). Sleeping %.2fs before retry.",
                        attempt,
                        self._retry_max_attempts,
                        sleep_s,
                    )
                    await asyncio.sleep(sleep_s)
                    backoff_s = min(60.0, backoff_s * 2)

        raise RuntimeError("Groq request failed after retries")

    def invoke(self, *args, **kwargs):  # type: ignore[override]
        backoff_s = 1.0
        for attempt in range(1, self._retry_max_attempts + 1):
            self._throttle.limiter.acquire_sync()
            with self._throttle.sync_semaphore:
                try:
                    return super().invoke(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not _is_rate_limit_error(exc) or attempt >= self._retry_max_attempts:
                        raise

                    retry_s = _extract_retry_delay_seconds(exc)
                    sleep_s = max(retry_s or 0.0, backoff_s)
                    sleep_s = min(60.0, sleep_s) + random.uniform(0.0, 0.5)
                    logger.warning(
                        "Groq rate-limited (attempt %s/%s). Sleeping %.2fs before retry.",
                        attempt,
                        self._retry_max_attempts,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    backoff_s = min(60.0, backoff_s * 2)

        raise RuntimeError("Groq request failed after retries")


@lru_cache
def get_llm() -> BaseChatModel:
    """
    Return a cached ChatGoogleGenerativeAI or ChatGroq instance.

    Uses gemini-2.5-pro or llama-3.3-70b-versatile by default.
    """
    settings = get_settings()
    provider = (settings.llm_provider or "gemini").lower().strip()

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "LLM_PROVIDER=groq requires the 'langchain-groq' package. "
                "Install it to use Groq."
            ) from exc

        if not settings.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY. "
                "Please get a free key from console.groq.com and set it in your .env file."
            )

        logger.info("Initializing Groq LLM: %s", settings.groq_model)
        return RateLimitedChatGroq(
            model_name=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.3,
            requests_per_minute=settings.groq_requests_per_minute,
            max_concurrency=1,
            retry_max_attempts=5,
        )

    if provider != "gemini":
        raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")

    logger.info("Initializing Gemini LLM: %s", settings.gemini_model)
    return RateLimitedChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
        max_tokens=4096,
        streaming=True,
        requests_per_minute=settings.gemini_requests_per_minute,
        max_concurrency=settings.gemini_max_concurrency,
        retry_max_attempts=settings.gemini_retry_max_attempts,
    )


@lru_cache
def get_fast_llm() -> BaseChatModel:
    """
    Return a faster, cheaper LLM instance for classification & routing.

    Uses gemini-2.5-flash or llama-3.1-8b-instant.
    """
    settings = get_settings()
    provider = (settings.fast_llm_provider or "groq").lower().strip()

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "FAST_LLM_PROVIDER=groq requires the 'langchain-groq' package. "
                "Install it to use Groq."
            ) from exc

        if not settings.groq_api_key:
            raise ValueError(
                "FAST_LLM_PROVIDER=groq requires GROQ_API_KEY. "
                "Please get a free key from console.groq.com and set it in your .env file."
            )

        logger.info("Initializing Groq Fast LLM: %s", settings.groq_fast_model)
        return RateLimitedChatGroq(
            model_name=settings.groq_fast_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.1,
            requests_per_minute=settings.groq_requests_per_minute,
            max_concurrency=1,
            retry_max_attempts=5,
        )

    if provider != "gemini":
        raise ValueError(f"Unknown fast_llm_provider: {settings.fast_llm_provider!r}")

    logger.info("Initializing Gemini Fast LLM: %s", settings.gemini_fast_model)
    return RateLimitedChatGoogleGenerativeAI(
        model=settings.gemini_fast_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
        max_tokens=1024,
        requests_per_minute=settings.gemini_fast_requests_per_minute,
        max_concurrency=settings.gemini_max_concurrency,
        retry_max_attempts=settings.gemini_retry_max_attempts,
    )
