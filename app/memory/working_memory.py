"""
Redis-backed working (short-term) memory.

Stores ephemeral session data that expires automatically:
- Conversation message buffers (Redis lists)
- Experiment state snapshots (Redis hashes)
- Recently-retrieved chunk caches (Redis sorted sets)
- Active session pointers per user (simple key-value)

All keys use the pattern ``session:{session_id}:<suffix>`` so they can be
cleaned up atomically when a session ends.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings
from app.core.exceptions import MemoryError

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Redis-backed short-term memory for active sessions."""

    # ── Key templates ────────────────────────────────────────
    _MSG_KEY = "session:{session_id}:messages"
    _STATE_KEY = "session:{session_id}:experiment_state"
    _CHUNKS_KEY = "session:{session_id}:cached_chunks"
    _ACTIVE_KEY = "user:{user_id}:active_session"

    def __init__(self, redis_client: aioredis.Redis) -> None:
        """
        Initialise with an async Redis client.

        Args:
            redis_client: An ``aioredis.Redis`` instance (connection-pooled).
        """
        self._redis = redis_client
        settings = get_settings()
        self._msg_ttl: int = settings.working_memory_ttl  # 2 h
        self._state_ttl: int = settings.experiment_state_ttl  # 24 h

    # ── Helpers ──────────────────────────────────────────────

    def _msg_key(self, session_id: str) -> str:
        return self._MSG_KEY.format(session_id=session_id)

    def _state_key(self, session_id: str) -> str:
        return self._STATE_KEY.format(session_id=session_id)

    def _chunks_key(self, session_id: str) -> str:
        return self._CHUNKS_KEY.format(session_id=session_id)

    def _active_key(self, user_id: str) -> str:
        return self._ACTIVE_KEY.format(user_id=user_id)

    # ── Conversation buffer ──────────────────────────────────

    async def store_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """
        Append a message to the session's conversation buffer.

        Each message is stored as a JSON blob in a Redis list.
        The list TTL is refreshed on every write so idle sessions
        expire after ``working_memory_ttl`` seconds.

        Args:
            session_id: Unique session identifier.
            role: Message role (``user``, ``assistant``, ``system``).
            content: Raw message text.
        """
        key = self._msg_key(session_id)
        message = json.dumps(
            {
                "role": role,
                "content": content,
                "timestamp": time.time(),
            }
        )
        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.rpush(key, message)
            pipe.expire(key, self._msg_ttl)
            await pipe.execute()
            logger.debug(
                "Stored message [%s] for session %s", role, session_id
            )
        except Exception as exc:
            logger.error(
                "Failed to store message for session %s: %s",
                session_id,
                exc,
            )
            raise MemoryError(
                f"Failed to store message for session {session_id}"
            ) from exc

    async def get_messages(
        self, session_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Return the most recent messages from the conversation buffer.

        Args:
            session_id: Unique session identifier.
            limit: Maximum number of messages to return (most recent first in
                   chronological order).

        Returns:
            List of dicts ``{"role": …, "content": …, "timestamp": …}``.
        """
        key = self._msg_key(session_id)
        try:
            # Fetch the last *limit* items from the list
            raw_messages = await self._redis.lrange(key, -limit, -1)
            return [json.loads(m) for m in raw_messages]
        except Exception as exc:
            logger.error(
                "Failed to get messages for session %s: %s",
                session_id,
                exc,
            )
            raise MemoryError(
                f"Failed to retrieve messages for session {session_id}"
            ) from exc

    # ── Experiment state ─────────────────────────────────────

    async def update_experiment_state(
        self, session_id: str, state: dict[str, Any]
    ) -> None:
        """
        Persist the current experiment state snapshot.

        The entire ``state`` dict is serialised to JSON and stored as a
        single Redis key with a 24-hour TTL.

        Args:
            session_id: Unique session identifier.
            state: Arbitrary state dict (current step, deviations, etc.).
        """
        key = self._state_key(session_id)
        try:
            await self._redis.set(
                key,
                json.dumps(state),
                ex=self._state_ttl,
            )
            logger.debug(
                "Updated experiment state for session %s", session_id
            )
        except Exception as exc:
            logger.error(
                "Failed to update experiment state for session %s: %s",
                session_id,
                exc,
            )
            raise MemoryError(
                f"Failed to update experiment state for session {session_id}"
            ) from exc

    async def get_experiment_state(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """
        Return the latest experiment state snapshot, or ``None`` if expired
        or absent.

        Args:
            session_id: Unique session identifier.

        Returns:
            State dict or ``None``.
        """
        key = self._state_key(session_id)
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.error(
                "Failed to get experiment state for session %s: %s",
                session_id,
                exc,
            )
            raise MemoryError(
                f"Failed to retrieve experiment state for session {session_id}"
            ) from exc

    # ── Cached chunks ────────────────────────────────────────

    async def cache_chunks(
        self, session_id: str, chunks: list[dict[str, Any]]
    ) -> None:
        """
        Cache recently-retrieved chunks as a sorted set scored by relevance.

        Each chunk dict must contain at least ``"chunk_id"`` and ``"score"``
        keys.  The sorted set is overwritten on each call so callers always
        see the freshest retrieval results.

        Args:
            session_id: Unique session identifier.
            chunks: List of chunk dicts; each must have a ``score`` field.
        """
        key = self._chunks_key(session_id)
        try:
            pipe = self._redis.pipeline(transaction=True)
            # Clear existing cached chunks before inserting new ones
            pipe.delete(key)
            for chunk in chunks:
                score = chunk.get("score", 0.0)
                pipe.zadd(key, {json.dumps(chunk): score})
            pipe.expire(key, self._msg_ttl)
            await pipe.execute()
            logger.debug(
                "Cached %d chunks for session %s", len(chunks), session_id
            )
        except Exception as exc:
            logger.error(
                "Failed to cache chunks for session %s: %s",
                session_id,
                exc,
            )
            raise MemoryError(
                f"Failed to cache chunks for session {session_id}"
            ) from exc

    async def get_cached_chunks(
        self, session_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Return the top-scored cached chunks for a session.

        Args:
            session_id: Unique session identifier.
            limit: Maximum number of chunks to return (highest score first).

        Returns:
            List of chunk dicts ordered by descending score.
        """
        key = self._chunks_key(session_id)
        try:
            # ZREVRANGE returns highest-score first
            raw_chunks = await self._redis.zrevrange(
                key, 0, limit - 1, withscores=False
            )
            return [json.loads(c) for c in raw_chunks]
        except Exception as exc:
            logger.error(
                "Failed to get cached chunks for session %s: %s",
                session_id,
                exc,
            )
            raise MemoryError(
                f"Failed to retrieve cached chunks for session {session_id}"
            ) from exc

    # ── Active session pointer ───────────────────────────────

    async def set_active_session(
        self, user_id: str, session_id: str
    ) -> None:
        """
        Set the active session pointer for a user.

        Args:
            user_id: User identifier.
            session_id: Session to mark as active.
        """
        key = self._active_key(user_id)
        try:
            await self._redis.set(key, session_id, ex=self._state_ttl)
            logger.debug(
                "Set active session for user %s → %s", user_id, session_id
            )
        except Exception as exc:
            logger.error(
                "Failed to set active session for user %s: %s",
                user_id,
                exc,
            )
            raise MemoryError(
                f"Failed to set active session for user {user_id}"
            ) from exc

    async def get_active_session(self, user_id: str) -> str | None:
        """
        Return the current active session id for a user, or ``None``.

        Args:
            user_id: User identifier.

        Returns:
            Session id string or ``None``.
        """
        key = self._active_key(user_id)
        try:
            return await self._redis.get(key)
        except Exception as exc:
            logger.error(
                "Failed to get active session for user %s: %s",
                user_id,
                exc,
            )
            raise MemoryError(
                f"Failed to get active session for user {user_id}"
            ) from exc

    # ── Cleanup ──────────────────────────────────────────────

    async def clear_session(self, session_id: str) -> None:
        """
        Delete **all** Redis keys associated with a session.

        This is called when a session ends or is explicitly cleaned up.

        Args:
            session_id: Unique session identifier.
        """
        keys = [
            self._msg_key(session_id),
            self._state_key(session_id),
            self._chunks_key(session_id),
        ]
        try:
            deleted = await self._redis.delete(*keys)
            logger.info(
                "Cleared %d keys for session %s", deleted, session_id
            )
        except Exception as exc:
            logger.error(
                "Failed to clear session %s: %s", session_id, exc
            )
            raise MemoryError(
                f"Failed to clear session {session_id}"
            ) from exc
