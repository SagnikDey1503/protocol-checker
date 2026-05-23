"""
Unified memory manager — façade that routes reads and writes across all
memory tiers (working, persistent, semantic, episodic).

Provides high-level operations that agents call without needing to know
which underlying store is involved:

- ``remember(...)`` — route a new memory to the right store.
- ``recall(...)`` — fan-out across memory types and merge results.
- ``get_full_context(...)`` — assemble a rich context dict for agent use.
- ``summarize_and_compress(...)`` — summarise old working-memory messages,
  persist them, and trim the buffer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import get_settings
from app.core.exceptions import MemoryError
from app.core.llm import get_fast_llm
from app.memory.episodic_memory import EpisodicMemoryStore
from app.memory.persistent_memory import PersistentMemory
from app.memory.semantic_memory import SemanticMemory
from app.memory.working_memory import WorkingMemory
from app.models.enums import MemoryType

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified façade across the four memory tiers."""

    def __init__(
        self,
        working: WorkingMemory,
        persistent: PersistentMemory,
        semantic: SemanticMemory,
        episodic: EpisodicMemoryStore,
    ) -> None:
        """
        Initialise the manager with all four memory backends.

        Args:
            working: Redis-backed short-term memory.
            persistent: PostgreSQL long-term memory.
            semantic: Pinecone vector semantic memory.
            episodic: Dual-storage episodic memory.
        """
        self._working = working
        self._persistent = persistent
        self._semantic = semantic
        self._episodic = episodic

    # ── Remember ─────────────────────────────────────────────

    async def remember(
        self,
        content: str,
        memory_type: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> None:
        """
        Route a memory to the appropriate store(s).

        Routing rules:
        - ``working``   → not applicable here (session-scoped, use
          ``WorkingMemory`` methods directly).
        - ``semantic``  → upserted into Pinecone.
        - ``episodic``  → written to both PostgreSQL and Pinecone.
        - ``persistent``→ (patterns/conversations are managed via dedicated
          methods on ``PersistentMemory``); semantic fallback.

        Args:
            content: Text content to store.
            memory_type: One of ``MemoryType`` enum values.
            user_id: Owning user UUID string.
            metadata: Extra metadata.
            importance: Importance score (0–1).
        """
        try:
            if memory_type == MemoryType.EPISODIC.value:
                episode_type = (metadata or {}).get("episode_type", "learning")
                await self._episodic.record_episode(
                    user_id=user_id,
                    episode_type=episode_type,
                    content=content,
                    metadata=metadata,
                    importance=importance,
                )
            elif memory_type == MemoryType.SEMANTIC.value:
                await self._semantic.store(
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata,
                    importance=importance,
                )
            elif memory_type == MemoryType.PERSISTENT.value:
                # Persistent-typed memories go to both semantic store
                # (for recall) and pattern tracking if applicable
                await self._semantic.store(
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata,
                    importance=importance,
                )
                # If it looks like a behavioural pattern, record it
                pattern_type = (metadata or {}).get("pattern_type")
                if pattern_type:
                    await self._persistent.update_user_pattern(
                        user_id=user_id,
                        pattern_type=pattern_type,
                        description=content,
                    )
            else:
                # Default fallback: semantic store
                await self._semantic.store(
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata,
                    importance=importance,
                )
            logger.info(
                "Remembered content (type=%s) for user %s",
                memory_type,
                user_id,
            )
        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to remember content: %s", exc)
            raise MemoryError("Failed to remember content") from exc

    # ── Recall ───────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        user_id: str,
        memory_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Fan-out search across specified memory types and merge results.

        Args:
            query: Natural-language search query.
            user_id: Owning user UUID string.
            memory_types: List of ``MemoryType`` values to search. Defaults
                          to ``["episodic", "semantic"]``.
            limit: Per-type result limit.

        Returns:
            Merged list of memory dicts sorted by descending score.
        """
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC.value, MemoryType.SEMANTIC.value]

        results: list[dict[str, Any]] = []

        try:
            tasks: list[asyncio.Task[Any]] = []

            if MemoryType.SEMANTIC.value in memory_types:
                tasks.append(
                    asyncio.create_task(
                        self._recall_semantic(query, user_id, limit)
                    )
                )

            if MemoryType.EPISODIC.value in memory_types:
                tasks.append(
                    asyncio.create_task(
                        self._recall_episodic(query, user_id, limit)
                    )
                )

            if MemoryType.PERSISTENT.value in memory_types:
                tasks.append(
                    asyncio.create_task(
                        self._recall_persistent(user_id, limit)
                    )
                )

            gathered = await asyncio.gather(*tasks, return_exceptions=True)

            for outcome in gathered:
                if isinstance(outcome, Exception):
                    logger.warning("Recall sub-task failed: %s", outcome)
                    continue
                results.extend(outcome)

            # Sort by similarity / importance score descending
            results.sort(
                key=lambda r: r.get("similarity_score", r.get("importance_score", 0.0)),
                reverse=True,
            )
            return results[:limit]

        except Exception as exc:
            logger.error("Failed to recall memories: %s", exc)
            raise MemoryError("Failed to recall memories") from exc

    async def _recall_semantic(
        self, query: str, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Recall from semantic memory."""
        entries = await self._semantic.recall(query, user_id, limit)
        return [
            {
                "id": e.id,
                "content": e.content,
                "memory_type": e.memory_type,
                "importance_score": e.importance_score,
                "similarity_score": e.similarity_score,
                "metadata": e.metadata,
                "created_at": e.created_at,
                "source": "semantic",
            }
            for e in entries
        ]

    async def _recall_episodic(
        self, query: str, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Recall from episodic memory."""
        episodes = await self._episodic.recall_relevant_episodes(
            user_id=user_id, query=query, limit=limit
        )
        for ep in episodes:
            ep["source"] = "episodic"
        return episodes

    async def _recall_persistent(
        self, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Recall from persistent memory (patterns & experiment history)."""
        patterns = await self._persistent.get_user_patterns(user_id)
        experiments = await self._persistent.get_user_experiments(
            user_id, limit=limit
        )
        results: list[dict[str, Any]] = []
        for p in patterns:
            results.append(
                {
                    "id": p["id"],
                    "content": p["description"],
                    "memory_type": "pattern",
                    "importance_score": min(p.get("frequency", 1) * 0.1, 1.0),
                    "similarity_score": 0.0,
                    "metadata": p,
                    "created_at": p.get("last_seen"),
                    "source": "persistent",
                }
            )
        for exp in experiments:
            results.append(
                {
                    "id": exp["id"],
                    "content": exp.get("title") or "Experiment",
                    "memory_type": "experiment",
                    "importance_score": 0.5,
                    "similarity_score": 0.0,
                    "metadata": exp,
                    "created_at": exp.get("started_at"),
                    "source": "persistent",
                }
            )
        return results

    # ── Full context assembly ────────────────────────────────

    async def get_full_context(
        self,
        session_id: str,
        user_id: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Assemble the richest possible context for an agent turn.

        Gathers:
        - Recent conversation messages (working memory)
        - Current experiment state (working memory)
        - Cached chunks (working memory)
        - Semantically recalled memories (semantic + episodic)
        - User profile summary (persistent)

        Args:
            session_id: Current session id.
            user_id: Owning user UUID string.
            query: The user's latest query (for recall).

        Returns:
            Dict with keys ``messages``, ``experiment_state``,
            ``cached_chunks``, ``memories``, ``user_profile``.
        """
        try:
            settings = get_settings()

            # Fire independent queries concurrently
            msgs_task = asyncio.create_task(
                self._working.get_messages(
                    session_id, limit=settings.conversation_buffer_size
                )
            )
            state_task = asyncio.create_task(
                self._working.get_experiment_state(session_id)
            )
            chunks_task = asyncio.create_task(
                self._working.get_cached_chunks(session_id)
            )
            recall_task = asyncio.create_task(
                self.recall(query, user_id, limit=5)
            )
            profile_task = asyncio.create_task(
                self._persistent.get_user_profile(user_id)
            )

            messages, state, chunks, memories, profile = await asyncio.gather(
                msgs_task, state_task, chunks_task, recall_task, profile_task,
                return_exceptions=False,
            )

            return {
                "messages": messages,
                "experiment_state": state,
                "cached_chunks": chunks,
                "memories": memories,
                "user_profile": profile,
            }

        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to build full context: %s", exc)
            raise MemoryError("Failed to build full context") from exc

    # ── Experiment context (convenience) ─────────────────────

    async def get_experiment_context(
        self, session_id: str
    ) -> dict[str, Any]:
        """
        Return the working-memory experiment state for a session.

        Thin wrapper around ``WorkingMemory.get_experiment_state`` for
        callers who only need the experiment state.

        Args:
            session_id: Current session id.

        Returns:
            State dict or empty dict if none found.
        """
        state = await self._working.get_experiment_state(session_id)
        return state or {}

    # ── User profile (convenience) ───────────────────────────

    async def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """
        Return the aggregate user profile from persistent memory.

        Args:
            user_id: User UUID string.

        Returns:
            Profile dict.
        """
        return await self._persistent.get_user_profile(user_id)

    # ── Summarize & compress ─────────────────────────────────

    async def summarize_and_compress(
        self, session_id: str, user_id: str
    ) -> str | None:
        """
        Summarise the working-memory conversation buffer, persist the
        summary, and trim the buffer to keep working memory lean.

        Flow:
        1. Read all messages from the session buffer.
        2. If the count exceeds the configured buffer size, use Haiku to
           generate a concise summary of the older messages.
        3. Save the summary as an episodic memory.
        4. Trim the buffer down to the most recent messages.

        Args:
            session_id: Session to compress.
            user_id: Owning user UUID string.

        Returns:
            The generated summary text, or ``None`` if compression was
            not needed.
        """
        try:
            settings = get_settings()
            buffer_size = settings.conversation_buffer_size
            messages = await self._working.get_messages(
                session_id, limit=1000
            )

            if len(messages) <= buffer_size:
                logger.debug(
                    "Session %s has %d messages (≤ %d), no compression needed",
                    session_id,
                    len(messages),
                    buffer_size,
                )
                return None

            # Messages to summarise (everything except the most recent N)
            keep_count = buffer_size // 2
            to_summarise = messages[: -keep_count]
            to_keep = messages[-keep_count:]

            # Build conversation text for summarisation
            conversation_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in to_summarise
            )

            llm = get_fast_llm()
            prompt = (
                "Summarise the following research lab conversation concisely. "
                "Preserve key decisions, experiment observations, safety "
                "concerns, and any deviations from protocols. Be factual.\n\n"
                f"{conversation_text}\n\n"
                "Summary:"
            )

            response = await llm.ainvoke(prompt)
            summary = response.content.strip()

            # Persist the summary as an episodic memory
            await self._episodic.record_episode(
                user_id=user_id,
                episode_type="learning",
                content=summary,
                metadata={
                    "session_id": session_id,
                    "compressed_message_count": len(to_summarise),
                    "source": "conversation_compression",
                },
                importance=0.6,
            )

            # Replace the Redis buffer with only the kept messages
            msg_key = self._working._msg_key(session_id)
            pipe = self._working._redis.pipeline(transaction=True)
            pipe.delete(msg_key)
            import json
            import time as _time

            for m in to_keep:
                pipe.rpush(
                    msg_key,
                    json.dumps(
                        {
                            "role": m["role"],
                            "content": m["content"],
                            "timestamp": m.get("timestamp", _time.time()),
                        }
                    ),
                )
            pipe.expire(msg_key, self._working._msg_ttl)
            await pipe.execute()

            logger.info(
                "Compressed session %s: summarised %d messages, kept %d",
                session_id,
                len(to_summarise),
                len(to_keep),
            )
            return summary

        except MemoryError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to summarize and compress session %s: %s",
                session_id,
                exc,
            )
            raise MemoryError(
                f"Failed to summarize and compress session {session_id}"
            ) from exc
