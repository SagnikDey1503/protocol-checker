"""
Episodic memory — dual-storage (PostgreSQL + Pinecone) episode recording.

Every episode is:
1. Written to PostgreSQL for durable, queryable storage.
2. Embedded and upserted into Pinecone for semantic recall.

An LLM (Haiku) scores the importance of each episode so that the recall
pipeline can prioritise high-value memories.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import MemoryError
from app.core.llm import get_fast_llm
from app.models.database import EpisodicMemory as EpisodicMemoryModel
from app.models.enums import EpisodeType
from app.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """Dual-storage episodic memory (DB + vector store)."""

    def __init__(
        self,
        db_session: AsyncSession,
        semantic_memory: SemanticMemory,
    ) -> None:
        """
        Initialise the episodic memory layer.

        Args:
            db_session: Async SQLAlchemy session for PostgreSQL writes.
            semantic_memory: ``SemanticMemory`` instance for vector upserts.
        """
        self._db = db_session
        self._semantic = semantic_memory

    # ── Record ───────────────────────────────────────────────

    async def record_episode(
        self,
        user_id: str,
        episode_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> str:
        """
        Record a new episode to both PostgreSQL and Pinecone.

        If ``importance`` is not supplied, the LLM scores it automatically
        via :meth:`score_importance`.

        Args:
            user_id: Owning user UUID string.
            episode_type: One of ``EpisodeType`` enum values.
            content: Free-text description of the episode.
            metadata: Optional metadata dict.
            importance: Optional pre-computed importance score (0–1).

        Returns:
            The episode UUID as a string.
        """
        try:
            # Score importance if not provided
            if importance is None:
                importance = await self.score_importance(content, episode_type)

            user_uuid = uuid.UUID(user_id)
            episode_id = uuid.uuid4()

            # ── 1. PostgreSQL ────────────────────────────────
            db_episode = EpisodicMemoryModel(
                id=episode_id,
                user_id=user_uuid,
                episode_type=episode_type,
                content=content,
                metadata_=metadata or {},
                importance_score=importance,
            )
            self._db.add(db_episode)
            await self._db.flush()

            # ── 2. Pinecone (semantic memory) ────────────────
            episode_metadata = {
                "episode_type": episode_type,
                "episode_id": str(episode_id),
                **(metadata or {}),
            }
            vector_id = await self._semantic.store(
                user_id=user_id,
                content=content,
                memory_type="episodic",
                metadata=episode_metadata,
                importance=importance,
            )

            # Link the embedding id back to the DB record
            db_episode.embedding_id = vector_id
            await self._db.flush()

            logger.info(
                "Recorded episode %s (type=%s, importance=%.2f) for user %s",
                episode_id,
                episode_type,
                importance,
                user_id,
            )
            return str(episode_id)

        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to record episode: %s", exc)
            raise MemoryError("Failed to record episode") from exc

    # ── Query (PostgreSQL) ───────────────────────────────────

    async def get_episodes(
        self,
        user_id: str,
        episode_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve episodes from PostgreSQL, optionally filtered by type.

        Args:
            user_id: Owning user UUID string.
            episode_type: Optional ``EpisodeType`` filter.
            limit: Max records to return.

        Returns:
            List of episode dicts ordered by most recent first.
        """
        try:
            user_uuid = uuid.UUID(user_id)
            stmt = (
                select(EpisodicMemoryModel)
                .where(EpisodicMemoryModel.user_id == user_uuid)
                .order_by(EpisodicMemoryModel.created_at.desc())
                .limit(limit)
            )
            if episode_type is not None:
                stmt = stmt.where(
                    EpisodicMemoryModel.episode_type == episode_type
                )

            result = await self._db.execute(stmt)
            episodes = result.scalars().all()
            return [
                {
                    "id": str(ep.id),
                    "episode_type": ep.episode_type,
                    "content": ep.content,
                    "importance_score": ep.importance_score,
                    "metadata": ep.metadata_,
                    "embedding_id": ep.embedding_id,
                    "created_at": ep.created_at.isoformat()
                    if ep.created_at
                    else None,
                }
                for ep in episodes
            ]
        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to get episodes for user %s: %s", user_id, exc)
            raise MemoryError(
                f"Failed to get episodes for user {user_id}"
            ) from exc

    # ── Semantic recall ──────────────────────────────────────

    async def recall_relevant_episodes(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Semantically search for episodes relevant to ``query``.

        Delegates to ``SemanticMemory.recall`` with a filter on
        ``memory_type == "episodic"``.

        Args:
            user_id: Owning user UUID string.
            query: Natural-language query.
            limit: Max number of results.

        Returns:
            List of dicts with ``content``, ``similarity_score``,
            ``importance_score``, etc.
        """
        try:
            entries = await self._semantic.recall(
                query=query,
                user_id=user_id,
                limit=limit,
            )
            # Filter to episodic only (belt-and-suspenders; Pinecone filter
            # may already handle this, but recall() is generic).
            episodic = [
                e for e in entries if e.memory_type == "episodic"
            ]
            return [
                {
                    "id": e.id,
                    "content": e.content,
                    "episode_type": e.metadata.get("episode_type", "unknown"),
                    "importance_score": e.importance_score,
                    "similarity_score": e.similarity_score,
                    "metadata": e.metadata,
                    "created_at": e.created_at,
                }
                for e in episodic
            ]
        except MemoryError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to recall relevant episodes for user %s: %s",
                user_id,
                exc,
            )
            raise MemoryError(
                f"Failed to recall relevant episodes for user {user_id}"
            ) from exc

    # ── Importance scoring ───────────────────────────────────

    async def score_importance(
        self, content: str, episode_type: str
    ) -> float:
        """
        Use Haiku to score how important an episode is on a 0–1 scale.

        Falls back to a heuristic default if the LLM call fails.

        Args:
            content: Episode text.
            episode_type: Episode category.

        Returns:
            Importance score between 0.0 and 1.0.
        """
        # Heuristic defaults by episode type
        type_defaults: dict[str, float] = {
            EpisodeType.EXPERIMENT.value: 0.6,
            EpisodeType.MISTAKE.value: 0.8,
            EpisodeType.SUCCESS.value: 0.7,
            EpisodeType.LEARNING.value: 0.7,
            EpisodeType.DEVIATION.value: 0.75,
        }
        default_score = type_defaults.get(episode_type, 0.5)

        try:
            llm = get_fast_llm()
            prompt = (
                "You are scoring the importance of a research lab episode for "
                "future recall. Score from 0.0 (trivial) to 1.0 (critical).\n\n"
                f"Episode type: {episode_type}\n"
                f"Content: {content}\n\n"
                "Respond with ONLY a decimal number between 0.0 and 1.0, "
                "nothing else."
            )
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()

            # Parse the score robustly
            score = float(raw)
            score = max(0.0, min(1.0, score))
            logger.debug(
                "LLM importance score for episode (%s): %.2f",
                episode_type,
                score,
            )
            return score

        except (ValueError, TypeError) as parse_err:
            logger.warning(
                "Could not parse LLM importance score (%s), using default %.2f",
                parse_err,
                default_score,
            )
            return default_score
        except Exception as exc:
            logger.warning(
                "LLM importance scoring failed (%s), using default %.2f",
                exc,
                default_score,
            )
            return default_score
