"""
PostgreSQL-backed persistent (long-term) memory.

Provides durable storage for:
- Conversation histories & summaries
- Completed experiment results
- User behavioural patterns
- Aggregate user profiles

All queries use SQLAlchemy 2.0 ``select()`` style with async sessions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import MemoryError
from app.models.database import (
    Conversation,
    ExperimentSession,
    Message,
    Protocol,
    User,
    UserPattern,
)
from app.models.enums import ExperimentStatus

logger = logging.getLogger(__name__)


class PersistentMemory:
    """PostgreSQL-backed long-term memory store."""

    def __init__(self, db_session: AsyncSession) -> None:
        """
        Initialise with an async SQLAlchemy session.

        Args:
            db_session: An ``AsyncSession`` instance (typically injected
                        via FastAPI's ``Depends(get_db)``).
        """
        self._db = db_session

    # ── Conversations ────────────────────────────────────────

    async def save_conversation(
        self,
        conversation_id: str,
        user_id: str,
        session_id: str | None,
        messages: list[dict[str, Any]],
        summary: str | None = None,
    ) -> str:
        """
        Save or update a conversation and its messages.

        If ``conversation_id`` already exists the existing record is updated
        (summary, message_count).  Otherwise a new row is created.

        Args:
            conversation_id: UUID string for the conversation.
            user_id: Owning user's UUID string.
            session_id: Optional linked experiment session UUID string.
            messages: List of ``{"role": …, "content": …}`` dicts.
            summary: Optional LLM-generated conversation summary.

        Returns:
            The conversation UUID as a string.
        """
        try:
            conv_uuid = uuid.UUID(conversation_id)
            user_uuid = uuid.UUID(user_id)
            session_uuid = uuid.UUID(session_id) if session_id else None

            # Check for existing conversation
            result = await self._db.execute(
                select(Conversation).where(Conversation.id == conv_uuid)
            )
            conversation = result.scalar_one_or_none()

            if conversation is None:
                conversation = Conversation(
                    id=conv_uuid,
                    user_id=user_uuid,
                    session_id=session_uuid,
                    summary=summary,
                    message_count=len(messages),
                )
                self._db.add(conversation)
                await self._db.flush()
            else:
                conversation.summary = summary or conversation.summary
                conversation.message_count = len(messages)

            # Persist individual messages
            for msg in messages:
                db_msg = Message(
                    conversation_id=conv_uuid,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    token_count=msg.get("token_count"),
                    sources=msg.get("sources"),
                    confidence=msg.get("confidence"),
                    agent_type=msg.get("agent_type"),
                    metadata_=msg.get("metadata"),
                )
                self._db.add(db_msg)

            await self._db.flush()
            logger.info(
                "Saved conversation %s with %d messages",
                conversation_id,
                len(messages),
            )
            return conversation_id

        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to save conversation %s: %s", conversation_id, exc)
            raise MemoryError(
                f"Failed to save conversation {conversation_id}"
            ) from exc

    async def get_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        Retrieve all messages for a conversation in chronological order.

        Args:
            conversation_id: UUID string of the conversation.

        Returns:
            List of message dicts with ``role``, ``content``, ``created_at``,
            and optional metadata fields.
        """
        try:
            conv_uuid = uuid.UUID(conversation_id)
            result = await self._db.execute(
                select(Message)
                .where(Message.conversation_id == conv_uuid)
                .order_by(Message.created_at.asc())
            )
            messages = result.scalars().all()
            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "token_count": msg.token_count,
                    "sources": msg.sources,
                    "confidence": msg.confidence,
                    "agent_type": msg.agent_type,
                    "metadata": msg.metadata_,
                    "created_at": msg.created_at.isoformat()
                    if msg.created_at
                    else None,
                }
                for msg in messages
            ]
        except MemoryError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to get conversation history %s: %s",
                conversation_id,
                exc,
            )
            raise MemoryError(
                f"Failed to get conversation history {conversation_id}"
            ) from exc

    # ── Experiments ──────────────────────────────────────────

    async def save_experiment_result(
        self,
        user_id: str,
        protocol_id: str | None,
        session_data: dict[str, Any],
    ) -> str:
        """
        Save a completed experiment session.

        If the session already exists (``session_data["id"]`` present) the
        record is updated; otherwise a new row is inserted.

        Args:
            user_id: Owning user UUID string.
            protocol_id: Optional protocol UUID string.
            session_data: Dict containing ``title``, ``status``, ``current_step``,
                          ``total_steps``, ``deviations``, ``notes``, ``timeline``.

        Returns:
            Experiment session UUID string.
        """
        try:
            user_uuid = uuid.UUID(user_id)
            protocol_uuid = uuid.UUID(protocol_id) if protocol_id else None

            session_id = session_data.get("id")
            if session_id:
                session_uuid = uuid.UUID(session_id)
                result = await self._db.execute(
                    select(ExperimentSession).where(
                        ExperimentSession.id == session_uuid
                    )
                )
                experiment = result.scalar_one_or_none()
            else:
                experiment = None
                session_uuid = uuid.uuid4()

            if experiment is None:
                experiment = ExperimentSession(
                    id=session_uuid,
                    user_id=user_uuid,
                    protocol_id=protocol_uuid,
                    title=session_data.get("title"),
                    status=session_data.get("status", ExperimentStatus.COMPLETED.value),
                    current_step=session_data.get("current_step", 0),
                    total_steps=session_data.get("total_steps", 0),
                    deviations=session_data.get("deviations"),
                    notes=session_data.get("notes"),
                    timeline=session_data.get("timeline"),
                    completed_at=datetime.now(timezone.utc),
                )
                self._db.add(experiment)
            else:
                experiment.status = session_data.get(
                    "status", ExperimentStatus.COMPLETED.value
                )
                experiment.current_step = session_data.get(
                    "current_step", experiment.current_step
                )
                experiment.deviations = session_data.get(
                    "deviations", experiment.deviations
                )
                experiment.notes = session_data.get("notes", experiment.notes)
                experiment.timeline = session_data.get(
                    "timeline", experiment.timeline
                )
                experiment.completed_at = datetime.now(timezone.utc)

            await self._db.flush()
            logger.info("Saved experiment result %s", session_uuid)
            return str(session_uuid)

        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to save experiment result: %s", exc)
            raise MemoryError("Failed to save experiment result") from exc

    async def get_user_experiments(
        self, user_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Retrieve a user's past experiment summaries.

        Args:
            user_id: User UUID string.
            limit: Max number of experiments to return.

        Returns:
            List of experiment summary dicts ordered by most recent first.
        """
        try:
            user_uuid = uuid.UUID(user_id)
            result = await self._db.execute(
                select(ExperimentSession)
                .where(ExperimentSession.user_id == user_uuid)
                .order_by(ExperimentSession.started_at.desc())
                .limit(limit)
            )
            experiments = result.scalars().all()
            return [
                {
                    "id": str(exp.id),
                    "protocol_id": str(exp.protocol_id) if exp.protocol_id else None,
                    "title": exp.title,
                    "status": exp.status,
                    "current_step": exp.current_step,
                    "total_steps": exp.total_steps,
                    "deviations": exp.deviations,
                    "notes": exp.notes,
                    "started_at": exp.started_at.isoformat()
                    if exp.started_at
                    else None,
                    "completed_at": exp.completed_at.isoformat()
                    if exp.completed_at
                    else None,
                }
                for exp in experiments
            ]
        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to get user experiments for %s: %s", user_id, exc)
            raise MemoryError(
                f"Failed to get user experiments for {user_id}"
            ) from exc

    # ── User Patterns ────────────────────────────────────────

    async def update_user_pattern(
        self,
        user_id: str,
        pattern_type: str,
        description: str,
    ) -> None:
        """
        Track or update a behavioural pattern for a user.

        If a pattern with the same ``(user_id, pattern_type, description)``
        already exists, its ``frequency`` is incremented and ``last_seen``
        is bumped.  Otherwise a new row is created.

        Args:
            user_id: User UUID string.
            pattern_type: One of ``PatternType`` enum values.
            description: Human-readable description of the pattern.
        """
        try:
            user_uuid = uuid.UUID(user_id)

            result = await self._db.execute(
                select(UserPattern).where(
                    UserPattern.user_id == user_uuid,
                    UserPattern.pattern_type == pattern_type,
                    UserPattern.description == description,
                )
            )
            pattern = result.scalar_one_or_none()

            if pattern is not None:
                pattern.frequency += 1
                pattern.last_seen = datetime.now(timezone.utc)
            else:
                pattern = UserPattern(
                    user_id=user_uuid,
                    pattern_type=pattern_type,
                    description=description,
                    frequency=1,
                    last_seen=datetime.now(timezone.utc),
                )
                self._db.add(pattern)

            await self._db.flush()
            logger.info(
                "Updated pattern [%s] for user %s: %s",
                pattern_type,
                user_id,
                description,
            )
        except MemoryError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to update user pattern for %s: %s", user_id, exc
            )
            raise MemoryError(
                f"Failed to update user pattern for {user_id}"
            ) from exc

    async def get_user_patterns(
        self, user_id: str
    ) -> list[dict[str, Any]]:
        """
        Return all recorded behavioural patterns for a user.

        Args:
            user_id: User UUID string.

        Returns:
            List of pattern dicts with ``pattern_type``, ``description``,
            ``frequency``, and ``last_seen``.
        """
        try:
            user_uuid = uuid.UUID(user_id)
            result = await self._db.execute(
                select(UserPattern)
                .where(UserPattern.user_id == user_uuid)
                .order_by(UserPattern.frequency.desc())
            )
            patterns = result.scalars().all()
            return [
                {
                    "id": str(p.id),
                    "pattern_type": p.pattern_type,
                    "description": p.description,
                    "frequency": p.frequency,
                    "last_seen": p.last_seen.isoformat() if p.last_seen else None,
                }
                for p in patterns
            ]
        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to get user patterns for %s: %s", user_id, exc)
            raise MemoryError(
                f"Failed to get user patterns for {user_id}"
            ) from exc

    # ── User Profile ─────────────────────────────────────────

    async def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """
        Build an aggregate user profile.

        Combines:
        - Basic user info
        - Total / completed experiment counts
        - Most common experiment types
        - Behavioural patterns
        - Inferred skill assessment

        Args:
            user_id: User UUID string.

        Returns:
            Profile dict suitable for schema ``UserProfileResponse``.
        """
        try:
            user_uuid = uuid.UUID(user_id)

            # User record
            user_result = await self._db.execute(
                select(User).where(User.id == user_uuid)
            )
            user = user_result.scalar_one_or_none()

            # Total experiments
            total_result = await self._db.execute(
                select(func.count())
                .select_from(ExperimentSession)
                .where(ExperimentSession.user_id == user_uuid)
            )
            total_experiments = total_result.scalar() or 0

            # Completed experiments
            completed_result = await self._db.execute(
                select(func.count())
                .select_from(ExperimentSession)
                .where(
                    ExperimentSession.user_id == user_uuid,
                    ExperimentSession.status == ExperimentStatus.COMPLETED.value,
                )
            )
            completed_experiments = completed_result.scalar() or 0

            # Common experiment types (via protocols linked to sessions)
            type_result = await self._db.execute(
                select(Protocol.experiment_type, func.count().label("cnt"))
                .join(
                    ExperimentSession,
                    ExperimentSession.protocol_id == Protocol.id,
                )
                .where(ExperimentSession.user_id == user_uuid)
                .where(Protocol.experiment_type.isnot(None))
                .group_by(Protocol.experiment_type)
                .order_by(func.count().desc())
                .limit(5)
            )
            common_types = [row[0] for row in type_result.all()]

            # Patterns
            patterns = await self.get_user_patterns(user_id)

            # Skill assessment heuristic
            skill_assessment = self._assess_skill(
                total_experiments, completed_experiments, patterns
            )

            return {
                "user_id": user_id,
                "full_name": user.full_name if user else None,
                "total_experiments": total_experiments,
                "completed_experiments": completed_experiments,
                "common_experiment_types": common_types,
                "patterns": patterns,
                "skill_assessment": skill_assessment,
            }

        except MemoryError:
            raise
        except Exception as exc:
            logger.error("Failed to get user profile for %s: %s", user_id, exc)
            raise MemoryError(
                f"Failed to get user profile for {user_id}"
            ) from exc

    # ── Private helpers ──────────────────────────────────────

    @staticmethod
    def _assess_skill(
        total: int, completed: int, patterns: list[dict[str, Any]]
    ) -> str:
        """
        Simple heuristic to estimate user skill level.

        Args:
            total: Total experiments attempted.
            completed: Completed experiments.
            patterns: User behavioural patterns.

        Returns:
            One of ``"beginner"``, ``"intermediate"``, ``"advanced"``.
        """
        if total == 0:
            return "beginner"

        completion_rate = completed / total if total > 0 else 0.0
        mistake_count = sum(
            p.get("frequency", 0)
            for p in patterns
            if p.get("pattern_type") == "common_mistake"
        )

        if total >= 20 and completion_rate >= 0.8 and mistake_count <= 2:
            return "advanced"
        if total >= 5 and completion_rate >= 0.5:
            return "intermediate"
        return "beginner"
