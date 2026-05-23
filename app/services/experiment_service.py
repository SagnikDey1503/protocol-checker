"""
Experiment session management service.

Handles the lifecycle of lab experiment sessions: creation, step tracking,
deviation recording, and timeline generation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import ExperimentNotFoundError, ProtocolNotFoundError
from app.models.database import ExperimentSession, Protocol
from app.models.enums import ExperimentStatus
from app.models.schemas import ExperimentResponse, ExperimentTimelineResponse

logger = logging.getLogger(__name__)


class ExperimentService:
    """Business logic for experiment session management.

    Args:
        db: Async SQLAlchemy session.
        redis: Async Redis client for experiment state caching.
    """

    def __init__(self, db: AsyncSession, redis: aioredis.Redis) -> None:
        self.db = db
        self.redis = redis
        self.settings = get_settings()

    # ── Start experiment ─────────────────────────────────────

    async def start_experiment(
        self,
        user_id: uuid.UUID,
        protocol_id: Optional[uuid.UUID] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ExperimentSession:
        """Create a new experiment session.

        If a ``protocol_id`` is provided, the experiment is linked to
        that protocol and its total steps are pre-populated.

        Args:
            user_id: The user starting the experiment.
            protocol_id: Optional protocol to follow.
            title: Human-readable title.
            description: Optional free-text description.

        Returns:
            The newly created ``ExperimentSession``.

        Raises:
            ProtocolNotFoundError: If the protocol doesn't exist.
        """
        total_steps = 0

        if protocol_id:
            result = await self.db.execute(
                select(Protocol).where(
                    Protocol.id == protocol_id,
                    Protocol.user_id == user_id,
                )
            )
            protocol = result.scalar_one_or_none()
            if protocol is None:
                raise ProtocolNotFoundError(str(protocol_id))
            total_steps = protocol.step_count or 0
            if not title:
                title = f"Experiment: {protocol.title}"

        experiment = ExperimentSession(
            user_id=user_id,
            protocol_id=protocol_id,
            title=title or f"Experiment {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            status=ExperimentStatus.ACTIVE.value,
            current_step=0,
            total_steps=total_steps,
            deviations=[],
            timeline=[
                {
                    "event": "experiment_started",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "description": description or "Experiment session created",
                }
            ],
            notes={},
        )
        self.db.add(experiment)
        await self.db.flush()
        await self.db.refresh(experiment)

        # Cache active experiment state in Redis
        await self.redis.setex(
            f"experiment_state:{experiment.id}",
            self.settings.experiment_state_ttl,
            experiment.status,
        )

        logger.info(
            "Experiment %s started by user %s (protocol=%s)",
            experiment.id,
            user_id,
            protocol_id,
        )
        return experiment

    # ── List experiments ─────────────────────────────────────

    async def get_experiments(
        self, user_id: uuid.UUID
    ) -> list[ExperimentSession]:
        """Return all experiments for a user, newest first.

        Args:
            user_id: The user's UUID.

        Returns:
            A list of ``ExperimentSession`` instances.
        """
        result = await self.db.execute(
            select(ExperimentSession)
            .where(ExperimentSession.user_id == user_id)
            .order_by(ExperimentSession.started_at.desc())
        )
        return list(result.scalars().all())

    # ── Get single experiment ────────────────────────────────

    async def get_experiment(
        self,
        experiment_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ExperimentSession:
        """Fetch a single experiment owned by the user.

        Args:
            experiment_id: The experiment's UUID.
            user_id: The owning user's UUID.

        Returns:
            The ``ExperimentSession``.

        Raises:
            ExperimentNotFoundError: If not found.
        """
        result = await self.db.execute(
            select(ExperimentSession).where(
                ExperimentSession.id == experiment_id,
                ExperimentSession.user_id == user_id,
            )
        )
        experiment = result.scalar_one_or_none()
        if experiment is None:
            raise ExperimentNotFoundError(str(experiment_id))
        return experiment

    # ── Update step ──────────────────────────────────────────

    async def update_step(
        self,
        experiment_id: uuid.UUID,
        user_id: uuid.UUID,
        step_number: int,
        notes: Optional[str] = None,
        deviation: Optional[str] = None,
    ) -> ExperimentSession:
        """Update the current step of an experiment.

        Records step progression, optional notes, and any deviations
        from the protocol.

        Args:
            experiment_id: The experiment's UUID.
            user_id: The owning user's UUID.
            step_number: The step number being completed/updated.
            notes: Optional observations or notes for this step.
            deviation: Optional description of protocol deviation.

        Returns:
            The updated ``ExperimentSession``.

        Raises:
            ExperimentNotFoundError: If not found or not active.
        """
        experiment = await self.get_experiment(experiment_id, user_id)

        if experiment.status != ExperimentStatus.ACTIVE.value:
            raise ExperimentNotFoundError(
                f"Experiment {experiment_id} is not active (status={experiment.status})"
            )

        now = datetime.now(timezone.utc)

        # Update current step
        experiment.current_step = step_number

        # Append timeline entry
        timeline = list(experiment.timeline or [])
        timeline_entry: dict[str, Any] = {
            "event": "step_updated",
            "step_number": step_number,
            "timestamp": now.isoformat(),
        }
        if notes:
            timeline_entry["notes"] = notes
        timeline.append(timeline_entry)
        experiment.timeline = timeline

        # Record notes
        exp_notes = dict(experiment.notes or {})
        if notes:
            exp_notes[f"step_{step_number}"] = notes
        experiment.notes = exp_notes

        # Record deviation
        if deviation:
            deviations = list(experiment.deviations or [])
            deviations.append(
                {
                    "step_number": step_number,
                    "deviation": deviation,
                    "timestamp": now.isoformat(),
                }
            )
            experiment.deviations = deviations

            # Also add deviation to timeline
            timeline.append(
                {
                    "event": "deviation_recorded",
                    "step_number": step_number,
                    "deviation": deviation,
                    "timestamp": now.isoformat(),
                }
            )
            experiment.timeline = timeline

        await self.db.flush()
        await self.db.refresh(experiment)

        logger.info(
            "Experiment %s step updated to %d%s",
            experiment_id,
            step_number,
            " (deviation)" if deviation else "",
        )
        return experiment

    # ── Complete experiment ───────────────────────────────────

    async def complete_experiment(
        self,
        experiment_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ExperimentSession:
        """Mark an experiment as completed.

        Sets the status to ``COMPLETED`` and records the completion
        timestamp.

        Args:
            experiment_id: The experiment's UUID.
            user_id: The owning user's UUID.

        Returns:
            The completed ``ExperimentSession``.

        Raises:
            ExperimentNotFoundError: If not found.
        """
        experiment = await self.get_experiment(experiment_id, user_id)

        now = datetime.now(timezone.utc)
        experiment.status = ExperimentStatus.COMPLETED.value
        experiment.completed_at = now

        # Append completion event to timeline
        timeline = list(experiment.timeline or [])
        timeline.append(
            {
                "event": "experiment_completed",
                "timestamp": now.isoformat(),
                "current_step": experiment.current_step,
                "total_steps": experiment.total_steps,
            }
        )
        experiment.timeline = timeline

        await self.db.flush()
        await self.db.refresh(experiment)

        # Clean up Redis state
        await self.redis.delete(f"experiment_state:{experiment_id}")

        logger.info("Experiment %s completed by user %s", experiment_id, user_id)
        return experiment

    # ── Timeline ─────────────────────────────────────────────

    async def get_timeline(
        self,
        experiment_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ExperimentTimelineResponse:
        """Build a structured timeline for an experiment.

        Args:
            experiment_id: The experiment's UUID.
            user_id: The owning user's UUID.

        Returns:
            An ``ExperimentTimelineResponse`` with steps, deviations,
            and duration.

        Raises:
            ExperimentNotFoundError: If not found.
        """
        experiment = await self.get_experiment(experiment_id, user_id)

        steps = [
            entry
            for entry in (experiment.timeline or [])
            if entry.get("event") == "step_updated"
        ]
        deviations = list(experiment.deviations or [])

        # Compute duration
        duration_seconds: Optional[float] = None
        if experiment.completed_at and experiment.started_at:
            delta = experiment.completed_at - experiment.started_at
            duration_seconds = delta.total_seconds()

        return ExperimentTimelineResponse(
            experiment_id=experiment.id,
            steps=steps,
            deviations=deviations,
            duration_seconds=duration_seconds,
        )
