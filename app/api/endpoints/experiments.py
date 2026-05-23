"""
Experiment session management API endpoints.
Handles starting, updating, completing, and checking timeline/status of experiment sessions.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, status
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis, get_current_user
from app.models.database import User
from app.models.schemas import (
    ExperimentStartRequest,
    ExperimentStepUpdate,
    ExperimentResponse,
    ExperimentTimelineResponse,
)
from app.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/start",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Experiments"],
)
async def start_experiment(
    request: ExperimentStartRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Start a new experiment session.
    Can optionally be linked to a pre-existing protocol.
    """
    service = ExperimentService(db, redis)
    logger.info(
        "Starting new experiment for user %s (protocol=%s)",
        current_user.id,
        request.protocol_id,
    )
    return await service.start_experiment(
        user_id=current_user.id,
        protocol_id=request.protocol_id,
        title=request.title,
        description=request.experiment_description,
    )


@router.get("/", response_model=list[ExperimentResponse], tags=["Experiments"])
async def list_experiments(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    List all experiment sessions for the current user.
    """
    service = ExperimentService(db, redis)
    return await service.get_experiments(current_user.id)


@router.get("/{experiment_id}", response_model=ExperimentResponse, tags=["Experiments"])
async def get_experiment(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Retrieve current status and details of a specific experiment session.
    """
    service = ExperimentService(db, redis)
    return await service.get_experiment(experiment_id, current_user.id)


@router.put("/{experiment_id}/step", response_model=ExperimentResponse, tags=["Experiments"])
async def update_experiment_step(
    experiment_id: uuid.UUID,
    update: ExperimentStepUpdate,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update progress in the experiment by marking a step active/complete,
    optionally logging notes or protocol deviations.
    """
    service = ExperimentService(db, redis)
    return await service.update_step(
        experiment_id=experiment_id,
        user_id=current_user.id,
        step_number=update.step_number,
        notes=update.notes,
        deviation=update.deviation,
    )


@router.post("/{experiment_id}/complete", response_model=ExperimentResponse, tags=["Experiments"])
async def complete_experiment(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Mark an experiment session as completed.
    """
    service = ExperimentService(db, redis)
    return await service.complete_experiment(experiment_id, current_user.id)


@router.get("/{experiment_id}/timeline", response_model=ExperimentTimelineResponse, tags=["Experiments"])
async def get_experiment_timeline(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get the structured timeline events for a specific experiment session.
    """
    service = ExperimentService(db, redis)
    return await service.get_timeline(experiment_id, current_user.id)
