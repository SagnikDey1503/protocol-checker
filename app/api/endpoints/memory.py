"""
Memory inspection API endpoints.
Provides search over semantic/episodic memory and retrieval of user patterns and history.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis, get_current_user, get_pinecone_index
from app.models.database import User
from app.models.schemas import (
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryEntry,
    UserProfileResponse,
)
from app.memory import (
    MemoryManager,
    WorkingMemory,
    PersistentMemory,
    SemanticMemory,
    EpisodicMemoryStore,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_memory_manager(db: AsyncSession, redis: aioredis.Redis) -> MemoryManager:
    """Initialize a MemoryManager façade using active request resources."""
    pinecone_index = get_pinecone_index()
    working = WorkingMemory(redis)
    persistent = PersistentMemory(db)
    semantic = SemanticMemory(pinecone_index)
    episodic = EpisodicMemoryStore(db, semantic)
    return MemoryManager(working, persistent, semantic, episodic)


@router.post("/recall", response_model=MemoryRecallResponse, tags=["Memory"])
async def recall_memories(
    request: MemoryRecallRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Search over the user's semantic and episodic memories.
    """
    manager = _get_memory_manager(db, redis)
    logger.info("Recalling memories for query: %s", request.query[:80])
    
    results = await manager.recall(
        query=request.query,
        user_id=str(current_user.id),
        memory_types=request.memory_types,
        limit=request.limit,
    )

    memories = [
        MemoryEntry(
            id=str(m["id"]),
            memory_type=m["memory_type"],
            content=m["content"],
            importance_score=m["importance_score"],
            created_at=m.get("created_at"),
            metadata=m.get("metadata"),
        )
        for m in results
    ]

    return MemoryRecallResponse(memories=memories, total=len(memories))


@router.get("/profile", response_model=UserProfileResponse, tags=["Memory"])
async def get_user_profile(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get the aggregated user profile, experience level, and behavioral patterns.
    """
    manager = _get_memory_manager(db, redis)
    profile = await manager.get_user_profile(str(current_user.id))
    
    return UserProfileResponse(
        user_id=current_user.id,
        total_experiments=profile.get("total_experiments", 0),
        completed_experiments=profile.get("completed_experiments", 0),
        common_experiment_types=profile.get("common_experiment_types", []),
        patterns=profile.get("patterns", []),
        skill_assessment=profile.get("skill_assessment"),
    )


@router.get("/experiments", tags=["Memory"])
async def get_memory_experiments(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get a list of past experiment summaries recorded in persistent memory.
    """
    manager = _get_memory_manager(db, redis)
    experiments = await manager._persistent.get_user_experiments(
        user_id=str(current_user.id),
        limit=20,
    )
    return {"experiments": experiments, "total": len(experiments)}
