"""
Health check endpoint.
Verifies PostgreSQL, Redis, and Pinecone database connections.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.config import get_settings
from app.dependencies import get_db, get_redis, get_pinecone_index
from app.models.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> HealthResponse:
    """
    Check the health of the application and its background dependencies.
    """
    settings = get_settings()
    services_status = {}

    # 1. Check PostgreSQL
    try:
        await db.execute(select(1))
        services_status["postgres"] = "connected"
    except Exception as e:
        logger.error("Health check: PostgreSQL connection failed: %s", e)
        services_status["postgres"] = "disconnected"

    # 2. Check Redis
    try:
        await redis.ping()
        services_status["redis"] = "connected"
    except Exception as e:
        logger.error("Health check: Redis connection failed: %s", e)
        services_status["redis"] = "disconnected"

    # 3. Check Pinecone
    try:
        index = get_pinecone_index()
        # Fetch stats to verify connectivity
        index.describe_index_stats()
        services_status["pinecone"] = "connected"
    except Exception as e:
        logger.error("Health check: Pinecone connection failed: %s", e)
        services_status["pinecone"] = "disconnected"

    overall_status = "healthy"
    if any(status == "disconnected" for status in services_status.values()):
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        services=services_status,
    )
