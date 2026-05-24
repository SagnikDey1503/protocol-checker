"""
Dependency injection for FastAPI.

Provides database sessions, Redis connections, Pinecone index,
and the current authenticated user as injectable dependencies.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import AsyncGenerator
from uuid import UUID

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pinecone import Pinecone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.core.auth import TokenValidationError, decode_user_id
from app.models.database import User

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


# ── Database ─────────────────────────────────────────────────


@lru_cache
def get_engine(database_url: str | None = None):
    """Create a cached async SQLAlchemy engine."""
    settings = get_settings()
    url = database_url or settings.database_url
    return create_async_engine(
        url,
        pool_pre_ping=True,       # Handle stale connections
        pool_size=20,
        max_overflow=10,
        echo=settings.debug,
    )


@lru_cache
def get_session_factory():
    """Create a cached async session factory."""
    engine = get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,  # CRITICAL for async — prevents DetachedInstanceError
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Redis ────────────────────────────────────────────────────


@lru_cache
def get_redis_pool():
    """Create a cached Redis connection pool."""
    settings = get_settings()
    return aioredis.ConnectionPool.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=50,
    )


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """FastAPI dependency: yields an async Redis connection."""
    pool = get_redis_pool()
    client = aioredis.Redis(connection_pool=pool)
    try:
        yield client
    finally:
        await client.aclose()


# ── Pinecone ─────────────────────────────────────────────────


@lru_cache
def get_pinecone_index():
    """Create a cached Pinecone index connection."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index_name)


# ── Auth ─────────────────────────────────────────────────────


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency: validate JWT and return the authenticated user."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = decode_user_id(credentials.credentials)
    except TokenValidationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user
