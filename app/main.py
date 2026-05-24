"""
Main entry point for the FastAPI application.

Configures ASGI app, includes routers, applies global middleware,
handles startup/shutdown lifecycle hooks, and mounts uploads directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import redis.asyncio as aioredis
from sqlalchemy import select

from app.config import get_settings
from app.api.router import api_router
from app.api.websocket import chat_ws
from app.api.middleware import ErrorHandlerMiddleware, RateLimiterMiddleware
from app.dependencies import get_engine, get_redis_pool, get_pinecone_index

# Setup logging
settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title=settings.app_name,
    description="Intelligent experimental lab assistant and research protocol protocol-aware companion.",
    version=settings.app_version,
)

# Custom middleware first (inner). CORS is added last so it wraps all responses.
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimiterMiddleware, max_requests=100, window_seconds=60)

cors_origins, allow_credentials = settings.resolved_cors_origins()
logger.info("CORS allow_origins=%s allow_credentials=%s", cors_origins, allow_credentials)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(chat_ws.router, prefix="/ws/chat")

# Mount upload static files path
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ── Startup & Shutdown events ─────────────────────────────────────────


@app.on_event("startup")
async def startup_event() -> None:
    """Verifies backend database and service connections on startup."""
    logger.info("FastAPI application starting up...")

    # 1. Verify PostgreSQL connection
    try:
        from app.dependencies import get_session_factory
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(select(1))
        logger.info("Startup check: PostgreSQL connection verified.")
    except Exception as e:
        logger.critical("Startup check: PostgreSQL connection failed: %s", e)

    # 2. Verify Redis connection
    try:
        redis_pool = get_redis_pool()
        redis_client = aioredis.Redis(connection_pool=redis_pool)
        await redis_client.ping()
        await redis_client.aclose()
        logger.info("Startup check: Redis connection verified.")
    except Exception as e:
        logger.critical("Startup check: Redis connection failed: %s", e)

    # 3. Verify Pinecone connection
    try:
        index = get_pinecone_index()
        index.describe_index_stats()
        logger.info("Startup check: Pinecone connection verified.")
    except Exception as e:
        logger.critical("Startup check: Pinecone connection failed: %s", e)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleans up resources during shutdown."""
    logger.info("FastAPI application shutting down...")
    
    # Close SQLAlchemy engine connections
    try:
        engine = get_engine()
        await engine.dispose()
        logger.info("SQLAlchemy engine disposed.")
    except Exception as e:
        logger.error("Failed to dispose SQLAlchemy engine: %s", e)
