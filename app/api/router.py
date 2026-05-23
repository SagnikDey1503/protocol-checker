"""
Main API router.
Assembles all endpoint-specific routers and applies appropriate path prefixes.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.endpoints import health, auth, protocols, chat, experiments, memory

api_router = APIRouter()

# Include endpoint routers with their standard v1 prefixes
api_router.include_router(health.router, prefix="")  # health.py defines "/health" route directly
api_router.include_router(auth.router, prefix="/auth")
api_router.include_router(protocols.router, prefix="/protocols")
api_router.include_router(chat.router, prefix="/chat")
api_router.include_router(experiments.router, prefix="/experiments")
api_router.include_router(memory.router, prefix="/memory")
