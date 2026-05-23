"""
Chat endpoints.
Provides synchronous and streaming conversation endpoints with the AI assistant.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis, get_current_user
from app.models.database import User
from app.models.schemas import ChatRequest, ChatResponse, ConversationHistoryResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Send a chat message and receive a complete response.
    """
    service = ChatService(db, redis)
    logger.info(
        "Received chat request from user %s (conversation_id=%s)",
        current_user.id,
        request.conversation_id,
    )
    return await service.process_message(
        message=request.message,
        user_id=current_user.id,
        conversation_id=request.conversation_id,
        experiment_id=request.experiment_id,
    )


@router.post("/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Send a chat message and stream the response back using Server-Sent Events (SSE).
    """
    service = ChatService(db, redis)
    logger.info(
        "Received streaming chat request from user %s (conversation_id=%s)",
        current_user.id,
        request.conversation_id,
    )
    
    generator = service.stream_message(
        message=request.message,
        user_id=current_user.id,
        conversation_id=request.conversation_id,
        experiment_id=request.experiment_id,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{conversation_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_history(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Retrieve the historical messages of a specific conversation thread.
    """
    service = ChatService(db, redis)
    return await service.get_history(conversation_id, current_user.id)
