"""
WebSocket chat handler.
Handles real-time, bi-directional message streaming for active experiment sessions.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from jose import jwt
import redis.asyncio as aioredis
from sqlalchemy import select

from app.config import get_settings
from app.dependencies import get_session_factory, get_redis_pool, get_pinecone_index
from app.models.database import User
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """
    Manages active WebSocket connections.
    """

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket connection established")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed")


manager = ConnectionManager()


async def get_websocket_user(websocket: WebSocket) -> Optional[User]:
    """Authenticate WebSocket connection using query parameter token."""
    token = websocket.query_params.get("token")
    if not token:
        logger.warning("WebSocket authentication: Token query param missing")
        return None

    try:
        settings = get_settings()
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id = payload.get("sub")
        if not user_id:
            logger.warning("WebSocket authentication: Sub claim missing in token")
            return None

        # Fetch user using a fresh DB session
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(User).where(User.id == uuid.UUID(user_id))
            )
            user = result.scalar_one_or_none()
            if user and user.is_active:
                return user
            
            logger.warning("WebSocket authentication: User %s not found or inactive", user_id)
            return None
    except Exception as e:
        logger.warning("WebSocket authentication failed: %s", e)
        return None


@router.websocket("/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """
    WebSocket endpoint for real-time streaming chat.
    Expects connection parameters: ws://host:port/ws/chat/{session_id}?token={jwt_token}
    """
    await manager.connect(websocket)

    # 0. Validate session_id format early to avoid silent handshake failures
    try:
        session_uuid = uuid.UUID(session_id)
    except Exception:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid session_id format (expected UUID)",
        )
        manager.disconnect(websocket)
        return

    # 1. Authenticate connection
    user = await get_websocket_user(websocket)
    if not user:
        logger.warning("WebSocket connection rejected: Authentication failed")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid or expired token",
        )
        manager.disconnect(websocket)
        return

    logger.info("WebSocket authenticated for user=%s session=%s", user.id, session_uuid)

    # Keep a pool for Redis
    redis_pool = get_redis_pool()
    redis_client = aioredis.Redis(connection_pool=redis_pool)
    session_factory = get_session_factory()

    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()
            logger.info("WebSocket received frame: %s", raw_data[:200])

            # Parse JSON message
            try:
                payload = json.loads(raw_data)
                message = payload.get("message", "")
                experiment_id_str = payload.get("experiment_id")
            except json.JSONDecodeError:
                message = raw_data
                experiment_id_str = None

            if not message or not message.strip():
                continue

            experiment_id = (
                uuid.UUID(experiment_id_str) if experiment_id_str else None
            )

            # Process the message with a fresh database session per message
            # to avoid transaction bleed or session reuse errors.
            async with session_factory() as session:
                service = ChatService(session, redis_client)
                generator = service.stream_message(
                    message=message,
                    user_id=user.id,
                    conversation_id=session_uuid,
                    experiment_id=experiment_id,
                )

                # Iterate through stream and send frames to client
                async for sse_event in generator:
                    # sse_event format: "data: {json_str}\n\n"
                    if sse_event.startswith("data: ") and sse_event.endswith("\n\n"):
                        json_str = sse_event[6:-2]
                        if json_str == "[DONE]":
                            await websocket.send_json({"type": "done"})
                        else:
                            try:
                                event_data = json.loads(json_str)
                                await websocket.send_json(event_data)
                            except Exception:
                                # Fallback if not JSON
                                await websocket.send_json(
                                    {"type": "token", "token": json_str}
                                )

    except WebSocketDisconnect as e:
        logger.info("WebSocket disconnected (code=%s)", getattr(e, "code", "unknown"))
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket session exception: %s", e, exc_info=True)
        manager.disconnect(websocket)
    finally:
        await redis_client.aclose()
