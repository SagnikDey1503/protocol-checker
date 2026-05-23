"""
Chat orchestration service.

Bridges the FastAPI endpoints to the LangGraph agent orchestrator.
Manages conversation persistence and provides both full-response and
streaming-response modes.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import AgentExecutionError, RetrievalError
from app.models.database import Conversation, Message
from app.models.enums import MessageRole
from app.models.schemas import (
    ChatResponse,
    ConversationHistoryResponse,
    MessageResponse,
    SafetyAlert,
    SourceCitation,
)

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates chat interactions between users and the AI agent system.

    Args:
        db: Async SQLAlchemy session.
        redis: Async Redis client for working memory / caching.
    """

    def __init__(self, db: AsyncSession, redis: aioredis.Redis) -> None:
        self.db = db
        self.redis = redis
        self.settings = get_settings()

    # ── Full (non-streaming) response ────────────────────────

    async def process_message(
        self,
        message: str,
        user_id: uuid.UUID,
        conversation_id: Optional[uuid.UUID] = None,
        experiment_id: Optional[uuid.UUID] = None,
    ) -> ChatResponse:
        """Process a chat message and return a complete response.

        1. Resolve or create the conversation.
        2. Persist the user message.
        3. Invoke the LangGraph orchestrator.
        4. Persist the assistant message.
        5. Return the structured ``ChatResponse``.

        Args:
            message: The user's chat message text.
            user_id: Authenticated user's UUID.
            conversation_id: Optional existing conversation to continue.
            experiment_id: Optional active experiment session.

        Returns:
            A populated ``ChatResponse`` with the assistant's reply,
            sources, safety alerts, etc.
        """
        conversation = await self._resolve_conversation(
            user_id, conversation_id, experiment_id
        )

        # Save user message
        await self._save_message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=message,
        )

        # Build orchestrator input
        agent_input = self._build_agent_input(
            message=message,
            user_id=user_id,
            conversation_id=conversation.id,
            experiment_id=experiment_id,
        )

        # Invoke the agent
        agent_output = await self._invoke_agent(agent_input)

        # Extract response fields
        response_text = agent_output.get("response", "I'm sorry, I couldn't generate a response.")
        agent_type = agent_output.get("agent_type", "general")
        confidence = agent_output.get("confidence", 0.0)
        sources = [
            SourceCitation(**s) for s in agent_output.get("sources", [])
        ]
        safety_alerts = [
            SafetyAlert(**a) for a in agent_output.get("safety_alerts", [])
        ]
        next_steps = agent_output.get("next_steps", [])
        deviation_detected = agent_output.get("deviation_detected", False)
        experiment_state = agent_output.get("experiment_state")

        # Save assistant message
        await self._save_message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            sources=[s.model_dump() for s in sources],
            confidence=confidence,
            agent_type=agent_type,
        )

        # Update conversation metadata
        conversation.message_count += 2
        await self.db.flush()

        return ChatResponse(
            message=response_text,
            conversation_id=conversation.id,
            agent_type=agent_type,
            confidence=confidence,
            sources=sources,
            safety_alerts=safety_alerts,
            next_steps=next_steps,
            deviation_detected=deviation_detected,
            experiment_state=experiment_state,
        )

    # ── Streaming response ───────────────────────────────────

    async def stream_message(
        self,
        message: str,
        user_id: uuid.UUID,
        conversation_id: Optional[uuid.UUID] = None,
        experiment_id: Optional[uuid.UUID] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat response as Server-Sent Events.

        Yields SSE-formatted strings (``data: ...\\n\\n``) as the
        agent generates tokens.

        Args:
            message: The user's chat message text.
            user_id: Authenticated user's UUID.
            conversation_id: Optional existing conversation to continue.
            experiment_id: Optional active experiment session.

        Yields:
            SSE-formatted strings for each token/chunk and a final
            ``[DONE]`` sentinel.
        """
        conversation = await self._resolve_conversation(
            user_id, conversation_id, experiment_id
        )

        await self._save_message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=message,
        )

        agent_input = self._build_agent_input(
            message=message,
            user_id=user_id,
            conversation_id=conversation.id,
            experiment_id=experiment_id,
        )

        full_response_parts: list[str] = []
        agent_type = "general"
        confidence = 0.0
        sources: list[dict[str, Any]] = []

        try:
            # Try to use the streaming orchestrator
            try:
                from app.agents.orchestrator import get_orchestrator

                orchestrator = get_orchestrator()
                config = {
                    "configurable": {
                        "thread_id": str(conversation.id),
                        "db": self.db,
                        "redis": self.redis,
                    }
                }
                async for event in orchestrator.astream(agent_input, config=config):
                    if isinstance(event, dict):
                        # Handle different event types from LangGraph
                        for node_name, node_output in event.items():
                            if isinstance(node_output, dict):
                                chunk = node_output.get("response_chunk", "")
                                if chunk:
                                    full_response_parts.append(chunk)
                                    yield f"data: {json.dumps({'token': chunk})}\n\n"

                                # Capture metadata from final node output
                                if "agent_type" in node_output:
                                    agent_type = node_output["agent_type"]
                                if "confidence" in node_output:
                                    confidence = node_output["confidence"]
                                if "sources" in node_output:
                                    sources = node_output["sources"]
            except ImportError:
                # Orchestrator not yet implemented — use direct LLM
                from app.core.llm import get_llm

                llm = get_llm()
                async for chunk in llm.astream(message):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_response_parts.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as exc:
            logger.error("Streaming error: %s", exc, exc_info=True)
            error_msg = "I encountered an error while generating a response."
            yield f"data: {json.dumps({'token': error_msg, 'error': True})}\n\n"
            full_response_parts = [error_msg]

        # Emit final metadata event
        full_response = "".join(full_response_parts)
        yield f"data: {json.dumps({'done': True, 'conversation_id': str(conversation.id), 'agent_type': agent_type})}\n\n"
        yield "data: [DONE]\n\n"

        # Persist assistant message
        await self._save_message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=full_response,
            sources=sources,
            confidence=confidence,
            agent_type=agent_type,
        )

        conversation.message_count += 2
        await self.db.flush()

    # ── History ──────────────────────────────────────────────

    async def get_history(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ConversationHistoryResponse:
        """Retrieve the full message history for a conversation.

        Args:
            conversation_id: UUID of the conversation.
            user_id: Authenticated user's UUID (for ownership check).

        Returns:
            A ``ConversationHistoryResponse`` containing all messages.

        Raises:
            RetrievalError: If the conversation doesn't exist or
                doesn't belong to the user.
        """
        result = await self.db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise RetrievalError(f"Conversation {conversation_id} not found")

        msg_result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        messages = msg_result.scalars().all()

        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=[
                MessageResponse(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    sources=m.sources,
                    confidence=m.confidence,
                    agent_type=m.agent_type,
                    created_at=m.created_at,
                )
                for m in messages
            ],
            total=len(messages),
        )

    # ── Private helpers ──────────────────────────────────────

    async def _resolve_conversation(
        self,
        user_id: uuid.UUID,
        conversation_id: Optional[uuid.UUID],
        experiment_id: Optional[uuid.UUID],
    ) -> Conversation:
        """Find an existing conversation or create a new one."""
        if conversation_id:
            result = await self.db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                return conversation

        # Create new conversation
        conversation = Conversation(
            user_id=user_id,
            session_id=experiment_id,
            title=f"Conversation {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            message_count=0,
        )
        self.db.add(conversation)
        await self.db.flush()
        await self.db.refresh(conversation)
        return conversation

    async def _save_message(
        self,
        conversation_id: uuid.UUID,
        role: MessageRole,
        content: str,
        sources: Optional[list[dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        agent_type: Optional[str] = None,
    ) -> Message:
        """Persist a message to the database."""
        msg = Message(
            conversation_id=conversation_id,
            role=role.value,
            content=content,
            sources=sources,
            confidence=confidence,
            agent_type=agent_type,
        )
        self.db.add(msg)
        await self.db.flush()
        return msg

    def _build_agent_input(
        self,
        message: str,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        experiment_id: Optional[uuid.UUID],
    ) -> dict[str, Any]:
        """Build the input dict for the LangGraph orchestrator."""
        return {
            "message": message,
            "user_query": message,
            "user_id": str(user_id),
            "conversation_id": str(conversation_id),
            "experiment_id": str(experiment_id) if experiment_id else None,
        }

    async def _invoke_agent(self, agent_input: dict[str, Any]) -> dict[str, Any]:
        """Invoke the LangGraph orchestrator and return its output.

        Falls back to a direct LLM call if the orchestrator module
        is not yet available.
        """
        try:
            from app.agents.orchestrator import get_orchestrator

            orchestrator = get_orchestrator()
            config = {
                "configurable": {
                    "thread_id": str(agent_input.get("conversation_id")),
                    "db": self.db,
                    "redis": self.redis,
                }
            }
            result = await orchestrator.ainvoke(agent_input, config=config)

            # LangGraph returns a dict of node outputs; extract the final state
            if isinstance(result, dict):
                # Try to get the response from the last node
                for key in ("generate_response", "response", "__end__"):
                    if key in result and isinstance(result[key], dict):
                        return result[key]
                # Flatten if it's already the final state
                return result

            return {"response": str(result), "agent_type": "general", "confidence": 0.5}
        except ImportError:
            logger.warning("Orchestrator not available; falling back to direct LLM.")
            return await self._direct_llm_fallback(agent_input)
        except Exception as exc:
            logger.error("Agent invocation failed: %s", exc, exc_info=True)
            raise AgentExecutionError(f"Agent failed: {exc}") from exc

    async def _direct_llm_fallback(
        self, agent_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Use the LLM directly when the orchestrator is unavailable."""
        from app.core.llm import get_llm

        llm = get_llm()
        result = await llm.ainvoke(agent_input["message"])
        return {
            "response": result.content,
            "agent_type": "general",
            "confidence": 0.5,
            "sources": [],
            "safety_alerts": [],
            "next_steps": [],
            "deviation_detected": False,
            "experiment_state": None,
        }
