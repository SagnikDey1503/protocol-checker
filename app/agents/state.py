"""
LangGraph Agent State Definition.

Defines the shared state schema that flows through the multi-agent graph.
Uses Annotated types with reducer functions (operator.add, add_messages)
so LangGraph knows how to merge partial updates from each node.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph multi-agent orchestrator.

    Attributes:
        messages: LangChain message list (auto-merged via add_messages).
        user_query: The raw user query text for the current turn.
        query_type: Classified intent (maps to QueryType enum values).
        current_protocol: Dict representation of the active protocol, if any.
        current_step: The step number the user is currently on.
        experiment_id: UUID string of the active experiment session.
        session_id: UUID string of the conversation session.
        user_id: UUID string of the authenticated user.
        retrieved_context: RAG-retrieved chunks relevant to the query.
        safety_alerts: Accumulated safety warnings (append-only via operator.add).
        memory_context: Recalled memories and user patterns for personalization.
        response: The final text response to return to the user.
        confidence: Confidence score [0.0–1.0] for the response.
        sources: Accumulated source citations (append-only via operator.add).
        next_steps: Suggested next actions for the user.
        deviation_detected: Whether a protocol deviation was detected.
    """

    messages: Annotated[list, add_messages]
    user_query: str
    query_type: str
    current_protocol: Optional[dict]
    current_step: Optional[int]
    experiment_id: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    retrieved_context: list[dict]
    safety_alerts: Annotated[list[dict], operator.add]
    memory_context: dict
    response: str
    confidence: float
    sources: Annotated[list[dict], operator.add]
    next_steps: list[str]
    deviation_detected: bool
