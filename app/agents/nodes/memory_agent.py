"""
Memory Agent Node.

Manages episodic and semantic memory recall (during input processing) and memory
saving (after response generation) using the unified MemoryManager.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.llm import get_fast_llm
from app.dependencies import get_pinecone_index
from app.memory import (
    MemoryManager,
    WorkingMemory,
    PersistentMemory,
    SemanticMemory,
    EpisodicMemoryStore,
)
from app.models.enums import MemoryType, EpisodeType

logger = logging.getLogger(__name__)


def _get_memory_manager(config: dict[str, Any]) -> MemoryManager:
    """Helper to initialize MemoryManager from request-scoped DB/Redis connections."""
    configurable = config.get("configurable", {})
    db = configurable.get("db")
    redis = configurable.get("redis")
    
    if not db or not redis:
        raise ValueError("Database and Redis clients must be provided in config['configurable']")
    
    pinecone_index = get_pinecone_index()
    
    working = WorkingMemory(redis)
    persistent = PersistentMemory(db)
    semantic = SemanticMemory(pinecone_index)
    episodic = EpisodicMemoryStore(db, semantic)
    
    return MemoryManager(working, persistent, semantic, episodic)


async def memory_recall_node(state: AgentState, config: dict[str, Any]) -> dict:
    """
    Recall relevant semantic/episodic memories and user profile info.

    Runs BEFORE main agent execution to populate memory_context.
    """
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    user_query = state.get("user_query")

    if not user_id or not user_query:
        return {"memory_context": {}}

    logger.info("Memory Agent recalling context for user %s", user_id)

    try:
        manager = _get_memory_manager(config)
        context = await manager.get_full_context(
            session_id=session_id or "",
            user_id=user_id,
            query=user_query,
        )
        return {"memory_context": context}
    except Exception as exc:
        logger.error("Failed to recall memories in memory_recall_node: %s", exc)
        return {"memory_context": {}}


async def memory_save_node(state: AgentState, config: dict[str, Any]) -> dict:
    """
    Analyze the current turn and save significant facts/insights to episodic or semantic memory.

    Runs AFTER response generation.
    """
    user_id = state.get("user_id")
    user_query = state.get("user_query")
    response = state.get("response")

    if not user_id or not user_query or not response:
        return {}

    logger.info("Memory Agent checking for facts to remember for user %s", user_id)

    try:
        manager = _get_memory_manager(config)
        llm = get_fast_llm()

        # Ask LLM if there is anything worth remembering
        system_prompt = (
            "You are a memory retention agent. Analyze the laboratory dialogue and extract any critical facts to remember long-term.\n"
            "Look for:\n"
            "- User preferences, habits, or skill level\n"
            "- Important experiment observations (e.g. 'the gel was smiling', 'got 50ng/ul DNA')\n"
            "- Mistakes or protocol deviations (e.g. 'I added buffer twice', 'skipped wash step')\n"
            "- Reagent or equipment configurations\n"
            "\n"
            "Format your response as a JSON object with keys:\n"
            "- 'should_remember': true or false\n"
            "- 'fact_to_remember': clear, concise summary of the fact (e.g., 'User skipped the 70% ethanol wash during plasmid extraction')\n"
            "- 'memory_type': 'episodic' (for mistakes/experiment specific events) or 'semantic' (for general user preferences/facts)\n"
            "- 'episode_type': 'mistake', 'success', 'learning', or 'deviation' (null if memory_type is semantic)\n"
            "\n"
            "Do NOT add conversational text. Return ONLY the JSON object."
        )

        user_content = f"User Query: {user_query}\nAssistant Response: {response}"
        
        analysis = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])

        import json
        analysis_text = analysis.content.strip()
        if analysis_text.startswith("```json"):
            analysis_text = analysis_text[7:]
        if analysis_text.endswith("```"):
            analysis_text = analysis_text[:-3]
        
        data = json.loads(analysis_text.strip())

        if data.get("should_remember") and data.get("fact_to_remember"):
            fact = data["fact_to_remember"]
            m_type = data.get("memory_type", "semantic")
            ep_type = data.get("episode_type")

            metadata = {"source": "memory_agent"}
            if m_type == "episodic" and ep_type:
                metadata["episode_type"] = ep_type

            await manager.remember(
                content=fact,
                memory_type=m_type,
                user_id=user_id,
                metadata=metadata,
                importance=0.7 if ep_type in ["mistake", "deviation"] else 0.5,
            )
            logger.info("Memory Agent saved new long-term memory: %s", fact)

        # Trigger conversation compression in working memory if needed
        session_id = state.get("session_id")
        if session_id:
            await manager.summarize_and_compress(session_id, user_id)

    except Exception as exc:
        logger.error("Failed to save memory in memory_save_node: %s", exc)

    return {}


# Legacy alias
async def memory_agent_node(state: AgentState, config: dict[str, Any]) -> dict:
    """Wrapper that delegates to memory_save_node by default."""
    return await memory_save_node(state, config)
