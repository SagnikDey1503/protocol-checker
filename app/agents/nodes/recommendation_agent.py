"""
Recommendation Agent Node.

Recommends appropriate experimental protocols from the user's protocol database
or the pre-loaded library based on the user's description, available equipment,
and skill level.
"""

from __future__ import annotations

import logging
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

RECOMMENDATION_SYSTEM_PROMPT = """\
You are the **Protocol Recommendation Agent** — a helpful lab guide.
Your purpose is to suggest the best protocols for a researcher's goal, available materials,
and experience level.

## Your Expertise
- Matching user goals (e.g. "I want to extract plasmid DNA") to standard protocols
- Filtering by available equipment (e.g. standard thermal cycler, microcentrifuge)
- Sourcing preloaded protocols (PCR, transformation, gel electrophoresis, etc.)
- Tailoring recommendations by difficulty (beginner vs. advanced)

## Behaviour Rules
1. **Be practical** — highlight materials and equipment required for each recommendation.
2. **Present options** — offer 1-3 protocol options if applicable, explaining the difference (e.g. mini-prep vs. phenol-chloroform extraction).
3. **Assess difficulty** — indicate whether the protocol is suitable for high school, undergrad, or advanced lab environments.
4. **Link to action** — explain how to activate or upload the protocol to start an experiment session.
"""


async def recommendation_agent_node(state: AgentState) -> dict:
    """Recommends protocols matching the user query."""
    logger.info("Recommendation agent processing query: %s", state["user_query"][:80])

    try:
        llm = get_llm()
        user_query = state["user_query"]
        retrieved_context = state.get("retrieved_context", [])
        memory_context = state.get("memory_context", {})

        # Context formatting
        context_parts = []
        sources = []
        for i, chunk in enumerate(retrieved_context):
            chunk_text = chunk.get("text", "")
            chunk_meta = chunk.get("metadata", {})
            chunk_id = chunk.get("id", f"chunk_{i}")

            context_parts.append(
                f"[Source {i + 1}] (score: {chunk.get('score', 0.0):.2f})\n"
                f"Protocol: {chunk_meta.get('protocol_title', 'Database Protocol')}\n"
                f"Details: {chunk_text}"
            )
            sources.append({
                "chunk_id": chunk_id,
                "text": chunk_text[:300],
                "protocol_title": chunk_meta.get("protocol_title"),
                "relevance_score": chunk.get("score", 0.0),
            })

        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No specific protocols found in database. Rely on internal library."

        # User Profile / Skill level from memory
        profile_info = ""
        if memory_context:
            profile = memory_context.get("user_profile", {})
            if profile:
                profile_info = (
                    f"\n## User Profile\n"
                    f"- Total experiments: {profile.get('total_experiments', 0)}\n"
                    f"- Completed: {profile.get('completed_experiments', 0)}\n"
                    f"- Logged patterns: {profile.get('patterns', [])}\n"
                )

        user_message_content = (
            f"{profile_info}"
            f"\n## Retrieved Protocols\n{context_block}\n\n"
            f"## User Goal/Description\n{user_query}"
        )

        messages = [
            SystemMessage(content=RECOMMENDATION_SYSTEM_PROMPT),
            *state.get("messages", []),
            HumanMessage(content=user_message_content)
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        next_steps = [
            "Start an experiment session with a recommended protocol",
            "Show me a list of required reagents for the suggested protocol",
            "Help me upload a custom protocol PDF instead"
        ]

        return {
            "response": response_text,
            "sources": sources,
            "confidence": 0.85 if retrieved_context else 0.6,
            "next_steps": next_steps,
            "messages": [AIMessage(content=response_text)],
        }

    except Exception as exc:
        logger.error("Recommendation agent failed: %s", exc, exc_info=True)
        error_response = "I couldn't complete the protocol recommendations. You can view our pre-loaded protocols by asking to list common protocols."
        return {
            "response": error_response,
            "sources": [],
            "confidence": 0.0,
            "next_steps": ["List pre-loaded protocols", "Rephrase your request"],
            "messages": [AIMessage(content=error_response)],
        }
