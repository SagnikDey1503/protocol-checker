"""
Research Assistant Agent Node.

Answers conceptual questions, explains biological mechanisms, chemical reactions,
and laboratory techniques using a combination of retrieved knowledge base context
and general scientific LLM reasoning.
"""

from __future__ import annotations

import logging
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
You are the **Research Assistant Agent** — a knowledgeable scientific advisor.
Your role is to explain biological principles, experimental mechanics, molecular pathways,
and laboratory procedures to students and lab technicians.

## Your Expertise
- Explaining scientific concepts (e.g. "How does Taq polymerase work?", "What is the role of SDS in SDS-PAGE?")
- Clarifying the chemical basis of steps (e.g. why ethanol precipitates DNA, how buffers maintain pH)
- Comparing different techniques (e.g. Sanger sequencing vs. NGS, PCR vs. qPCR)
- Advising on experimental design (controls, duplicate samples, troubleshooting theory)

## Behaviour Rules
1. **Explain the 'Why'** — do not just state what to do. Explain the underlying biological or chemical mechanism.
2. **Be pedagogically clear** — break down complex pathways into step-by-step molecular processes.
3. **Use analogies** where helpful for beginners, but maintain scientific accuracy.
4. **Be objective** — present limitations, advantages, and alternative approaches.
5. **Always cite your sources** if retrieving information from the knowledge base.
"""


async def research_agent_node(state: AgentState) -> dict:
    """Answers general or concept-oriented biology and lab technique questions.

    Uses retrieved_context (if present) and LLM knowledge.
    """
    logger.info("Research agent processing query: %s", state["user_query"][:80])

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
                f"Reference: {chunk_meta.get('title', 'Knowledge Base')}\n"
                f"Content: {chunk_text}"
            )
            sources.append({
                "chunk_id": chunk_id,
                "text": chunk_text[:300],
                "title": chunk_meta.get("title", "Scientific Reference"),
                "relevance_score": chunk.get("score", 0.0),
            })

        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No specific research context retrieved."

        # Memories
        memory_info = ""
        if memory_context:
            memories = memory_context.get("relevant_memories", [])
            if memories:
                mem_lines = [f"- {m.get('content', '')}" for m in memories[:3]]
                memory_info = f"\n## Relevant User History\n" + "\n".join(mem_lines) + "\n"

        user_message_content = (
            f"{memory_info}"
            f"\n## Retrieved Scientific Context\n{context_block}\n\n"
            f"## User Query\n{user_query}"
        )

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            *state.get("messages", []),
            HumanMessage(content=user_message_content)
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Determine confidence
        confidence = 0.8
        if retrieved_context:
            top_score = max(c.get("score", 0.0) for c in retrieved_context)
            confidence = min(top_score + 0.15, 1.0)

        next_steps = [
            "Ask about the chemical mechanism of a specific step",
            "Ask about troubleshooting patterns for this procedure",
            "Start tracking an experiment using an uploaded protocol"
        ]

        return {
            "response": response_text,
            "sources": sources,
            "confidence": confidence,
            "next_steps": next_steps,
            "messages": [AIMessage(content=response_text)],
        }

    except Exception as exc:
        logger.error("Research agent failed: %s", exc, exc_info=True)
        error_response = "I encountered an error while researching your query. Let me try answering using my general scientific knowledge instead."
        return {
            "response": error_response,
            "sources": [],
            "confidence": 0.5,
            "next_steps": ["Ask about a different concept", "Rephrase your question"],
            "messages": [AIMessage(content=error_response)],
        }
