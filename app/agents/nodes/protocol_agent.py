"""
Protocol Understanding Agent Node.

Specialises in explaining uploaded protocol PDFs — step details,
dependencies between steps, reagent preparation, timing, and
providing clear, contextual guidance for each phase of a protocol.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, SystemMessage

from app.agents.state import AgentState
from app.core.exceptions import AgentExecutionError
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

PROTOCOL_SYSTEM_PROMPT = """\
You are the **Protocol Understanding Agent** — an expert lab assistant that \
helps researchers understand and follow experimental protocols.

## Your Expertise
- Interpreting protocol PDF content (steps, notes, reagent lists, timing)
- Explaining step-by-step procedures in clear, actionable language
- Identifying dependencies between protocol steps (e.g., "Step 5 requires \
the incubation from Step 3 to be complete")
- Clarifying ambiguous protocol language
- Highlighting critical parameters (temperatures, concentrations, timing)
- Warning about common mistakes at specific steps

## Behaviour Rules
1. **Always cite your sources** — reference the protocol section, step number, \
or page when providing information.
2. **Be specific** — include exact values (temperatures, volumes, times) rather \
than vague instructions.
3. **Identify prerequisites** — when explaining a step, mention what must be \
prepared or completed first.
4. **Flag safety notes** — if a step involves hazardous materials, mention \
relevant precautions inline.
5. **Suggest next steps** — after answering, suggest 1-3 logical next actions.
6. If you don't have enough context from the protocol, say so clearly and \
suggest what additional information would help.

## Response Format
- Use clear headings and bullet points for readability.
- Include the step number(s) you are referencing.
- When listing reagents or materials, include concentrations and volumes.
"""


async def protocol_agent_node(state: AgentState) -> dict:
    """Process a protocol-related query using RAG context and LLM reasoning.

    Receives the shared agent state (including retrieved_context from the
    RAG pipeline), constructs a specialised prompt with protocol context,
    and generates a detailed, cited response.

    Args:
        state: Current LangGraph agent state.

    Returns:
        Partial state update with response, sources, confidence, and next_steps.
    """
    logger.info("Protocol agent processing query: %s", state["user_query"][:80])

    try:
        llm = get_llm()
        user_query = state["user_query"]
        retrieved_context = state.get("retrieved_context", [])
        current_protocol = state.get("current_protocol")
        current_step = state.get("current_step")
        memory_context = state.get("memory_context", {})

        # Build context block from retrieved chunks
        context_parts: list[str] = []
        sources: list[dict] = []

        for i, chunk in enumerate(retrieved_context):
            chunk_text = chunk.get("text", "")
            chunk_meta = chunk.get("metadata", {})
            chunk_id = chunk.get("id", f"chunk_{i}")

            context_parts.append(
                f"[Source {i + 1}] (score: {chunk.get('score', 0.0):.2f})\n"
                f"Protocol: {chunk_meta.get('protocol_title', 'Unknown')}\n"
                f"Section: {chunk_meta.get('section', 'N/A')} | "
                f"Step: {chunk_meta.get('step_number', 'N/A')}\n"
                f"Content: {chunk_text}"
            )
            sources.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text[:300],
                    "protocol_title": chunk_meta.get("protocol_title"),
                    "page_number": chunk_meta.get("page_number"),
                    "step_number": chunk_meta.get("step_number"),
                    "relevance_score": chunk.get("score", 0.0),
                }
            )

        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No protocol context retrieved."

        # Build active-protocol info
        protocol_info = ""
        if current_protocol:
            protocol_info = (
                f"\n## Active Protocol\n"
                f"- Title: {current_protocol.get('title', 'Unknown')}\n"
                f"- Total Steps: {current_protocol.get('step_count', 'Unknown')}\n"
                f"- Current Step: {current_step or 'Not started'}\n"
                f"- Experiment Type: {current_protocol.get('experiment_type', 'Unknown')}\n"
            )

        # Build memory info
        memory_info = ""
        if memory_context:
            memories = memory_context.get("relevant_memories", [])
            if memories:
                mem_lines = [f"- {m.get('content', '')}" for m in memories[:5]]
                memory_info = f"\n## Relevant History\n" + "\n".join(mem_lines) + "\n"

        # Compose the user message
        user_message_content = (
            f"{protocol_info}"
            f"{memory_info}"
            f"\n## Retrieved Protocol Context\n{context_block}\n\n"
            f"## User Question\n{user_query}"
        )

        messages = [
            SystemMessage(content=PROTOCOL_SYSTEM_PROMPT),
            *state.get("messages", []),
        ]
        # We put the enriched content as a new user-style message
        from langchain_core.messages import HumanMessage

        messages.append(HumanMessage(content=user_message_content))

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Estimate confidence based on context quality
        if retrieved_context:
            top_score = max(c.get("score", 0.0) for c in retrieved_context)
            confidence = min(top_score + 0.1, 1.0)
        else:
            confidence = 0.4  # lower confidence without context

        # Suggest next steps
        next_steps: list[str] = []
        if current_step is not None and current_protocol:
            total = current_protocol.get("step_count", 0)
            if current_step < total:
                next_steps.append(f"Ask about Step {current_step + 1}")
            next_steps.append("Ask about required reagents for the next step")
        if not next_steps:
            next_steps = [
                "Ask a follow-up question about this protocol",
                "Upload a protocol PDF if you haven't already",
                "Start an experiment session to track your progress",
            ]

        logger.info("Protocol agent completed. Confidence: %.2f, Sources: %d", confidence, len(sources))

        return {
            "response": response_text,
            "sources": sources,
            "confidence": confidence,
            "next_steps": next_steps,
            "messages": [AIMessage(content=response_text)],
        }

    except Exception as exc:
        logger.error("Protocol agent failed: %s", exc, exc_info=True)
        error_response = (
            "I encountered an error while processing your protocol question. "
            "Please try rephrasing your question or ensure a protocol has been uploaded."
        )
        return {
            "response": error_response,
            "sources": [],
            "confidence": 0.0,
            "next_steps": ["Try rephrasing your question", "Upload a protocol PDF"],
            "messages": [AIMessage(content=error_response)],
        }
