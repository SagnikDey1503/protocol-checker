"""
Troubleshooting Agent Node.

Diagnoses experimental problems by matching symptoms against common
error patterns, retrieving relevant troubleshooting knowledge via RAG,
and suggesting structured recovery steps.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

TROUBLESHOOTING_SYSTEM_PROMPT = """\
You are the **Troubleshooting Agent** — an expert lab diagnostician who helps \
researchers identify and solve experimental problems.

## Your Expertise
- Diagnosing failed experiments (no bands on gel, low yield, contamination)
- Identifying common error patterns in molecular biology protocols
- Suggesting systematic debugging approaches
- Recommending recovery steps when experiments go wrong
- Differentiating between protocol errors, reagent issues, and equipment problems

## Common Error Patterns You Know
1. **PCR failures**: No bands, non-specific bands, primer dimers, smearing
2. **Gel electrophoresis issues**: Smiling, fuzzy bands, no migration, wrong sizes
3. **Cloning problems**: No colonies, all-background colonies, wrong inserts
4. **Protein expression**: No expression, inclusion bodies, degradation
5. **Cell culture**: Contamination, poor growth, morphology changes
6. **RNA work**: Degradation, low yield, genomic DNA contamination
7. **Western blot**: No signal, high background, unexpected bands
8. **Spectrophotometry**: Inconsistent readings, abnormal ratios (260/280, 260/230)

## Diagnostic Approach
1. **Gather symptoms**: Ask clarifying questions if needed
2. **Form hypotheses**: List 2-4 most likely causes, ranked by probability
3. **Suggest diagnostics**: Quick tests to distinguish between causes
4. **Recommend solutions**: Step-by-step recovery actions for each hypothesis
5. **Preventive advice**: How to avoid the issue next time

## Response Format
- Use a structured diagnostic format:
  - **Symptoms**: Brief summary of the problem
  - **Most Likely Causes**: Numbered list with probability estimates
  - **Diagnostic Steps**: How to narrow down the cause
  - **Recommended Actions**: Concrete steps to fix the issue
  - **Prevention**: How to avoid this in the future
- Be specific with concentrations, temperatures, and times in your suggestions.
- Reference the specific protocol step where the issue might have originated.
"""


async def troubleshooting_agent_node(state: AgentState) -> dict:
    """Diagnose experimental problems and suggest recovery steps.

    Uses RAG-retrieved context and the LLM to match symptoms against
    known error patterns and provide structured troubleshooting guidance.

    Args:
        state: Current LangGraph agent state.

    Returns:
        Partial state update with response, sources, and confidence.
    """
    logger.info("Troubleshooting agent processing: %s", state["user_query"][:80])

    try:
        llm = get_llm()
        user_query = state["user_query"]
        retrieved_context = state.get("retrieved_context", [])
        current_protocol = state.get("current_protocol")
        current_step = state.get("current_step")
        memory_context = state.get("memory_context", {})

        # Build context from RAG results
        context_parts: list[str] = []
        sources: list[dict] = []
        for i, chunk in enumerate(retrieved_context):
            chunk_text = chunk.get("text", "")
            chunk_meta = chunk.get("metadata", {})
            chunk_id = chunk.get("id", f"chunk_{i}")

            context_parts.append(
                f"[Source {i + 1}] {chunk_meta.get('protocol_title', 'Unknown')}: "
                f"{chunk_text[:300]}"
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

        context_block = "\n\n".join(context_parts) if context_parts else "No relevant troubleshooting context found."

        # Experiment context
        experiment_info = ""
        if current_protocol:
            experiment_info = (
                f"\n## Current Experiment\n"
                f"- Protocol: {current_protocol.get('title', 'Unknown')}\n"
                f"- Type: {current_protocol.get('experiment_type', 'Unknown')}\n"
                f"- Current Step: {current_step or 'Unknown'}\n"
            )

        # Previous issues from memory
        history_info = ""
        if memory_context:
            past_issues = memory_context.get("past_issues", [])
            if past_issues:
                history_info = "\n## Previous Issues\n"
                for issue in past_issues[-3:]:
                    history_info += f"- {issue.get('content', '')}\n"

        user_message = (
            f"{experiment_info}"
            f"{history_info}"
            f"\n## Relevant Knowledge Base\n{context_block}\n\n"
            f"## Problem Description\n{user_query}\n\n"
            f"Please diagnose this problem using your structured diagnostic approach."
        )

        messages = [
            SystemMessage(content=TROUBLESHOOTING_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Estimate confidence
        if retrieved_context:
            avg_score = sum(c.get("score", 0.0) for c in retrieved_context) / len(retrieved_context)
            confidence = min(avg_score + 0.15, 0.95)
        else:
            confidence = 0.5  # reasonable baseline for LLM-only diagnosis

        logger.info("Troubleshooting agent completed. Confidence: %.2f", confidence)

        return {
            "response": response_text,
            "sources": sources,
            "confidence": confidence,
            "messages": [AIMessage(content=response_text)],
        }

    except Exception as exc:
        logger.error("Troubleshooting agent failed: %s", exc, exc_info=True)
        error_response = (
            "I encountered an error while diagnosing your problem. "
            "Please describe the issue in detail — include what you expected "
            "to happen vs. what actually happened, and any error messages."
        )
        return {
            "response": error_response,
            "sources": [],
            "confidence": 0.0,
            "messages": [AIMessage(content=error_response)],
        }
