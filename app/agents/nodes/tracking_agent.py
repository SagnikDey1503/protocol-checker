"""
Experiment Tracking Agent Node.

Tracks experiment progress, detects skipped or out-of-order steps,
predicts the next likely step, updates working memory, and flags
protocol deviations that might compromise the experiment.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

TRACKING_SYSTEM_PROMPT = """\
You are the **Experiment Tracking Agent** — an intelligent lab notebook assistant \
that monitors experiment progress in real time.

## Your Responsibilities
1. **Track step progress**: Know which step the researcher is on and what comes next.
2. **Detect skipped steps**: If the researcher jumps ahead, identify which steps \
were skipped and assess whether they are prerequisites.
3. **Predict next steps**: Based on the current position, suggest what to do next.
4. **Detect deviations**: If the researcher deviates from the protocol \
(wrong concentrations, different timing, skipped steps), flag it clearly.
5. **Record context**: Note any important observations the researcher mentions \
(unexpected results, modifications, timing changes).

## Deviation Detection Rules
- **Minor**: Single step skipped that is likely independent (e.g., optional wash)
- **Moderate**: 2-3 steps skipped, or a timing modification
- **Major**: Critical prerequisite step skipped, wrong reagent used, or unsafe modification
- **Critical**: Action that could compromise safety or render the entire experiment invalid

## Response Format
- Start with a brief status update ("You are on Step X of Y")
- Note any concerns or deviations
- Clearly state the next expected step(s)
- If a deviation is detected, explain the potential impact
- Suggest corrective actions when appropriate

Always be supportive — deviations happen in real research. Focus on helping \
the researcher understand implications rather than being judgmental.
"""


async def tracking_agent_node(state: AgentState) -> dict:
    """Track experiment progress and detect protocol deviations.

    Analyses the user's message in the context of their active experiment,
    detects any step changes or deviations, and provides status updates
    with next-step predictions.

    Args:
        state: Current LangGraph agent state.

    Returns:
        Partial state update with response, next_steps, deviation_detected,
        and safety_alerts if a deviation is found.
    """
    logger.info("Tracking agent processing: %s", state["user_query"][:80])

    try:
        llm = get_llm()
        user_query = state["user_query"]
        current_protocol = state.get("current_protocol")
        current_step = state.get("current_step")
        experiment_id = state.get("experiment_id")
        retrieved_context = state.get("retrieved_context", [])
        memory_context = state.get("memory_context", {})

        # Build experiment status block
        experiment_info = "## Experiment Status\n"
        if current_protocol:
            experiment_info += (
                f"- Protocol: {current_protocol.get('title', 'Unknown')}\n"
                f"- Total Steps: {current_protocol.get('step_count', 0)}\n"
                f"- Current Step: {current_step or 0}\n"
                f"- Experiment ID: {experiment_id or 'Not set'}\n"
            )
            # Include step details from protocol
            steps = current_protocol.get("steps", [])
            if steps:
                experiment_info += "\n### Protocol Steps:\n"
                for i, step in enumerate(steps[:20], 1):  # Limit to 20 steps
                    marker = "→ " if i == current_step else "  "
                    step_text = step.get("text", step) if isinstance(step, dict) else str(step)
                    experiment_info += f"{marker}Step {i}: {str(step_text)[:120]}\n"
        else:
            experiment_info += "- No active protocol loaded\n"

        # Include timeline / previous deviations from memory
        previous_deviations = ""
        if memory_context:
            experiment_memories = memory_context.get("experiment_state", {})
            devs = experiment_memories.get("deviations", [])
            if devs:
                previous_deviations = "\n### Previous Deviations:\n"
                for d in devs[-5:]:
                    previous_deviations += (
                        f"- Step {d.get('step', '?')}: {d.get('description', 'unknown')}\n"
                    )

        # Include retrieved context
        context_text = ""
        if retrieved_context:
            context_parts = []
            for chunk in retrieved_context[:5]:
                context_parts.append(chunk.get("text", ""))
            context_text = "\n\n## Relevant Protocol Context\n" + "\n---\n".join(context_parts)

        user_message = (
            f"{experiment_info}"
            f"{previous_deviations}"
            f"{context_text}\n\n"
            f"## Researcher's Message\n{user_query}"
        )

        messages = [
            SystemMessage(content=TRACKING_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Ask the LLM to respond AND assess deviation in structured format
        analysis_prompt = (
            f"\n\nAfter your response, provide a JSON block on a new line "
            f"tagged with ```json\n that contains:\n"
            f'{{"deviation_detected": true/false, "deviation_severity": '
            f'"none|minor|moderate|major|critical", "new_step": <int or null>, '
            f'"skipped_steps": [<int>], "next_predicted_step": <int>}}\n```'
        )
        messages[-1] = HumanMessage(content=user_message + analysis_prompt)

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse structured output from response
        deviation_detected = False
        safety_alerts: list[dict] = []
        next_steps: list[str] = []
        clean_response = response_text

        # Try to extract JSON block
        if "```json" in response_text:
            try:
                json_start = response_text.index("```json") + 7
                json_end = response_text.index("```", json_start)
                json_str = response_text[json_start:json_end].strip()
                analysis = json.loads(json_str)

                deviation_detected = analysis.get("deviation_detected", False)
                severity = analysis.get("deviation_severity", "none")
                new_step = analysis.get("new_step")
                skipped = analysis.get("skipped_steps", [])
                next_pred = analysis.get("next_predicted_step")

                # Remove JSON block from visible response
                clean_response = response_text[:response_text.index("```json")].strip()

                # Build next steps
                if next_pred:
                    next_steps.append(f"Proceed to Step {next_pred}")
                if skipped:
                    next_steps.append(
                        f"Consider completing skipped step(s): {', '.join(str(s) for s in skipped)}"
                    )

                # Generate safety alerts for deviations
                if deviation_detected and severity in ("major", "critical"):
                    safety_alerts.append(
                        {
                            "level": "high" if severity == "major" else "critical",
                            "message": (
                                f"Protocol deviation detected ({severity}): "
                                f"{'Skipped steps: ' + str(skipped) if skipped else 'Unexpected step change'}. "
                                f"This may affect experiment validity."
                            ),
                            "related_step": new_step,
                        }
                    )
                elif deviation_detected:
                    safety_alerts.append(
                        {
                            "level": "medium" if severity == "moderate" else "low",
                            "message": (
                                f"Minor protocol deviation noted ({severity}). "
                                f"Review to ensure experiment integrity."
                            ),
                            "related_step": new_step,
                        }
                    )
            except (ValueError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not parse tracking analysis JSON: %s", exc)
                clean_response = response_text

        if not next_steps:
            if current_step is not None and current_protocol:
                total = current_protocol.get("step_count", 0)
                if current_step < total:
                    next_steps.append(f"Continue to Step {current_step + 1}")
                else:
                    next_steps.append("Experiment complete — review your results")
            else:
                next_steps.append("Provide an update on your current progress")

        logger.info(
            "Tracking agent completed. Deviation: %s, Alerts: %d",
            deviation_detected,
            len(safety_alerts),
        )

        return {
            "response": clean_response,
            "next_steps": next_steps,
            "deviation_detected": deviation_detected,
            "safety_alerts": safety_alerts,
            "messages": [AIMessage(content=clean_response)],
        }

    except Exception as exc:
        logger.error("Tracking agent failed: %s", exc, exc_info=True)
        error_response = (
            "I had trouble tracking your experiment progress. "
            "Please provide your current step number and any observations."
        )
        return {
            "response": error_response,
            "next_steps": ["Tell me which step you're on"],
            "deviation_detected": False,
            "safety_alerts": [],
            "messages": [AIMessage(content=error_response)],
        }
