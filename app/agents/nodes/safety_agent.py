"""
Safety Agent Node.

Monitors conversations for safety concerns — hazardous reagent mentions,
unsafe mixing, contamination risks, PPE gaps, and critical protocol steps.
Contains a comprehensive SAFETY_DATABASE for common lab reagents.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.agents.tools.safety_tools import SAFETY_DATABASE
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

SAFETY_SYSTEM_PROMPT = """\
You are the **Safety Agent** — a vigilant lab safety officer embedded in the \
research assistant. Your mission is to keep researchers safe.

## Your Responsibilities
1. **Identify hazardous reagents** mentioned in the conversation and provide \
appropriate safety warnings.
2. **Detect dangerous combinations**: Flag mixing of incompatible chemicals.
3. **Check contamination risks**: Warn about cross-contamination scenarios.
4. **PPE reminders**: Ensure researchers are wearing appropriate protective equipment.
5. **Procedure safety**: Flag unsafe procedures (e.g., mouth pipetting, \
improper waste disposal, heating sealed containers).

## Hazard Classification
- **LOW**: Minor irritants, standard PPE sufficient
- **MEDIUM**: Significant irritants or sensitizers, additional precautions needed
- **HIGH**: Toxic, corrosive, or carcinogenic — fume hood and extra PPE required
- **CRITICAL**: Immediately dangerous — stop and verify procedures

## Response Rules
1. If no safety concerns are found, provide a brief "all clear" confirmation.
2. If concerns exist, be **specific and actionable** — don't just say "be careful".
3. Always include **required PPE** for the materials being used.
4. For HIGH/CRITICAL hazards, clearly mark them with ⚠️ or 🚨.
5. Never downplay safety risks — err on the side of caution.
6. Include first-aid information for HIGH/CRITICAL reagents.

## Known Hazardous Reagents Database
{reagent_info}
"""

# Pre-build the reagent reference for the system prompt
_reagent_lines: list[str] = []
for name, info in SAFETY_DATABASE.items():
    _reagent_lines.append(
        f"- **{name.title()}**: {', '.join(info['hazards'])} "
        f"(Risk: {info['risk_level'].upper()})"
    )
REAGENT_REFERENCE = "\n".join(_reagent_lines)


def _detect_reagents_in_text(text: str) -> list[str]:
    """Scan text for mentions of known hazardous reagents.

    Returns a list of matched reagent names from the SAFETY_DATABASE.
    """
    text_lower = text.lower()
    found: list[str] = []
    for reagent_name in SAFETY_DATABASE:
        # Match whole word or hyphenated form
        pattern = r"\b" + re.escape(reagent_name) + r"\b"
        if re.search(pattern, text_lower):
            found.append(reagent_name)
    return found


async def safety_agent_node(state: AgentState) -> dict:
    """Analyse the conversation for safety concerns and generate alerts.

    Scans the user query and retrieved context for hazardous reagent
    mentions, checks for incompatible combinations, and uses the LLM
    to identify procedural safety risks.

    Args:
        state: Current LangGraph agent state.

    Returns:
        Partial state update with safety_alerts list and response text.
    """
    logger.info("Safety agent processing: %s", state["user_query"][:80])

    try:
        llm = get_llm()
        user_query = state["user_query"]
        retrieved_context = state.get("retrieved_context", [])
        current_protocol = state.get("current_protocol")
        current_step = state.get("current_step")

        # Collect all text to scan for reagent mentions
        all_text = user_query
        for chunk in retrieved_context:
            all_text += " " + chunk.get("text", "")
        if current_protocol:
            reagents = current_protocol.get("reagents", [])
            if reagents:
                all_text += " " + " ".join(str(r) for r in reagents)

        # Detect known hazardous reagents
        detected_reagents = _detect_reagents_in_text(all_text)
        logger.info("Detected hazardous reagents: %s", detected_reagents)

        # Build safety alerts from database lookups
        safety_alerts: list[dict] = []
        reagent_warnings: list[str] = []

        for reagent_name in detected_reagents:
            info = SAFETY_DATABASE[reagent_name]
            risk = info["risk_level"]

            reagent_warnings.append(
                f"**{reagent_name.title()}** (Risk: {risk.upper()}):\n"
                f"  - Hazards: {', '.join(info['hazards'])}\n"
                f"  - PPE: {', '.join(info['ppe'])}\n"
                f"  - Handling: {info['handling']}"
            )

            if risk in ("high", "critical"):
                safety_alerts.append(
                    {
                        "level": risk,
                        "message": (
                            f"{'🚨' if risk == 'critical' else '⚠️'} "
                            f"{reagent_name.title()}: {', '.join(info['hazards'])}. "
                            f"Required PPE: {', '.join(info['ppe'])}. "
                            f"{info['handling']}"
                        ),
                        "reagents": [reagent_name],
                        "related_step": current_step,
                    }
                )

        # Check for dangerous combinations
        if len(detected_reagents) >= 2:
            for i, reagent_a in enumerate(detected_reagents):
                for reagent_b in detected_reagents[i + 1 :]:
                    info_a = SAFETY_DATABASE[reagent_a]
                    info_b = SAFETY_DATABASE[reagent_b]
                    # Check cross-incompatibilities
                    for incompat in info_a.get("incompatible_with", []):
                        if reagent_b in incompat.lower() or incompat.lower() in reagent_b:
                            safety_alerts.append(
                                {
                                    "level": "high",
                                    "message": (
                                        f"⚠️ INCOMPATIBILITY: {reagent_a.title()} is "
                                        f"incompatible with {reagent_b.title()} "
                                        f"(category: {incompat}). Do NOT mix directly."
                                    ),
                                    "reagents": [reagent_a, reagent_b],
                                    "related_step": current_step,
                                }
                            )

        # Use LLM for deeper procedural safety analysis
        safety_prompt = SAFETY_SYSTEM_PROMPT.format(reagent_info=REAGENT_REFERENCE)

        context_text = ""
        if retrieved_context:
            context_text = "\n".join(c.get("text", "")[:200] for c in retrieved_context[:3])

        reagent_warnings_text = "\n\n".join(reagent_warnings) if reagent_warnings else "No known hazardous reagents detected."

        user_message = (
            f"## Detected Reagents\n{reagent_warnings_text}\n\n"
            f"## Protocol Context\n{context_text or 'No context available'}\n\n"
            f"## Current Step: {current_step or 'Unknown'}\n\n"
            f"## User Message\n{user_query}\n\n"
            f"Please analyse for safety concerns. If the user is specifically "
            f"asking about safety, provide a comprehensive safety briefing. "
            f"If no safety concerns are found, confirm that the procedure "
            f"appears safe with current information."
        )

        messages = [
            SystemMessage(content=safety_prompt),
            HumanMessage(content=user_message),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # If no explicit alerts were generated but LLM identified concerns,
        # add a general advisory alert
        if not safety_alerts and any(
            keyword in response_text.lower()
            for keyword in ["warning", "caution", "danger", "⚠️", "🚨", "hazard"]
        ):
            safety_alerts.append(
                {
                    "level": "medium",
                    "message": "Safety considerations identified — review the response for details.",
                    "related_step": current_step,
                }
            )

        logger.info("Safety agent completed. Alerts: %d", len(safety_alerts))

        return {
            "safety_alerts": safety_alerts,
            "response": response_text,
            "messages": [AIMessage(content=response_text)],
        }

    except Exception as exc:
        logger.error("Safety agent failed: %s", exc, exc_info=True)
        fallback = (
            "I was unable to complete the safety analysis. As a precaution, "
            "please wear appropriate PPE (gloves, goggles, lab coat) and "
            "consult the relevant Safety Data Sheets (SDS) for all reagents."
        )
        return {
            "safety_alerts": [
                {
                    "level": "medium",
                    "message": "Safety analysis incomplete — exercise extra caution.",
                }
            ],
            "response": fallback,
            "messages": [AIMessage(content=fallback)],
        }
