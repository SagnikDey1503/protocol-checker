"""
Experiment State Management Tools.

LangChain @tool-decorated functions for tracking experiment progress,
updating step state, reading timelines, and detecting protocol deviations.
State is stored in Redis for fast access with TTL-based expiry.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool

from app.config import get_settings
from app.dependencies import get_redis_pool

logger = logging.getLogger(__name__)


def _redis_key(experiment_id: str) -> str:
    """Build the Redis key for experiment state."""
    return f"experiment:{experiment_id}:state"


async def _get_redis():
    """Get a Redis client from the pool (non-FastAPI context)."""
    import redis.asyncio as aioredis

    pool = get_redis_pool()
    return aioredis.Redis(connection_pool=pool)


@tool
async def get_current_step(experiment_id: str) -> dict:
    """Get the current step information for an active experiment.

    Reads the experiment state from Redis working memory to determine
    which step the researcher is currently on, along with step metadata.

    Args:
        experiment_id: UUID string of the experiment session.

    Returns:
        Dict with current_step number, total_steps, status, and step_info.
    """
    logger.info("Getting current step for experiment %s", experiment_id)

    try:
        redis_client = await _get_redis()
        key = _redis_key(experiment_id)
        state_raw = await redis_client.get(key)
        await redis_client.aclose()

        if not state_raw:
            return {
                "current_step": 0,
                "total_steps": 0,
                "status": "not_found",
                "step_info": None,
                "message": f"No active experiment found with ID {experiment_id}",
            }

        state = json.loads(state_raw)
        current = state.get("current_step", 0)
        total = state.get("total_steps", 0)
        steps = state.get("steps", [])

        step_info = None
        if 0 < current <= len(steps):
            step_info = steps[current - 1]

        return {
            "current_step": current,
            "total_steps": total,
            "status": state.get("status", "active"),
            "step_info": step_info,
            "started_at": state.get("started_at"),
            "protocol_title": state.get("protocol_title", "Unknown"),
        }

    except Exception as exc:
        logger.error("Failed to get current step: %s", exc)
        return {
            "current_step": 0,
            "total_steps": 0,
            "status": "error",
            "step_info": None,
            "message": f"Error retrieving experiment state: {exc}",
        }


@tool
async def update_step(
    experiment_id: str,
    step_number: int,
    notes: Optional[str] = None,
    deviation: Optional[str] = None,
) -> dict:
    """Update the current step in an active experiment.

    Records step completion, optional notes, and any deviations from
    the protocol. Updates the experiment timeline in Redis.

    Args:
        experiment_id: UUID string of the experiment session.
        step_number: The step number being completed / moved to.
        notes: Optional researcher notes for this step.
        deviation: Optional description of any protocol deviation.

    Returns:
        Dict confirming the update with previous and new step numbers.
    """
    logger.info(
        "Updating experiment %s to step %d (deviation=%s)",
        experiment_id,
        step_number,
        deviation,
    )

    try:
        redis_client = await _get_redis()
        key = _redis_key(experiment_id)
        state_raw = await redis_client.get(key)

        if not state_raw:
            await redis_client.aclose()
            return {
                "success": False,
                "message": f"No active experiment found with ID {experiment_id}",
            }

        state = json.loads(state_raw)
        previous_step = state.get("current_step", 0)
        state["current_step"] = step_number

        # Record in timeline
        timeline_entry = {
            "step_number": step_number,
            "previous_step": previous_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        if deviation:
            timeline_entry["deviation"] = deviation
            deviations = state.get("deviations", [])
            deviations.append(
                {
                    "step": step_number,
                    "description": deviation,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            state["deviations"] = deviations

        timeline = state.get("timeline", [])
        timeline.append(timeline_entry)
        state["timeline"] = timeline

        # Check if experiment is complete
        if step_number >= state.get("total_steps", 0):
            state["status"] = "completed"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()

        settings = get_settings()
        await redis_client.set(
            key, json.dumps(state), ex=settings.experiment_state_ttl
        )
        await redis_client.aclose()

        return {
            "success": True,
            "previous_step": previous_step,
            "current_step": step_number,
            "status": state.get("status", "active"),
            "deviation_recorded": deviation is not None,
        }

    except Exception as exc:
        logger.error("Failed to update step: %s", exc)
        return {"success": False, "message": f"Error updating step: {exc}"}


@tool
async def get_experiment_timeline(experiment_id: str) -> dict:
    """Get the full timeline of an experiment, including all step transitions.

    Returns a chronological record of every step change, with timestamps,
    notes, and any recorded deviations.

    Args:
        experiment_id: UUID string of the experiment session.

    Returns:
        Dict with timeline list, deviation count, and experiment metadata.
    """
    logger.info("Getting timeline for experiment %s", experiment_id)

    try:
        redis_client = await _get_redis()
        key = _redis_key(experiment_id)
        state_raw = await redis_client.get(key)
        await redis_client.aclose()

        if not state_raw:
            return {
                "experiment_id": experiment_id,
                "timeline": [],
                "deviations": [],
                "message": "No active experiment found",
            }

        state = json.loads(state_raw)
        timeline = state.get("timeline", [])
        deviations = state.get("deviations", [])

        # Compute duration if both start and latest timestamps exist
        duration_seconds = None
        started_at = state.get("started_at")
        if started_at and timeline:
            try:
                start = datetime.fromisoformat(started_at)
                last_ts = timeline[-1].get("timestamp")
                if last_ts:
                    end = datetime.fromisoformat(last_ts)
                    duration_seconds = (end - start).total_seconds()
            except (ValueError, TypeError):
                pass

        return {
            "experiment_id": experiment_id,
            "protocol_title": state.get("protocol_title", "Unknown"),
            "current_step": state.get("current_step", 0),
            "total_steps": state.get("total_steps", 0),
            "status": state.get("status", "active"),
            "timeline": timeline,
            "deviations": deviations,
            "deviation_count": len(deviations),
            "duration_seconds": duration_seconds,
        }

    except Exception as exc:
        logger.error("Failed to get timeline: %s", exc)
        return {
            "experiment_id": experiment_id,
            "timeline": [],
            "deviations": [],
            "message": f"Error: {exc}",
        }


@tool
async def detect_deviation(
    experiment_id: str,
    expected_step: int,
    actual_step: int,
    context: Optional[str] = None,
) -> dict:
    """Detect whether a protocol deviation has occurred.

    Compares the expected next step against the actual step being performed,
    and analyses whether skipping or reordering is safe based on protocol
    dependency information.

    Args:
        experiment_id: UUID string of the experiment session.
        expected_step: The step number that should come next per the protocol.
        actual_step: The step number the researcher is actually performing.
        context: Optional additional context about what the researcher is doing.

    Returns:
        Dict indicating whether a deviation was detected, its severity, and advice.
    """
    logger.info(
        "Checking deviation for experiment %s: expected=%d, actual=%d",
        experiment_id,
        expected_step,
        actual_step,
    )

    deviation_detected = expected_step != actual_step

    if not deviation_detected:
        return {
            "deviation_detected": False,
            "severity": "none",
            "message": f"Step {actual_step} matches expected sequence.",
        }

    # Determine severity based on step gap
    step_gap = abs(actual_step - expected_step)
    if step_gap == 1:
        severity = "low"
        advice = (
            f"Minor deviation: moved to step {actual_step} instead of "
            f"expected step {expected_step}. This may be acceptable if "
            f"the steps are independent."
        )
    elif step_gap <= 3:
        severity = "medium"
        advice = (
            f"Moderate deviation: skipped {step_gap} step(s). Please verify "
            f"that steps {expected_step}–{actual_step - 1 if actual_step > expected_step else actual_step + 1} "
            f"are not prerequisites for step {actual_step}."
        )
    else:
        severity = "high"
        advice = (
            f"Significant deviation: jumped from step {expected_step - 1} to "
            f"step {actual_step} (skipped {step_gap} steps). This could "
            f"compromise the experiment. Check protocol dependencies carefully."
        )

    # Record the deviation in Redis
    try:
        redis_client = await _get_redis()
        key = _redis_key(experiment_id)
        state_raw = await redis_client.get(key)
        if state_raw:
            state = json.loads(state_raw)
            deviations = state.get("deviations", [])
            deviations.append(
                {
                    "expected_step": expected_step,
                    "actual_step": actual_step,
                    "severity": severity,
                    "context": context,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            state["deviations"] = deviations
            settings = get_settings()
            await redis_client.set(
                key, json.dumps(state), ex=settings.experiment_state_ttl
            )
        await redis_client.aclose()
    except Exception as exc:
        logger.error("Failed to record deviation: %s", exc)

    return {
        "deviation_detected": True,
        "severity": severity,
        "expected_step": expected_step,
        "actual_step": actual_step,
        "step_gap": step_gap,
        "advice": advice,
        "context": context,
    }
