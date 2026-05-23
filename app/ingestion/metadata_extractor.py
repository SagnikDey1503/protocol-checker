"""
Metadata extraction for the Research Protocol Assistant chunking pipeline.

Extracts structured metadata (reagents, equipment, temperature, timing, safety levels,
and step dependencies) from raw text chunks using a combination of regex matching
and LLM-based processing via Gemini (Flash).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.llm import get_fast_llm
from app.ingestion.chunker import DocumentChunk
from app.models.enums import SafetyLevel

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Enriches DocumentChunks with structured metadata.

    Uses ChatGoogleGenerativeAI (Gemini Flash) to extract reagents, equipment, safety levels,
    and dependencies, combined with local regex fallback for physical parameters
    like temperature and timing.
    """

    def __init__(self) -> None:
        self._llm = get_fast_llm()
        # Temperature regex: matches e.g. 37°C, 4 °C, 95 degrees Celsius, -80C
        self._temp_pattern = re.compile(
            r"(-?\d+(?:\.\d+)?\s*(?:°C|°F|°\s*[CF]|degrees?\s*(?:Celsius|Fahrenheit|Centigrade|C|F)|C|F\b))",
            re.IGNORECASE,
        )
        # Timing regex: matches e.g. 5 min, 10 minutes, 2 hours, 30s, 1 hr, 45 sec
        self._time_pattern = re.compile(
            r"(\b\d+(?:\.\d+)?\s*(?:sec|second|min|minute|hr|hour|day|wk|week)s?\b)",
            re.IGNORECASE,
        )

    async def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Enrich a list of DocumentChunks with extracted metadata.

        Processes chunks asynchronously. Implements error recovery to ensure
        that extraction failures on individual chunks do not block the pipeline.
        """
        logger.info("Extracting metadata for %d chunks", len(chunks))
        
        # We can extract metadata in parallel or sequentially.
        # To avoid hitting rate limits too fast, we extract sequentially or with a limit.
        import asyncio
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests

        async def process_chunk(chunk: DocumentChunk) -> DocumentChunk:
            async with semaphore:
                try:
                    # 1. Regex extracts first
                    temp = self._extract_temperature(chunk.text)
                    if temp:
                        chunk.temperature = temp
                    
                    timing = self._extract_timing(chunk.text)
                    if timing:
                        chunk.timing = timing

                    # 2. LLM extracts the rest (reagents, equipment, safety, dependencies)
                    # We send a single prompt asking for JSON to minimize API calls.
                    prompt_system = (
                        "You are a scientific metadata extractor. Extract laboratory metadata from the provided protocol text chunk.\n"
                        "Return a JSON object with these EXACT keys:\n"
                        "- 'reagents': list of strings (chemical substances, buffer names, reagents, enzymes, etc.)\n"
                        "- 'equipment': list of strings (instruments like centrifuge, PCR cycler, pipette, gel box, etc.)\n"
                        "- 'safety_level': one of: 'low', 'medium', 'high', 'critical'. Classify based on reagents and procedure.\n"
                        "- 'dependencies': list of step numbers or actions this step depends on (e.g. ['Step 3', 'centrifugation']).\n"
                        "- 'temperature': string or null (if not mentioned in text)\n"
                        "- 'timing': string or null (if not mentioned in text)\n"
                        "\nDo NOT add any conversational explanation. Only return valid JSON."
                    )

                    prompt_human = f"Protocol Text Chunk:\n\"\"\"\n{chunk.text}\n\"\"\""

                    response = await self._llm.ainvoke([
                        SystemMessage(content=prompt_system),
                        HumanMessage(content=prompt_human)
                    ])

                    response_text = response.content.strip()
                    # Clean markdown wrappers if present
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                    try:
                        data = json.loads(response_text)
                        
                        # Populate reagents
                        if isinstance(data.get("reagents"), list):
                            chunk.reagents = [str(r).strip() for r in data["reagents"] if r]
                        
                        # Populate equipment
                        if isinstance(data.get("equipment"), list):
                            chunk.equipment = [str(e).strip() for e in data["equipment"] if e]
                        
                        # Populate safety level
                        safety_str = data.get("safety_level", "low").lower()
                        if safety_str in ["low", "medium", "high", "critical"]:
                            chunk.safety_level = SafetyLevel(safety_str)
                        else:
                            chunk.safety_level = SafetyLevel.LOW

                        # Populate dependencies
                        if isinstance(data.get("dependencies"), list):
                            chunk.dependencies = [str(d).strip() for d in data["dependencies"] if d]

                        # If LLM extracted temp/timing and regex didn't, use LLM values
                        if not chunk.temperature and data.get("temperature"):
                            chunk.temperature = str(data["temperature"]).strip()
                        if not chunk.timing and data.get("timing"):
                            chunk.timing = str(data["timing"]).strip()

                    except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON response from metadata extractor: %s", response_text)
                        # Fallback: execute helper functions
                        chunk.reagents = await self._extract_reagents(chunk.text)
                        chunk.equipment = await self._extract_equipment(chunk.text)
                        chunk.safety_level = await self._classify_safety(chunk.text, chunk.reagents)
                        chunk.dependencies = await self._detect_dependencies(chunk.text, chunk.step_number)

                except Exception as e:
                    logger.error("Error extracting metadata for chunk: %s", e)
                return chunk

        tasks = [process_chunk(chunk) for chunk in chunks]
        enriched_chunks = await asyncio.gather(*tasks)
        return list(enriched_chunks)

    # ── Fallback Methods ──────────────────────────────────────────

    async def _extract_reagents(self, text: str) -> list[str]:
        """LLM-based fallback to extract reagents."""
        try:
            prompt = f"List all chemical reagents, enzymes, buffers, or biological samples mentioned in this text as a simple JSON list of strings:\n\n{text}"
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            cleaned = response.content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception:
            return []

    async def _extract_equipment(self, text: str) -> list[str]:
        """LLM-based fallback to extract equipment."""
        try:
            prompt = f"List all laboratory equipment or instruments mentioned in this text as a simple JSON list of strings:\n\n{text}"
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            cleaned = response.content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception:
            return []

    async def _classify_safety(self, text: str, reagents: list[str]) -> SafetyLevel:
        """LLM-based fallback to classify safety levels."""
        try:
            prompt = (
                f"Classify the safety hazard level of performing this step based on the text and reagents: {reagents}.\n"
                f"Text:\n{text}\n\n"
                "Respond with EXACTLY one word: 'low', 'medium', 'high', or 'critical'."
            )
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            res = response.content.strip().lower()
            if res in ["low", "medium", "high", "critical"]:
                return SafetyLevel(res)
            return SafetyLevel.LOW
        except Exception:
            return SafetyLevel.LOW

    async def _detect_dependencies(self, text: str, step_num: Optional[int]) -> list[str]:
        """LLM-based fallback to detect step dependencies."""
        try:
            prompt = (
                f"Does this step (Step {step_num if step_num else ''}) depend on any previous steps, incubations, or preparations?\n"
                f"Text:\n{text}\n\n"
                "List any detected step dependencies as a simple JSON list of strings (e.g. ['Step 2', 'primer dilution']). Return empty list if none."
            )
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            cleaned = response.content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception:
            return []

    # ── Regex Extractor Helpers ────────────────────────────────────

    def _extract_timing(self, text: str) -> Optional[str]:
        """Regex-based duration extraction."""
        match = self._time_pattern.search(text)
        return match.group(1) if match else None

    def _extract_temperature(self, text: str) -> Optional[str]:
        """Regex-based temperature extraction."""
        match = self._temp_pattern.search(text)
        return match.group(1) if match else None
