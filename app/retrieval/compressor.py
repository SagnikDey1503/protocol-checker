"""
Context Compressor — LLM-powered extraction and deduplication.

After retrieval and reranking the candidate chunks may still contain
irrelevant sentences or redundant information across chunks.  The
compressor uses Claude Haiku (via ``get_fast_llm()``) to:

1. Extract only the sentences relevant to the query from each chunk.
2. Remove cross-chunk redundancy.
3. Drop chunks whose relevance is below a configurable threshold.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.exceptions import RetrievalError
from app.core.llm import get_fast_llm
from app.retrieval.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Chunks with a relevance score from the compressor below this
# threshold are silently dropped.
_RELEVANCE_THRESHOLD = 0.3

_COMPRESSION_SYSTEM_PROMPT = """\
You are a precise information extractor for a scientific research assistant.

Given a user query and a document chunk, your job is to:
1. Extract ONLY the sentences/phrases from the chunk that are directly relevant to the query.
2. Preserve exact scientific terminology, numbers, measurements and procedure steps.
3. Do NOT add any information that is not in the original chunk.
4. Do NOT paraphrase — copy relevant text verbatim where possible.
5. If the chunk contains NO relevant information, respond with exactly: IRRELEVANT

Return the extracted text only, no commentary.
"""

_DEDUP_SYSTEM_PROMPT = """\
You are a deduplication engine for a scientific research assistant.

You will receive a list of text excerpts that may contain overlapping information.
Remove redundant content while keeping ALL unique facts.  Preserve the ordering.
Output one excerpt per numbered line, or drop an excerpt entirely if its content
is fully covered by another excerpt.

Return the deduplicated excerpts only, numbered like the input.
"""


class ContextCompressor:
    """Compress and deduplicate retrieval context using an LLM.

    Usage::

        compressor = ContextCompressor()
        compressed = await compressor.compress("primer melting temp?", chunks)
    """

    def __init__(self) -> None:
        """Initialise with a fast LLM handle."""
        self._llm = get_fast_llm()
        logger.info("ContextCompressor initialised")

    # ── public API ────────────────────────────────────────────

    async def compress(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Compress and deduplicate *chunks* against the *query*.

        Pipeline:
            1. Extract relevant passages from each chunk (parallel).
            2. Filter out chunks marked ``IRRELEVANT``.
            3. Remove cross-chunk redundancy in a single LLM call.

        Args:
            query: User query providing context for relevance.
            chunks: Candidate chunks to compress.

        Returns:
            Compressed, deduplicated list of :class:`RetrievedChunk`.

        Raises:
            RetrievalError: If the LLM calls fail irrecoverably.
        """
        if not chunks:
            return []

        try:
            # Step 1: Extract relevant parts from each chunk (parallel).
            extracted = await self._extract_relevant_parts(query, chunks)

            # Step 2: Filter below-threshold chunks.
            filtered = [c for c in extracted if c is not None]

            if not filtered:
                logger.info("Compressor: all chunks dropped as irrelevant")
                return []

            # Step 3: Deduplicate across remaining chunks.
            deduped = await self._deduplicate(filtered)

            logger.debug(
                "Compressor: %d → %d extracted → %d deduped",
                len(chunks),
                len(filtered),
                len(deduped),
            )
            return deduped

        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("Context compression failed")
            raise RetrievalError(f"Context compression failed: {exc}") from exc

    # ── private helpers ──────────────────────────────────────

    async def _extract_relevant_parts(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk | None]:
        """Run per-chunk extraction concurrently.

        Returns a list parallel to *chunks*, with ``None`` for irrelevant
        entries.
        """
        tasks = [self._extract_single(query, chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    async def _extract_single(
        self,
        query: str,
        chunk: RetrievedChunk,
    ) -> RetrievedChunk | None:
        """Extract the relevant portion of one chunk.

        Returns ``None`` if the LLM determines the chunk is irrelevant
        or if the extraction score falls below *_RELEVANCE_THRESHOLD*.
        """
        try:
            messages = [
                SystemMessage(content=_COMPRESSION_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Query: {query}\n\n"
                        f"Document chunk (source: {chunk.source}):\n"
                        f"{chunk.text}"
                    )
                ),
            ]

            response = await self._llm.ainvoke(messages)
            extracted_text = response.content.strip()

            if not extracted_text or extracted_text.upper() == "IRRELEVANT":
                return None

            # Rough relevance heuristic: ratio of extracted to original length.
            # Very short extractions relative to the query likely indicate low
            # relevance.
            extraction_ratio = len(extracted_text) / max(len(chunk.text), 1)
            if extraction_ratio < 0.05 and chunk.score < _RELEVANCE_THRESHOLD:
                return None

            # Return a *new* chunk with the compressed text, preserving metadata.
            return RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=extracted_text,
                score=chunk.score,
                metadata=chunk.metadata,
                source=chunk.source,
            )

        except Exception as exc:
            # Failing to compress a single chunk should not kill the
            # pipeline — fall back to the original.
            logger.warning("Extraction failed for chunk %s: %s", chunk.chunk_id, exc)
            return chunk

    async def _deduplicate(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Remove cross-chunk redundancy using the LLM.

        If there are ≤ 2 chunks the deduplication step is skipped as the
        LLM overhead is not worthwhile.
        """
        if len(chunks) <= 2:
            return chunks

        try:
            numbered_excerpts = "\n".join(
                f"{i + 1}. {c.text}" for i, c in enumerate(chunks)
            )

            messages = [
                SystemMessage(content=_DEDUP_SYSTEM_PROMPT),
                HumanMessage(content=numbered_excerpts),
            ]

            response = await self._llm.ainvoke(messages)
            deduped_text = response.content.strip()

            # Parse out the surviving numbered items.
            surviving_indices = self._parse_numbered_response(deduped_text, len(chunks))

            if not surviving_indices:
                # Parsing failed — return all chunks unmodified.
                logger.warning("Dedup parsing returned no indices — keeping all chunks")
                return chunks

            deduped_chunks: list[RetrievedChunk] = []
            for idx in surviving_indices:
                if 0 <= idx < len(chunks):
                    original_chunk = chunks[idx]
                    # Optionally update text with deduplicated version.
                    deduped_chunks.append(original_chunk)

            return deduped_chunks if deduped_chunks else chunks

        except Exception as exc:
            # Dedup failure is non-fatal — return the unmodified list.
            logger.warning("Deduplication failed: %s — keeping all chunks", exc)
            return chunks

    @staticmethod
    def _parse_numbered_response(text: str, total: int) -> list[int]:
        """Extract surviving 1-based indices from the LLM response.

        The LLM is expected to return numbered lines like:

            1. …extracted text…
            3. …extracted text…

        We parse the leading number from each line.
        """
        import re

        indices: list[int] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\.", line)
            if match:
                idx = int(match.group(1)) - 1  # Convert to 0-based.
                if 0 <= idx < total:
                    indices.append(idx)

        return indices
