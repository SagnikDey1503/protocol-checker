"""
Vector Retriever — Pinecone-based semantic search.

Embeds the incoming query with sentence-transformers, queries a Pinecone
index, and returns scored `RetrievedChunk` objects with metadata.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.core.embeddings import embed_query
from app.core.exceptions import RetrievalError
from app.dependencies import get_pinecone_index

logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """A single chunk returned by any retrieval component.

    Attributes:
        chunk_id: Unique identifier of the chunk (Pinecone vector ID).
        text: The full text content of the chunk.
        score: Normalised relevance score in [0, 1].
        metadata: Arbitrary key/value metadata stored alongside the chunk.
        source: Human-readable source description (e.g. protocol title, page).
    """

    chunk_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


# ── Vector Retriever ─────────────────────────────────────────


class VectorRetriever:
    """Retrieve chunks from Pinecone using dense-vector similarity search.

    Typical usage::

        retriever = VectorRetriever()
        chunks = await retriever.retrieve("How do I pipette?", namespace="protocols", top_k=10)
    """

    def __init__(self) -> None:
        """Initialise the retriever with a cached Pinecone index handle."""
        self._index = get_pinecone_index()
        self._settings = get_settings()
        logger.info("VectorRetriever initialised (index=%s)", self._settings.pinecone_index_name)

    # ── public API ────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed *query* and search Pinecone for the closest vectors.

        Args:
            query: Natural-language search string.
            namespace: Pinecone namespace to search within (e.g. ``"protocols"``).
            top_k: Maximum number of results to return.  Defaults to the
                ``retrieval_top_k`` setting.
            filters: Optional Pinecone metadata filter dict using operators
                such as ``$eq``, ``$gte``, ``$in``.

        Returns:
            A list of :class:`RetrievedChunk` sorted by descending score.

        Raises:
            RetrievalError: When embedding or Pinecone query fails.
        """
        top_k = top_k or self._settings.retrieval_top_k

        try:
            # Embed in a thread to avoid blocking the event-loop (CPU-bound).
            query_embedding = await asyncio.to_thread(embed_query, query)

            # Build Pinecone query kwargs.
            query_kwargs: dict[str, Any] = {
                "vector": query_embedding,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True,
            }

            if filters:
                query_kwargs["filter"] = self._build_pinecone_filter(filters)

            # Pinecone SDK is synchronous — offload to a thread.
            response = await asyncio.to_thread(self._index.query, **query_kwargs)

            chunks = self._parse_response(response)
            logger.debug(
                "VectorRetriever returned %d chunks (namespace=%s, top_k=%d)",
                len(chunks),
                namespace,
                top_k,
            )
            return chunks

        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("Pinecone vector retrieval failed")
            raise RetrievalError(f"Vector retrieval failed: {exc}") from exc

    # ── private helpers ──────────────────────────────────────

    @staticmethod
    def _build_pinecone_filter(filters: dict[str, Any]) -> dict[str, Any]:
        """Convert a flat filter dict into a Pinecone metadata filter.

        Accepted input styles:

        * **Simple equality**: ``{"experiment_type": "PCR"}``
          → ``{"experiment_type": {"$eq": "PCR"}}``
        * **Operator dicts**: ``{"step_number": {"$gte": 3}}``
          → passed through unchanged.
        * **List values**:  ``{"tags": ["safety", "reagent"]}``
          → ``{"tags": {"$in": ["safety", "reagent"]}}``

        Multiple keys are wrapped in a ``$and`` clause automatically.
        """
        clauses: list[dict[str, Any]] = []

        for key, value in filters.items():
            if isinstance(value, dict):
                # Already an operator dict — pass through.
                clauses.append({key: value})
            elif isinstance(value, list):
                clauses.append({key: {"$in": value}})
            else:
                clauses.append({key: {"$eq": value}})

        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _parse_response(response: Any) -> list[RetrievedChunk]:
        """Parse Pinecone query response into :class:`RetrievedChunk` objects.

        Similarity scores (cosine, range [-1, 1] but typically [0, 1] for
        normalised vectors) are mapped to a confidence in [0, 1].
        """
        chunks: list[RetrievedChunk] = []

        matches = getattr(response, "matches", None) or []
        for match in matches:
            metadata: dict[str, Any] = dict(getattr(match, "metadata", {}) or {})
            text = metadata.pop("text", "")

            # Convert raw cosine-similarity score → confidence in [0, 1].
            raw_score: float = float(getattr(match, "score", 0.0))
            confidence = VectorRetriever._score_to_confidence(raw_score)

            source = metadata.get("protocol_title", metadata.get("source", ""))

            chunks.append(
                RetrievedChunk(
                    chunk_id=str(match.id),
                    text=text,
                    score=confidence,
                    metadata=metadata,
                    source=str(source),
                )
            )

        return chunks

    @staticmethod
    def _score_to_confidence(raw_score: float) -> float:
        """Map a raw similarity score to a 0-1 confidence value.

        For normalised embeddings the cosine similarity already lies in
        [0, 1] (negative similarity is theoretically possible but practically
        absent).  We clamp to [0, 1] for safety.
        """
        return max(0.0, min(1.0, raw_score))
