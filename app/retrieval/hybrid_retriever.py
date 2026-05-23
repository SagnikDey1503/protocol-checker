"""
Hybrid Retriever — Fuses dense vector and sparse BM25 results via RRF.

Reciprocal Rank Fusion (RRF) is used because BM25 scores and cosine
similarity scores live on fundamentally different scales.  RRF only
considers the *rank* of each result, making it a robust, parameter-light
fusion strategy.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from app.config import get_settings
from app.core.exceptions import RetrievalError
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import RetrievedChunk, VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine vector and BM25 retrieval results using Reciprocal Rank Fusion.

    Usage::

        hybrid = HybridRetriever(vector_retriever, bm25_retriever)
        chunks = await hybrid.retrieve("primer annealing temperature", namespace="protocols")
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
    ) -> None:
        """Initialise with pre-configured sub-retrievers.

        Args:
            vector_retriever: Dense vector search component.
            bm25_retriever: Sparse keyword search component.
        """
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._settings = get_settings()
        logger.info("HybridRetriever initialised")

    # ── public API ────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Execute both sub-retrievers and fuse via RRF.

        The vector retriever is queried with the full *top_k*; the BM25
        retriever is also queried with the same *top_k*.  Both result
        lists are then merged with :meth:`_reciprocal_rank_fusion`.

        Args:
            query: Natural-language query.
            namespace: Pinecone namespace for the vector search leg.
            top_k: Number of final results to return.
            filters: Optional metadata filters forwarded to the vector
                retriever.

        Returns:
            Fused, deduplicated list of :class:`RetrievedChunk` sorted by
            RRF score (descending).

        Raises:
            RetrievalError: If both sub-retrievers fail.
        """
        top_k = top_k or self._settings.retrieval_top_k

        # Fetch a larger candidate pool from each leg so RRF has material.
        candidate_k = max(top_k * 2, 20)

        vector_results: list[RetrievedChunk] = []
        bm25_results: list[RetrievedChunk] = []

        # Run both retrievers concurrently.
        vector_error: Exception | None = None
        bm25_error: Exception | None = None

        async def _run_vector() -> list[RetrievedChunk]:
            return await self._vector.retrieve(
                query=query,
                namespace=namespace,
                top_k=candidate_k,
                filters=filters,
            )

        async def _run_bm25() -> list[RetrievedChunk]:
            # BM25 is CPU-bound & synchronous — wrap in a thread.
            return await asyncio.to_thread(self._bm25.retrieve, query, candidate_k)

        try:
            vector_task = asyncio.create_task(_run_vector())
            bm25_task = asyncio.create_task(_run_bm25())

            # Gather with return_exceptions so one failure doesn't cancel the other.
            results = await asyncio.gather(vector_task, bm25_task, return_exceptions=True)

            if isinstance(results[0], Exception):
                vector_error = results[0]
                logger.warning("Vector retrieval failed: %s", vector_error)
            else:
                vector_results = results[0]

            if isinstance(results[1], Exception):
                bm25_error = results[1]
                logger.warning("BM25 retrieval failed: %s", bm25_error)
            else:
                bm25_results = results[1]

        except Exception as exc:
            logger.exception("Hybrid retrieval gather failed")
            raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc

        # If both legs failed, propagate.
        if vector_error and bm25_error:
            raise RetrievalError(
                f"Both retrieval legs failed. Vector: {vector_error} | BM25: {bm25_error}"
            )

        # Fuse whatever we have.
        results_lists: list[list[RetrievedChunk]] = []
        if vector_results:
            results_lists.append(vector_results)
        if bm25_results:
            results_lists.append(bm25_results)

        if not results_lists:
            return []

        fused = self._reciprocal_rank_fusion(
            results_lists,
            k=self._settings.rrf_k,
        )

        logger.debug(
            "Hybrid retrieval fused %d vector + %d BM25 → %d results (returning top %d)",
            len(vector_results),
            len(bm25_results),
            len(fused),
            top_k,
        )

        return fused[:top_k]

    # ── RRF algorithm ────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        results_lists: list[list[RetrievedChunk]],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.

        For each result list the score contribution of a chunk at rank *r*
        (0-indexed) is ``1 / (k + r + 1)``.  Scores are summed across
        lists and the final output is sorted by the aggregated RRF score.

        Duplicate chunks (same ``chunk_id``) are merged; the metadata and
        text from the first occurrence are kept.

        Args:
            results_lists: Two or more ranked result lists.
            k: RRF constant controlling how much weight is given to lower
                ranks.  The default of 60 is the value proposed in the
                original RRF paper.

        Returns:
            Deduplicated list of :class:`RetrievedChunk` sorted descending
            by RRF score.
        """
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for result_list in results_lists:
            for rank, chunk in enumerate(result_list):
                rrf_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
                # Keep the first-seen version of the chunk.
                if chunk.chunk_id not in chunk_map:
                    chunk_map[chunk.chunk_id] = chunk

        # Sort by RRF score descending.
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        fused_chunks: list[RetrievedChunk] = []
        for cid in sorted_ids:
            chunk = chunk_map[cid]
            # Overwrite the original score with the RRF score.
            chunk.score = rrf_scores[cid]
            fused_chunks.append(chunk)

        return fused_chunks
