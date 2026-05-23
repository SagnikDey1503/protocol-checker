"""
RAG Pipeline Orchestrator — End-to-end retrieval-augmented generation.

Orchestrates the full retrieval flow:

    MultiQuery → HybridRetrieval → Rerank → Compress → Cite

Each stage is optional / toggleable so the pipeline can be tailored to
latency vs. quality requirements.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.core.exceptions import RetrievalError
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.compressor import ContextCompressor
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.multi_query import MultiQueryGenerator
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import RetrievedChunk, VectorRetriever

logger = logging.getLogger(__name__)


# ── Result Data Class ────────────────────────────────────────


@dataclass
class RAGResult:
    """Container for the output of a full RAG pipeline run.

    Attributes:
        chunks: Final list of retrieved & processed chunks.
        citations: Source citation dicts ready for the API response.
        confidence: Aggregate retrieval confidence in [0, 1].
        query_variations: All query strings used (original + generated).
    """

    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    query_variations: list[str] = field(default_factory=list)


# ── Pipeline ─────────────────────────────────────────────────


class RAGPipeline:
    """Full RAG pipeline orchestrating multi-query, hybrid retrieval,
    cross-encoder reranking, and context compression.

    Usage::

        pipeline = RAGPipeline()
        result = await pipeline.retrieve(
            query="What temperature should I use for primer annealing?",
            namespace="protocols",
        )
        for chunk in result.chunks:
            print(chunk.text, chunk.score)
    """

    def __init__(self) -> None:
        """Initialise all sub-components.

        Heavy models (cross-encoder, sentence-transformer) are loaded here
        at construction time so that the first query does not pay the cold-
        start penalty.
        """
        self._settings = get_settings()

        # Sub-components.
        self._vector_retriever = VectorRetriever()
        self._bm25_retriever = BM25Retriever()
        self._hybrid_retriever = HybridRetriever(
            vector_retriever=self._vector_retriever,
            bm25_retriever=self._bm25_retriever,
        )
        self._reranker = CrossEncoderReranker()
        self._compressor = ContextCompressor()
        self._multi_query = MultiQueryGenerator()

        logger.info("RAGPipeline fully initialised")

    # ── Accessors for sub-components ─────────────────────────

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Expose the BM25 retriever so callers can build its index."""
        return self._bm25_retriever

    @property
    def vector_retriever(self) -> VectorRetriever:
        """Expose the vector retriever."""
        return self._vector_retriever

    # ── public API ────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_reranking: bool = True,
        use_multi_query: bool = True,
    ) -> RAGResult:
        """Execute the full RAG pipeline.

        Stages (in order):
            1. **Multi-query expansion** (optional) — generate diverse query
               formulations to broaden recall.
            2. **Hybrid retrieval** — run vector + BM25 search per query,
               fuse with RRF.
            3. **Reranking** (optional) — cross-encoder rescoring of the
               top candidates.
            4. **Compression** — LLM-based extraction & deduplication.
            5. **Citation generation** — map chunks to source documents.
            6. **Confidence computation** — aggregate retrieval confidence.

        Args:
            query: User's natural-language question.
            namespace: Pinecone namespace to search.
            top_k: Maximum number of final chunks to return.
            filters: Optional Pinecone metadata filters.
            use_reranking: Whether to apply the cross-encoder reranking
                stage.  Disable for lower latency.
            use_multi_query: Whether to generate query variations.  Disable
                for lower latency.

        Returns:
            A :class:`RAGResult` containing chunks, citations, confidence,
            and the query variations used.

        Raises:
            RetrievalError: If the pipeline encounters an unrecoverable error.
        """
        top_k = top_k or self._settings.retrieval_top_k

        try:
            # ── Stage 1: Multi-query expansion ────────────────
            if use_multi_query:
                query_variations = await self._multi_query.generate_queries(query)
            else:
                query_variations = [query]

            logger.info(
                "RAG pipeline: %d query variation(s) for '%s'",
                len(query_variations),
                query[:80],
            )

            # ── Stage 2: Hybrid retrieval per query ───────────
            # Retrieve a broader candidate pool for reranking.
            candidate_k = top_k * 3 if use_reranking else top_k

            all_chunks: list[RetrievedChunk] = []
            retrieval_tasks = [
                self._hybrid_retriever.retrieve(
                    query=q,
                    namespace=namespace,
                    top_k=candidate_k,
                    filters=filters,
                )
                for q in query_variations
            ]

            retrieval_results = await asyncio.gather(
                *retrieval_tasks, return_exceptions=True
            )

            for i, result in enumerate(retrieval_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Retrieval for query variation %d failed: %s",
                        i,
                        result,
                    )
                else:
                    all_chunks.extend(result)

            if not all_chunks:
                logger.warning("RAG pipeline: no chunks retrieved for any query variation")
                return RAGResult(
                    chunks=[],
                    citations=[],
                    confidence=0.0,
                    query_variations=query_variations,
                )

            # Deduplicate chunks across query variations (keep highest score).
            all_chunks = self._deduplicate_chunks(all_chunks)

            logger.debug(
                "RAG pipeline: %d unique chunks after multi-query retrieval",
                len(all_chunks),
            )

            # ── Stage 3: Reranking ────────────────────────────
            if use_reranking and all_chunks:
                all_chunks = await self._reranker.rerank(
                    query=query,
                    chunks=all_chunks,
                    top_k=min(top_k * 2, len(all_chunks)),
                )
                logger.debug(
                    "RAG pipeline: %d chunks after reranking", len(all_chunks)
                )

            # ── Stage 4: Compression ──────────────────────────
            compressed = await self._compressor.compress(query, all_chunks[:top_k])
            if compressed:
                all_chunks = compressed
            else:
                # Compression dropped everything — fall back to pre-compression.
                all_chunks = all_chunks[:top_k]

            logger.debug(
                "RAG pipeline: %d chunks after compression", len(all_chunks)
            )

            # ── Stage 5 & 6: Citations + Confidence ──────────
            citations = self._generate_citations(all_chunks)
            confidence = self._compute_confidence(all_chunks)

            return RAGResult(
                chunks=all_chunks,
                citations=citations,
                confidence=confidence,
                query_variations=query_variations,
            )

        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("RAG pipeline failed")
            raise RetrievalError(f"RAG pipeline failed: {exc}") from exc

    # ── citation generation ──────────────────────────────────

    @staticmethod
    def _generate_citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        """Map chunks to structured source citation dicts.

        Groups chunks by source document and returns one citation entry
        per unique source.

        Returns:
            A list of dicts, each with keys:
                - ``chunk_id``
                - ``text`` (truncated preview)
                - ``protocol_title``
                - ``page_number``
                - ``step_number``
                - ``relevance_score``
        """
        citations: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)

            citations.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:300],
                    "protocol_title": chunk.metadata.get(
                        "protocol_title", chunk.source
                    ),
                    "page_number": chunk.metadata.get("page_number"),
                    "step_number": chunk.metadata.get("step_number"),
                    "relevance_score": round(chunk.score, 4),
                }
            )

        return citations

    # ── confidence scoring ───────────────────────────────────

    @staticmethod
    def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
        """Compute an overall retrieval confidence in [0, 1].

        Strategy:
            * 0 chunks → 0.0
            * Weighted average of chunk scores, where higher-ranked chunks
              contribute more (exponential decay weights).
            * Boosted slightly when multiple high-scoring chunks agree.
        """
        if not chunks:
            return 0.0

        # Exponentially decaying weights: top chunk gets weight 1.0,
        # second gets 0.7, third 0.49, …
        decay = 0.7
        weighted_sum = 0.0
        weight_total = 0.0

        for i, chunk in enumerate(chunks):
            w = decay ** i
            weighted_sum += chunk.score * w
            weight_total += w

        avg_conf = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Agreement bonus: if ≥ 3 chunks score above 0.6, bump confidence.
        high_scorers = sum(1 for c in chunks if c.score > 0.6)
        if high_scorers >= 3:
            agreement_bonus = min(0.1, high_scorers * 0.02)
            avg_conf = min(1.0, avg_conf + agreement_bonus)

        return round(avg_conf, 4)

    # ── deduplication ────────────────────────────────────────

    @staticmethod
    def _deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Deduplicate chunks by ``chunk_id``, keeping the highest score.

        Args:
            chunks: Potentially overlapping chunks from multiple queries.

        Returns:
            Deduplicated list sorted by score descending.
        """
        best: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk

        deduped = list(best.values())
        deduped.sort(key=lambda c: c.score, reverse=True)
        return deduped
