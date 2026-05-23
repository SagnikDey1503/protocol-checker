"""
Cross-Encoder Reranker — Precise relevance scoring with a cross-encoder.

Re-scores a short candidate list (typically 20-50 chunks) produced by the
hybrid retriever, using a cross-encoder model that jointly encodes the
query and each document for fine-grained relevance estimation.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.core.exceptions import RerankingError
from app.retrieval.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Hard upper bound on candidates sent to the cross-encoder.
_MAX_RERANK_CANDIDATES = 50


class CrossEncoderReranker:
    """Rerank retrieval candidates using a cross-encoder model.

    The cross-encoder is heavier than bi-encoder similarity but far more
    accurate because it jointly attends to query and document tokens.
    It should only be run on a small candidate set (≤ 50 chunks).

    Usage::

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("How to calibrate a pipette?", chunks, top_k=5)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
        """
        self._model_name = model_name
        self._settings = get_settings()

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
            logger.info("CrossEncoderReranker loaded model: %s", model_name)
        except Exception as exc:
            logger.exception("Failed to load cross-encoder model: %s", model_name)
            raise RerankingError(f"Could not load reranker model '{model_name}': {exc}") from exc

    # ── public API ────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-score and re-sort *chunks* with the cross-encoder.

        Args:
            query: The user query.
            chunks: Candidate chunks from the retrieval stage.
            top_k: Number of results to return after reranking.  Defaults to
                ``rerank_top_k`` from settings.

        Returns:
            Top-k :class:`RetrievedChunk` objects sorted by cross-encoder
            score (descending), with ``chunk.score`` updated.

        Raises:
            RerankingError: If cross-encoder inference fails.
        """
        top_k = top_k or self._settings.rerank_top_k

        if not chunks:
            return []

        # Limit candidates to avoid OOM / excessive latency.
        candidates = chunks[:_MAX_RERANK_CANDIDATES]
        if len(chunks) > _MAX_RERANK_CANDIDATES:
            logger.warning(
                "Reranker received %d chunks — truncating to %d",
                len(chunks),
                _MAX_RERANK_CANDIDATES,
            )

        try:
            # Build query-document pairs for the cross-encoder.
            pairs: list[list[str]] = [[query, c.text] for c in candidates]

            # Cross-encoder inference is CPU/GPU-bound — run in a thread.
            raw_scores: list[float] = await asyncio.to_thread(
                self._predict_scores, pairs
            )

            # Normalise scores to [0, 1] using sigmoid-like mapping.
            normalised = self._normalise_scores(raw_scores)

            # Attach scores back to chunks and sort.
            for chunk, score in zip(candidates, normalised):
                chunk.score = score

            candidates.sort(key=lambda c: c.score, reverse=True)

            result = candidates[:top_k]
            logger.debug(
                "Reranker: %d candidates → top %d (best=%.4f, worst=%.4f)",
                len(candidates),
                len(result),
                result[0].score if result else 0.0,
                result[-1].score if result else 0.0,
            )
            return result

        except RerankingError:
            raise
        except Exception as exc:
            logger.exception("Cross-encoder reranking failed")
            raise RerankingError(f"Reranking failed: {exc}") from exc

    # ── private helpers ──────────────────────────────────────

    def _predict_scores(self, pairs: list[list[str]]) -> list[float]:
        """Run cross-encoder prediction synchronously.

        Args:
            pairs: List of [query, document] string pairs.

        Returns:
            Raw logit scores from the cross-encoder.
        """
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]

    @staticmethod
    def _normalise_scores(raw_scores: list[float]) -> list[float]:
        """Map raw cross-encoder logits to [0, 1] via min-max scaling.

        If all scores are identical the function returns 0.5 for every item
        to avoid division by zero.
        """
        if not raw_scores:
            return []

        min_s = min(raw_scores)
        max_s = max(raw_scores)
        span = max_s - min_s

        if span < 1e-9:
            return [0.5] * len(raw_scores)

        return [(s - min_s) / span for s in raw_scores]
