"""
BM25 Retriever — Sparse keyword search using Okapi BM25.

Maintains an in-memory BM25 index over document texts for fast lexical
retrieval.  Designed to complement the dense vector retriever in a
hybrid pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.exceptions import RetrievalError
from app.retrieval.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# A compact set of English stopwords covering the most frequent
# non-informative tokens.  Kept small on purpose so that domain terms
# (e.g. "the PCR protocol") are not accidentally dropped.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "not", "is", "am", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might",
        "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "about", "between", "through", "during",
        "before", "after", "above", "below", "up", "down", "out", "off",
        "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "if", "while", "this", "that", "these", "those", "it",
        "its", "i", "me", "my", "we", "us", "our", "you", "your", "he",
        "him", "his", "she", "her", "they", "them", "their", "what",
        "which", "who", "whom",
    }
)


class BM25Retriever:
    """Sparse keyword retriever using the BM25 Okapi algorithm.

    Usage::

        bm25 = BM25Retriever()
        bm25.build_index([
            {"chunk_id": "c1", "text": "Add 5 µL of primer mix", "metadata": {}, "source": "PCR Protocol"},
            ...
        ])
        results = bm25.retrieve("primer concentration", top_k=5)
    """

    def __init__(self) -> None:
        """Create an empty BM25 retriever.

        Call :meth:`build_index` before attempting retrieval.
        """
        self._bm25: BM25Okapi | None = None
        self._documents: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        logger.info("BM25Retriever initialised (index not yet built)")

    # ── public API ────────────────────────────────────────────

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Build (or rebuild) the BM25 index from a list of documents.

        Args:
            documents: Each dict **must** contain at least a ``"text"`` key.
                Optional keys: ``"chunk_id"``, ``"metadata"``, ``"source"``.

        Raises:
            RetrievalError: If the document list is empty or lacks text.
        """
        if not documents:
            raise RetrievalError("Cannot build BM25 index from an empty document list")

        self._documents = documents
        self._tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in documents]

        try:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception as exc:
            logger.exception("Failed to build BM25 index")
            raise RetrievalError(f"BM25 index construction failed: {exc}") from exc

        logger.info("BM25 index built with %d documents", len(documents))

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Search the BM25 index and return the top-k scoring chunks.

        Args:
            query: Raw user query string.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`RetrievedChunk` sorted by descending BM25 score.

        Raises:
            RetrievalError: If the index has not been built yet.
        """
        if self._bm25 is None or not self._documents:
            raise RetrievalError(
                "BM25 index has not been built — call build_index() first"
            )

        try:
            tokenized_query = self._tokenize(query)
            if not tokenized_query:
                logger.warning("BM25 query tokenised to empty — returning no results")
                return []

            raw_scores = self._bm25.get_scores(tokenized_query)

            # Build (index, score) pairs, drop zero-score matches, sort desc.
            scored_indices = [
                (idx, float(score))
                for idx, score in enumerate(raw_scores)
                if score > 0.0
            ]
            scored_indices.sort(key=lambda pair: pair[1], reverse=True)
            scored_indices = scored_indices[:top_k]

            if not scored_indices:
                return []

            # Normalise BM25 scores into [0, 1] relative to this result set.
            max_score = scored_indices[0][1]
            chunks: list[RetrievedChunk] = []
            for idx, score in scored_indices:
                doc = self._documents[idx]
                normalised = score / max_score if max_score > 0 else 0.0
                chunks.append(
                    RetrievedChunk(
                        chunk_id=doc.get("chunk_id", f"bm25-{idx}"),
                        text=doc.get("text", ""),
                        score=normalised,
                        metadata=doc.get("metadata", {}),
                        source=doc.get("source", ""),
                    )
                )

            logger.debug("BM25 retrieval returned %d chunks for query '%s'", len(chunks), query[:60])
            return chunks

        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("BM25 retrieval failed")
            raise RetrievalError(f"BM25 retrieval failed: {exc}") from exc

    # ── private helpers ──────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise *text* into lowercased, stopword-filtered tokens.

        Splits on non-alphanumeric characters, discards single-character
        tokens and stopwords.

        Args:
            text: Raw string to tokenise.

        Returns:
            List of cleaned tokens.
        """
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        return [
            tok for tok in tokens
            if tok and len(tok) > 1 and tok not in _STOPWORDS
        ]
