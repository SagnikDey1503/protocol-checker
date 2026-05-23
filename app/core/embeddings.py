"""
Embedding model initialization and utilities.

Uses sentence-transformers for local embedding generation (free, no API costs).
The model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """
    Return a cached SentenceTransformer embedding model.

    Default: all-MiniLM-L6-v2 (384-dim, fast, good quality)
    Can be swapped to a larger model via EMBEDDING_MODEL env var.
    """
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info(
        "Embedding model loaded. Dimension: %d", model.get_sentence_embedding_dimension()
    )
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed.
        batch_size: Batch size for encoding (larger = faster but more memory).

    Returns:
        List of embedding vectors as Python lists of floats.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # Cosine similarity works best with normalized vectors
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Generate a single embedding for a search query.

    Optimized path for single-query encoding during retrieval.
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
