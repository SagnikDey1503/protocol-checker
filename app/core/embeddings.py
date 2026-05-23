"""
Embedding model initialization and utilities using Google Generative AI (Gemini).
Uses models/gemini-embedding-2 with custom output dimension of 384.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached GoogleGenerativeAIEmbeddings model.
    """
    settings = get_settings()
    logger.info("Initializing GoogleGenerativeAIEmbeddings...")
    model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimension
    )
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Google Generative AI.
    """
    model = get_embedding_model()
    # GoogleGenerativeAIEmbeddings uses embed_documents for multiple texts
    return model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Generate a single embedding for a search query.
    """
    model = get_embedding_model()
    return model.embed_query(query)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
