"""
RAG Retrieval Tools for the Multi-Agent System.

LangChain @tool-decorated functions that search the Pinecone vector
database for protocol content, safety information, and general
biology/chemistry knowledge.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

from app.config import get_settings
from app.core.embeddings import embed_query
from app.dependencies import get_pinecone_index
from app.models.enums import PineconeNamespace

logger = logging.getLogger(__name__)


def _search_pinecone(
    query: str,
    namespace: str,
    top_k: int = 5,
    filter_dict: Optional[dict] = None,
) -> list[dict]:
    """Shared helper that embeds a query and searches a Pinecone namespace.

    Args:
        query: Natural-language search string.
        namespace: Pinecone namespace to search within.
        top_k: Number of results to return.
        filter_dict: Optional metadata filter for Pinecone query.

    Returns:
        List of dicts with keys: id, text, score, metadata.
    """
    try:
        settings = get_settings()
        index = get_pinecone_index()
        query_embedding = embed_query(query)

        query_kwargs: dict = {
            "vector": query_embedding,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True,
        }
        if filter_dict:
            query_kwargs["filter"] = filter_dict

        results = index.query(**query_kwargs)

        hits: list[dict] = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            hits.append(
                {
                    "id": match["id"],
                    "text": metadata.get("text", ""),
                    "score": float(match.get("score", 0.0)),
                    "metadata": metadata,
                }
            )
        return hits

    except Exception as exc:
        logger.error("Pinecone search failed (namespace=%s): %s", namespace, exc)
        return []


@tool
def search_protocol(query: str, protocol_id: Optional[str] = None) -> list[dict]:
    """Search the protocol database for relevant protocol steps, sections, and notes.

    Use this tool when you need to find specific protocol information such as
    step details, reagent lists, equipment requirements, timing, or any
    content from uploaded protocol PDFs.

    Args:
        query: Natural-language search query describing what to find.
        protocol_id: Optional protocol UUID to restrict search to a single protocol.

    Returns:
        List of matching protocol chunks with text, score, and metadata.
    """
    logger.info("Searching protocols: query=%r, protocol_id=%s", query, protocol_id)
    filter_dict = None
    if protocol_id:
        filter_dict = {"protocol_id": {"$eq": protocol_id}}

    settings = get_settings()
    return _search_pinecone(
        query=query,
        namespace=PineconeNamespace.PROTOCOLS.value,
        top_k=settings.retrieval_top_k,
        filter_dict=filter_dict,
    )


@tool
def search_safety_info(query: str) -> list[dict]:
    """Search safety information related to lab procedures, chemicals, and protocols.

    Use this tool when you need to look up safety data, hazard warnings,
    PPE requirements, chemical incompatibilities, or any safety-related
    protocol content.

    Args:
        query: Natural-language query about safety concerns.

    Returns:
        List of matching safety-related chunks with text, score, and metadata.
    """
    logger.info("Searching safety info: query=%r", query)
    settings = get_settings()
    return _search_pinecone(
        query=query,
        namespace=PineconeNamespace.SAFETY.value,
        top_k=settings.retrieval_top_k,
    )


@tool
def search_knowledge(query: str) -> list[dict]:
    """Search general biology, chemistry, and lab technique knowledge base.

    Use this tool for conceptual questions about mechanisms, techniques,
    reagent functions, or any general scientific knowledge that isn't
    specific to a single uploaded protocol.

    Args:
        query: Natural-language search query for general knowledge.

    Returns:
        List of matching knowledge-base chunks with text, score, and metadata.
    """
    logger.info("Searching knowledge base: query=%r", query)
    settings = get_settings()
    return _search_pinecone(
        query=query,
        namespace=PineconeNamespace.KNOWLEDGE.value,
        top_k=settings.retrieval_top_k,
    )
