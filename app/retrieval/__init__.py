"""
RAG Retrieval Pipeline — Package Initialization.

Provides vector search, BM25 keyword search, hybrid retrieval with
Reciprocal Rank Fusion, cross-encoder reranking, context compression,
multi-query expansion, and the full orchestrated RAG pipeline.
"""

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.compressor import ContextCompressor
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.multi_query import MultiQueryGenerator
from app.retrieval.pipeline import RAGPipeline, RAGResult
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import RetrievedChunk, VectorRetriever

__all__ = [
    "BM25Retriever",
    "ContextCompressor",
    "CrossEncoderReranker",
    "HybridRetriever",
    "MultiQueryGenerator",
    "RAGPipeline",
    "RAGResult",
    "RetrievedChunk",
    "VectorRetriever",
]
