"""
Document Processing & Ingestion Pipeline for the Research Protocol Assistant.

This package handles the full lifecycle of converting raw PDF protocol
documents into searchable, semantically-enriched vector embeddings:

    1. PDF Parsing — Extract text, tables, and structural elements
    2. Chunking — Split into semantically meaningful chunks
    3. Metadata Extraction — Enrich chunks with reagents, equipment, safety info
    4. Embedding Generation — Create vector embeddings and store in Pinecone
    5. Pipeline Orchestration — Coordinate the full ingestion workflow

Usage:
    from app.ingestion import IngestionPipeline

    pipeline = IngestionPipeline()
    result = await pipeline.ingest(
        file_path="/path/to/protocol.pdf",
        user_id="user-uuid",
        title="Western Blot Protocol",
        experiment_type="western_blot",
    )
"""

from app.ingestion.pipeline import IngestionPipeline

__all__ = ["IngestionPipeline"]
