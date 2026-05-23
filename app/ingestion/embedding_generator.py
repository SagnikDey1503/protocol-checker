"""
Embedding generator and vector store indexing for the ingestion pipeline.

Generates dense vector embeddings using local sentence-transformers (via get_embedding_model)
and stores them in Pinecone using logical namespaces.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.embeddings import embed_texts
from app.dependencies import get_pinecone_index
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings and indexing records for Pinecone.

    Handles chunk text extraction, embedding calls, formatting into Pinecone
    compatible records, and batch upsert operations.
    """

    def __init__(self) -> None:
        self._index = get_pinecone_index()

    async def generate_and_store(
        self,
        chunks: list[DocumentChunk],
        protocol_id: str,
        namespace: str,
    ) -> dict[str, Any]:
        """
        Generate embeddings for a list of chunks and upsert them to Pinecone.

        Args:
            chunks: List of DocumentChunk instances.
            protocol_id: UUID of the protocol.
            namespace: logical Pinecone namespace.

        Returns:
            Dict containing processing stats.
        """
        if not chunks:
            logger.warning("No chunks provided to embedding generator")
            return {"chunks_embedded": 0, "vectors_upserted": 0}

        logger.info(
            "Generating embeddings for %d chunks (protocol_id=%s, namespace=%s)",
            len(chunks),
            protocol_id,
            namespace,
        )

        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings
        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", e)
            raise

        # Prepare records for Pinecone
        records = self._prepare_pinecone_records(chunks, embeddings, protocol_id)

        # Batch upsert to Pinecone (groups of 100)
        batch_size = 100
        total_upserted = 0

        try:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                # Pinecone upsert expects list of dicts/tuples: (id, vector, metadata)
                self._index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                logger.debug("Upserted batch of %d vectors to Pinecone", len(batch))

            logger.info("Successfully indexed %d vectors in Pinecone", total_upserted)
            return {"chunks_embedded": len(chunks), "vectors_upserted": total_upserted}
        except Exception as e:
            logger.error("Pinecone upsert failed: %s", e)
            raise

    def _prepare_pinecone_records(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        protocol_id: str,
    ) -> list[dict[str, Any]]:
        """
        Format chunk metadata and embeddings into Pinecone upsert format.

        Returns list of structures like:
            {"id": chunk_id, "values": vector, "metadata": metadata}
        """
        records: list[dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings):
            # Flatten dataclass to dict suitable for Pinecone
            metadata = chunk.to_pinecone_metadata()
            
            # Make sure protocol_id is in metadata
            metadata["protocol_id"] = protocol_id

            records.append({
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": metadata,
            })

        return records
