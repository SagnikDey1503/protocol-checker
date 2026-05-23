"""
Ingestion pipeline orchestrator for the Research Protocol Assistant.

Coordinates PDF parsing, text cleaning, chunking, metadata extraction,
embedding generation, vector store indexing in Pinecone, and persistent database storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import DocumentChunker, ChunkStrategy
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.embedding_generator import EmbeddingGenerator
from app.models.database import Protocol
from app.models.enums import ChunkType

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the document ingestion workflow.

    Usage::

        pipeline = IngestionPipeline()
        stats = await pipeline.ingest(
            file_path="/path/to/protocol.pdf",
            user_id="user-uuid",
            title="Western Blot Protocol",
            experiment_type="western_blot",
            db=db_session,
            protocol_id="protocol-uuid",
            pinecone_namespace="protocol_namespace"
        )
    """

    def __init__(self) -> None:
        self.parser = PDFParser()
        self.chunker = DocumentChunker()
        self.extractor = MetadataExtractor()
        self.embedding_generator = EmbeddingGenerator()

    async def ingest(
        self,
        file_path: str,
        user_id: str,
        title: str,
        experiment_type: str,
        db: AsyncSession,
        protocol_id: str,
        pinecone_namespace: str,
    ) -> dict:
        """Runs the parsing, chunking, metadata, embedding, and storage workflow."""
        import time

        logger.info("Ingestion starting for file: %s", file_path)
        pipeline_start = time.monotonic()

        # 1. Parse PDF
        parse_start = time.monotonic()
        parsed_doc = await self.parser.parse(file_path)
        logger.info(
            "Ingestion step: parse_pdf completed in %.2fs (pages=%d, tables=%d, chars=%d)",
            time.monotonic() - parse_start,
            parsed_doc.page_count,
            len(parsed_doc.tables),
            parsed_doc.total_text_length,
        )

        # 2. Chunk Document
        # Defaulting to step-aware chunking as the assistant works with experimental procedures.
        chunk_start = time.monotonic()
        strategy = ChunkStrategy.STEP_AWARE
        chunks = self.chunker.chunk_document(parsed_doc, strategy=strategy)
        logger.info(
            "Ingestion step: chunk_document completed in %.2fs (chunks=%d)",
            time.monotonic() - chunk_start,
            len(chunks),
        )

        # 3. Extract Metadata
        meta_start = time.monotonic()
        chunks = await self.extractor.extract(chunks)
        logger.info(
            "Ingestion step: metadata_extraction completed in %.2fs",
            time.monotonic() - meta_start,
        )

        # 4. Generate & Store Embeddings in Pinecone
        embed_start = time.monotonic()
        pinecone_stats = await self.embedding_generator.generate_and_store(
            chunks=chunks,
            protocol_id=protocol_id,
            namespace=pinecone_namespace,
        )
        logger.info(
            "Ingestion step: embedding_store completed in %.2fs",
            time.monotonic() - embed_start,
        )

        # 5. Extract unique reagents and equipment across all chunks
        reagents_set = set()
        equipment_set = set()
        for chunk in chunks:
            if chunk.reagents:
                reagents_set.update(chunk.reagents)
            if chunk.equipment:
                equipment_set.update(chunk.equipment)

        reagents = sorted(list(reagents_set))
        equipment = sorted(list(equipment_set))

        # 6. Build structured steps list
        steps_start = time.monotonic()
        steps = []
        for chunk in chunks:
            if chunk.chunk_type == ChunkType.STEP and chunk.step_number is not None:
                steps.append({
                    "step_number": chunk.step_number,
                    "text": chunk.text,
                    "section_title": chunk.section_title or f"Step {chunk.step_number}",
                    "reagents": chunk.reagents,
                    "equipment": chunk.equipment,
                    "temperature": chunk.temperature,
                    "timing": chunk.timing,
                    "safety_level": chunk.safety_level.value,
                    "dependencies": chunk.dependencies,
                })
        
        # Sort steps chronologically
        steps.sort(key=lambda s: s["step_number"])

        logger.info(
            "Ingestion step: build_steps completed in %.2fs (steps=%d)",
            time.monotonic() - steps_start,
            len(steps),
        )

        logger.info(
            "Ingestion completed in %.2fs for file: %s",
            time.monotonic() - pipeline_start,
            file_path,
        )

        return {
            "parsed_doc": parsed_doc,
            "chunks": chunks,
            "reagents": reagents,
            "equipment": equipment,
            "steps": steps,
            "pinecone_stats": pinecone_stats,
        }


async def run_pipeline(protocol: Protocol, db: AsyncSession) -> None:
    """
    Entry point for background tasks to ingest a protocol.

    Finds the file on disk, runs the orchestrator, and updates the database record.
    """
    logger.info("Background task running ingestion for protocol %s", protocol.id)

    # Locate file path on disk
    upload_dir = Path("uploads") / str(protocol.user_id)
    matching_files = list(upload_dir.glob(f"*_{protocol.original_filename}"))
    if not matching_files:
        raise FileNotFoundError(
            f"Could not find uploaded file for protocol {protocol.id} with name {protocol.original_filename}"
        )
    
    file_path = str(matching_files[0])

    pipeline = IngestionPipeline()
    result = await pipeline.ingest(
        file_path=file_path,
        user_id=str(protocol.user_id),
        title=protocol.title,
        experiment_type=protocol.experiment_type or "",
        db=db,
        protocol_id=str(protocol.id),
        pinecone_namespace=protocol.pinecone_namespace or f"protocol_{protocol.id.hex[:12]}",
    )

    # Save to database
    protocol.parsed_content = {
        "raw_text": result["parsed_doc"].raw_text,
        "metadata": result["parsed_doc"].metadata,
    }
    protocol.steps = result["steps"]
    protocol.step_count = len(result["steps"])
    protocol.reagents = result["reagents"]
    protocol.equipment = result["equipment"]
    protocol.is_processed = True

    logger.info(
        "Background task completed ingestion for protocol %s (steps=%d, reagents=%d)",
        protocol.id,
        protocol.step_count,
        len(protocol.reagents),
    )
