"""
Protocol management service.

Handles uploading, listing, retrieving, and deleting research protocols.
Delegates PDF ingestion to a background task that runs the ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import (
    PDFParsingError,
    ProtocolNotFoundError,
)
from app.models.database import Protocol

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")


class ProtocolService:
    """Business logic for research protocol CRUD operations.

    Args:
        db: Async SQLAlchemy session.
        redis: Async Redis client (used for cache invalidation).
    """

    def __init__(self, db: AsyncSession, redis: aioredis.Redis) -> None:
        self.db = db
        self.redis = redis
        self.settings = get_settings()

    # ── Upload ───────────────────────────────────────────────

    async def upload_protocol(
        self,
        file_content: bytes,
        filename: str,
        user_id: uuid.UUID,
        title: Optional[str] = None,
    ) -> Protocol:
        """Save uploaded PDF to disk and create a Protocol record.

        The file is stored under ``uploads/<user_id>/<unique_name>.pdf``.
        The caller is responsible for triggering ingestion via
        ``BackgroundTasks`` after this method returns.

        Args:
            file_content: Raw bytes of the uploaded PDF.
            filename: Original filename from the client.
            user_id: Owning user's UUID.
            title: Optional human-readable title; defaults to filename stem.

        Returns:
            The persisted ``Protocol`` ORM instance.

        Raises:
            PDFParsingError: If the file is empty or cannot be stored.
        """
        if not file_content:
            raise PDFParsingError("Uploaded file is empty")

        # Compute file hash for dedup
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Check for duplicate
        existing = await self.db.execute(
            select(Protocol).where(
                Protocol.user_id == user_id,
                Protocol.file_hash == file_hash,
            )
        )
        existing_protocol = existing.scalar_one_or_none()
        if existing_protocol:
            logger.info(
                "Duplicate protocol detected for user %s (hash=%s)",
                user_id,
                file_hash[:12],
            )
            return existing_protocol

        # Persist file
        user_upload_dir = UPLOAD_DIR / str(user_id)
        user_upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = user_upload_dir / safe_name

        try:
            file_path.write_bytes(file_content)
        except OSError as exc:
            logger.error("Failed to write upload to %s: %s", file_path, exc)
            raise PDFParsingError(f"Failed to save uploaded file: {exc}") from exc

        # Create DB record
        protocol = Protocol(
            user_id=user_id,
            title=title or Path(filename).stem.replace("_", " ").replace("-", " ").title(),
            original_filename=filename,
            file_hash=file_hash,
            is_processed=False,
            step_count=0,
            pinecone_namespace=f"protocol_{uuid.uuid4().hex[:12]}",
        )
        self.db.add(protocol)
        await self.db.flush()
        await self.db.refresh(protocol)

        logger.info(
            "Protocol %s created for user %s (file=%s)",
            protocol.id,
            user_id,
            filename,
        )
        return protocol

    # ── List ─────────────────────────────────────────────────

    async def get_protocols(self, user_id: uuid.UUID) -> list[Protocol]:
        """Return all protocols belonging to a user, newest first.

        Args:
            user_id: The owning user's UUID.

        Returns:
            A list of ``Protocol`` instances ordered by creation date descending.
        """
        result = await self.db.execute(
            select(Protocol)
            .where(Protocol.user_id == user_id)
            .order_by(Protocol.created_at.desc())
        )
        return list(result.scalars().all())

    # ── Detail ───────────────────────────────────────────────

    async def get_protocol(
        self,
        protocol_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Protocol:
        """Fetch a single protocol owned by the given user.

        Args:
            protocol_id: The protocol's UUID.
            user_id: The owning user's UUID.

        Returns:
            The ``Protocol`` instance.

        Raises:
            ProtocolNotFoundError: If the protocol does not exist or
                belongs to a different user.
        """
        result = await self.db.execute(
            select(Protocol).where(
                Protocol.id == protocol_id,
                Protocol.user_id == user_id,
            )
        )
        protocol = result.scalar_one_or_none()
        if protocol is None:
            raise ProtocolNotFoundError(str(protocol_id))
        return protocol

    # ── Delete ───────────────────────────────────────────────

    async def delete_protocol(
        self,
        protocol_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Delete a protocol and clean up associated resources.

        Removes the database record, the on-disk file, and any
        vectors stored in Pinecone under the protocol's namespace.

        Args:
            protocol_id: The protocol's UUID.
            user_id: The owning user's UUID.

        Raises:
            ProtocolNotFoundError: If the protocol does not exist.
        """
        protocol = await self.get_protocol(protocol_id, user_id)

        # Clean up Pinecone vectors
        if protocol.pinecone_namespace:
            try:
                from app.dependencies import get_pinecone_index

                index = get_pinecone_index()
                index.delete(delete_all=True, namespace=protocol.pinecone_namespace)
                logger.info(
                    "Deleted Pinecone namespace %s", protocol.pinecone_namespace
                )
            except Exception as exc:
                logger.warning(
                    "Failed to delete Pinecone namespace %s: %s",
                    protocol.pinecone_namespace,
                    exc,
                )

        # Clean up file
        if protocol.original_filename:
            upload_dir = UPLOAD_DIR / str(user_id)
            for f in upload_dir.glob(f"*_{protocol.original_filename}"):
                try:
                    f.unlink()
                except OSError:
                    logger.warning("Could not remove file %s", f)

        # Delete from DB (cascades will handle related records)
        await self.db.delete(protocol)
        await self.db.flush()

        # Invalidate any cached data
        await self.redis.delete(f"protocol:{protocol_id}")

        logger.info("Protocol %s deleted for user %s", protocol_id, user_id)

    # ── Ingestion trigger (called from background task) ──────

    @staticmethod
    async def run_ingestion(protocol_id: uuid.UUID) -> None:
        """Run the full ingestion pipeline for a protocol.

        This is intended to be called from a ``BackgroundTasks`` callback.
        It creates its own DB session so it can commit independently.

        Args:
            protocol_id: UUID of the protocol to ingest.
        """
        from app.dependencies import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            try:
                result = await session.execute(
                    select(Protocol).where(Protocol.id == protocol_id)
                )
                protocol = result.scalar_one_or_none()
                if protocol is None:
                    logger.error("Ingestion: Protocol %s not found", protocol_id)
                    return

                # Attempt to import and run the ingestion pipeline
                try:
                    from app.ingestion.pipeline import run_pipeline

                    await run_pipeline(protocol, session)
                except ImportError:
                    logger.warning(
                        "Ingestion pipeline not implemented yet. "
                        "Marking protocol %s as processed with stub data.",
                        protocol_id,
                    )
                    # Stub: mark as processed so the API doesn't hang
                    protocol.is_processed = True
                    protocol.step_count = 0
                    protocol.steps = []

                await session.commit()
                logger.info("Ingestion completed for protocol %s", protocol_id)
            except Exception as exc:
                await session.rollback()
                logger.error(
                    "Ingestion failed for protocol %s: %s", protocol_id, exc,
                    exc_info=True,
                )
