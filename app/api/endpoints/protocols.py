"""
Protocol management API endpoints.
Handles PDF uploading, listing, fetching, deleting, and step retrieval.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
    BackgroundTasks,
    status,
)
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis, get_current_user
from app.models.database import User
from app.models.schemas import (
    ProtocolUploadResponse,
    ProtocolDetailResponse,
    ProtocolListResponse,
)
from app.services.protocol_service import ProtocolService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    response_model=ProtocolUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Protocols"],
)
async def upload_protocol(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    experiment_type: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Upload a protocol PDF.
    Saves the file and triggers the background ingestion pipeline.
    """
    service = ProtocolService(db, redis)
    file_content = await file.read()

    # Create protocol record
    protocol = await service.upload_protocol(
        file_content=file_content,
        filename=file.filename or "uploaded_protocol.pdf",
        user_id=current_user.id,
        title=title,
    )

    # Queue background task if the protocol is new/unprocessed
    if not protocol.is_processed:
        if experiment_type:
            protocol.experiment_type = experiment_type
            db.add(protocol)
            await db.flush()

        background_tasks.add_task(ProtocolService.run_ingestion, protocol.id)
        logger.info("Queued background ingestion for protocol %s", protocol.id)

    return protocol


@router.get("/", response_model=ProtocolListResponse, tags=["Protocols"])
async def list_protocols(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    List all protocols uploaded by the current user.
    """
    service = ProtocolService(db, redis)
    protocols = await service.get_protocols(current_user.id)
    return ProtocolListResponse(protocols=protocols, total=len(protocols))


@router.get("/{protocol_id}", response_model=ProtocolDetailResponse, tags=["Protocols"])
async def get_protocol(
    protocol_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get detailed information about a specific protocol.
    """
    service = ProtocolService(db, redis)
    protocol = await service.get_protocol(protocol_id, current_user.id)
    
    # Map model metadata_ to schema metadata
    metadata = getattr(protocol, "metadata_", {})
    
    return ProtocolDetailResponse(
        id=protocol.id,
        title=protocol.title,
        experiment_type=protocol.experiment_type,
        original_filename=protocol.original_filename,
        step_count=protocol.step_count,
        is_processed=protocol.is_processed,
        created_at=protocol.created_at,
        steps=protocol.steps or [],
        reagents=protocol.reagents or [],
        equipment=protocol.equipment or [],
        metadata=metadata,
    )


@router.get("/{protocol_id}/steps", tags=["Protocols"])
async def get_protocol_steps(
    protocol_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get the structured steps of a specific protocol.
    """
    service = ProtocolService(db, redis)
    protocol = await service.get_protocol(protocol_id, current_user.id)
    return {"protocol_id": protocol.id, "steps": protocol.steps or [], "step_count": protocol.step_count}


@router.delete("/{protocol_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Protocols"])
async def delete_protocol(
    protocol_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Delete a protocol, its uploaded file, and its Pinecone vectors.
    """
    service = ProtocolService(db, redis)
    await service.delete_protocol(protocol_id, current_user.id)
