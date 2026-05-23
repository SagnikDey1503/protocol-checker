"""
Pydantic schemas for API request/response validation.

These are the data transfer objects (DTOs) that flow between
the API layer and internal services.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── Auth ─────────────────────────────────────────────────────


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Protocols ────────────────────────────────────────────────


class ProtocolUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    experiment_type: Optional[str]
    original_filename: str
    step_count: int
    is_processed: bool
    created_at: datetime


class ProtocolDetailResponse(ProtocolUploadResponse):
    steps: Optional[list[dict[str, Any]]] = None
    reagents: Optional[list[str]] = None
    equipment: Optional[list[str]] = None
    metadata_: Optional[dict[str, Any]] = Field(None, alias="metadata")


class ProtocolListResponse(BaseModel):
    protocols: list[ProtocolUploadResponse]
    total: int


# ── Experiment Sessions ──────────────────────────────────────


class ExperimentStartRequest(BaseModel):
    protocol_id: Optional[uuid.UUID] = None
    title: Optional[str] = None
    experiment_description: Optional[str] = None


class ExperimentStepUpdate(BaseModel):
    step_number: int
    notes: Optional[str] = None
    deviation: Optional[str] = None


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    protocol_id: Optional[uuid.UUID]
    title: Optional[str]
    status: str
    current_step: int
    total_steps: int
    deviations: Optional[list[dict[str, Any]]]
    timeline: Optional[list[dict[str, Any]]]
    started_at: datetime
    completed_at: Optional[datetime]


class ExperimentTimelineResponse(BaseModel):
    experiment_id: uuid.UUID
    steps: list[dict[str, Any]]
    deviations: list[dict[str, Any]]
    duration_seconds: Optional[float]


# ── Chat ─────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[uuid.UUID] = None
    experiment_id: Optional[uuid.UUID] = None


class SourceCitation(BaseModel):
    chunk_id: str
    text: str
    protocol_title: Optional[str] = None
    page_number: Optional[int] = None
    step_number: Optional[int] = None
    relevance_score: float = 0.0


class SafetyAlert(BaseModel):
    level: str  # low, medium, high, critical
    message: str
    related_step: Optional[int] = None
    reagents: Optional[list[str]] = None


class ChatResponse(BaseModel):
    message: str
    conversation_id: uuid.UUID
    agent_type: str
    confidence: float = 0.0
    sources: list[SourceCitation] = []
    safety_alerts: list[SafetyAlert] = []
    next_steps: list[str] = []
    deviation_detected: bool = False
    experiment_state: Optional[dict[str, Any]] = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    sources: Optional[list[dict[str, Any]]]
    confidence: Optional[float]
    agent_type: Optional[str]
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    conversation_id: uuid.UUID
    messages: list[MessageResponse]
    total: int


# ── Memory ───────────────────────────────────────────────────


class MemoryRecallRequest(BaseModel):
    query: str
    memory_types: list[str] = ["episodic", "semantic"]
    limit: int = Field(5, ge=1, le=20)


class MemoryEntry(BaseModel):
    id: str
    memory_type: str
    content: str
    importance_score: float
    created_at: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None


class MemoryRecallResponse(BaseModel):
    memories: list[MemoryEntry]
    total: int


class UserProfileResponse(BaseModel):
    user_id: uuid.UUID
    total_experiments: int
    completed_experiments: int
    common_experiment_types: list[str]
    patterns: list[dict[str, Any]]
    skill_assessment: Optional[str] = None


# ── Health ───────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    services: dict[str, str]  # service_name -> "connected" | "disconnected"


# ── Protocol Recommendation ─────────────────────────────────


class ProtocolRecommendationRequest(BaseModel):
    description: str = Field(..., min_length=5)
    difficulty_level: Optional[str] = None  # beginner, intermediate, advanced
    available_equipment: Optional[list[str]] = None


class ProtocolRecommendation(BaseModel):
    title: str
    experiment_type: str
    difficulty: str
    description: str
    required_materials: list[str]
    estimated_duration: Optional[str] = None
    source: Optional[str] = None
    relevance_score: float = 0.0


class ProtocolRecommendationResponse(BaseModel):
    recommendations: list[ProtocolRecommendation]
    total: int
