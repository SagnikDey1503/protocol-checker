"""
Enumerations used across the Research Protocol Assistant.
"""

from __future__ import annotations

import enum


class ExperimentStatus(str, enum.Enum):
    """Lifecycle states for an experiment session."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ChunkType(str, enum.Enum):
    """Categories of document chunks produced by the ingestion pipeline."""

    STEP = "step"
    SECTION = "section"
    NOTE = "note"
    SAFETY = "safety"
    TABLE = "table"
    OVERVIEW = "overview"


class SafetyLevel(str, enum.Enum):
    """Risk classification for protocol steps and chunks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryType(str, enum.Enum):
    """Categories of memories stored by the memory system."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERSISTENT = "persistent"


class EpisodeType(str, enum.Enum):
    """Classification of episodic memory entries."""

    EXPERIMENT = "experiment"
    MISTAKE = "mistake"
    SUCCESS = "success"
    LEARNING = "learning"
    DEVIATION = "deviation"


class PatternType(str, enum.Enum):
    """Types of user behavioural patterns tracked by the memory agent."""

    COMMON_MISTAKE = "common_mistake"
    PREFERENCE = "preference"
    SKILL_LEVEL = "skill_level"
    EXPERIMENT_TYPE = "experiment_type"


class QueryType(str, enum.Enum):
    """Intent classifications for incoming user queries."""

    PROTOCOL_QUESTION = "protocol_question"
    EXPERIMENT_UPDATE = "experiment_update"
    SAFETY_CONCERN = "safety_concern"
    TROUBLESHOOTING = "troubleshooting"
    CONCEPTUAL_QUESTION = "conceptual_question"
    PROTOCOL_REQUEST = "protocol_request"
    GENERAL_CHAT = "general_chat"


class MessageRole(str, enum.Enum):
    """Roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PineconeNamespace(str, enum.Enum):
    """Logical namespaces inside the Pinecone index."""

    PROTOCOLS = "protocols"
    MEMORIES = "memories"
    SAFETY = "safety"
    KNOWLEDGE = "knowledge"
