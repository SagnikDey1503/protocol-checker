"""
Memory system for the AI Research Protocol Assistant.

Provides a multi-tier memory architecture:
- **WorkingMemory**: Redis-backed short-term session state and conversation buffers.
- **PersistentMemory**: PostgreSQL long-term storage for conversations, experiments, patterns.
- **SemanticMemory**: Pinecone vector-based semantic recall for similarity search.
- **EpisodicMemoryStore**: Dual-storage (DB + vector) episode recording and recall.
- **MemoryManager**: Unified façade that routes reads/writes across all tiers.
"""

from app.memory.episodic_memory import EpisodicMemoryStore
from app.memory.manager import MemoryManager
from app.memory.persistent_memory import PersistentMemory
from app.memory.semantic_memory import MemoryEntry, SemanticMemory
from app.memory.working_memory import WorkingMemory

__all__ = [
    "WorkingMemory",
    "PersistentMemory",
    "SemanticMemory",
    "MemoryEntry",
    "EpisodicMemoryStore",
    "MemoryManager",
]
