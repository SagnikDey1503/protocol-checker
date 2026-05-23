"""
Pinecone-backed semantic (vector) memory.

Stores memory entries as embeddings in a dedicated Pinecone namespace and
supports similarity-based recall for questions like *"Have I seen anything
like this before?"*.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.embeddings import embed_query, embed_texts
from app.core.exceptions import MemoryError
from app.models.enums import PineconeNamespace

logger = logging.getLogger(__name__)


# ── Data transfer object ─────────────────────────────────────


@dataclass
class MemoryEntry:
    """
    A single memory record returned by semantic recall.

    Attributes:
        id: Unique identifier of the memory vector.
        content: Original text content that was embedded.
        memory_type: Category tag (e.g. ``episodic``, ``semantic``).
        importance_score: Application-assigned importance (0–1).
        metadata: Arbitrary metadata stored alongside the vector.
        created_at: ISO-8601 timestamp of creation.
        similarity_score: Cosine similarity to the recall query (0–1).
    """

    id: str
    content: str
    memory_type: str
    importance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    similarity_score: float = 0.0


class SemanticMemory:
    """Pinecone-backed semantic memory for similarity search."""

    _NAMESPACE = PineconeNamespace.MEMORIES.value

    def __init__(self, pinecone_index: Any) -> None:
        """
        Initialise with a Pinecone ``Index`` object.

        Args:
            pinecone_index: Pinecone index instance obtained via
                            ``get_pinecone_index()``.
        """
        self._index = pinecone_index

    # ── Store ────────────────────────────────────────────────

    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> str:
        """
        Embed and upsert a memory entry into Pinecone.

        Args:
            user_id: Owning user UUID string.
            content: Text content to embed and store.
            memory_type: Category tag (see ``MemoryType`` enum).
            metadata: Extra metadata dict (merged into vector metadata).
            importance: Importance score (0–1).

        Returns:
            The generated vector id.
        """
        try:
            vector_id = str(uuid.uuid4())
            embedding = embed_texts([content])[0]

            pinecone_metadata: dict[str, Any] = {
                "user_id": user_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if metadata:
                pinecone_metadata.update(metadata)

            self._index.upsert(
                vectors=[
                    {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": pinecone_metadata,
                    }
                ],
                namespace=self._NAMESPACE,
            )
            logger.info(
                "Stored semantic memory %s for user %s (type=%s)",
                vector_id,
                user_id,
                memory_type,
            )
            return vector_id

        except Exception as exc:
            logger.error("Failed to store semantic memory: %s", exc)
            raise MemoryError("Failed to store semantic memory") from exc

    # ── Recall ───────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """
        Semantically search for memories relevant to ``query``.

        Only memories belonging to ``user_id`` are considered.

        Args:
            query: Natural-language search query.
            user_id: Owning user UUID string.
            limit: Max number of results.

        Returns:
            List of ``MemoryEntry`` objects sorted by descending similarity.
        """
        try:
            query_embedding = embed_query(query)

            results = self._index.query(
                vector=query_embedding,
                top_k=limit,
                namespace=self._NAMESPACE,
                include_metadata=True,
                filter={"user_id": {"$eq": user_id}},
            )

            entries: list[MemoryEntry] = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                entries.append(
                    MemoryEntry(
                        id=match["id"],
                        content=meta.get("content", ""),
                        memory_type=meta.get("memory_type", "unknown"),
                        importance_score=meta.get("importance", 0.5),
                        metadata={
                            k: v
                            for k, v in meta.items()
                            if k
                            not in {
                                "content",
                                "memory_type",
                                "importance",
                                "created_at",
                                "user_id",
                            }
                        },
                        created_at=meta.get("created_at"),
                        similarity_score=match.get("score", 0.0),
                    )
                )

            logger.debug(
                "Recalled %d memories for user %s (query=%r)",
                len(entries),
                user_id,
                query[:60],
            )
            return entries

        except Exception as exc:
            logger.error("Failed to recall semantic memories: %s", exc)
            raise MemoryError("Failed to recall semantic memories") from exc

    # ── Specialised recalls ──────────────────────────────────

    async def recall_similar_experiences(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """
        Find memories that represent *similar past experiences*.

        Filters to ``episodic`` and ``experiment`` memory types only.

        Args:
            query: Natural-language description of the current experience.
            user_id: Owning user UUID string.
            limit: Max number of results.

        Returns:
            List of ``MemoryEntry`` objects.
        """
        try:
            query_embedding = embed_query(query)

            results = self._index.query(
                vector=query_embedding,
                top_k=limit,
                namespace=self._NAMESPACE,
                include_metadata=True,
                filter={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {
                            "memory_type": {
                                "$in": ["episodic", "experiment"]
                            }
                        },
                    ]
                },
            )

            entries: list[MemoryEntry] = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                entries.append(
                    MemoryEntry(
                        id=match["id"],
                        content=meta.get("content", ""),
                        memory_type=meta.get("memory_type", "unknown"),
                        importance_score=meta.get("importance", 0.5),
                        metadata={
                            k: v
                            for k, v in meta.items()
                            if k
                            not in {
                                "content",
                                "memory_type",
                                "importance",
                                "created_at",
                                "user_id",
                            }
                        },
                        created_at=meta.get("created_at"),
                        similarity_score=match.get("score", 0.0),
                    )
                )

            logger.debug(
                "Recalled %d similar experiences for user %s",
                len(entries),
                user_id,
            )
            return entries

        except Exception as exc:
            logger.error("Failed to recall similar experiences: %s", exc)
            raise MemoryError(
                "Failed to recall similar experiences"
            ) from exc
