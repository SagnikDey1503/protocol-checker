"""
Document chunking strategies for the Research Protocol Assistant.

Provides three complementary chunking strategies for research protocols:

1. **Semantic chunking** — Splits text at natural semantic boundaries by
   measuring embedding similarity between sentences. Adjacent sentences
   with high similarity stay together; drops in similarity trigger splits.

2. **Parent-child chunking** — Creates large parent chunks for context
   and small child chunks for precise retrieval. Child chunks reference
   their parent via parent_chunk_id.

3. **Step-aware chunking** — Detects numbered protocol steps and keeps
   each step as an individual chunk, ideal for step-by-step guidance.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from app.config import get_settings
from app.core.embeddings import cosine_similarity, embed_texts
from app.core.exceptions import ChunkingError
from app.models.enums import ChunkType, SafetyLevel
from app.utils.text_cleaning import extract_numbered_steps, normalize_whitespace

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""

    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"
    STEP_AWARE = "step_aware"


@dataclass
class DocumentChunk:
    """
    A single chunk of document text with rich metadata.

    This is the core data structure that flows through the entire ingestion
    pipeline — from chunking through metadata extraction to embedding storage.
    """

    chunk_id: str = ""
    text: str = ""
    chunk_type: ChunkType = ChunkType.SECTION
    step_number: Optional[int] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    experiment_type: Optional[str] = None
    reagents: list[str] = field(default_factory=list)
    equipment: list[str] = field(default_factory=list)
    temperature: Optional[str] = None
    timing: Optional[str] = None
    safety_level: SafetyLevel = SafetyLevel.LOW
    dependencies: list[str] = field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate a unique chunk_id if not provided."""
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())

    @property
    def char_count(self) -> int:
        """Character count of the chunk text."""
        return len(self.text)

    def to_pinecone_metadata(self) -> dict[str, Any]:
        """
        Convert chunk metadata to a flat dict suitable for Pinecone.

        Pinecone metadata values must be strings, numbers, booleans,
        or lists of strings. This method flattens the dataclass
        accordingly.
        """
        meta: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "safety_level": self.safety_level.value,
            "text": self.text[:1000],  # Pinecone metadata size limit
        }
        if self.step_number is not None:
            meta["step_number"] = self.step_number
        if self.section_title:
            meta["section_title"] = self.section_title
        if self.page_number is not None:
            meta["page_number"] = self.page_number
        if self.experiment_type:
            meta["experiment_type"] = self.experiment_type
        if self.reagents:
            meta["reagents"] = self.reagents
        if self.equipment:
            meta["equipment"] = self.equipment
        if self.temperature:
            meta["temperature"] = self.temperature
        if self.timing:
            meta["timing"] = self.timing
        if self.dependencies:
            meta["dependencies"] = self.dependencies
        if self.parent_chunk_id:
            meta["parent_chunk_id"] = self.parent_chunk_id
        return meta


# ── Document Chunker ─────────────────────────────────────────────


class DocumentChunker:
    """
    Splits parsed documents into enriched chunks using configurable strategies.

    Reads chunk_size and chunk_overlap from application settings and
    supports three strategies: semantic, parent-child, and step-aware.

    Example::

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(parsed_doc, strategy=ChunkStrategy.STEP_AWARE)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.chunk_size: int = settings.chunk_size
        self.chunk_overlap: int = settings.chunk_overlap
        self._similarity_threshold: float = 0.5  # For semantic chunking

    def chunk_document(
        self,
        parsed_doc: Any,
        strategy: ChunkStrategy = ChunkStrategy.SEMANTIC,
    ) -> list[DocumentChunk]:
        """
        Split a parsed document into chunks using the specified strategy.

        Dispatches to the appropriate internal method based on the
        strategy parameter. Falls back to semantic chunking if the
        requested strategy produces no results.

        Args:
            parsed_doc: A ParsedDocument instance from the PDF parser.
            strategy: Chunking strategy to apply.

        Returns:
            List of DocumentChunk instances.

        Raises:
            ChunkingError: If chunking fails or produces no chunks.
        """
        text = parsed_doc.raw_text
        if not text or not text.strip():
            raise ChunkingError("Cannot chunk empty document text")

        # Build per-page metadata lookup
        page_metadata = self._build_page_metadata(parsed_doc)

        logger.info(
            "Chunking document (%d chars) with strategy=%s",
            len(text),
            strategy.value,
        )

        try:
            if strategy == ChunkStrategy.SEMANTIC:
                chunks = self._semantic_chunk(text, page_metadata)
            elif strategy == ChunkStrategy.PARENT_CHILD:
                chunks = self._parent_child_chunk(text, page_metadata)
            elif strategy == ChunkStrategy.STEP_AWARE:
                chunks = self._step_aware_chunk(text, page_metadata)
            else:
                raise ChunkingError(f"Unknown chunking strategy: {strategy}")

            # Handle table chunks separately
            table_chunks = self._chunk_tables(parsed_doc)
            chunks.extend(table_chunks)

            if not chunks:
                raise ChunkingError(
                    "Chunking produced no results — document may be too short "
                    "or not contain recognizable content"
                )

            logger.info(
                "Chunking complete: %d chunks (strategy=%s)",
                len(chunks),
                strategy.value,
            )
            return chunks

        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(f"Chunking failed: {e}") from e

    # ── Semantic Chunking ────────────────────────────────────────

    def _semantic_chunk(
        self,
        text: str,
        page_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """
        Split text at semantic boundaries using embedding similarity.

        The algorithm:
        1. Split text into sentences.
        2. Embed each sentence.
        3. Compute cosine similarity between adjacent sentences.
        4. Split where similarity drops below the threshold.
        5. Merge small resulting chunks to meet minimum size.

        Args:
            text: Full document text to chunk.
            page_metadata: Mapping of page numbers to metadata.

        Returns:
            List of semantically coherent DocumentChunks.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [self._make_chunk(text, page_metadata)]

        # For very short documents, just return one chunk
        if len(sentences) <= 3 or len(text) <= self.chunk_size:
            return [self._make_chunk(text, page_metadata)]

        # Embed all sentences
        try:
            embeddings = embed_texts([s for s in sentences])
        except Exception as e:
            logger.warning("Embedding failed for semantic chunking: %s. Falling back to fixed-size.", e)
            return self._fixed_size_chunk(text, page_metadata)

        # Find split points based on similarity drops
        split_indices: list[int] = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self._similarity_threshold:
                split_indices.append(i + 1)

        # Build chunks from split points
        chunks: list[DocumentChunk] = []
        prev_idx = 0

        for split_idx in split_indices:
            chunk_text = " ".join(sentences[prev_idx:split_idx]).strip()
            if chunk_text:
                chunks.append(self._make_chunk(chunk_text, page_metadata))
            prev_idx = split_idx

        # Add the last chunk
        remaining = " ".join(sentences[prev_idx:]).strip()
        if remaining:
            chunks.append(self._make_chunk(remaining, page_metadata))

        # Merge small chunks and split large ones
        chunks = self._enforce_size_constraints(chunks, page_metadata)

        return chunks

    # ── Parent-Child Chunking ────────────────────────────────────

    def _parent_child_chunk(
        self,
        text: str,
        page_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """
        Create large parent chunks with small child chunks.

        Parent chunks provide broader context (2x chunk_size), while
        child chunks (chunk_size) are used for precise retrieval.
        Each child stores a reference to its parent via parent_chunk_id.

        Args:
            text: Full document text to chunk.
            page_metadata: Mapping of page numbers to metadata.

        Returns:
            List containing both parent and child DocumentChunks.
        """
        parent_size = self.chunk_size * 2
        child_size = self.chunk_size
        overlap = self.chunk_overlap

        chunks: list[DocumentChunk] = []

        # Create parent chunks
        parents = self._split_text_fixed(text, parent_size, overlap)

        for parent_text in parents:
            parent_chunk = DocumentChunk(
                text=parent_text,
                chunk_type=ChunkType.SECTION,
                metadata={"is_parent": True},
            )

            # Assign page number from position
            self._assign_page_number(parent_chunk, text, parent_text, page_metadata)
            chunks.append(parent_chunk)

            # Create child chunks within this parent
            children = self._split_text_fixed(parent_text, child_size, overlap // 2)
            for child_text in children:
                child_chunk = DocumentChunk(
                    text=child_text,
                    chunk_type=ChunkType.SECTION,
                    parent_chunk_id=parent_chunk.chunk_id,
                    metadata={"is_parent": False},
                )
                self._assign_page_number(child_chunk, text, child_text, page_metadata)
                chunks.append(child_chunk)

        return chunks

    # ── Step-Aware Chunking ──────────────────────────────────────

    def _step_aware_chunk(
        self,
        text: str,
        page_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """
        Detect protocol steps and keep each as an individual chunk.

        Specifically designed for research protocols where each numbered
        step should remain as a cohesive unit. Non-step text (preamble,
        materials lists, notes) is chunked using fixed-size splitting.

        Args:
            text: Full document text to chunk.
            page_metadata: Mapping of page numbers to metadata.

        Returns:
            List of DocumentChunks, one per protocol step plus chunks
            for non-step content.
        """
        steps = extract_numbered_steps(text)

        if not steps:
            logger.info("No numbered steps found; falling back to semantic chunking")
            return self._semantic_chunk(text, page_metadata)

        chunks: list[DocumentChunk] = []

        # Find the position of the first step to extract preamble
        first_step_text = steps[0][1]
        first_step_pos = text.find(first_step_text)

        # Chunk the preamble (text before steps)
        if first_step_pos > 0:
            preamble = text[:first_step_pos].strip()
            if preamble:
                preamble_chunks = self._fixed_size_chunk(preamble, page_metadata)
                for pc in preamble_chunks:
                    pc.chunk_type = ChunkType.OVERVIEW
                    pc.section_title = "Preamble"
                chunks.extend(preamble_chunks)

        # Create a chunk for each step
        for step_num, step_text in steps:
            step_text = normalize_whitespace(step_text)

            if len(step_text) > self.chunk_size * 2:
                # Step is very long — split it but keep step association
                sub_chunks = self._split_text_fixed(step_text, self.chunk_size, self.chunk_overlap)
                for i, sub_text in enumerate(sub_chunks):
                    chunk = DocumentChunk(
                        text=sub_text,
                        chunk_type=ChunkType.STEP,
                        step_number=step_num,
                        section_title=f"Step {step_num} (part {i + 1})",
                        metadata={"step_part": i + 1, "total_parts": len(sub_chunks)},
                    )
                    self._assign_page_number(chunk, text, sub_text, page_metadata)
                    chunks.append(chunk)
            else:
                chunk = DocumentChunk(
                    text=step_text,
                    chunk_type=ChunkType.STEP,
                    step_number=step_num,
                    section_title=f"Step {step_num}",
                )
                self._assign_page_number(chunk, text, step_text, page_metadata)
                chunks.append(chunk)

        # Handle text after the last step (notes, cleanup, etc.)
        last_step_text = steps[-1][1]
        last_step_end = text.rfind(last_step_text)
        if last_step_end >= 0:
            epilogue_start = last_step_end + len(last_step_text)
            epilogue = text[epilogue_start:].strip()
            if epilogue and len(epilogue) > 50:  # Skip trivial trailing text
                epilogue_chunks = self._fixed_size_chunk(epilogue, page_metadata)
                for ec in epilogue_chunks:
                    ec.chunk_type = ChunkType.NOTE
                    ec.section_title = "Notes"
                chunks.extend(epilogue_chunks)

        return chunks

    # ── Helper Methods ───────────────────────────────────────────

    def _fixed_size_chunk(
        self,
        text: str,
        page_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """
        Split text into fixed-size chunks with overlap.

        A simple baseline chunking strategy that splits on word
        boundaries to avoid breaking mid-word.

        Args:
            text: Text to chunk.
            page_metadata: Mapping of page numbers to metadata.

        Returns:
            List of fixed-size DocumentChunks.
        """
        parts = self._split_text_fixed(text, self.chunk_size, self.chunk_overlap)
        chunks = []
        for part in parts:
            chunk = self._make_chunk(part, page_metadata)
            self._assign_page_number(chunk, text, part, page_metadata)
            chunks.append(chunk)
        return chunks

    def _split_text_fixed(
        self,
        text: str,
        size: int,
        overlap: int,
    ) -> list[str]:
        """
        Split text into fixed-size pieces with overlap, breaking at word boundaries.

        Args:
            text: Text to split.
            size: Maximum chunk size in characters.
            overlap: Number of characters to overlap between chunks.

        Returns:
            List of text pieces.
        """
        if len(text) <= size:
            return [text]

        words = text.split()
        pieces: list[str] = []
        current_piece: list[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current_piece else 0)  # +1 for space

            if current_len + word_len > size and current_piece:
                pieces.append(" ".join(current_piece))

                # Calculate overlap: keep last N characters worth of words
                overlap_words: list[str] = []
                overlap_len = 0
                for w in reversed(current_piece):
                    if overlap_len + len(w) + 1 > overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_len += len(w) + 1

                current_piece = overlap_words
                current_len = sum(len(w) for w in current_piece) + max(0, len(current_piece) - 1)

            current_piece.append(word)
            current_len += word_len

        if current_piece:
            pieces.append(" ".join(current_piece))

        return pieces

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex-based heuristics.

        Handles common abbreviations (Dr., Mr., etc.) and decimal
        numbers to avoid false splits.

        Args:
            text: Text to split into sentences.

        Returns:
            List of sentence strings.
        """
        # Protect abbreviations and decimal numbers from splitting
        protected = text
        abbreviations = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Fig.", "Eq.", "Vol.",
                         "No.", "vs.", "etc.", "e.g.", "i.e.", "approx.", "ca.",
                         "min.", "max.", "temp.", "conc.", "vol.", "wt.", "rpm."]
        placeholders: dict[str, str] = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholders[placeholder] = abbr
            protected = protected.replace(abbr, placeholder)

        # Split on sentence-ending punctuation followed by space + capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

        # Restore abbreviations
        result: list[str] = []
        for sent in sentences:
            for placeholder, abbr in placeholders.items():
                sent = sent.replace(placeholder, abbr)
            sent = sent.strip()
            if sent:
                result.append(sent)

        return result

    def _enforce_size_constraints(
        self,
        chunks: list[DocumentChunk],
        page_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """
        Merge small chunks and split oversized chunks.

        Ensures every chunk falls within the target size range:
        [chunk_size // 4, chunk_size * 1.5].

        Args:
            chunks: Chunks to size-adjust.
            page_metadata: Mapping of page numbers to metadata.

        Returns:
            Size-adjusted list of DocumentChunks.
        """
        min_size = self.chunk_size // 4
        max_size = int(self.chunk_size * 1.5)

        result: list[DocumentChunk] = []
        buffer_text = ""

        for chunk in chunks:
            if chunk.char_count < min_size:
                # Accumulate small chunks
                buffer_text += (" " if buffer_text else "") + chunk.text
                if len(buffer_text) >= min_size:
                    result.append(self._make_chunk(buffer_text, page_metadata))
                    buffer_text = ""
            elif chunk.char_count > max_size:
                # Flush buffer first
                if buffer_text:
                    result.append(self._make_chunk(buffer_text, page_metadata))
                    buffer_text = ""
                # Split oversized chunk
                sub_parts = self._split_text_fixed(
                    chunk.text, self.chunk_size, self.chunk_overlap
                )
                for part in sub_parts:
                    result.append(self._make_chunk(part, page_metadata))
            else:
                # Flush buffer first
                if buffer_text:
                    # Try merging buffer with this chunk
                    combined = buffer_text + " " + chunk.text
                    if len(combined) <= max_size:
                        result.append(self._make_chunk(combined, page_metadata))
                    else:
                        result.append(self._make_chunk(buffer_text, page_metadata))
                        result.append(chunk)
                    buffer_text = ""
                else:
                    result.append(chunk)

        # Flush remaining buffer
        if buffer_text:
            if result and result[-1].char_count + len(buffer_text) + 1 <= max_size:
                # Merge with last chunk
                result[-1].text += " " + buffer_text
            else:
                result.append(self._make_chunk(buffer_text, page_metadata))

        return result

    def _chunk_tables(self, parsed_doc: Any) -> list[DocumentChunk]:
        """
        Create chunks from extracted tables.

        Each table is converted to a text representation and stored
        as a TABLE-type chunk.

        Args:
            parsed_doc: ParsedDocument with tables attribute.

        Returns:
            List of table DocumentChunks.
        """
        chunks: list[DocumentChunk] = []
        tables = getattr(parsed_doc, "tables", [])

        for i, table in enumerate(tables):
            if not table:
                continue

            # Convert table to readable text format
            table_text = self._table_to_text(table)
            if not table_text or len(table_text) < 10:
                continue

            chunk = DocumentChunk(
                text=table_text,
                chunk_type=ChunkType.TABLE,
                section_title=f"Table {i + 1}",
                metadata={"table_index": i, "row_count": len(table)},
            )
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _table_to_text(table: list[list[str]]) -> str:
        """
        Convert a table (list of rows) to a readable text representation.

        Args:
            table: Table as list of rows, each row being a list of cells.

        Returns:
            Formatted text representation of the table.
        """
        if not table:
            return ""

        lines: list[str] = []
        # Use first row as header
        if len(table) > 1:
            header = " | ".join(table[0])
            lines.append(f"Table Header: {header}")
            separator = "-" * len(header)
            lines.append(separator)
            for row in table[1:]:
                lines.append(" | ".join(row))
        else:
            lines.append(" | ".join(table[0]))

        return "\n".join(lines)

    @staticmethod
    def _make_chunk(text: str, page_metadata: dict[str, Any]) -> DocumentChunk:
        """Create a basic DocumentChunk from text."""
        return DocumentChunk(
            text=normalize_whitespace(text),
            chunk_type=ChunkType.SECTION,
        )

    def _assign_page_number(
        self,
        chunk: DocumentChunk,
        full_text: str,
        chunk_text: str,
        page_metadata: dict[str, Any],
    ) -> None:
        """
        Estimate which page a chunk belongs to based on its position.

        Uses the character offset of the chunk text within the full
        document text to estimate the page number.

        Args:
            chunk: The chunk to annotate with page_number.
            full_text: Full document text.
            chunk_text: The chunk's text to locate.
            page_metadata: Mapping with 'page_offsets' key.
        """
        if not page_metadata or "page_offsets" not in page_metadata:
            return

        # Find position of chunk text in full document
        pos = full_text.find(chunk_text[:100])
        if pos < 0:
            return

        page_offsets = page_metadata["page_offsets"]
        for page_num, (start, end) in page_offsets.items():
            if start <= pos < end:
                chunk.page_number = int(page_num)
                return

    @staticmethod
    def _build_page_metadata(parsed_doc: Any) -> dict[str, Any]:
        """
        Build a page offset map from a ParsedDocument.

        Creates a mapping of page numbers to their character offset
        ranges within the concatenated raw text.

        Args:
            parsed_doc: ParsedDocument with pages attribute.

        Returns:
            Dict with 'page_offsets' mapping page_num → (start, end).
        """
        pages = getattr(parsed_doc, "pages", [])
        if not pages:
            return {}

        offsets: dict[int, tuple[int, int]] = {}
        current_offset = 0

        for page in pages:
            page_len = len(page.text) + 2  # +2 for \n\n separator
            offsets[page.page_number] = (current_offset, current_offset + page_len)
            current_offset += page_len

        return {"page_offsets": offsets}
