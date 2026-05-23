"""
PDF text extraction for the Research Protocol Assistant.

Provides a robust PDF parser that uses `pdfplumber` as the primary engine.
Extracted content is returned as a structured ParsedDocument dataclass.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.core.exceptions import PDFParsingError
from app.utils.text_cleaning import clean_ocr_text, normalize_whitespace

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────


@dataclass
class PageContent:
    """Represents the extracted content of a single PDF page."""

    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SectionContent:
    """Represents a detected document section."""

    title: str
    content: str
    page_range: tuple[int, int] = (0, 0)
    level: int = 1


@dataclass
class ParsedDocument:
    """
    Complete parsed output from a PDF document.

    Contains the structured extraction results including per-page text,
    detected sections, extracted tables, raw concatenated text, and
    document metadata (page count, file hash, etc.).
    """

    pages: list[PageContent] = field(default_factory=list)
    sections: list[SectionContent] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return len(self.pages)

    @property
    def has_tables(self) -> bool:
        """Whether any tables were extracted."""
        return len(self.tables) > 0

    @property
    def total_text_length(self) -> int:
        """Total character count of raw text."""
        return len(self.raw_text)


# ── PDF Parser ───────────────────────────────────────────────────


class PDFParser:
    """
    Robust PDF text extraction using pdfplumber.

    Uses ``pdfplumber`` as the primary parser optimized for text and table extraction.

    Example::

        parser = PDFParser()
        doc = await parser.parse("/path/to/protocol.pdf")
        print(doc.raw_text[:500])
        print(f"Found {len(doc.tables)} tables across {doc.page_count} pages")
    """

    async def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF file and return structured content.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            ParsedDocument with pages, sections, tables, raw text, and metadata.

        Raises:
            PDFParsingError: If the file doesn't exist, isn't a PDF, or
                the parser fails to extract any text.
        """
        file_path = str(Path(file_path).resolve())

        # Validate file
        if not os.path.exists(file_path):
            raise PDFParsingError(f"PDF file not found: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise PDFParsingError(f"File is not a PDF: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise PDFParsingError(f"PDF file is empty: {file_path}")

        import time

        logger.info("Parsing PDF: %s (%.2f MB)", file_path, file_size / 1_048_576)
        parse_start = time.monotonic()

        # Compute file hash for deduplication
        file_hash = self._compute_file_hash(file_path)

        parsed_doc = ParsedDocument(
            metadata={
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size_bytes": file_size,
                "file_hash": file_hash,
            }
        )

        try:
            primary_start = time.monotonic()
            primary_result = await self._parse_with_pdfplumber(file_path)
            parsed_doc.pages = primary_result.pages
            parsed_doc.sections = primary_result.sections
            parsed_doc.raw_text = primary_result.raw_text
            parsed_doc.tables = primary_result.tables
            parsed_doc.metadata.update(primary_result.metadata)
            logger.info(
                "pdfplumber parser extracted %d pages, %d chars (%.2fs)",
                len(parsed_doc.pages),
                len(parsed_doc.raw_text),
                time.monotonic() - primary_start,
            )
        except Exception as e:
            logger.warning("pdfplumber parser failed: %s", e)

        # Final validation
        if not parsed_doc.raw_text:
            raise PDFParsingError(
                f"Parser failed to extract text from: {file_path}"
            )

        # Clean the extracted text
        parsed_doc.raw_text = self._clean_text(parsed_doc.raw_text)
        for page in parsed_doc.pages:
            page.text = self._clean_text(page.text)

        parsed_doc.metadata["total_chars"] = len(parsed_doc.raw_text)
        parsed_doc.metadata["page_count"] = len(parsed_doc.pages)
        parsed_doc.metadata["table_count"] = len(parsed_doc.tables)

        logger.info(
            "PDF parsing complete: %d pages, %d tables, %d chars",
            parsed_doc.page_count,
            len(parsed_doc.tables),
            parsed_doc.total_text_length,
        )

        logger.info("PDF parsing total time: %.2fs", time.monotonic() - parse_start)

        return parsed_doc

    async def _parse_with_pdfplumber(self, file_path: str) -> ParsedDocument:
        """
        Parse PDF using ``pdfplumber``, optimized for table extraction.

        pdfplumber excels at detecting and extracting tabular data from
        PDFs. It also provides reliable page-level text extraction for
        text-based (non-scanned) PDFs.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            ParsedDocument populated from pdfplumber page data.

        Raises:
            PDFParsingError: If pdfplumber fails to open/read the PDF.
        """
        import asyncio

        try:
            import pdfplumber
        except ImportError as exc:
            raise PDFParsingError(
                "pdfplumber is not installed. Install with: pip install pdfplumber"
            ) from exc

        def _extract() -> ParsedDocument:
            pages: list[PageContent] = []
            tables: list[list[list[str]]] = []
            all_text_parts: list[str] = []

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    page_text = page.extract_text() or ""
                    page_tables: list[list[list[str]]] = []

                    # Extract tables
                    raw_tables = page.extract_tables() or []
                    for raw_table in raw_tables:
                        if raw_table:
                            cleaned_table = [
                                [str(cell) if cell is not None else "" for cell in row]
                                for row in raw_table
                                if row  # Skip empty rows
                            ]
                            if cleaned_table:
                                page_tables.append(cleaned_table)
                                tables.append(cleaned_table)

                    pages.append(
                        PageContent(
                            page_number=page_num,
                            text=page_text,
                            tables=page_tables,
                        )
                    )
                    if page_text:
                        all_text_parts.append(page_text)

            return ParsedDocument(
                pages=pages,
                tables=tables,
                raw_text="\n\n".join(all_text_parts),
                metadata={
                    "parser": "pdfplumber",
                    "page_count": len(pages),
                },
            )

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _extract)
        except Exception as e:
            raise PDFParsingError(
                f"pdfplumber extraction failed: {e}"
            ) from e

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing OCR artifacts and normalizing whitespace.

        Applies the project-wide text cleaning utilities in sequence:
        1. OCR artifact removal (ligatures, curly quotes, broken hyphens)
        2. Whitespace normalization (collapse runs, strip lines)

        Args:
            text: Raw extracted text from a PDF parser.

        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""
        text = clean_ocr_text(text)
        text = normalize_whitespace(text)
        return text

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """
        Compute SHA-256 hash of a file for deduplication.

        Reads the file in 8KB chunks to handle large PDFs efficiently.

        Args:
            file_path: Path to the file.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _is_duplicate_table(
        candidate: list[list[str]],
        existing_tables: list[list[list[str]]],
    ) -> bool:
        """
        Check whether a table is a duplicate of any existing table.

        Uses a simple text-flattening approach: if the concatenated cell
        text of two tables is identical, they are considered duplicates.

        Args:
            candidate: The table to check.
            existing_tables: List of tables already collected.

        Returns:
            True if the candidate is a duplicate.
        """
        candidate_text = " ".join(
            " ".join(cell for cell in row) for row in candidate
        ).strip()

        for existing in existing_tables:
            existing_text = " ".join(
                " ".join(cell for cell in row) for row in existing
            ).strip()
            if candidate_text == existing_text:
                return True

        return False
