"""
PDF text extraction for the Research Protocol Assistant.

Provides a robust PDF parser that uses `unstructured` as the primary engine
(with hi-res strategy for layout-aware extraction) and falls back to
`pdfplumber` for table-heavy documents. Extracted content is returned as
a structured ParsedDocument dataclass.
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
    Robust PDF text extraction with multi-strategy parsing.

    Uses ``unstructured`` with hi-res strategy as the primary parser
    for layout-aware extraction, and ``pdfplumber`` as a secondary
    parser optimized for table extraction. Results from both are
    merged into a unified ParsedDocument.

    Example::

        parser = PDFParser()
        doc = await parser.parse("/path/to/protocol.pdf")
        print(doc.raw_text[:500])
        print(f"Found {len(doc.tables)} tables across {doc.page_count} pages")
    """

    async def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF file and return structured content.

        This is the main entry point. It validates the file, runs both
        the unstructured and pdfplumber parsers, merges results, and
        returns a fully populated ParsedDocument.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            ParsedDocument with pages, sections, tables, raw text, and metadata.

        Raises:
            PDFParsingError: If the file doesn't exist, isn't a PDF, or
                both parsers fail to extract any text.
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

        # Parse with primary strategy (unstructured)
        parsed_doc = ParsedDocument(
            metadata={
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size_bytes": file_size,
                "file_hash": file_hash,
            }
        )

        # Try primary parser
        try:
            primary_start = time.monotonic()
            primary_result = await self._parse_with_unstructured(file_path)
            parsed_doc.pages = primary_result.pages
            parsed_doc.sections = primary_result.sections
            parsed_doc.raw_text = primary_result.raw_text
            parsed_doc.tables = primary_result.tables
            parsed_doc.metadata.update(primary_result.metadata)
            logger.info(
                "Unstructured parser extracted %d pages, %d chars (%.2fs)",
                len(parsed_doc.pages),
                len(parsed_doc.raw_text),
                time.monotonic() - primary_start,
            )
        except Exception as e:
            logger.warning("Unstructured parser failed: %s. Falling back.", e)

        # Enhance tables with pdfplumber (always try, it may find tables the
        # primary parser missed)
        try:
            table_start = time.monotonic()
            table_result = await self._parse_with_pdfplumber(file_path)
            if table_result.tables:
                # Merge tables — deduplicate by content similarity
                existing_table_count = len(parsed_doc.tables)
                for table in table_result.tables:
                    if table and not self._is_duplicate_table(table, parsed_doc.tables):
                        parsed_doc.tables.append(table)
                logger.info(
                    "pdfplumber added %d new tables (%.2fs)",
                    len(parsed_doc.tables) - existing_table_count,
                    time.monotonic() - table_start,
                )

            # If primary parser failed, use pdfplumber text as fallback
            if not parsed_doc.raw_text and table_result.raw_text:
                parsed_doc.pages = table_result.pages
                parsed_doc.raw_text = table_result.raw_text
                parsed_doc.metadata["parser_fallback"] = "pdfplumber"
                logger.info(
                    "Using pdfplumber text as fallback (%d chars, %.2fs)",
                    len(parsed_doc.raw_text),
                    time.monotonic() - table_start,
                )
        except Exception as e:
            logger.warning("pdfplumber parser failed: %s", e)

        # Final validation
        if not parsed_doc.raw_text:
            raise PDFParsingError(
                f"Both parsers failed to extract text from: {file_path}"
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

    async def _parse_with_unstructured(self, file_path: str) -> ParsedDocument:
        """
        Parse PDF using the ``unstructured`` library with hi-res strategy.

        This strategy uses layout detection and OCR for scanned documents,
        producing high-quality structured output with element type
        classification (Title, NarrativeText, Table, ListItem, etc.).

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            ParsedDocument populated from unstructured elements.

        Raises:
            PDFParsingError: If unstructured partitioning fails.
        """
        import asyncio

        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise PDFParsingError(
                "unstructured[pdf] is not installed. "
                "Install with: pip install 'unstructured[pdf]'"
            ) from exc

        try:
            # partition_pdf is synchronous — run in thread pool
            loop = asyncio.get_event_loop()
            elements = await loop.run_in_executor(
                None,
                lambda: partition_pdf(
                    filename=file_path,
                    strategy="hi_res",
                    include_page_breaks=True,
                    infer_table_structure=True,
                ),
            )
        except Exception as e:
            raise PDFParsingError(
                f"unstructured partition_pdf failed: {e}"
            ) from e

        # Organize elements by page
        pages_dict: dict[int, PageContent] = {}
        sections: list[SectionContent] = []
        tables: list[list[list[str]]] = []
        all_text_parts: list[str] = []

        for element in elements:
            elem_metadata = element.metadata if hasattr(element, "metadata") else None
            page_num = getattr(elem_metadata, "page_number", 1) if elem_metadata else 1
            elem_type = type(element).__name__
            elem_text = str(element).strip()

            if not elem_text:
                continue

            # Initialize page if needed
            if page_num not in pages_dict:
                pages_dict[page_num] = PageContent(page_number=page_num, text="")

            pages_dict[page_num].text += elem_text + "\n"
            all_text_parts.append(elem_text)

            # Detect sections from Title elements
            if elem_type == "Title":
                sections.append(
                    SectionContent(
                        title=elem_text,
                        content="",
                        page_range=(page_num, page_num),
                    )
                )
            elif sections:
                # Accumulate content into the current section
                sections[-1].content += elem_text + "\n"
                sections[-1].page_range = (
                    sections[-1].page_range[0],
                    page_num,
                )

            # Extract tables
            if elem_type == "Table":
                table_data = self._parse_table_element(element)
                if table_data:
                    tables.append(table_data)
                    pages_dict[page_num].tables.append(table_data)

        # Build sorted page list
        pages = [pages_dict[k] for k in sorted(pages_dict.keys())]

        return ParsedDocument(
            pages=pages,
            sections=sections,
            tables=tables,
            raw_text="\n\n".join(all_text_parts),
            metadata={"parser": "unstructured", "element_count": len(elements)},
        )

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
    def _parse_table_element(element: Any) -> Optional[list[list[str]]]:
        """
        Attempt to extract structured table data from an unstructured Table element.

        Falls back to splitting the element's text by newlines and tabs
        if the element doesn't have structured metadata.

        Args:
            element: An unstructured Table element.

        Returns:
            Table as a list of rows (each row is a list of cell strings),
            or None if the table is empty.
        """
        text = str(element).strip()
        if not text:
            return None

        # Try to get structured table from metadata
        metadata = element.metadata if hasattr(element, "metadata") else None
        if metadata and hasattr(metadata, "text_as_html"):
            try:
                return PDFParser._parse_html_table(metadata.text_as_html)
            except Exception:
                pass

        # Fallback: split text into rows/columns
        rows = text.split("\n")
        table: list[list[str]] = []
        for row in rows:
            cells = [c.strip() for c in row.split("\t") if c.strip()]
            if not cells:
                cells = [c.strip() for c in row.split("  ") if c.strip()]
            if cells:
                table.append(cells)

        return table if table else None

    @staticmethod
    def _parse_html_table(html: str) -> Optional[list[list[str]]]:
        """
        Parse an HTML table string into a list of rows.

        Args:
            html: HTML string containing a <table> element.

        Returns:
            Parsed table as list of rows, or None.
        """
        import re

        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
        if not rows:
            return None

        table: list[list[str]] = []
        for row_html in rows:
            cells = re.findall(
                r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>",
                row_html,
                re.DOTALL | re.IGNORECASE,
            )
            cleaned_cells = [
                re.sub(r"<[^>]+>", "", cell).strip() for cell in cells
            ]
            if cleaned_cells:
                table.append(cleaned_cells)

        return table if table else None

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
