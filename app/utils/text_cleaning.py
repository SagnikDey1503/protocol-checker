"""
Text cleaning and normalization utilities for the Research Protocol Assistant.

Provides functions to clean OCR artifacts, normalize whitespace, extract
numbered steps from protocol text, and split documents by section headers.
These utilities are used heavily by the ingestion pipeline to prepare raw
PDF text for chunking and embedding.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


# ── Common OCR artifact patterns ────────────────────────────────
_OCR_REPLACEMENTS: list[tuple[str, str]] = [
    # Ligature normalization
    ("\ufb01", "fi"),
    ("\ufb02", "fl"),
    ("\ufb03", "ffi"),
    ("\ufb04", "ffl"),
    ("\ufb00", "ff"),
    # Mis-recognized characters
    ("\u2018", "'"),   # Left single curly quote → straight
    ("\u2019", "'"),   # Right single curly quote → straight
    ("\u201c", '"'),   # Left double curly quote → straight
    ("\u201d", '"'),   # Right double curly quote → straight
    ("\u2013", "-"),   # En dash → hyphen
    ("\u2014", "--"),  # Em dash → double hyphen
    ("\u2026", "..."), # Ellipsis
    ("\u00b0", "°"),   # Degree sign normalization
    ("\u00b5", "µ"),   # Micro sign normalization
    ("\u2022", "•"),   # Bullet normalization
]

# Pattern for broken hyphenation across lines (e.g., "cen-\ntrifuge" → "centrifuge")
_HYPHENATION_PATTERN = re.compile(r"(\w+)-\s*\n\s*(\w+)")

# Pattern for multiple consecutive blank lines
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")

# Pattern for multiple spaces (but not newlines)
_MULTI_SPACES = re.compile(r"[ \t]{2,}")

# Pattern for page number artifacts (e.g., "Page 3 of 15", "- 3 -", "3")
_PAGE_NUMBER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|-\s*\d+\s*-|\[\d+\])\s*(?:\n|$)",
    re.IGNORECASE,
)

# Pattern for header/footer repetitions (common in scanned PDFs)
_HEADER_FOOTER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Confidential|Draft|Internal Use Only|CONFIDENTIAL)\s*(?:\n|$)",
    re.IGNORECASE,
)

# ── Section header patterns ─────────────────────────────────────
_SECTION_HEADER_PATTERNS = [
    # Numbered sections: "1. Introduction", "2.1 Methods", "III. Results"
    re.compile(
        r"^(?P<header>(?:\d+\.)+\s*.+?)$",
        re.MULTILINE,
    ),
    # All-caps headers: "INTRODUCTION", "MATERIALS AND METHODS"
    re.compile(
        r"^(?P<header>[A-Z][A-Z\s&]{2,})$",
        re.MULTILINE,
    ),
    # Title-case headers with colon or on their own line
    re.compile(
        r"^(?P<header>(?:Abstract|Introduction|Background|Objective|Purpose|"
        r"Materials?\s*(?:and|&)\s*Methods?|Reagents?|Equipment|"
        r"Procedure|Protocol|Steps?|Methods?|Results?|Discussion|"
        r"Conclusion|References?|Appendix|Notes?|"
        r"Safety\s*(?:Precautions?|Notes?|Information)?|"
        r"Troubleshooting|Tips?|Warnings?|Caution|"
        r"Preparation|Setup|Cleanup|Storage|Disposal)"
        r"[:\s]*.*)$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# ── Step numbering patterns ─────────────────────────────────────
_STEP_PATTERNS = [
    # "Step 1:", "Step 1.", "Step 1 -"
    re.compile(
        r"(?:^|\n)\s*(?:Step\s+)?(\d+)\s*[.:)\-]\s*(.+?)(?=\n\s*(?:Step\s+)?\d+\s*[.:)\-]|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Bulleted with numbers: "1) ...", "1. ..."
    re.compile(
        r"(?:^|\n)\s*(\d+)\s*[.)]\s+(.+?)(?=\n\s*\d+\s*[.)]|\Z)",
        re.DOTALL,
    ),
    # Roman numeral steps: "i. ...", "ii. ...", "I. ...", "II. ..."
    re.compile(
        r"(?:^|\n)\s*((?:i{1,3}|iv|vi{0,3}|ix|x{0,3}))\s*[.)]\s+(.+?)(?=\n\s*(?:i{1,3}|iv|vi{0,3}|ix|x{0,3})\s*[.)]|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Letter steps: "a. ...", "b. ...", "a) ..."
    re.compile(
        r"(?:^|\n)\s*([a-z])\s*[.)]\s+(.+?)(?=\n\s*[a-z]\s*[.)]|\Z)",
        re.DOTALL,
    ),
]

# Roman numeral lookup for conversion
_ROMAN_MAP = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}


def clean_ocr_text(text: str) -> str:
    """
    Remove common OCR artifacts from extracted PDF text.

    Handles ligature normalization, curly quote conversion, broken
    hyphenation across lines, page number removal, and Unicode
    normalization (NFC form).

    Args:
        text: Raw text extracted from a PDF via OCR or text layer.

    Returns:
        Cleaned text with OCR artifacts removed.
    """
    if not text:
        return ""

    result = text

    # Apply character-level replacements
    for old, new in _OCR_REPLACEMENTS:
        result = result.replace(old, new)

    # Fix broken hyphenation across lines
    result = _HYPHENATION_PATTERN.sub(r"\1\2", result)

    # Remove page number artifacts
    result = _PAGE_NUMBER_PATTERN.sub("\n", result)

    # Remove common header/footer boilerplate
    result = _HEADER_FOOTER_PATTERN.sub("\n", result)

    # Normalize Unicode to NFC form (composed characters)
    result = unicodedata.normalize("NFC", result)

    # Remove null bytes and other control characters (except newline/tab)
    result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", result)

    logger.debug("Cleaned OCR text: %d → %d chars", len(text), len(result))
    return result


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces and newlines into canonical form.

    Reduces runs of 2+ spaces/tabs to a single space, and runs
    of 3+ newlines to a double newline (paragraph break). Strips
    leading/trailing whitespace from each line.

    Args:
        text: Text with potentially irregular whitespace.

    Returns:
        Text with normalized whitespace.
    """
    if not text:
        return ""

    # Replace tabs with spaces
    result = text.replace("\t", " ")

    # Collapse multiple spaces to single space (preserve newlines)
    result = _MULTI_SPACES.sub(" ", result)

    # Strip each line individually
    lines = [line.strip() for line in result.splitlines()]
    result = "\n".join(lines)

    # Collapse 3+ consecutive blank lines to 2 (paragraph break)
    result = _MULTI_BLANK_LINES.sub("\n\n", result)

    # Strip leading/trailing whitespace from the whole text
    result = result.strip()

    return result


def extract_numbered_steps(text: str) -> list[tuple[int, str]]:
    """
    Parse numbered and bulleted lists from protocol text.

    Detects multiple numbering formats including Arabic numerals
    (1., 2., 3.), Roman numerals (i., ii., iii.), letter-based
    (a., b., c.), and explicit "Step N:" patterns.

    Each step's text is cleaned and whitespace-normalized.
    Steps are returned sorted by their ordinal number.

    Args:
        text: Text containing numbered steps or bulleted lists.

    Returns:
        List of (step_number, step_text) tuples sorted by step_number.
        For letter-based steps, 'a'→1, 'b'→2, etc.
        For Roman numerals, 'i'→1, 'ii'→2, etc.
    """
    if not text:
        return []

    steps: list[tuple[int, str]] = []
    seen_step_numbers: set[int] = set()

    for pattern in _STEP_PATTERNS:
        matches = pattern.findall(text)
        if not matches:
            continue

        for match in matches:
            raw_num, step_text = match[0], match[1]
            step_text = normalize_whitespace(step_text.strip())

            if not step_text:
                continue

            # Convert the step identifier to an integer
            step_num = _parse_step_number(raw_num)
            if step_num is None:
                continue

            # Avoid duplicates from overlapping patterns
            if step_num not in seen_step_numbers:
                steps.append((step_num, step_text))
                seen_step_numbers.add(step_num)

        # If we found steps with this pattern, don't try less-specific ones
        if steps:
            break

    # Sort by step number
    steps.sort(key=lambda x: x[0])

    logger.debug("Extracted %d numbered steps from text", len(steps))
    return steps


def _parse_step_number(raw: str) -> Optional[int]:
    """
    Convert a raw step identifier to an integer.

    Handles Arabic numerals, Roman numerals, and lowercase letters.

    Args:
        raw: Raw step identifier string (e.g., "3", "iii", "b").

    Returns:
        Integer step number, or None if parsing fails.
    """
    raw = raw.strip().lower()

    # Try Arabic numeral first
    try:
        return int(raw)
    except ValueError:
        pass

    # Try Roman numeral
    if raw in _ROMAN_MAP:
        return _ROMAN_MAP[raw]

    # Try single letter (a=1, b=2, ...)
    if len(raw) == 1 and raw.isalpha():
        return ord(raw) - ord("a") + 1

    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    """
    Split document text by section headers.

    Detects headers by numbered sections (e.g., "1. Introduction"),
    all-caps lines, and common protocol section keywords (Materials,
    Procedure, Results, etc.).

    The first chunk of text before any header is assigned the
    section title "Preamble".

    Args:
        text: Full document text to split into sections.

    Returns:
        List of (section_title, section_content) tuples.
        Section titles are stripped and normalized.
        Section content includes all text until the next header.
    """
    if not text:
        return []

    # Collect all header positions and titles
    headers: list[tuple[int, int, str]] = []  # (start, end, title)

    for pattern in _SECTION_HEADER_PATTERNS:
        for match in pattern.finditer(text):
            header_text = match.group("header").strip()
            # Skip very short or very long "headers" (likely false positives)
            if len(header_text) < 3 or len(header_text) > 100:
                continue
            # Skip lines that are actually numbered step instructions
            # (they should be handled by step extraction, not section splitting)
            if re.match(r"^\d+\.\s+[a-z]", header_text):
                continue
            headers.append((match.start(), match.end(), header_text))

    if not headers:
        # No sections found — return the whole text as a single section
        cleaned = normalize_whitespace(text)
        if cleaned:
            return [("Document", cleaned)]
        return []

    # Sort by position and deduplicate overlapping headers
    headers.sort(key=lambda h: h[0])
    deduped: list[tuple[int, int, str]] = []
    for start, end, title in headers:
        if deduped and start < deduped[-1][1]:
            # Overlapping — keep the longer match
            if len(title) > len(deduped[-1][2]):
                deduped[-1] = (start, end, title)
            continue
        deduped.append((start, end, title))

    sections: list[tuple[str, str]] = []

    # Handle preamble (text before first header)
    preamble = text[: deduped[0][0]].strip()
    if preamble:
        preamble = normalize_whitespace(preamble)
        if preamble:
            sections.append(("Preamble", preamble))

    # Extract each section
    for i, (start, end, title) in enumerate(deduped):
        # Content runs from after this header to the start of the next header
        content_start = end
        content_end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)

        content = text[content_start:content_end].strip()
        content = normalize_whitespace(content)

        if content:
            sections.append((title, content))

    logger.debug("Split text into %d sections", len(sections))
    return sections
