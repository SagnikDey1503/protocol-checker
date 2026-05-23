"""
Multi-Query Generator — LLM-powered query expansion for better recall.

Generates semantically diverse query variations so the retrieval stage
can surface chunks that a single query formulation would miss.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.exceptions import RetrievalError
from app.core.llm import get_fast_llm

logger = logging.getLogger(__name__)

_MULTI_QUERY_SYSTEM_PROMPT = """\
You are a query expansion engine for a scientific research protocol assistant.

Given a user query, generate {num_queries} alternative versions of the same
question.  Each variation should:

1. Preserve the original intent and meaning.
2. Use different vocabulary, synonyms, or phrasing.
3. Approach the topic from a slightly different angle (e.g. more specific,
   more general, focused on a sub-aspect).
4. Be self-contained — each variation must make sense without the original query.

Output ONLY the alternative queries, one per line, numbered 1 through {num_queries}.
Do NOT include explanations, commentary, or the original query.
"""


class MultiQueryGenerator:
    """Generate query variations to improve retrieval recall.

    Usage::

        gen = MultiQueryGenerator()
        queries = await gen.generate_queries("How do I denature DNA?", num_queries=3)
        # → ["How do I denature DNA?", "What temperature denatures …", …]
    """

    def __init__(self) -> None:
        """Initialise with a fast LLM."""
        self._llm = get_fast_llm()
        logger.info("MultiQueryGenerator initialised")

    async def generate_queries(
        self,
        original_query: str,
        num_queries: int = 3,
    ) -> list[str]:
        """Generate *num_queries* alternative formulations of *original_query*.

        The original query is **always** included as the first element of
        the returned list.

        Args:
            original_query: The user's raw query.
            num_queries: Number of **additional** alternative queries to
                generate (default 3).

        Returns:
            A list of ``num_queries + 1`` query strings (original first).

        Raises:
            RetrievalError: If the LLM call fails.
        """
        if not original_query.strip():
            return [original_query]

        try:
            system_prompt = _MULTI_QUERY_SYSTEM_PROMPT.format(num_queries=num_queries)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Original query: {original_query}"),
            ]

            response = await self._llm.ainvoke(messages)
            raw_text = response.content.strip()

            variations = self._parse_queries(raw_text, num_queries)

            # Always prepend the original query.
            result = [original_query] + variations
            logger.debug(
                "MultiQuery: original='%s' → %d variations generated",
                original_query[:60],
                len(variations),
            )
            return result

        except RetrievalError:
            raise
        except Exception as exc:
            logger.warning(
                "Multi-query generation failed (%s) — falling back to original query",
                exc,
            )
            # Graceful degradation: just use the original query.
            return [original_query]

    # ── private helpers ──────────────────────────────────────

    @staticmethod
    def _parse_queries(raw_text: str, expected: int) -> list[str]:
        """Parse numbered query lines from the LLM response.

        Tolerates missing numbers and blank lines.

        Args:
            raw_text: Raw LLM output.
            expected: Expected number of queries (best-effort).

        Returns:
            Up to *expected* non-empty query strings.
        """
        import re

        queries: list[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip optional leading number + punctuation, e.g. "1. ", "2) ".
            cleaned = re.sub(r"^\d+[\.\)\-:]\s*", "", line).strip()
            if cleaned:
                queries.append(cleaned)

        return queries[:expected]
