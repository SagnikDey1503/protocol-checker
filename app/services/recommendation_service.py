"""
Protocol recommendation service.

Provides protocol recommendations from a built-in library of common
synthetic biology / molecular biology protocols, scored against user
descriptions using simple keyword and embedding similarity.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.embeddings import cosine_similarity, embed_query, embed_texts
from app.models.schemas import (
    ProtocolRecommendation,
    ProtocolRecommendationResponse,
)

logger = logging.getLogger(__name__)

# ── Built-in Protocol Library ────────────────────────────────

PROTOCOL_LIBRARY: list[dict[str, Any]] = [
    {
        "title": "Standard PCR Amplification",
        "experiment_type": "PCR",
        "difficulty": "beginner",
        "description": (
            "Polymerase Chain Reaction for amplifying specific DNA sequences. "
            "Includes primer design considerations, thermal cycling parameters, "
            "and optimization tips for various template types."
        ),
        "required_materials": [
            "Taq DNA polymerase",
            "dNTPs",
            "Forward primer",
            "Reverse primer",
            "Template DNA",
            "PCR buffer",
            "MgCl2",
            "Thermal cycler",
            "PCR tubes",
        ],
        "estimated_duration": "2-3 hours",
        "source": "Standard molecular biology",
        "keywords": [
            "pcr", "amplification", "dna", "polymerase", "primer",
            "thermal cycler", "gene", "fragment",
        ],
    },
    {
        "title": "Agarose Gel Electrophoresis",
        "experiment_type": "Gel Electrophoresis",
        "difficulty": "beginner",
        "description": (
            "Separation and visualization of DNA or RNA fragments by size using "
            "agarose gel electrophoresis. Covers gel preparation, loading, "
            "running conditions, and imaging with ethidium bromide or SYBR Safe."
        ),
        "required_materials": [
            "Agarose",
            "TAE or TBE buffer",
            "DNA ladder",
            "Loading dye",
            "Ethidium bromide or SYBR Safe",
            "Gel casting tray",
            "Electrophoresis chamber",
            "Power supply",
            "UV transilluminator",
        ],
        "estimated_duration": "1-2 hours",
        "source": "Standard molecular biology",
        "keywords": [
            "gel", "electrophoresis", "agarose", "dna", "rna",
            "separation", "band", "fragment", "size",
        ],
    },
    {
        "title": "Genomic DNA Extraction (Spin Column)",
        "experiment_type": "DNA Extraction",
        "difficulty": "beginner",
        "description": (
            "Isolation of high-quality genomic DNA from bacterial or mammalian "
            "cells using silica spin-column purification. Includes cell lysis, "
            "protein removal, and DNA elution steps."
        ),
        "required_materials": [
            "DNA extraction kit (spin column)",
            "Proteinase K",
            "Lysis buffer",
            "Wash buffer",
            "Elution buffer",
            "Microcentrifuge",
            "Microcentrifuge tubes",
            "Vortex mixer",
        ],
        "estimated_duration": "1-2 hours",
        "source": "Standard molecular biology",
        "keywords": [
            "dna", "extraction", "isolation", "purification", "genomic",
            "spin column", "lysis", "cell",
        ],
    },
    {
        "title": "Bacterial Transformation (Heat Shock)",
        "experiment_type": "Transformation",
        "difficulty": "intermediate",
        "description": (
            "Introduction of plasmid DNA into competent E. coli cells using "
            "the heat-shock method. Covers competent cell preparation, "
            "transformation protocol, and colony selection on antibiotic plates."
        ),
        "required_materials": [
            "Competent E. coli cells",
            "Plasmid DNA",
            "SOC or LB medium",
            "LB agar plates with antibiotic",
            "Water bath (42°C)",
            "Ice",
            "Sterile spreader",
            "Incubator (37°C)",
        ],
        "estimated_duration": "3-4 hours (plus overnight incubation)",
        "source": "Standard molecular biology",
        "keywords": [
            "transformation", "competent", "heat shock", "plasmid",
            "bacteria", "e. coli", "antibiotic", "colony",
        ],
    },
    {
        "title": "Molecular Cloning (Restriction Digest & Ligation)",
        "experiment_type": "Cloning",
        "difficulty": "intermediate",
        "description": (
            "Classic molecular cloning workflow: restriction enzyme digestion "
            "of insert and vector, gel purification, T4 DNA ligase ligation, "
            "transformation, and colony screening."
        ),
        "required_materials": [
            "Restriction enzymes",
            "T4 DNA ligase",
            "Ligase buffer",
            "Vector DNA",
            "Insert DNA",
            "Gel extraction kit",
            "Competent cells",
            "LB agar plates with antibiotic",
            "Agarose gel reagents",
        ],
        "estimated_duration": "2-3 days",
        "source": "Standard molecular biology",
        "keywords": [
            "cloning", "restriction", "ligation", "digest", "vector",
            "insert", "ligase", "plasmid", "construct",
        ],
    },
    {
        "title": "Mammalian Cell Culture (Adherent Cells)",
        "experiment_type": "Cell Culture",
        "difficulty": "intermediate",
        "description": (
            "Routine maintenance of adherent mammalian cell lines including "
            "thawing frozen stocks, passaging with trypsin, cell counting, "
            "and cryopreservation. Covers sterile technique and contamination "
            "prevention."
        ),
        "required_materials": [
            "Cell culture medium (DMEM/RPMI)",
            "Fetal bovine serum (FBS)",
            "Trypsin-EDTA",
            "PBS (sterile)",
            "T-75 culture flasks",
            "Hemocytometer",
            "Biosafety cabinet",
            "CO2 incubator (37°C, 5% CO2)",
            "Cryovials",
            "Freezing medium",
        ],
        "estimated_duration": "30 min per passage",
        "source": "Standard cell biology",
        "keywords": [
            "cell culture", "mammalian", "adherent", "passaging",
            "trypsin", "media", "incubator", "sterile",
        ],
    },
    {
        "title": "Western Blot Analysis",
        "experiment_type": "Protein Analysis",
        "difficulty": "advanced",
        "description": (
            "Detection and quantification of specific proteins using SDS-PAGE "
            "separation, transfer to PVDF/nitrocellulose membrane, antibody "
            "probing, and chemiluminescent detection."
        ),
        "required_materials": [
            "SDS-PAGE gel",
            "Running buffer",
            "Transfer buffer",
            "PVDF or nitrocellulose membrane",
            "Primary antibody",
            "Secondary antibody (HRP-conjugated)",
            "ECL substrate",
            "Blocking buffer (BSA or milk)",
            "Electrophoresis apparatus",
            "Transfer apparatus",
        ],
        "estimated_duration": "1-2 days",
        "source": "Standard biochemistry",
        "keywords": [
            "western blot", "protein", "antibody", "sds-page",
            "membrane", "detection", "expression",
        ],
    },
    {
        "title": "CRISPR-Cas9 Gene Editing",
        "experiment_type": "Gene Editing",
        "difficulty": "advanced",
        "description": (
            "Targeted gene knockout or knock-in using CRISPR-Cas9 system. "
            "Covers guide RNA design, Cas9 delivery (plasmid or RNP), "
            "transfection, selection, and genotyping of edited clones."
        ),
        "required_materials": [
            "Cas9 protein or plasmid",
            "Guide RNA (sgRNA)",
            "Transfection reagent",
            "Target cells",
            "Selection antibiotic or marker",
            "Genotyping primers",
            "T7 endonuclease I (for mismatch assay)",
            "Cell culture reagents",
        ],
        "estimated_duration": "2-4 weeks",
        "source": "Standard gene editing",
        "keywords": [
            "crispr", "cas9", "gene editing", "knockout", "knock-in",
            "guide rna", "sgrna", "genome", "edit",
        ],
    },
]


class RecommendationService:
    """Recommend protocols from the built-in library.

    Uses a hybrid scoring approach:
    1. Keyword overlap (fast, always available)
    2. Embedding cosine similarity (semantic, more accurate)

    The final score is a weighted average of both.
    """

    def __init__(self) -> None:
        self._embeddings_cache: dict[str, list[float]] = {}

    async def recommend(
        self,
        description: str,
        difficulty: Optional[str] = None,
        equipment: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> ProtocolRecommendationResponse:
        """Score and return the most relevant protocols.

        Args:
            description: User's free-text description of what they want to do.
            difficulty: Optional difficulty filter (beginner/intermediate/advanced).
            equipment: Optional list of equipment the user has available.
            top_k: Number of results to return (default 5).

        Returns:
            A ``ProtocolRecommendationResponse`` with scored recommendations.
        """
        candidates = list(PROTOCOL_LIBRARY)

        # Filter by difficulty if specified
        if difficulty:
            difficulty_lower = difficulty.lower()
            candidates = [
                p for p in candidates if p["difficulty"] == difficulty_lower
            ] or candidates  # Fall back to all if no match

        # Score each candidate
        scored: list[tuple[float, dict[str, Any]]] = []
        for protocol in candidates:
            score = self._compute_score(description, protocol, equipment)
            scored.append((score, protocol))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build response
        recommendations = [
            ProtocolRecommendation(
                title=p["title"],
                experiment_type=p["experiment_type"],
                difficulty=p["difficulty"],
                description=p["description"],
                required_materials=p["required_materials"],
                estimated_duration=p.get("estimated_duration"),
                source=p.get("source"),
                relevance_score=round(score, 3),
            )
            for score, p in scored[:top_k]
        ]

        return ProtocolRecommendationResponse(
            recommendations=recommendations,
            total=len(recommendations),
        )

    def _compute_score(
        self,
        description: str,
        protocol: dict[str, Any],
        equipment: Optional[list[str]],
    ) -> float:
        """Compute a relevance score for a protocol.

        Uses keyword overlap + optional equipment matching.
        Tries embedding similarity but falls back gracefully.
        """
        description_lower = description.lower()
        score = 0.0

        # 1. Keyword overlap score (0–1)
        keywords = protocol.get("keywords", [])
        if keywords:
            matches = sum(
                1 for kw in keywords if kw.lower() in description_lower
            )
            keyword_score = matches / len(keywords)
            score += keyword_score * 0.4  # 40% weight

        # 2. Title/description text overlap (0–1)
        protocol_text = f"{protocol['title']} {protocol['description']}".lower()
        desc_words = set(description_lower.split())
        proto_words = set(protocol_text.split())
        if desc_words and proto_words:
            overlap = len(desc_words & proto_words) / max(len(desc_words), 1)
            score += min(overlap, 1.0) * 0.2  # 20% weight

        # 3. Embedding cosine similarity (0–1)
        try:
            embedding_score = self._embedding_similarity(description, protocol)
            score += embedding_score * 0.3  # 30% weight
        except Exception:
            # Embeddings unavailable — redistribute weight to keywords
            score += keyword_score * 0.15 if keywords else 0
            logger.debug("Embedding scoring unavailable; using keyword-only.")

        # 4. Equipment match bonus (0–0.1)
        if equipment:
            required = [m.lower() for m in protocol.get("required_materials", [])]
            available = [e.lower() for e in equipment]
            equipment_matches = sum(
                1
                for req in required
                if any(avail in req or req in avail for avail in available)
            )
            if required:
                score += (equipment_matches / len(required)) * 0.1

        return min(score, 1.0)

    def _embedding_similarity(
        self,
        description: str,
        protocol: dict[str, Any],
    ) -> float:
        """Compute cosine similarity between description and protocol embeddings."""
        query_embedding = embed_query(description)

        cache_key = protocol["title"]
        if cache_key not in self._embeddings_cache:
            protocol_text = f"{protocol['title']}. {protocol['description']}"
            embeddings = embed_texts([protocol_text])
            self._embeddings_cache[cache_key] = embeddings[0]

        proto_embedding = self._embeddings_cache[cache_key]
        similarity = cosine_similarity(query_embedding, proto_embedding)
        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0
