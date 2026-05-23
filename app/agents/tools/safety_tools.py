"""
Safety Lookup Tools.

LangChain @tool-decorated functions for checking chemical compatibility,
looking up safety data sheets, and determining PPE requirements.
Contains a built-in safety database of common lab reagents.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Built-In Safety Database ─────────────────────────────────
# Common hazardous reagents used in molecular biology / chemistry labs.
# Each entry maps a reagent name (lowercased) to its safety profile.

SAFETY_DATABASE: dict[str, dict] = {
    "ethidium bromide": {
        "hazards": ["mutagen", "suspected carcinogen", "irritant"],
        "ppe": ["nitrile gloves", "lab coat", "safety goggles"],
        "handling": "Handle in fume hood. Use dedicated waste containers. Do not pour down sink.",
        "incompatible_with": ["strong oxidizers", "strong acids"],
        "first_aid": "Skin: wash with soap and water. Eyes: flush with water 15 min. Ingestion: seek medical help.",
        "risk_level": "high",
    },
    "formaldehyde": {
        "hazards": ["carcinogen", "toxic", "corrosive", "sensitizer"],
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood required"],
        "handling": "Must be used in a chemical fume hood. Keep containers sealed. Use formaldehyde-rated filters.",
        "incompatible_with": ["strong oxidizers", "acids", "bases", "isocyanates"],
        "first_aid": "Skin: remove clothing, wash 15 min. Inhalation: move to fresh air. Eyes: flush 20 min.",
        "risk_level": "critical",
    },
    "chloroform": {
        "hazards": ["toxic", "suspected carcinogen", "irritant", "CNS depressant"],
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood required"],
        "handling": "Use in fume hood only. Store in dark, cool place away from light and heat.",
        "incompatible_with": ["strong bases", "strong oxidizers", "metals like aluminum and magnesium"],
        "first_aid": "Inhalation: move to fresh air. Skin: wash with soap. Eyes: flush 15 min.",
        "risk_level": "high",
    },
    "phenol": {
        "hazards": ["toxic", "corrosive", "burns skin rapidly", "systemic toxicity"],
        "ppe": ["double nitrile gloves", "lab coat", "face shield", "fume hood required"],
        "handling": "Extremely corrosive to skin — absorbs rapidly. Keep PEG 300/400 nearby for decontamination.",
        "incompatible_with": ["strong oxidizers", "aldehydes", "metals"],
        "first_aid": "Skin: do NOT use water first — apply PEG 300, then wash. Seek immediate medical attention.",
        "risk_level": "critical",
    },
    "trizol": {
        "hazards": ["toxic", "corrosive", "contains phenol and guanidinium thiocyanate"],
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood required"],
        "handling": "Contains phenol — handle as phenol. Use in fume hood. Avoid skin contact at all costs.",
        "incompatible_with": ["bleach", "strong oxidizers"],
        "first_aid": "Skin: apply PEG 300 then wash. Inhalation: move to fresh air. Eyes: flush 20 min.",
        "risk_level": "critical",
    },
    "sds": {
        "hazards": ["irritant", "harmful if swallowed"],
        "ppe": ["gloves", "safety goggles"],
        "handling": "Avoid generating dust. Do not inhale powder.",
        "incompatible_with": ["strong oxidizers"],
        "first_aid": "Skin: wash with water. Eyes: flush 10 min. Ingestion: rinse mouth, seek medical advice.",
        "risk_level": "low",
    },
    "dmso": {
        "hazards": ["skin penetration enhancer", "mild irritant"],
        "ppe": ["nitrile gloves (change frequently)", "lab coat", "safety goggles"],
        "handling": "Rapidly penetrates skin and carries dissolved chemicals into body. Change gloves often.",
        "incompatible_with": ["strong oxidizers", "acid chlorides"],
        "first_aid": "Skin: wash immediately with water. Will carry dissolved reagents through skin.",
        "risk_level": "medium",
    },
    "acrylamide": {
        "hazards": ["neurotoxin", "suspected carcinogen", "reproductive toxin"],
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood for powder"],
        "handling": "Do not weigh as powder if possible — use premixed solutions. Avoid inhalation and skin contact.",
        "incompatible_with": ["strong oxidizers", "acids", "bases"],
        "first_aid": "Skin: wash thoroughly. Ingestion: seek medical help immediately. Chronic exposure: neurotoxic.",
        "risk_level": "high",
    },
    "beta-mercaptoethanol": {
        "hazards": ["toxic", "strong odor", "irritant", "corrosive"],
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood required"],
        "handling": "Must use in fume hood — extremely pungent. Store sealed. Small quantities only.",
        "incompatible_with": ["strong oxidizers", "acids", "metals"],
        "first_aid": "Inhalation: move to fresh air. Skin: wash 15 min. Eyes: flush 15 min.",
        "risk_level": "high",
    },
    "dtt": {
        "hazards": ["irritant", "harmful if swallowed", "reducing agent"],
        "ppe": ["gloves", "safety goggles"],
        "handling": "Prepare fresh. Store frozen aliquots. Less toxic than beta-mercaptoethanol.",
        "incompatible_with": ["strong oxidizers"],
        "first_aid": "Skin: wash with water. Eyes: flush 10 min.",
        "risk_level": "low",
    },
    "hydrochloric acid": {
        "hazards": ["corrosive", "toxic fumes", "causes severe burns"],
        "ppe": ["acid-resistant gloves", "lab coat", "face shield", "fume hood required"],
        "handling": "Always add acid to water, never water to acid. Use in fume hood.",
        "incompatible_with": ["bases", "metals", "oxidizers", "cyanides"],
        "first_aid": "Skin: flush 20 min. Eyes: flush 20 min. Inhalation: move to fresh air.",
        "risk_level": "high",
    },
    "sodium hydroxide": {
        "hazards": ["corrosive", "causes severe burns", "exothermic when dissolved"],
        "ppe": ["nitrile gloves", "lab coat", "face shield", "safety goggles"],
        "handling": "Add to water slowly — generates heat. Do not use with aluminum containers.",
        "incompatible_with": ["acids", "aluminum", "zinc", "chlorinated solvents"],
        "first_aid": "Skin: flush 20 min, remove contaminated clothing. Eyes: flush 30 min.",
        "risk_level": "high",
    },
    "liquid nitrogen": {
        "hazards": ["cryogenic burn", "asphyxiation risk", "pressure build-up"],
        "ppe": ["cryogenic gloves", "face shield", "lab coat", "closed-toe shoes"],
        "handling": "Use in well-ventilated areas only. Never seal in airtight container. O₂ monitor required.",
        "incompatible_with": ["sealed containers", "organic materials at cryo temperatures"],
        "first_aid": "Burn: do not rub affected area, warm slowly with lukewarm water. Seek medical help.",
        "risk_level": "high",
    },
    "methanol": {
        "hazards": ["toxic", "flammable", "causes blindness if ingested"],
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood required"],
        "handling": "Highly flammable — keep away from heat sources. Use in fume hood.",
        "incompatible_with": ["strong oxidizers", "acids", "peroxides"],
        "first_aid": "Ingestion: EMERGENCY — seek immediate medical help. Inhalation: fresh air.",
        "risk_level": "high",
    },
    "hydrogen peroxide": {
        "hazards": ["oxidizer", "corrosive at high concentrations", "irritant"],
        "ppe": ["nitrile gloves", "safety goggles", "lab coat"],
        "handling": "Store in original container. Keep away from metals, organics, and reducing agents.",
        "incompatible_with": ["reducing agents", "metals", "organic materials", "flammable solvents"],
        "first_aid": "Skin: wash with water. Eyes: flush 15 min. Ingestion: do not induce vomiting.",
        "risk_level": "medium",
    },
}


def _normalize_reagent_name(name: str) -> str:
    """Normalize reagent name for lookup (lowercase, strip whitespace)."""
    return name.strip().lower()


@tool
def check_chemical_compatibility(
    chemical_a: str, chemical_b: str
) -> dict:
    """Check whether two chemicals are compatible for mixing or close storage.

    Looks up both chemicals in the safety database and checks their
    incompatibility lists for cross-matches.

    Args:
        chemical_a: Name of the first chemical / reagent.
        chemical_b: Name of the second chemical / reagent.

    Returns:
        Dict with compatibility status, warnings, and recommendations.
    """
    logger.info("Checking compatibility: %s + %s", chemical_a, chemical_b)

    a_key = _normalize_reagent_name(chemical_a)
    b_key = _normalize_reagent_name(chemical_b)

    a_info = SAFETY_DATABASE.get(a_key)
    b_info = SAFETY_DATABASE.get(b_key)

    warnings: list[str] = []
    compatible = True

    # Check A's incompatibility list against B
    if a_info:
        for incompat in a_info.get("incompatible_with", []):
            if b_key in incompat.lower() or incompat.lower() in b_key:
                warnings.append(
                    f"{chemical_a} is incompatible with {chemical_b}: "
                    f"listed under '{incompat}'."
                )
                compatible = False

        # Check chemical category matches
        b_categories = set()
        if b_info:
            for hazard in b_info.get("hazards", []):
                b_categories.add(hazard.lower())
        for incompat in a_info.get("incompatible_with", []):
            incompat_lower = incompat.lower()
            if any(cat in incompat_lower for cat in b_categories):
                warnings.append(
                    f"{chemical_a} is incompatible with '{incompat}' — "
                    f"{chemical_b} may fall into this category."
                )
                compatible = False

    # Check B's incompatibility list against A
    if b_info:
        for incompat in b_info.get("incompatible_with", []):
            if a_key in incompat.lower() or incompat.lower() in a_key:
                if not any(a_key in w.lower() for w in warnings):
                    warnings.append(
                        f"{chemical_b} is incompatible with {chemical_a}: "
                        f"listed under '{incompat}'."
                    )
                    compatible = False

    # Handle unknown chemicals
    unknown: list[str] = []
    if not a_info:
        unknown.append(chemical_a)
    if not b_info:
        unknown.append(chemical_b)

    if unknown:
        warnings.append(
            f"Safety data not found for: {', '.join(unknown)}. "
            f"Please consult the relevant SDS (Safety Data Sheet) manually."
        )

    # Deduplicate warnings
    warnings = list(dict.fromkeys(warnings))

    return {
        "chemical_a": chemical_a,
        "chemical_b": chemical_b,
        "compatible": compatible,
        "warnings": warnings,
        "recommendation": (
            "These chemicals appear compatible based on available data."
            if compatible
            else "⚠️ Potential incompatibility detected — review warnings carefully before proceeding."
        ),
    }


@tool
def get_safety_info(reagent_name: str) -> dict:
    """Get comprehensive safety information for a laboratory reagent.

    Looks up the reagent in the built-in safety database and returns
    hazard information, PPE requirements, handling instructions,
    incompatibilities, and first-aid procedures.

    Args:
        reagent_name: Name of the chemical or reagent to look up.

    Returns:
        Dict with full safety profile or a not-found message.
    """
    logger.info("Looking up safety info for: %s", reagent_name)

    key = _normalize_reagent_name(reagent_name)
    info = SAFETY_DATABASE.get(key)

    if info:
        return {
            "reagent": reagent_name,
            "found": True,
            "hazards": info["hazards"],
            "ppe_required": info["ppe"],
            "handling_instructions": info["handling"],
            "incompatible_with": info["incompatible_with"],
            "first_aid": info["first_aid"],
            "risk_level": info["risk_level"],
        }

    # Try partial matching
    partial_matches: list[str] = []
    for db_key in SAFETY_DATABASE:
        if key in db_key or db_key in key:
            partial_matches.append(db_key)

    return {
        "reagent": reagent_name,
        "found": False,
        "message": (
            f"No safety data found for '{reagent_name}'. "
            f"Please consult the manufacturer's SDS (Safety Data Sheet)."
        ),
        "similar_reagents": partial_matches,
        "general_advice": (
            "When handling unknown reagents: wear nitrile gloves, lab coat, "
            "and safety goggles. Work in a fume hood if volatile. "
            "Always consult the SDS before use."
        ),
    }


@tool
def check_ppe_requirements(reagents: list[str]) -> dict:
    """Determine the PPE requirements for working with a set of reagents.

    Aggregates PPE needs from all listed reagents and returns the
    maximum protection level required.

    Args:
        reagents: List of reagent names being used in the procedure.

    Returns:
        Dict with combined PPE requirements and the overall risk level.
    """
    logger.info("Checking PPE requirements for: %s", reagents)

    all_ppe: set[str] = set()
    risk_levels: list[str] = []
    reagent_details: list[dict] = []
    unknown_reagents: list[str] = []

    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    for reagent in reagents:
        key = _normalize_reagent_name(reagent)
        info = SAFETY_DATABASE.get(key)

        if info:
            all_ppe.update(info["ppe"])
            risk_levels.append(info["risk_level"])
            reagent_details.append(
                {
                    "reagent": reagent,
                    "risk_level": info["risk_level"],
                    "ppe": info["ppe"],
                    "key_hazards": info["hazards"][:3],
                }
            )
        else:
            unknown_reagents.append(reagent)

    # Default PPE baseline
    if not all_ppe:
        all_ppe = {"gloves", "safety goggles", "lab coat"}

    # Determine highest risk level
    overall_risk = "low"
    if risk_levels:
        overall_risk = max(risk_levels, key=lambda r: risk_order.get(r, 0))

    # Add baseline requirements
    all_ppe.add("closed-toe shoes")
    if overall_risk in ("high", "critical"):
        all_ppe.add("fume hood required")

    # Build reminders
    reminders: list[str] = [
        "Tie back long hair and remove dangling jewelry.",
        "Know the location of the nearest eyewash station and safety shower.",
        "Have spill kits readily available.",
    ]
    if overall_risk == "critical":
        reminders.insert(0, "⚠️ CRITICAL: Ensure a second person is present in the lab.")
        reminders.append("Have emergency contact numbers posted.")

    return {
        "required_ppe": sorted(all_ppe),
        "overall_risk_level": overall_risk,
        "reagent_details": reagent_details,
        "unknown_reagents": unknown_reagents,
        "safety_reminders": reminders,
    }
