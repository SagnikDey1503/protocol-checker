"""
Lab Calculation Tools.

LangChain @tool-decorated functions for common laboratory calculations:
dilution, molarity, temperature conversion, and PCR annealing temperature
estimation.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def calculate_dilution(
    stock_concentration: float,
    desired_concentration: float,
    desired_volume: float,
    concentration_unit: str = "M",
    volume_unit: str = "mL",
) -> dict:
    """Calculate the volume of stock solution needed for a dilution (C1V1 = C2V2).

    Uses the dilution equation: C₁ × V₁ = C₂ × V₂
    Solves for V₁ (volume of stock to add) and computes the diluent volume.

    Args:
        stock_concentration: Concentration of the stock solution (C₁).
        desired_concentration: Target concentration after dilution (C₂).
        desired_volume: Total volume desired after dilution (V₂).
        concentration_unit: Unit for concentrations (e.g., M, mM, µM, mg/mL).
        volume_unit: Unit for volumes (e.g., mL, µL, L).

    Returns:
        Dict with stock_volume, diluent_volume, and the equation used.
    """
    logger.info(
        "Dilution: C1=%.4f, C2=%.4f, V2=%.4f %s",
        stock_concentration,
        desired_concentration,
        desired_volume,
        volume_unit,
    )

    if stock_concentration <= 0:
        return {"error": "Stock concentration must be greater than zero."}
    if desired_concentration <= 0:
        return {"error": "Desired concentration must be greater than zero."}
    if desired_volume <= 0:
        return {"error": "Desired volume must be greater than zero."}
    if desired_concentration > stock_concentration:
        return {
            "error": (
                "Desired concentration cannot exceed stock concentration. "
                "You cannot concentrate a solution by dilution."
            )
        }

    # C1 * V1 = C2 * V2  →  V1 = (C2 * V2) / C1
    stock_volume = (desired_concentration * desired_volume) / stock_concentration
    diluent_volume = desired_volume - stock_volume

    dilution_factor = stock_concentration / desired_concentration

    return {
        "stock_volume": round(stock_volume, 4),
        "diluent_volume": round(diluent_volume, 4),
        "total_volume": round(desired_volume, 4),
        "dilution_factor": round(dilution_factor, 2),
        "stock_concentration": stock_concentration,
        "desired_concentration": desired_concentration,
        "concentration_unit": concentration_unit,
        "volume_unit": volume_unit,
        "equation": (
            f"C₁V₁ = C₂V₂ → {stock_concentration} {concentration_unit} × V₁ = "
            f"{desired_concentration} {concentration_unit} × {desired_volume} {volume_unit}"
        ),
        "instructions": (
            f"Add {round(stock_volume, 4)} {volume_unit} of "
            f"{stock_concentration} {concentration_unit} stock solution to "
            f"{round(diluent_volume, 4)} {volume_unit} of diluent "
            f"for a final volume of {desired_volume} {volume_unit} at "
            f"{desired_concentration} {concentration_unit}."
        ),
    }


@tool
def calculate_molarity(
    mass_grams: float,
    molecular_weight: float,
    volume_liters: float,
) -> dict:
    """Calculate the molarity of a solution given mass, molecular weight, and volume.

    Molarity (M) = moles / volume (L) = (mass_g / MW) / volume_L

    Args:
        mass_grams: Mass of solute in grams.
        molecular_weight: Molecular weight of the solute in g/mol.
        volume_liters: Volume of solution in liters.

    Returns:
        Dict with molarity in M, mM, and µM, plus the moles of solute.
    """
    logger.info(
        "Molarity: mass=%.4f g, MW=%.2f g/mol, volume=%.4f L",
        mass_grams,
        molecular_weight,
        volume_liters,
    )

    if mass_grams < 0:
        return {"error": "Mass cannot be negative."}
    if molecular_weight <= 0:
        return {"error": "Molecular weight must be greater than zero."}
    if volume_liters <= 0:
        return {"error": "Volume must be greater than zero."}

    moles = mass_grams / molecular_weight
    molarity = moles / volume_liters
    molarity_mm = molarity * 1_000
    molarity_um = molarity * 1_000_000

    return {
        "molarity_M": round(molarity, 6),
        "molarity_mM": round(molarity_mm, 4),
        "molarity_µM": round(molarity_um, 2),
        "moles": round(moles, 6),
        "mass_grams": mass_grams,
        "molecular_weight": molecular_weight,
        "volume_liters": volume_liters,
        "equation": f"M = ({mass_grams} g / {molecular_weight} g/mol) / {volume_liters} L",
        "summary": (
            f"Dissolving {mass_grams} g of solute (MW {molecular_weight} g/mol) "
            f"in {volume_liters} L gives a {round(molarity, 6)} M "
            f"({round(molarity_mm, 4)} mM) solution."
        ),
    }


@tool
def convert_temperature(
    value: float,
    from_unit: str,
    to_unit: str,
) -> dict:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: Temperature value to convert.
        from_unit: Source unit — one of 'C', 'F', 'K' (case-insensitive).
        to_unit: Target unit — one of 'C', 'F', 'K' (case-insensitive).

    Returns:
        Dict with the converted temperature value and both units.
    """
    from_u = from_unit.upper().strip()
    to_u = to_unit.upper().strip()

    valid = {"C", "F", "K"}
    if from_u not in valid or to_u not in valid:
        return {
            "error": f"Invalid unit. Use one of: {valid}. Got from={from_unit}, to={to_unit}.",
        }

    # Convert to Celsius as intermediate
    if from_u == "C":
        celsius = value
    elif from_u == "F":
        celsius = (value - 32) * 5 / 9
    else:  # K
        celsius = value - 273.15

    # Convert from Celsius to target
    if to_u == "C":
        result = celsius
    elif to_u == "F":
        result = celsius * 9 / 5 + 32
    else:  # K
        result = celsius + 273.15

    return {
        "input_value": value,
        "input_unit": from_u,
        "output_value": round(result, 2),
        "output_unit": to_u,
        "summary": f"{value}°{from_u} = {round(result, 2)}°{to_u}",
    }


@tool
def calculate_pcr_annealing_temp(
    primer_sequence: str,
    primer_name: Optional[str] = None,
) -> dict:
    """Estimate the PCR annealing temperature for a DNA primer.

    Uses the Wallace rule for short primers (<14 nt):
        Tm = 2(A+T) + 4(G+C)

    And the basic salt-adjusted formula for longer primers:
        Tm = 64.9 + 41 × (G+C - 16.4) / N

    The recommended annealing temperature is Tm − 5°C.

    Args:
        primer_sequence: DNA sequence of the primer (5'→3'). Accepts
            A, T, G, C characters (case-insensitive). Ambiguous bases
            and non-ATGC characters are ignored with a warning.
        primer_name: Optional name/label for the primer.

    Returns:
        Dict with Tm estimate, recommended annealing temp, GC content,
        and primer statistics.
    """
    logger.info("PCR Tm calculation for primer: %s", primer_name or primer_sequence[:20])

    # Clean sequence
    seq = primer_sequence.upper().strip()
    valid_bases = set("ATGC")
    clean_seq = "".join(c for c in seq if c in valid_bases)
    invalid_chars = set(seq) - valid_bases - {" ", "\n", "\t"}

    if len(clean_seq) < 10:
        return {
            "error": "Primer is too short (< 10 nt). Provide at least 10 valid bases.",
            "valid_bases_found": len(clean_seq),
        }

    n = len(clean_seq)
    gc_count = clean_seq.count("G") + clean_seq.count("C")
    at_count = clean_seq.count("A") + clean_seq.count("T")
    gc_content = gc_count / n * 100

    # Calculate Tm
    if n < 14:
        # Wallace rule
        tm = 2 * at_count + 4 * gc_count
        method = "Wallace rule: Tm = 2(A+T) + 4(G+C)"
    else:
        # Basic salt-adjusted
        tm = 64.9 + 41 * (gc_count - 16.4) / n
        method = "Basic formula: Tm = 64.9 + 41 × (G+C − 16.4) / N"

    annealing_temp = tm - 5.0

    # Quality warnings
    warnings: list[str] = []
    if gc_content < 40:
        warnings.append("Low GC content (<40%) — primer may have weak binding.")
    elif gc_content > 60:
        warnings.append("High GC content (>60%) — may cause secondary structures.")
    if n > 30:
        warnings.append("Primer is quite long (>30 nt) — consider shorter design.")
    if n < 18:
        warnings.append("Primer is short (<18 nt) — specificity may be low.")
    if invalid_chars:
        warnings.append(
            f"Ignored non-ATGC characters: {', '.join(sorted(invalid_chars))}"
        )

    # Check for self-complementarity (basic 3' dimer check)
    last_4 = clean_seq[-4:]
    complement = str.maketrans("ATGC", "TACG")
    rev_comp_last_4 = last_4.translate(complement)[::-1]
    if rev_comp_last_4 in clean_seq:
        warnings.append(
            "Potential primer dimer: 3' end may self-anneal. Check with a primer design tool."
        )

    return {
        "primer_name": primer_name or "unnamed",
        "sequence": clean_seq,
        "length": n,
        "gc_count": gc_count,
        "at_count": at_count,
        "gc_content_percent": round(gc_content, 1),
        "melting_temperature_C": round(tm, 1),
        "recommended_annealing_C": round(annealing_temp, 1),
        "calculation_method": method,
        "warnings": warnings,
        "summary": (
            f"Primer '{primer_name or 'unnamed'}' ({n} nt, {round(gc_content, 1)}% GC): "
            f"Tm ≈ {round(tm, 1)}°C, recommended annealing ≈ {round(annealing_temp, 1)}°C"
        ),
    }
