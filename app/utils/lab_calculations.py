"""
Laboratory calculation utilities.

Provides common lab math functions used by the calculation tools
and the research assistant agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class DilutionResult:
    """Result of a dilution calculation."""

    stock_volume: float
    diluent_volume: float
    final_volume: float
    stock_concentration: float
    final_concentration: float
    unit: str


@dataclass
class MolarityResult:
    """Result of a molarity calculation."""

    molarity: float
    mass_grams: float
    volume_liters: float
    molecular_weight: float
    unit: str = "M"


def calculate_dilution(
    stock_concentration: float,
    final_concentration: float,
    final_volume: float,
    unit: str = "x",
) -> DilutionResult:
    """
    Calculate dilution volumes using C1V1 = C2V2.

    Args:
        stock_concentration: Concentration of stock solution.
        final_concentration: Desired final concentration.
        final_volume: Desired final volume (in mL).
        unit: Concentration unit (e.g., "x", "M", "mM", "µM", "ng/µL").

    Returns:
        DilutionResult with stock and diluent volumes.

    Raises:
        ValueError: If concentrations are invalid.
    """
    if stock_concentration <= 0 or final_concentration <= 0:
        raise ValueError("Concentrations must be positive")
    if final_concentration > stock_concentration:
        raise ValueError("Final concentration cannot exceed stock concentration")
    if final_volume <= 0:
        raise ValueError("Final volume must be positive")

    # C1V1 = C2V2  =>  V1 = (C2 * V2) / C1
    stock_volume = (final_concentration * final_volume) / stock_concentration
    diluent_volume = final_volume - stock_volume

    return DilutionResult(
        stock_volume=round(stock_volume, 3),
        diluent_volume=round(diluent_volume, 3),
        final_volume=final_volume,
        stock_concentration=stock_concentration,
        final_concentration=final_concentration,
        unit=unit,
    )


def calculate_molarity(
    mass_grams: Optional[float] = None,
    volume_liters: Optional[float] = None,
    molecular_weight: Optional[float] = None,
    molarity: Optional[float] = None,
) -> MolarityResult:
    """
    Calculate molarity or any missing component.

    M = mass / (MW × volume)

    Provide any three of the four parameters and the fourth will be calculated.

    Args:
        mass_grams: Mass of solute in grams.
        volume_liters: Volume of solution in liters.
        molecular_weight: Molecular weight in g/mol.
        molarity: Molar concentration in M.

    Returns:
        MolarityResult with all four values.

    Raises:
        ValueError: If not exactly one parameter is None.
    """
    params = [mass_grams, volume_liters, molecular_weight, molarity]
    none_count = sum(1 for p in params if p is None)

    if none_count != 1:
        raise ValueError("Provide exactly three of the four parameters")

    if molarity is None:
        molarity = mass_grams / (molecular_weight * volume_liters)
    elif mass_grams is None:
        mass_grams = molarity * molecular_weight * volume_liters
    elif volume_liters is None:
        volume_liters = mass_grams / (molarity * molecular_weight)
    elif molecular_weight is None:
        molecular_weight = mass_grams / (molarity * volume_liters)

    return MolarityResult(
        molarity=round(molarity, 6),
        mass_grams=round(mass_grams, 4),
        volume_liters=round(volume_liters, 6),
        molecular_weight=round(molecular_weight, 2),
    )


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: Temperature value.
        from_unit: Source unit ('C', 'F', or 'K').
        to_unit: Target unit ('C', 'F', or 'K').

    Returns:
        Converted temperature value.
    """
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    # Convert to Celsius first
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Convert from Celsius to target
    if to_unit == "C":
        return round(celsius, 2)
    elif to_unit == "F":
        return round(celsius * 9 / 5 + 32, 2)
    elif to_unit == "K":
        return round(celsius + 273.15, 2)
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def calculate_pcr_annealing_temp(
    primer_sequence: str,
    method: str = "basic",
) -> dict:
    """
    Estimate PCR annealing temperature from primer sequence.

    Args:
        primer_sequence: DNA primer sequence (5' to 3').
        method: Calculation method — 'basic' (Wallace rule) or 'salt_adjusted'.

    Returns:
        Dict with tm (melting temp), recommended_annealing, and method used.
    """
    seq = primer_sequence.upper().strip()

    # Count bases
    a_count = seq.count("A")
    t_count = seq.count("T")
    g_count = seq.count("G")
    c_count = seq.count("C")
    length = len(seq)

    if length == 0:
        raise ValueError("Primer sequence is empty")

    if method == "basic":
        # Wallace rule: for primers < 14 nt
        # Tm = 2(A+T) + 4(G+C)
        if length < 14:
            tm = 2 * (a_count + t_count) + 4 * (g_count + c_count)
        else:
            # For longer primers: Tm = 64.9 + 41*(G+C-16.4) / length
            gc_content = (g_count + c_count) / length
            tm = 64.9 + 41 * (g_count + c_count - 16.4) / length
    elif method == "salt_adjusted":
        # Salt-adjusted: Tm = 100.5 + (41 * GC_content) - (820/length) + 16.6*log10([Na+])
        gc_content = (g_count + c_count) / length
        na_concentration = 0.05  # Default 50mM Na+
        tm = 100.5 + (41 * gc_content) - (820 / length) + 16.6 * math.log10(na_concentration)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Annealing temp is typically 5°C below Tm
    annealing = tm - 5

    return {
        "melting_temperature_c": round(tm, 1),
        "recommended_annealing_c": round(annealing, 1),
        "primer_length": length,
        "gc_content_percent": round((g_count + c_count) / length * 100, 1),
        "method": method,
    }


def calculate_dna_concentration(
    absorbance_260: float,
    dilution_factor: float = 1.0,
    nucleic_acid_type: str = "dsDNA",
) -> dict:
    """
    Calculate nucleic acid concentration from A260 reading.

    Uses Beer-Lambert law with standard extinction coefficients:
    - dsDNA: 50 µg/mL per A260
    - ssDNA: 33 µg/mL per A260
    - RNA: 40 µg/mL per A260

    Args:
        absorbance_260: A260 reading.
        dilution_factor: Dilution factor of the sample.
        nucleic_acid_type: Type of nucleic acid.

    Returns:
        Dict with concentration and quality metrics.
    """
    coefficients = {
        "dsDNA": 50,
        "ssDNA": 33,
        "RNA": 40,
    }

    if nucleic_acid_type not in coefficients:
        raise ValueError(f"Unknown nucleic acid type: {nucleic_acid_type}")

    coefficient = coefficients[nucleic_acid_type]
    concentration = absorbance_260 * coefficient * dilution_factor

    return {
        "concentration_ug_ml": round(concentration, 2),
        "concentration_ng_ul": round(concentration, 2),  # Same numeric value, different unit
        "nucleic_acid_type": nucleic_acid_type,
        "extinction_coefficient": coefficient,
    }
