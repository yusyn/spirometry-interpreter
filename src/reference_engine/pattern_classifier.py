"""ATS/ERS Physiological Pattern Classifier for Spirometry.

Implements standardized physiological pattern classification per ATS/ERS guidelines:
- Airflow obstruction test based on Lower Limit of Normal (LLN) or fixed ratio (0.70).
- Vital capacity evaluation for restriction suspicion.
- Severity grading based on FEV1 % predicted (Mild, Moderate, Severe, Very Severe).
- Borderline and discordance detection between fixed ratio and LLN.
- Mandatory clinical disclaimers regarding Total Lung Capacity (TLC) requirement for restriction.
"""

from dataclasses import dataclass
from typing import Optional

from src.reference_engine.calculator import ReferenceOutput


@dataclass
class PatternResult:
    """Standardized ATS/ERS physiological pattern classification output.

    Attributes
    ----------
    pattern : str
        Standard physiological pattern: 'Normal Spirometry', 'Obstructive',
        'Suspected Restrictive', or 'Mixed'.
    severity : Optional[str]
        Severity grading for obstructive and mixed defects:
        'Mild' (>=80%), 'Moderate' (50-79%), 'Severe' (30-49%),
        or 'Very Severe' (<30% FEV1 % pred). None for normal or isolated restriction.
    clinical_note : str
        Clinical interpretive note and caveats. Mandatorily states
        'TLC measurement required to confirm true restrictive lung defect'
        for suspected restrictive or mixed patterns.
    is_borderline : bool
        True if FEV1/FVC ratio is in close proximity to threshold or
        if fixed ratio (0.70) and LLN yield discordant classifications.
    used_fixed_ratio : bool
        True if GOLD fixed ratio (0.70) was evaluated; False if ATS/ERS LLN was used.
    """

    pattern: str
    severity: Optional[str]
    clinical_note: str
    is_borderline: bool
    used_fixed_ratio: bool


def _determine_severity(fev1_pct_pred: float) -> str:
    """Determine ATS/ERS obstruction severity based on FEV1 % predicted.

    Parameters
    ----------
    fev1_pct_pred : float
        Measured FEV1 expressed as percentage of predicted value.

    Returns
    -------
    str
        'Mild', 'Moderate', 'Severe', or 'Very Severe'.
    """
    if fev1_pct_pred >= 80.0:
        return "Mild"
    if fev1_pct_pred >= 50.0:
        return "Moderate"
    if fev1_pct_pred >= 30.0:
        return "Severe"
    return "Very Severe"


def classify_physiological_pattern(
    fev1: float,
    fvc: float,
    ref: ReferenceOutput,
    use_fixed_ratio: bool = False,
) -> PatternResult:
    """Classify spirometry into ATS/ERS physiological patterns.

    Parameters
    ----------
    fev1 : float
        Measured Forced Expiratory Volume in 1 second in litres.
    fvc : float
        Measured Forced Vital Capacity in litres.
    ref : ReferenceOutput
        Deterministic reference output computed for patient demographics.
    use_fixed_ratio : bool, default=False
        If True, use GOLD fixed ratio (FEV1/FVC < 0.70).
        If False (default), use ATS/ERS Lower Limit of Normal (ratio < ratio_lln).

    Returns
    -------
    PatternResult
        Classification result containing pattern, severity, clinical note,
        borderline flag, and criterion mode.

    Raises
    ------
    ValueError
        If fev1 or fvc are non-positive or fev1 > fvc.
    """
    if fev1 <= 0.0 or fvc <= 0.0:
        raise ValueError("FEV1 and FVC must be positive numbers.")
    if fev1 > fvc:
        raise ValueError("FEV1 cannot exceed FVC in acceptable spirometry.")

    measured_ratio_pct = (fev1 / fvc) * 100.0

    # 1. Obstruction evaluations
    obs_fixed = (fev1 / fvc) < 0.70
    obs_lln = measured_ratio_pct < ref.ratio_lln

    # Discrepancy and borderline detection
    ratio_discrepancy = obs_fixed != obs_lln
    near_lln = (
        abs(ref.ratio_zscore - (-1.645)) <= 0.20
        or abs(measured_ratio_pct - ref.ratio_lln) <= 2.0
    )
    near_fixed = abs(measured_ratio_pct - 70.0) <= 2.0

    is_borderline = ratio_discrepancy or near_lln or near_fixed

    # Active obstruction test
    is_obstructive = obs_fixed if use_fixed_ratio else obs_lln

    # 2. Vital capacity / restriction evaluation
    if use_fixed_ratio:
        is_low_fvc = ref.fvc_pct_pred < 80.0
    else:
        is_low_fvc = fvc < ref.fvc_lln

    # 3. Physiological pattern assignment & severity grading
    if is_obstructive and is_low_fvc:
        pattern = "Mixed"
        severity = _determine_severity(ref.fev1_pct_pred)
        primary_note = (
            "Mixed obstructive-restrictive physiological pattern. "
            "TLC measurement required to confirm true restrictive lung defect."
        )
    elif is_obstructive:
        pattern = "Obstructive"
        severity = _determine_severity(ref.fev1_pct_pred)
        primary_note = f"Airflow obstruction physiological pattern ({severity} severity)."
    elif is_low_fvc:
        pattern = "Suspected Restrictive"
        severity = None
        primary_note = (
            "Suspected restrictive physiological pattern. "
            "TLC measurement required to confirm true restrictive lung defect."
        )
    else:
        pattern = "Normal Spirometry"
        severity = None
        primary_note = "Normal spirometry: physiological volumes and ratio within normal reference limits."

    # Contextual note assembly
    notes = [primary_note]
    if is_borderline:
        if ratio_discrepancy:
            notes.append(
                "Discrepancy detected between fixed ratio (0.70) and ATS/ERS LLN criteria; "
                "borderline airflow limitation."
            )
        else:
            notes.append(
                "Borderline FEV1/FVC ratio near physiological threshold; clinical correlation recommended."
            )

    return PatternResult(
        pattern=pattern,
        severity=severity,
        clinical_note=" ".join(notes),
        is_borderline=is_borderline,
        used_fixed_ratio=use_fixed_ratio,
    )
