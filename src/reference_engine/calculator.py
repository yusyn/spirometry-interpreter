"""Deterministic Reference & Z-Score Calculator for Spirometry.

Implements standard reference equations (ECSC / Quanjer 1993) to calculate
predicted values, percentage of predicted (% predicted), standardized
residual deviations (Z-scores), and Lower Limits of Normal (LLN = Predicted - 1.645 * RSD).
"""

from dataclasses import dataclass
from src.reference_engine.validator import validate_spirometry_input


@dataclass
class ReferenceOutput:
    """Deterministic reference values and statistical indices.

    Attributes
    ----------
    fev1_pred : float
        Predicted FEV1 in litres.
    fvc_pred : float
        Predicted FVC in litres.
    ratio_pred : float
        Predicted FEV1/FVC ratio as a percentage (e.g. 81.81).
    fev1_pct_pred : float
        Measured FEV1 as a percentage of predicted FEV1.
    fvc_pct_pred : float
        Measured FVC as a percentage of predicted FVC.
    fev1_zscore : float
        Standardized Z-score for FEV1: (measured - predicted) / RSD.
    fvc_zscore : float
        Standardized Z-score for FVC: (measured - predicted) / RSD.
    ratio_zscore : float
        Standardized Z-score for FEV1/FVC ratio: (measured_ratio - predicted_ratio) / RSD.
    fev1_lln : float
        Lower Limit of Normal for FEV1 (pred - 1.645 * RSD).
    fvc_lln : float
        Lower Limit of Normal for FVC (pred - 1.645 * RSD).
    ratio_lln : float
        Lower Limit of Normal for FEV1/FVC ratio (pred - 1.645 * RSD).
    """

    fev1_pred: float
    fvc_pred: float
    ratio_pred: float
    fev1_pct_pred: float
    fvc_pct_pred: float
    fev1_zscore: float
    fvc_zscore: float
    ratio_zscore: float
    fev1_lln: float
    fvc_lln: float
    ratio_lln: float


# 1.645 corresponds to the 5th percentile lower limit of normal (one-tailed)
LLN_Z_FACTOR = 1.645


def calculate_reference_values(
    age: float,
    height_cm: float,
    sex: str,
    fev1: float,
    fvc: float,
) -> ReferenceOutput:
    """Calculate deterministic reference values, % predicted, Z-scores, and LLN.

    Parameters
    ----------
    age : float
        Patient age in years (6 to 80).
    height_cm : float
        Patient height in centimeters (100 to 220) or meters (<= 2.5).
    sex : str
        Biological sex ('Male', 'Female', 'M', 'F', case-insensitive).
    fev1 : float
        Measured Forced Expiratory Volume in 1 second in litres.
    fvc : float
        Measured Forced Vital Capacity in litres.

    Returns
    -------
    ReferenceOutput
        Dataclass containing predicted, % predicted, Z-scores, and LLN.

    Raises
    ------
    ValueError
        If inputs fail physiological or boundary validation.
    """
    input_data = {
        "age": age,
        "height_cm": height_cm,
        "sex": sex,
        "fev1": fev1,
        "fvc": fvc,
    }
    is_valid, err_msg = validate_spirometry_input(input_data)
    if not is_valid:
        raise ValueError(err_msg)

    # Use validated & normalized values
    norm_age = float(input_data["age"])
    norm_height_m = float(input_data["height_cm"]) / 100.0
    norm_sex = input_data["sex"]
    norm_fev1 = float(input_data["fev1"])
    norm_fvc = float(input_data["fvc"])

    # ECSC / Quanjer 1993 standard reference equations
    if norm_sex == "Male":
        fev1_pred = 4.301 * norm_height_m - 0.0290 * norm_age - 2.492
        fvc_pred = 5.761 * norm_height_m - 0.0260 * norm_age - 4.343
        ratio_pred = -0.18 * norm_age + 87.21
        rsd_fev1 = 0.51
        rsd_fvc = 0.61
        rsd_ratio = 7.17
    else:  # Female
        fev1_pred = 3.953 * norm_height_m - 0.0251 * norm_age - 2.604
        fvc_pred = 4.433 * norm_height_m - 0.0260 * norm_age - 2.892
        ratio_pred = -0.19 * norm_age + 89.10
        rsd_fev1 = 0.38
        rsd_fvc = 0.43
        rsd_ratio = 6.51

    # % Predicted
    fev1_pct_pred = (norm_fev1 / fev1_pred) * 100.0 if fev1_pred > 0 else 0.0
    fvc_pct_pred = (norm_fvc / fvc_pred) * 100.0 if fvc_pred > 0 else 0.0

    # Measured Ratio (%)
    measured_ratio = (norm_fev1 / norm_fvc) * 100.0

    # Z-scores: (measured - predicted) / RSD
    fev1_zscore = (norm_fev1 - fev1_pred) / rsd_fev1
    fvc_zscore = (norm_fvc - fvc_pred) / rsd_fvc
    ratio_zscore = (measured_ratio - ratio_pred) / rsd_ratio

    # Lower Limit of Normal (LLN): predicted - 1.645 * RSD
    fev1_lln = fev1_pred - LLN_Z_FACTOR * rsd_fev1
    fvc_lln = fvc_pred - LLN_Z_FACTOR * rsd_fvc
    ratio_lln = ratio_pred - LLN_Z_FACTOR * rsd_ratio

    return ReferenceOutput(
        fev1_pred=fev1_pred,
        fvc_pred=fvc_pred,
        ratio_pred=ratio_pred,
        fev1_pct_pred=fev1_pct_pred,
        fvc_pct_pred=fvc_pct_pred,
        fev1_zscore=fev1_zscore,
        fvc_zscore=fvc_zscore,
        ratio_zscore=ratio_zscore,
        fev1_lln=fev1_lln,
        fvc_lln=fvc_lln,
        ratio_lln=ratio_lln,
    )
