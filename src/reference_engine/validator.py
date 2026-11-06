"""Input validation and physiological sanity checker for spirometry data."""

import math
from typing import Optional, Tuple


MIN_AGE_YEARS = 6.0
MAX_AGE_YEARS = 80.0
MIN_HEIGHT_CM = 100.0
MAX_HEIGHT_CM = 220.0

VALID_SEX_MAP = {
    "male": "Male",
    "m": "Male",
    "female": "Female",
    "f": "Female",
}


def _lookup_field(data: dict, *keys):
    """Look up a key in data with fallback case-insensitivity."""
    for key in keys:
        if key in data:
            return data[key]
    for key in keys:
        target = key.lower()
        for k, v in data.items():
            if str(k).strip().lower() == target:
                return v
    return None


def validate_spirometry_input(data: dict) -> Tuple[bool, Optional[str]]:
    """Validate spirometry input parameters against physiological and demographic boundaries.

    Parameters
    ----------
    data : dict
        Dictionary containing patient and spirometry measurements:
        - age: float (6 to 80 years)
        - sex: str ('Male', 'Female', 'M', 'F')
        - height or height_cm: float (100 to 220 cm; values <= 2.5 interpreted as meters)
        - fev1: float (litres, positive)
        - fvc: float (litres, positive, >= fev1)

    Returns
    -------
    Tuple[bool, Optional[str]]
        (True, None) if validation succeeds, or (False, error_message) if invalid.
    """
    if not isinstance(data, dict):
        return False, "Input data must be a dictionary."

    # 1. Check required fields
    age_raw = _lookup_field(data, "age")
    sex_raw = _lookup_field(data, "sex")
    height_raw = _lookup_field(data, "height_cm", "height")
    fev1_raw = _lookup_field(data, "fev1")
    fvc_raw = _lookup_field(data, "fvc")

    missing = []
    if age_raw is None or (isinstance(age_raw, str) and not age_raw.strip()):
        missing.append("age")
    if sex_raw is None or (isinstance(sex_raw, str) and not sex_raw.strip()):
        missing.append("sex")
    if height_raw is None or (isinstance(height_raw, str) and not height_raw.strip()):
        missing.append("height/height_cm")
    if fev1_raw is None or (isinstance(fev1_raw, str) and not fev1_raw.strip()):
        missing.append("fev1")
    if fvc_raw is None or (isinstance(fvc_raw, str) and not fvc_raw.strip()):
        missing.append("fvc")

    if missing:
        return False, f"Missing required fields: {', '.join(missing)}."

    # 2. Numeric conversion and finiteness checks
    try:
        age = float(age_raw)
        if math.isnan(age) or math.isinf(age):
            return False, "Age must be a valid finite numeric value."
    except (ValueError, TypeError):
        return False, f"Invalid numeric value for age: {age_raw!r}."

    try:
        height = float(height_raw)
        if math.isnan(height) or math.isinf(height):
            return False, "Height must be a valid finite numeric value."
    except (ValueError, TypeError):
        return False, f"Invalid numeric value for height: {height_raw!r}."

    try:
        fev1 = float(fev1_raw)
        if math.isnan(fev1) or math.isinf(fev1):
            return False, "FEV1 must be a valid finite numeric value."
    except (ValueError, TypeError):
        return False, f"Invalid numeric value for FEV1: {fev1_raw!r}."

    try:
        fvc = float(fvc_raw)
        if math.isnan(fvc) or math.isinf(fvc):
            return False, "FVC must be a valid finite numeric value."
    except (ValueError, TypeError):
        return False, f"Invalid numeric value for FVC: {fvc_raw!r}."

    # 3. Positive values check
    if age <= 0:
        return False, "Age must be greater than zero."
    if height <= 0:
        return False, "Height must be greater than zero."
    if fev1 <= 0:
        return False, "FEV1 must be greater than zero."
    if fvc <= 0:
        return False, "FVC must be greater than zero."

    # 4. Height normalization (convert meters to cm if entered <= 2.5) and bounds check
    height_cm = height * 100.0 if height <= 2.5 else height
    if height_cm < MIN_HEIGHT_CM or height_cm > MAX_HEIGHT_CM:
        return False, f"Height must be between {int(MIN_HEIGHT_CM)} and {int(MAX_HEIGHT_CM)} cm."

    # 5. Age boundary check [6, 80]
    if age < MIN_AGE_YEARS or age > MAX_AGE_YEARS:
        return False, f"Age must be between {int(MIN_AGE_YEARS)} and {int(MAX_AGE_YEARS)} years."

    # 6. Sex validation
    if not isinstance(sex_raw, str):
        return False, "Sex must be a string ('Male', 'Female', 'M', 'F')."

    sex_key = sex_raw.strip().lower()
    if sex_key not in VALID_SEX_MAP:
        return False, f"Sex must be one of: Male, Female, M, F (got {sex_raw!r})."

    # 7. Physiological consistency: FEV1 cannot exceed FVC
    if fev1 > fvc:
        return False, f"FEV1 cannot exceed FVC (physiologically impossible in acceptable spirometry: FEV1={fev1:.2f}L, FVC={fvc:.2f}L)."

    # Normalize data dictionary in-place for downstream convenience
    try:
        data["height_cm"] = float(height_cm)
        data["sex"] = VALID_SEX_MAP[sex_key]
        data["age"] = float(age)
        data["fev1"] = float(fev1)
        data["fvc"] = float(fvc)
    except Exception:
        pass

    return True, None
