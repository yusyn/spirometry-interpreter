"""
GLI spirometry reference equations (Global Lung Function Initiative).

Supported methods
-----------------
* ``gli_global``     – race-neutral GLI Global (Bowerman et al., AJRCCM 2023)
* ``gli_2012_other`` – GLI-2012 Other/Mixed composite (Quanjer et al., ERJ 2012)

Important limitation (PoC)
--------------------------
The official age-dependent M-spline and S-spline look-up tables that are an
integral part of both equation sets are **not** redistributed with this
repository.  They are obtained from the GLI Network
(https://www.lungfunction.org) under a CC-BY-NC licence and must be placed
by the integrator under ``src/reference/data/`` following the schema
documented in ``docs/reference_equations.md``.

Until those tables are present the numerical path raises
``DataNotAvailableError``.  The public API, validation, LMS mathematics and
contract tests are fully implemented and ready for the tables.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .base import (
    LLN_Z,
    IndexResult,
    ReferenceMethod,
    ReferenceResult,
    Sex,
    SpirometryIndex,
    UnsupportedIndexError,
    ValidationError,
    lms_from_z,
    lms_z_score,
    require_finite,
    validate_age_years,
    validate_height_cm,
    validate_measured,
    validate_method,
    validate_sex,
)

# ---------------------------------------------------------------------------
# Domain limits taken from the published papers
# ---------------------------------------------------------------------------
# GLI-2012 / GLI Global: ages 3–95 years, height roughly 100–220 cm
# (exact height bounds are soft; extreme heights produce large extrapolation
# warnings rather than hard rejection in the original software).
_AGE_MIN = 3.0
_AGE_MAX = 95.0
_HEIGHT_MIN_CM = 100.0
_HEIGHT_MAX_CM = 220.0

# Version strings that will appear in every result once tables are loaded.
_GLI_GLOBAL_VERSION = "2022-bowerman-v1"  # placeholder until tables arrive
_GLI_2012_OTHER_VERSION = "2012-quanjer-other-v1"

# Path where official tables are expected (relative to this package).
_DATA_DIR = Path(__file__).resolve().parent / "data"


class DataNotAvailableError(RuntimeError):
    """
    Raised when the official coefficient / spline tables required for a
    numerical evaluation are missing from ``src/reference/data/``.
    """


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_reference(
    *,
    method: ReferenceMethod | str,
    sex: Sex | str,
    age_years: float,
    height_cm: float,
    measured: Optional[Mapping[str, float]] = None,
    indices: Optional[Sequence[SpirometryIndex | str]] = None,
) -> ReferenceResult:
    """
    Compute predicted, z-score and LLN for the requested spirometric indices.

    Parameters
    ----------
    method :
        Explicit reference-equation identifier.  Must be one of
        ``ReferenceMethod.GLI_GLOBAL`` or ``ReferenceMethod.GLI_2012_OTHER``.
        No silent default is applied.
    sex :
        Biological sex (``male`` / ``female``) as required by the equations.
    age_years :
        Age in years (may be fractional).  Documented domain: [3, 95].
    height_cm :
        Standing height in centimetres.  Documented soft domain: [100, 220].
    measured :
        Optional mapping of index name → observed value (litres for volumes,
        dimensionless ratio for FEV1/FVC).  Keys may be ``FEV1``, ``FVC``,
        ``FEV1/FVC`` (case-insensitive).  Missing keys yield ``measured=None``
        in the corresponding ``IndexResult``.
    indices :
        Which indices to evaluate.  Defaults to all three supported indices.

    Returns
    -------
    ReferenceResult
        Structured container holding one ``IndexResult`` per requested index
        plus metadata that identifies the equation set and version.

    Raises
    ------
    ValidationError
        Invalid or out-of-domain inputs, unknown method, non-finite numbers.
    DataNotAvailableError
        Official spline / coefficient tables are not present on disk.
    """
    method = validate_method(method)
    sex = validate_sex(sex)
    age = validate_age_years(age_years, min_age=_AGE_MIN, max_age=_AGE_MAX)
    height = validate_height_cm(
        height_cm, min_cm=_HEIGHT_MIN_CM, max_cm=_HEIGHT_MAX_CM
    )

    if indices is None:
        indices = list(SpirometryIndex)
    else:
        indices = [
            SpirometryIndex(i) if not isinstance(i, SpirometryIndex) else i
            for i in indices
        ]

    measured_norm = validate_measured(measured, indices)

    # Route to the independent implementation path for each method.
    if method is ReferenceMethod.GLI_GLOBAL:
        return _compute_gli_global(sex, age, height, measured_norm, indices)
    if method is ReferenceMethod.GLI_2012_OTHER:
        return _compute_gli_2012_other(sex, age, height, measured_norm, indices)

    # Defensive – validate_method should already have caught this.
    raise ValidationError(f"Unhandled reference method: {method}")


# ---------------------------------------------------------------------------
# Method-specific paths (kept deliberately separate)
# ---------------------------------------------------------------------------

def _compute_gli_global(
    sex: Sex,
    age: float,
    height: float,
    measured: Mapping[str, Optional[float]],
    indices: Sequence[SpirometryIndex],
) -> ReferenceResult:
    """Independent code path for GLI Global race-neutral equations."""
    _ensure_tables_present("gli_global")
    # --- numerical evaluation would go here once tables are loaded ---
    # The placeholder below never executes when tables are missing.
    raise DataNotAvailableError(  # pragma: no cover
        "Internal error: tables reported present but evaluation not implemented"
    )


def _compute_gli_2012_other(
    sex: Sex,
    age: float,
    height: float,
    measured: Mapping[str, Optional[float]],
    indices: Sequence[SpirometryIndex],
) -> ReferenceResult:
    """Independent code path for GLI-2012 Other/Mixed equations."""
    _ensure_tables_present("gli_2012_other")
    raise DataNotAvailableError(  # pragma: no cover
        "Internal error: tables reported present but evaluation not implemented"
    )


def _ensure_tables_present(method_key: str) -> None:
    """
    Check that the required data files exist.

    Expected layout (to be supplied by the integrator from official GLI sources):

        data/
            gli_global/
                coefficients.json   # intercepts, height/age exponents, fixed L
                m_spline.csv        # age → Mspline lookup
                s_spline.csv        # age → Sspline lookup
            gli_2012_other/
                coefficients.json
                m_spline.csv
                s_spline.csv
    """
    base = _DATA_DIR / method_key
    required = [
        base / "coefficients.json",
        base / "m_spline.csv",
        base / "s_spline.csv",
    ]
    missing = [str(p.relative_to(_DATA_DIR.parent)) for p in required if not p.is_file()]
    if missing:
        raise DataNotAvailableError(
            f"Official GLI coefficient/spline tables for '{method_key}' are "
            f"not present.  Missing files: {missing}.  "
            f"Obtain the tables from the GLI Network (https://www.lungfunction.org) "
            f"under the CC-BY-NC licence, place them under "
            f"src/reference/data/{method_key}/ following the schema in "
            f"docs/reference_equations.md, then re-run.  "
            f"No numerical results are fabricated."
        )


# ---------------------------------------------------------------------------
# LMS helpers exposed for unit tests of the pure mathematics
# ---------------------------------------------------------------------------

def compute_lms_metrics(
    measured: float,
    L: float,
    M: float,
    S: float,
) -> tuple[float, float]:
    """
    Pure LMS calculation used by both equation families.

    Returns
    -------
    (z_score, lln)
    """
    z = lms_z_score(measured, L, M, S)
    lln = lms_from_z(LLN_Z, L, M, S)
    return z, lln
