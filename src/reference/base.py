"""
Core types, validation helpers and abstract contract for spirometry
reference equations.

No patient data is logged or persisted.  All functions are pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class Sex(str, Enum):
    """Biological sex as required by the published GLI equations."""

    MALE = "male"
    FEMALE = "female"


class SpirometryIndex(str, Enum):
    """Supported spirometric indices."""

    FEV1 = "FEV1"
    FVC = "FVC"
    FEV1_FVC = "FEV1/FVC"


class ReferenceMethod(str, Enum):
    """
    Explicit identifiers for reference-equation families.

    Callers **must** select one; the engine never applies a silent default.
    """

    GLI_GLOBAL = "gli_global"
    """GLI Global race-neutral equations (Bowerman et al., 2022/2023)."""

    GLI_2012_OTHER = "gli_2012_other"
    """GLI-2012 composite 'Other/Mixed' equations (Quanjer et al., 2012)."""


class ValidationError(ValueError):
    """Raised when input values are missing, non-finite or out of documented range."""


class UnsupportedIndexError(ValueError):
    """Raised when a requested index is not supported by the chosen method."""


@dataclass(frozen=True)
class IndexResult:
    """
    Structured result for a single spirometric index.

    Attributes
    ----------
    index :
        Identifier of the index (e.g. ``FEV1``).
    measured :
        Observed value supplied by the caller (same unit as predicted).
    predicted :
        Predicted median (M) from the reference equation, or ``None`` if
        the index is not supported by the selected method.
    z_score :
        LMS z-score, or ``None`` when predicted is unavailable.
    lln :
        Lower limit of normal (typically the 5th percentile, z ≈ −1.645),
        or ``None`` when predicted is unavailable.
    unit :
        Unit of measured / predicted / lln (``L`` for volumes, ``ratio``
        for FEV1/FVC).
    reference_id :
        Stable identifier of the equation set (matches ``ReferenceMethod``).
    reference_version :
        Version string of the coefficient / spline tables used.
    warnings :
        Human-readable notes (e.g. age near domain boundary).
    unsupported_reason :
        Populated only when predicted is ``None``.
    """

    index: str
    measured: Optional[float]
    predicted: Optional[float]
    z_score: Optional[float]
    lln: Optional[float]
    unit: str
    reference_id: str
    reference_version: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    unsupported_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "measured": self.measured,
            "predicted": self.predicted,
            "z_score": self.z_score,
            "lln": self.lln,
            "unit": self.unit,
            "reference_id": self.reference_id,
            "reference_version": self.reference_version,
            "warnings": list(self.warnings),
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class ReferenceResult:
    """Aggregate result for all requested indices under one reference method."""

    method: ReferenceMethod
    sex: Sex
    age_years: float
    height_cm: float
    results: Mapping[str, IndexResult]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "sex": self.sex.value,
            "age_years": self.age_years,
            "height_cm": self.height_cm,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Validation helpers (pure, deterministic)
# ---------------------------------------------------------------------------

def require_finite(name: str, value: float) -> float:
    """Reject NaN / Inf."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def validate_sex(sex: Sex | str) -> Sex:
    if isinstance(sex, Sex):
        return sex
    try:
        return Sex(str(sex).lower())
    except ValueError as exc:
        raise ValidationError(
            f"sex must be one of {[s.value for s in Sex]}, got {sex!r}"
        ) from exc


def validate_method(method: ReferenceMethod | str) -> ReferenceMethod:
    if isinstance(method, ReferenceMethod):
        return method
    try:
        return ReferenceMethod(str(method).lower())
    except ValueError as exc:
        raise ValidationError(
            f"Unknown reference method {method!r}. "
            f"Supported: {[m.value for m in ReferenceMethod]}"
        ) from exc


def validate_age_years(age: float, *, min_age: float, max_age: float) -> float:
    age = require_finite("age_years", age)
    if age < min_age or age > max_age:
        raise ValidationError(
            f"age_years={age} is outside the documented domain "
            f"[{min_age}, {max_age}] for the selected reference method"
        )
    return age


def validate_height_cm(height: float, *, min_cm: float, max_cm: float) -> float:
    height = require_finite("height_cm", height)
    if height < min_cm or height > max_cm:
        raise ValidationError(
            f"height_cm={height} is outside the documented domain "
            f"[{min_cm}, {max_cm}] for the selected reference method"
        )
    return height


def validate_measured(
    measured: Optional[Mapping[str, float]],
    indices: Sequence[SpirometryIndex],
) -> dict[str, Optional[float]]:
    """
    Normalise measured dict.  Missing keys become ``None``.
    Values that are present must be finite and non-negative (volumes)
    or in (0, 1.5] for the ratio.
    """
    out: dict[str, Optional[float]] = {idx.value: None for idx in indices}
    if measured is None:
        return out
    for key, raw in measured.items():
        canon = key.upper().replace(" ", "").replace("_", "/")
        if canon in ("FEV1/FVC", "FEV1FVC", "FEV1_FVC"):
            canon = SpirometryIndex.FEV1_FVC.value
        elif canon in ("FEV1", "FVC"):
            pass
        else:
            raise ValidationError(f"Unknown measured index key: {key!r}")
        val = require_finite(f"measured[{key}]", raw)
        if canon in (SpirometryIndex.FEV1.value, SpirometryIndex.FVC.value):
            if val < 0:
                raise ValidationError(f"measured {canon} must be ≥ 0, got {val}")
        elif canon == SpirometryIndex.FEV1_FVC.value:
            if not (0 < val <= 1.5):
                raise ValidationError(
                    f"measured FEV1/FVC must be in (0, 1.5], got {val}"
                )
        out[canon] = val
    return out


# ---------------------------------------------------------------------------
# LMS utilities (pure mathematics, independent of coefficient tables)
# ---------------------------------------------------------------------------

def lms_z_score(measured: float, L: float, M: float, S: float) -> float:
    """
    Compute the LMS z-score.

    When L ≠ 0:
        z = ((measured / M) ** L − 1) / (L · S)
    When L = 0 (log-normal limit):
        z = log(measured / M) / S

    Raises ValidationError on non-positive measured or M, or non-positive S.
    """
    if measured <= 0 or M <= 0 or S <= 0:
        raise ValidationError(
            f"LMS requires measured > 0, M > 0, S > 0; "
            f"got measured={measured}, M={M}, S={S}"
        )
    if abs(L) < 1e-12:  # treat as L == 0
        return math.log(measured / M) / S
    return ((measured / M) ** L - 1.0) / (L * S)


def lms_from_z(z: float, L: float, M: float, S: float) -> float:
    """
    Invert the LMS transform: recover the absolute value that corresponds
    to a given z-score (used for LLN).
    """
    if M <= 0 or S <= 0:
        raise ValidationError(f"LMS requires M > 0, S > 0; got M={M}, S={S}")
    if abs(L) < 1e-12:
        return M * math.exp(z * S)
    return M * (1.0 + L * S * z) ** (1.0 / L)


# Conventional lower-limit z-score used by GLI (approx. 5th percentile)
LLN_Z = -1.64485
