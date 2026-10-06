"""
Schema validation for GLI coefficient and spline data files.

This module validates the *structure* of user-supplied tables.  It never
embeds official numeric coefficients.  Tables must be obtained by the
integrator from the GLI Network under the applicable licence and placed
under ``src/reference/data/{method}/``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """Raised when a coefficient/spline file fails structural validation."""


# Indices that both equation families are expected to cover once tables exist.
REQUIRED_INDICES = ("FEV1", "FVC", "FEV1/FVC")
REQUIRED_SEXES = ("male", "female")

# Soft domain bounds used only for schema plausibility checks (not numerical eval).
_AGE_MIN_PLAUABLE = 2.5
_AGE_MAX_PLAUABLE = 96.0


@dataclass(frozen=True)
class CoefficientsTable:
    """Validated coefficients.json content."""

    method: str
    version: str
    source: str
    licence: str
    accessed: str
    parameters: Mapping[str, Mapping[str, Mapping[str, float]]]


@dataclass(frozen=True)
class SplineTable:
    """Validated m_spline.csv or s_spline.csv content."""

    ages: tuple[float, ...]
    values: tuple[float, ...]

    def interpolate(self, age: float) -> float:
        """
        Linear interpolation between tabulated ages.

        Raises SchemaError if age is outside the tabulated range
        (no extrapolation; matches the documented GLI practice of
        requiring the age to fall within the look-up table).
        """
        if age < self.ages[0] or age > self.ages[-1]:
            raise SchemaError(
                f"age={age} outside spline table domain "
                f"[{self.ages[0]}, {self.ages[-1]}]; extrapolation is not permitted"
            )
        # Exact match
        for i, a in enumerate(self.ages):
            if abs(a - age) < 1e-12:
                return self.values[i]
        # Linear interpolate between bracketing points
        for i in range(len(self.ages) - 1):
            a0, a1 = self.ages[i], self.ages[i + 1]
            if a0 <= age <= a1:
                t = (age - a0) / (a1 - a0) if a1 != a0 else 0.0
                return self.values[i] + t * (self.values[i + 1] - self.values[i])
        raise SchemaError(f"internal interpolation failure for age={age}")


def load_coefficients(path: Path) -> CoefficientsTable:
    """Load and validate coefficients.json."""
    if not path.is_file():
        raise SchemaError(f"coefficients file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"coefficients.json is not valid JSON: {exc}") from exc

    for key in ("method", "version", "source", "licence", "accessed", "parameters"):
        if key not in raw:
            raise SchemaError(f"coefficients.json missing required key: {key!r}")

    params = raw["parameters"]
    if not isinstance(params, dict):
        raise SchemaError("coefficients.json 'parameters' must be an object")

    for index in REQUIRED_INDICES:
        if index not in params:
            raise SchemaError(f"coefficients.json missing index: {index!r}")
        for sex in REQUIRED_SEXES:
            if sex not in params[index]:
                raise SchemaError(
                    f"coefficients.json missing sex {sex!r} under index {index!r}"
                )
            entry = params[index][sex]
            if not isinstance(entry, dict):
                raise SchemaError(
                    f"coefficients for {index}/{sex} must be an object"
                )
            # Required numeric keys for the LMS mean model
            for coeff_key in ("a", "b_height", "c_age", "L"):
                if coeff_key not in entry:
                    raise SchemaError(
                        f"coefficients for {index}/{sex} missing key {coeff_key!r}"
                    )
                val = entry[coeff_key]
                if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                    raise SchemaError(
                        f"coefficients for {index}/{sex}.{coeff_key} must be finite number"
                    )

    return CoefficientsTable(
        method=str(raw["method"]),
        version=str(raw["version"]),
        source=str(raw["source"]),
        licence=str(raw["licence"]),
        accessed=str(raw["accessed"]),
        parameters=params,
    )


def load_spline_csv(path: Path) -> SplineTable:
    """Load and validate an m_spline.csv or s_spline.csv file."""
    if not path.is_file():
        raise SchemaError(f"spline file not found: {path}")

    ages: list[float] = []
    values: list[float] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SchemaError(f"{path.name}: empty CSV")
        normalised = {h.strip().lower(): h for h in reader.fieldnames}
        if "age" not in normalised or "spline" not in normalised:
            raise SchemaError(
                f"{path.name}: CSV must have 'age' and 'spline' columns; "
                f"got {list(reader.fieldnames)}"
            )
        age_col = normalised["age"]
        spline_col = normalised["spline"]
        for row_num, row in enumerate(reader, start=2):
            try:
                age = float(row[age_col])
                val = float(row[spline_col])
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaError(
                    f"{path.name} row {row_num}: non-numeric age/spline ({exc})"
                ) from exc
            if not math.isfinite(age) or not math.isfinite(val):
                raise SchemaError(
                    f"{path.name} row {row_num}: non-finite age or spline value"
                )
            ages.append(age)
            values.append(val)

    if len(ages) < 2:
        raise SchemaError(f"{path.name}: need at least 2 spline points, got {len(ages)}")

    # Must be strictly increasing ages
    for i in range(1, len(ages)):
        if ages[i] <= ages[i - 1]:
            raise SchemaError(
                f"{path.name}: ages must be strictly increasing; "
                f"saw {ages[i - 1]} then {ages[i]} at index {i}"
            )

    if ages[0] > _AGE_MIN_PLAUABLE + 1 or ages[-1] < _AGE_MAX_PLAUABLE - 5:
        # Soft warning-level check only via SchemaError for clearly wrong tables
        pass  # do not hard-fail on soft domain; numerical layer enforces domain

    return SplineTable(ages=tuple(ages), values=tuple(values))


def validate_method_data_dir(data_dir: Path, method_key: str) -> dict[str, Any]:
    """
    Validate the full directory layout for one method.

    Expected files:
        coefficients.json, m_spline.csv, s_spline.csv

    Returns a summary dict suitable for diagnostics.  Raises SchemaError
    on any structural problem.
    """
    base = data_dir / method_key
    if not base.is_dir():
        raise SchemaError(f"data directory missing: {base}")

    coeffs = load_coefficients(base / "coefficients.json")
    if coeffs.method != method_key:
        raise SchemaError(
            f"coefficients.json method={coeffs.method!r} does not match "
            f"directory key {method_key!r}"
        )
    m_spline = load_spline_csv(base / "m_spline.csv")
    s_spline = load_spline_csv(base / "s_spline.csv")

    if len(m_spline.ages) != len(s_spline.ages) or m_spline.ages != s_spline.ages:
        raise SchemaError(
            f"{method_key}: m_spline and s_spline age grids must be identical"
        )

    return {
        "method": coeffs.method,
        "version": coeffs.version,
        "source": coeffs.source,
        "licence": coeffs.licence,
        "accessed": coeffs.accessed,
        "n_spline_points": len(m_spline.ages),
        "age_range": (m_spline.ages[0], m_spline.ages[-1]),
        "indices": list(REQUIRED_INDICES),
    }
