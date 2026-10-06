"""
Spirometry reference-equation engine (PoC).

This package computes predicted values, z-scores and lower limits of normal
(LLN) for spirometric indices relative to published physiological reference
equations.  It does **not** produce a medical diagnosis.

Supported reference methods (must be chosen explicitly by the caller):

* ``gli_global``     – GLI Global race-neutral (2022 / Bowerman et al.)
* ``gli_2012_other`` – GLI-2012 Other/Mixed composite equation

Public API
----------
>>> from reference import compute_reference, ReferenceMethod, Sex
>>> result = compute_reference(
...     method=ReferenceMethod.GLI_GLOBAL,
...     sex=Sex.FEMALE,
...     age_years=45.0,
...     height_cm=165.0,
...     measured={"FEV1": 2.80, "FVC": 3.50},
... )
"""

from .base import (
    IndexResult,
    ReferenceMethod,
    ReferenceResult,
    Sex,
    SpirometryIndex,
    UnsupportedIndexError,
    ValidationError,
)
from .gli import DataNotAvailableError, compute_reference
from .schema import SchemaError

__all__ = [
    "Sex",
    "SpirometryIndex",
    "ReferenceMethod",
    "IndexResult",
    "ReferenceResult",
    "ValidationError",
    "UnsupportedIndexError",
    "DataNotAvailableError",
    "SchemaError",
    "compute_reference",
]

__version__ = "0.1.1-poc"
