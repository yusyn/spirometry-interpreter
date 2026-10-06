"""
Contract, validation and pure-LMS tests for the GLI reference module.

Numerical acceptance tests that require official coefficient tables are
guarded by a pytest skip when the tables are absent.  No synthetic
coefficients are invented.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make the package importable without installation
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reference import (  # noqa: E402
    IndexResult,
    ReferenceMethod,
    ReferenceResult,
    Sex,
    SpirometryIndex,
    ValidationError,
    compute_reference,
)
from reference.base import (  # noqa: E402
    LLN_Z,
    lms_from_z,
    lms_z_score,
)
from reference.gli import (  # noqa: E402
    DataNotAvailableError,
    compute_lms_metrics,
)


# ---------------------------------------------------------------------------
# Pure LMS mathematics (always runnable)
# ---------------------------------------------------------------------------

class TestLMS:
    def test_z_score_L_nonzero(self):
        # Classic example: L=1, M=3.0, S=0.1, measured=3.0 → z=0
        z = lms_z_score(3.0, L=1.0, M=3.0, S=0.1)
        assert abs(z) < 1e-12

    def test_z_score_L_zero(self):
        # L→0 limit: z = ln(y/M)/S
        z = lms_z_score(math.e, L=0.0, M=1.0, S=1.0)
        assert abs(z - 1.0) < 1e-12

    def test_lln_roundtrip(self):
        L, M, S = 1.2, 3.5, 0.12
        lln = lms_from_z(LLN_Z, L, M, S)
        z_back = lms_z_score(lln, L, M, S)
        assert abs(z_back - LLN_Z) < 1e-9

    def test_lms_metrics_wrapper(self):
        z, lln = compute_lms_metrics(2.8, L=1.0, M=3.0, S=0.1)
        assert abs(z - ((2.8 / 3.0) - 1) / 0.1) < 1e-12
        assert lln < 3.0  # LLN must be below the median

    def test_lms_rejects_non_positive(self):
        with pytest.raises(ValidationError):
            lms_z_score(0.0, 1.0, 3.0, 0.1)
        with pytest.raises(ValidationError):
            lms_z_score(2.0, 1.0, -1.0, 0.1)
        with pytest.raises(ValidationError):
            lms_z_score(2.0, 1.0, 3.0, 0.0)


# ---------------------------------------------------------------------------
# Input validation (always runnable)
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_method(self):
        with pytest.raises(ValidationError, match="Unknown reference method"):
            compute_reference(
                method="nhanes_iii",
                sex=Sex.MALE,
                age_years=40.0,
                height_cm=175.0,
            )

    def test_invalid_sex(self):
        with pytest.raises(ValidationError, match="sex must be"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex="unknown",
                age_years=40.0,
                height_cm=175.0,
            )

    def test_age_out_of_domain(self):
        with pytest.raises(ValidationError, match="outside the documented domain"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.FEMALE,
                age_years=2.0,
                height_cm=165.0,
            )
        with pytest.raises(ValidationError, match="outside the documented domain"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.FEMALE,
                age_years=96.0,
                height_cm=165.0,
            )

    def test_height_out_of_domain(self):
        with pytest.raises(ValidationError, match="outside the documented domain"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.MALE,
                age_years=40.0,
                height_cm=80.0,
            )

    def test_nan_age(self):
        with pytest.raises(ValidationError, match="finite"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.MALE,
                age_years=float("nan"),
                height_cm=175.0,
            )

    def test_inf_height(self):
        with pytest.raises(ValidationError, match="finite"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.MALE,
                age_years=40.0,
                height_cm=float("inf"),
            )

    def test_negative_measured_volume(self):
        with pytest.raises(ValidationError, match="must be ≥ 0"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.MALE,
                age_years=40.0,
                height_cm=175.0,
                measured={"FEV1": -0.5},
            )

    def test_invalid_ratio(self):
        with pytest.raises(ValidationError, match="FEV1/FVC"):
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.MALE,
                age_years=40.0,
                height_cm=175.0,
                measured={"FEV1/FVC": 2.0},
            )


# ---------------------------------------------------------------------------
# Method routing – independent paths (always runnable)
# ---------------------------------------------------------------------------

class TestMethodRouting:
    def test_gli_global_and_gli_2012_other_are_distinct(self):
        """Both methods must raise DataNotAvailableError independently;
        the error message must name the correct method key."""
        with pytest.raises(DataNotAvailableError) as exc_g:
            compute_reference(
                method=ReferenceMethod.GLI_GLOBAL,
                sex=Sex.FEMALE,
                age_years=45.0,
                height_cm=165.0,
            )
        assert "gli_global" in str(exc_g.value)

        with pytest.raises(DataNotAvailableError) as exc_o:
            compute_reference(
                method=ReferenceMethod.GLI_2012_OTHER,
                sex=Sex.FEMALE,
                age_years=45.0,
                height_cm=165.0,
            )
        assert "gli_2012_other" in str(exc_o.value)

        # The two error messages must differ (independent code paths)
        assert str(exc_g.value) != str(exc_o.value)

    def test_string_method_accepted(self):
        with pytest.raises(DataNotAvailableError):
            compute_reference(
                method="gli_global",
                sex="female",
                age_years=30.0,
                height_cm=160.0,
            )


# ---------------------------------------------------------------------------
# Metadata contract (always runnable once tables appear;
# for now we only verify the exception carries the method identity)
# ---------------------------------------------------------------------------

class TestMetadataContract:
    def test_result_would_carry_reference_id(self):
        """
        When tables become available the IndexResult must expose
        reference_id and reference_version.  Until then we document the
        expected contract via the dataclass fields themselves.
        """
        fields = {f.name for f in IndexResult.__dataclass_fields__.values()}
        assert "reference_id" in fields
        assert "reference_version" in fields
        assert "unsupported_reason" in fields

    def test_reference_result_carries_method(self):
        fields = {f.name for f in ReferenceResult.__dataclass_fields__.values()}
        assert "method" in fields
        assert "metadata" in fields


# ---------------------------------------------------------------------------
# Numerical tests – skipped until official tables are present
# ---------------------------------------------------------------------------

def _tables_present(method_key: str) -> bool:
    data = ROOT / "src" / "reference" / "data" / method_key
    return all(
        (data / name).is_file()
        for name in ("coefficients.json", "m_spline.csv", "s_spline.csv")
    )


@pytest.mark.skipif(
    not _tables_present("gli_global"),
    reason="Official GLI Global coefficient/spline tables not present under "
           "src/reference/data/gli_global/.  Numerical acceptance blocked.",
)
class TestGLIGlobalNumerical:
    """Placeholder for official worked examples once tables arrive."""

    def test_placeholder(self):
        # Replace with concrete assertions from Bowerman et al. supplement
        # or from a trusted independent implementation (e.g. rspiro).
        result = compute_reference(
            method=ReferenceMethod.GLI_GLOBAL,
            sex=Sex.MALE,
            age_years=40.0,
            height_cm=175.0,
            measured={"FEV1": 3.5, "FVC": 4.5},
        )
        assert result.method is ReferenceMethod.GLI_GLOBAL
        assert "FEV1" in result.results
        assert result.results["FEV1"].reference_id == "gli_global"
        assert result.results["FEV1"].predicted is not None


@pytest.mark.skipif(
    not _tables_present("gli_2012_other"),
    reason="Official GLI-2012 Other/Mixed coefficient/spline tables not present "
           "under src/reference/data/gli_2012_other/.  Numerical acceptance blocked.",
)
class TestGLI2012OtherNumerical:
    """Placeholder for official worked examples once tables arrive."""

    def test_placeholder(self):
        result = compute_reference(
            method=ReferenceMethod.GLI_2012_OTHER,
            sex=Sex.FEMALE,
            age_years=50.0,
            height_cm=160.0,
            measured={"FEV1": 2.5, "FVC": 3.2},
        )
        assert result.method is ReferenceMethod.GLI_2012_OTHER
        assert result.results["FEV1"].reference_id == "gli_2012_other"
