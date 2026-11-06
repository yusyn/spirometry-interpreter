"""Tests for deterministic reference and z-score calculator."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_engine.calculator import ReferenceOutput, calculate_reference_values


def test_reference_calculation_male_adult():
    """Verify standard reference calculation for an adult male per brief requirements."""
    res = calculate_reference_values(age=30, height_cm=175, sex="Male", fev1=4.0, fvc=5.0)
    assert res.fev1_pred > 3.0 and res.fev1_pred < 5.0
    assert res.fvc_pred > 4.0 and res.fvc_pred < 6.0
    assert res.fev1_pct_pred > 80.0
    assert isinstance(res.fev1_zscore, float)
    assert res.fev1_lln < res.fev1_pred


def test_reference_calculation_exact_male_values():
    """Verify exact numerical equations for adult male (ECSC / Quanjer 1993)."""
    # Male formulas:
    # H = 1.75 m, A = 30 yr
    # FEV1_pred = 4.301 * 1.75 - 0.0290 * 30 - 2.492 = 4.16475 (RSD=0.51)
    # FVC_pred  = 5.761 * 1.75 - 0.0260 * 30 - 4.343 = 4.95875 (RSD=0.61)
    # Ratio_pred = -0.18 * 30 + 87.21 = 81.81 (RSD=7.17)
    # LLN = pred - 1.645 * RSD
    res = calculate_reference_values(age=30, height_cm=175, sex="Male", fev1=4.0, fvc=5.0)

    assert res.fev1_pred == pytest.approx(4.16475, rel=1e-5)
    assert res.fvc_pred == pytest.approx(4.95875, rel=1e-5)
    assert res.ratio_pred == pytest.approx(81.81, rel=1e-5)

    assert res.fev1_pct_pred == pytest.approx((4.0 / 4.16475) * 100.0, rel=1e-5)
    assert res.fvc_pct_pred == pytest.approx((5.0 / 4.95875) * 100.0, rel=1e-5)

    assert res.fev1_zscore == pytest.approx((4.0 - 4.16475) / 0.51, rel=1e-5)
    assert res.fvc_zscore == pytest.approx((5.0 - 4.95875) / 0.61, rel=1e-5)
    measured_ratio = (4.0 / 5.0) * 100.0
    assert res.ratio_zscore == pytest.approx((measured_ratio - 81.81) / 7.17, rel=1e-5)

    assert res.fev1_lln == pytest.approx(4.16475 - 1.645 * 0.51, rel=1e-5)
    assert res.fvc_lln == pytest.approx(4.95875 - 1.645 * 0.61, rel=1e-5)
    assert res.ratio_lln == pytest.approx(81.81 - 1.645 * 7.17, rel=1e-5)


def test_reference_calculation_exact_female_values():
    """Verify exact numerical equations for adult female (ECSC / Quanjer 1993)."""
    # Female formulas:
    # H = 1.65 m, A = 25 yr
    # FEV1_pred = 3.953 * 1.65 - 0.0251 * 25 - 2.604 = 3.29095 (RSD=0.38)
    # FVC_pred  = 4.433 * 1.65 - 0.0260 * 25 - 2.892 = 3.77245 (RSD=0.43)
    # Ratio_pred = -0.19 * 25 + 89.10 = 84.35 (RSD=6.51)
    res = calculate_reference_values(age=25, height_cm=165, sex="Female", fev1=3.0, fvc=3.5)

    assert res.fev1_pred == pytest.approx(3.29095, rel=1e-5)
    assert res.fvc_pred == pytest.approx(3.77245, rel=1e-5)
    assert res.ratio_pred == pytest.approx(84.35, rel=1e-5)

    assert res.fev1_pct_pred == pytest.approx((3.0 / 3.29095) * 100.0, rel=1e-5)
    assert res.fvc_pct_pred == pytest.approx((3.5 / 3.77245) * 100.0, rel=1e-5)

    assert res.fev1_zscore == pytest.approx((3.0 - 3.29095) / 0.38, rel=1e-5)
    assert res.fvc_zscore == pytest.approx((3.5 - 3.77245) / 0.43, rel=1e-5)
    measured_ratio = (3.0 / 3.5) * 100.0
    assert res.ratio_zscore == pytest.approx((measured_ratio - 84.35) / 6.51, rel=1e-5)

    assert res.fev1_lln == pytest.approx(3.29095 - 1.645 * 0.38, rel=1e-5)
    assert res.fvc_lln == pytest.approx(3.77245 - 1.645 * 0.43, rel=1e-5)
    assert res.ratio_lln == pytest.approx(84.35 - 1.645 * 6.51, rel=1e-5)


def test_sex_normalization_and_height_in_meters():
    """Verify that case-insensitive sex and height in meters (<= 2.5) are properly normalized."""
    res_cm = calculate_reference_values(age=30, height_cm=175.0, sex="Male", fev1=4.0, fvc=5.0)
    res_m = calculate_reference_values(age=30, height_cm=1.75, sex="m", fev1=4.0, fvc=5.0)
    assert res_cm.fev1_pred == pytest.approx(res_m.fev1_pred)
    assert res_cm.fvc_pred == pytest.approx(res_m.fvc_pred)
    assert res_cm.ratio_pred == pytest.approx(res_m.ratio_pred)

    res_f = calculate_reference_values(age=40, height_cm=160.0, sex="f", fev1=2.8, fvc=3.2)
    res_female = calculate_reference_values(age=40, height_cm=160.0, sex="FEMALE", fev1=2.8, fvc=3.2)
    assert res_f.fev1_pred == pytest.approx(res_female.fev1_pred)


def test_dataclass_fields():
    """Verify ReferenceOutput contains all 11 required fields."""
    res = calculate_reference_values(age=30, height_cm=175, sex="Male", fev1=4.0, fvc=5.0)
    expected_fields = {
        "fev1_pred",
        "fvc_pred",
        "ratio_pred",
        "fev1_pct_pred",
        "fvc_pct_pred",
        "fev1_zscore",
        "fvc_zscore",
        "ratio_zscore",
        "fev1_lln",
        "fvc_lln",
        "ratio_lln",
    }
    assert set(res.__dataclass_fields__.keys()) == expected_fields


@pytest.mark.parametrize(
    "kwargs,err_match",
    [
        (dict(age=5, height_cm=170, sex="Male", fev1=3.0, fvc=3.5), "Age must be between 6 and 80"),
        (dict(age=85, height_cm=170, sex="Male", fev1=3.0, fvc=3.5), "Age must be between 6 and 80"),
        (dict(age=30, height_cm=80, sex="Male", fev1=3.0, fvc=3.5), "Height must be between 100 and 220"),
        (dict(age=30, height_cm=250, sex="Male", fev1=3.0, fvc=3.5), "Height must be between 100 and 220"),
        (dict(age=30, height_cm=170, sex="Unknown", fev1=3.0, fvc=3.5), "Sex must be one of"),
        (dict(age=30, height_cm=170, sex="Male", fev1=4.5, fvc=4.0), "FEV1 cannot exceed FVC"),
        (dict(age=30, height_cm=170, sex="Male", fev1=-1.0, fvc=4.0), "FEV1 must be greater than zero"),
    ],
)
def test_validation_errors_raise_value_error(kwargs, err_match):
    """Verify input validation fails with ValueError on boundary / physiological violations."""
    with pytest.raises(ValueError, match=err_match):
        calculate_reference_values(**kwargs)
