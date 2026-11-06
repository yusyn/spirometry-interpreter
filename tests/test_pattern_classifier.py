"""Tests for ATS/ERS Physiological Pattern Classifier."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_engine.calculator import calculate_reference_values
from src.reference_engine.pattern_classifier import (
    PatternResult,
    classify_physiological_pattern,
)


def test_obstructive_pattern():
    """Verify obstructive pattern detection and moderate/severe grading."""
    ref = calculate_reference_values(age=50, height_cm=170, sex="Male", fev1=1.8, fvc=3.8)
    res = classify_physiological_pattern(fev1=1.8, fvc=3.8, ref=ref)
    assert res.pattern == "Obstructive"
    assert res.severity in ["Moderate", "Severe"]
    assert res.severity == "Moderate"
    assert res.used_fixed_ratio is False
    assert res.is_borderline is False


def test_suspected_restrictive_pattern():
    """Verify suspected restrictive pattern and mandatory TLC disclaimer."""
    ref = calculate_reference_values(age=40, height_cm=170, sex="Female", fev1=2.2, fvc=2.4)
    res = classify_physiological_pattern(fev1=2.2, fvc=2.4, ref=ref)
    assert res.pattern == "Suspected Restrictive"
    assert res.severity is None
    assert "TLC measurement required to confirm true restrictive lung defect" in res.clinical_note
    assert "TLC measurement required" in res.clinical_note


def test_normal_spirometry_pattern():
    """Verify normal spirometry when ratio and FVC are both above LLN."""
    ref = calculate_reference_values(age=25, height_cm=175, sex="Male", fev1=4.0, fvc=4.8)
    res = classify_physiological_pattern(fev1=4.0, fvc=4.8, ref=ref)
    assert res.pattern == "Normal Spirometry"
    assert res.severity is None
    assert res.is_borderline is False


def test_mixed_pattern():
    """Verify mixed obstructive-restrictive pattern with low ratio and low FVC."""
    # 60yo Male, 170cm: fvc_pred ~ 3.88, fvc_lln ~ 2.88. fvc=2.4 is low.
    # ratio_lln ~ 64.6%. fev1=1.3, fvc=2.4 -> ratio = 54.17% (obstructive).
    ref = calculate_reference_values(age=60, height_cm=170, sex="Male", fev1=1.3, fvc=2.4)
    res = classify_physiological_pattern(fev1=1.3, fvc=2.4, ref=ref)
    assert res.pattern == "Mixed"
    assert res.severity in ["Moderate", "Severe", "Very Severe"]
    assert "TLC measurement required to confirm true restrictive lung defect" in res.clinical_note


def test_severity_levels():
    """Verify ATS/ERS severity levels: Mild, Moderate, Severe, Very Severe."""
    # 50yo Male, 170cm: fev1_pred = 3.3697, fvc_pred = 4.1507
    # Ratio LLN = 66.42%
    # Normal FVC = 4.0 (> LLN ~ 3.15)
    # Mild: fev1 % pred >= 80% -> fev1 = 2.75 (81.6%), ratio = 2.75/4.5 = 61.1% (< 66.42%)
    ref_mild = calculate_reference_values(age=50, height_cm=170, sex="Male", fev1=2.75, fvc=4.5)
    res_mild = classify_physiological_pattern(fev1=2.75, fvc=4.5, ref=ref_mild)
    assert res_mild.pattern == "Obstructive"
    assert res_mild.severity == "Mild"

    # Moderate: 50% <= fev1 % pred < 80% -> fev1 = 2.0 (59.4%), fvc = 4.0 -> ratio = 50%
    ref_mod = calculate_reference_values(age=50, height_cm=170, sex="Male", fev1=2.0, fvc=4.0)
    res_mod = classify_physiological_pattern(fev1=2.0, fvc=4.0, ref=ref_mod)
    assert res_mod.pattern == "Obstructive"
    assert res_mod.severity == "Moderate"

    # Severe: 30% <= fev1 % pred < 50% -> fev1 = 1.3 (38.6%), fvc = 3.5 -> ratio = 37.1%
    ref_sev = calculate_reference_values(age=50, height_cm=170, sex="Male", fev1=1.3, fvc=3.5)
    res_sev = classify_physiological_pattern(fev1=1.3, fvc=3.5, ref=ref_sev)
    assert res_sev.pattern == "Obstructive"
    assert res_sev.severity == "Severe"

    # Very Severe: fev1 % pred < 30% -> fev1 = 0.9 (26.7%), fvc = 3.5 -> ratio = 25.7%
    ref_vsev = calculate_reference_values(age=50, height_cm=170, sex="Male", fev1=0.9, fvc=3.5)
    res_vsev = classify_physiological_pattern(fev1=0.9, fvc=3.5, ref=ref_vsev)
    assert res_vsev.pattern == "Obstructive"
    assert res_vsev.severity == "Very Severe"


def test_fixed_ratio_mode():
    """Verify behavior when use_fixed_ratio is True vs False."""
    # 70yo Male, 170cm:
    # ratio_pred = 74.61%, ratio_lln = 62.82%
    # If fev1=2.6, fvc=3.9: ratio = 66.67%.
    # 66.67% < 70% (obstructive by fixed ratio)
    # 66.67% > 62.82% (normal by LLN)
    ref = calculate_reference_values(age=70, height_cm=170, sex="Male", fev1=2.6, fvc=3.9)

    # LLN mode (default): should be Normal Spirometry
    res_lln = classify_physiological_pattern(fev1=2.6, fvc=3.9, ref=ref, use_fixed_ratio=False)
    assert res_lln.pattern == "Normal Spirometry"
    assert res_lln.used_fixed_ratio is False
    assert res_lln.is_borderline is True  # Discrepancy between fixed and LLN!

    # Fixed ratio mode: should be Obstructive
    res_fixed = classify_physiological_pattern(fev1=2.6, fvc=3.9, ref=ref, use_fixed_ratio=True)
    assert res_fixed.pattern == "Obstructive"
    assert res_fixed.used_fixed_ratio is True
    assert res_fixed.is_borderline is True


def test_borderline_near_lln():
    """Verify borderline detection when ratio is within proximity of LLN threshold."""
    # 50yo Male, 170cm: ratio_lln = 66.415%
    # Suppose measured ratio is 66.0% (just below LLN, e.g. fev1=2.508, fvc=3.8 -> 66.0%)
    ref = calculate_reference_values(age=50, height_cm=170, sex="Male", fev1=2.508, fvc=3.8)
    res = classify_physiological_pattern(fev1=2.508, fvc=3.8, ref=ref)
    assert res.is_borderline is True
    assert "Borderline" in res.clinical_note or "threshold" in res.clinical_note


def test_dataclass_structure():
    """Verify PatternResult dataclass attributes and types."""
    ref = calculate_reference_values(age=30, height_cm=175, sex="Male", fev1=3.8, fvc=4.5)
    res = classify_physiological_pattern(fev1=3.8, fvc=4.5, ref=ref)
    assert isinstance(res, PatternResult)
    assert hasattr(res, "pattern")
    assert hasattr(res, "severity")
    assert hasattr(res, "clinical_note")
    assert hasattr(res, "is_borderline")
    assert hasattr(res, "used_fixed_ratio")


def test_invalid_physiological_inputs_raise():
    """Verify that impossible physiological combinations raise ValueError."""
    ref = calculate_reference_values(age=30, height_cm=175, sex="Male", fev1=3.8, fvc=4.5)
    with pytest.raises(ValueError, match="FEV1 cannot exceed FVC"):
        classify_physiological_pattern(fev1=4.8, fvc=4.5, ref=ref)
    with pytest.raises(ValueError, match="FEV1 and FVC must be positive"):
        classify_physiological_pattern(fev1=0.0, fvc=4.5, ref=ref)
