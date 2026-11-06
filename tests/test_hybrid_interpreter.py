"""Tests for Dual-Engine Spirometry Interpreter and Discrepancy Analyzer."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_interpreter import (
    hybrid_interpret,
    interpret_spirometry_dual_engine,
)


def test_dual_engine_output_structure():
    """Verify exact schema of dual-engine output dictionary."""
    data = {
        "age": 55,
        "height": 170,
        "weight": 75,
        "sex": "Male",
        "fev1": 2.1,
        "fvc": 3.5,
        "race": "White",
    }
    result = interpret_spirometry_dual_engine(data)

    # Top-level keys
    assert "physiological_engine" in result
    assert "ml_surrogate" in result
    assert "comparison" in result
    assert "disclaimer" in result

    # Physiological Engine schema
    pe = result["physiological_engine"]
    assert "predicted" in pe
    assert set(pe["predicted"].keys()) == {"fev1", "fvc", "ratio"}
    assert "lln" in pe
    assert set(pe["lln"].keys()) == {"fev1", "fvc", "ratio"}
    assert "zscore" in pe
    assert set(pe["zscore"].keys()) == {"fev1", "fvc", "ratio"}
    assert "pct_predicted" in pe
    assert set(pe["pct_predicted"].keys()) == {"fev1", "fvc"}
    assert "pattern" in pe
    assert "severity" in pe
    assert "clinical_note" in pe
    assert "is_borderline" in pe
    assert isinstance(pe["is_borderline"], bool)

    # ML Surrogate schema
    ml = result["ml_surrogate"]
    assert "pattern" in ml
    assert "probabilities" in ml
    assert isinstance(ml["probabilities"], dict)
    assert "shap_explanation" in ml
    assert isinstance(ml["shap_explanation"], list)

    # Comparison schema
    cmp = result["comparison"]
    assert "concordant" in cmp
    assert isinstance(cmp["concordant"], bool)
    assert "discrepancy_detected" in cmp
    assert isinstance(cmp["discrepancy_detected"], bool)
    assert cmp["concordant"] == (not cmp["discrepancy_detected"])
    assert "discrepancy_explanation" in cmp
    assert isinstance(cmp["discrepancy_explanation"], str)

    # Disclaimer check
    assert isinstance(result["disclaimer"], str)
    assert len(result["disclaimer"]) > 20


def test_dual_engine_concordant_obstructive():
    """Verify concordant obstructive pattern between physiological engine and ML."""
    data = {
        "age": 55,
        "height": 170,
        "weight": 75,
        "sex": "Male",
        "fev1": 2.1,
        "fvc": 3.5,
        "race": "White",
    }
    result = interpret_spirometry_dual_engine(data)

    assert result["physiological_engine"]["pattern"] == "Obstructive"
    assert result["ml_surrogate"]["pattern"] == "Obstructive"
    assert result["comparison"]["concordant"] is True
    assert result["comparison"]["discrepancy_detected"] is False
    assert result["physiological_engine"]["severity"] in ["Moderate", "Severe"]


def test_dual_engine_concordant_restrictive_tlc_caveat():
    """Verify normalization of Suspected Restrictive vs Restrictive and TLC caveat."""
    data = {
        "age": 40,
        "height": 170,
        "weight": 70,
        "sex": "Female",
        "fev1": 2.2,
        "fvc": 2.4,
        "race": "White",
    }
    result = interpret_spirometry_dual_engine(data)

    assert result["physiological_engine"]["pattern"] == "Suspected Restrictive"
    assert result["ml_surrogate"]["pattern"] == "Restrictive"
    # Normalized patterns should be concordant
    assert result["comparison"]["concordant"] is True
    assert result["comparison"]["discrepancy_detected"] is False
    # Restriction MUST include TLC requirement
    assert "TLC" in result["physiological_engine"]["clinical_note"]
    assert "TLC" in result["disclaimer"]


def test_dual_engine_concordant_normal():
    """Verify concordant normal spirometry pattern."""
    data = {
        "age": 25,
        "height": 180,
        "weight": 75,
        "sex": "Male",
        "fev1": 4.5,
        "fvc": 5.2,
        "race": "White",
    }
    result = interpret_spirometry_dual_engine(data)

    assert result["physiological_engine"]["pattern"] == "Normal Spirometry"
    assert result["ml_surrogate"]["pattern"] == "Normal"
    assert result["comparison"]["concordant"] is True
    assert result["comparison"]["discrepancy_detected"] is False


def test_dual_engine_discrepancy_detection():
    """Verify divergence detection when ML and physiological engine disagree."""
    # FEV1=2.3, FVC=3.6 at age 60: ratio=63.9%, LLN=64.6% -> Phys: Obstructive, ML: Normal
    data = {
        "age": 60,
        "height": 170,
        "weight": 75,
        "sex": "Male",
        "fev1": 2.3,
        "fvc": 3.6,
        "race": "White",
    }
    result = interpret_spirometry_dual_engine(data)

    assert result["physiological_engine"]["pattern"] == "Obstructive"
    assert result["ml_surrogate"]["pattern"] == "Normal"
    assert result["comparison"]["concordant"] is False
    assert result["comparison"]["discrepancy_detected"] is True

    explanation = result["comparison"]["discrepancy_explanation"]
    assert "Discrepancy detected" in explanation
    assert "Obstructive" in explanation
    assert "Normal" in explanation


def test_dual_engine_validation_error():
    """Verify physiological validation failure raises ValueError."""
    # FEV1 > FVC is physiologically impossible
    invalid_data = {
        "age": 55,
        "height": 170,
        "weight": 75,
        "sex": "Male",
        "fev1": 4.0,
        "fvc": 3.0,
        "race": "White",
    }
    with pytest.raises(ValueError, match="FEV1 cannot exceed FVC"):
        interpret_spirometry_dual_engine(invalid_data)


def test_dual_engine_defaults_handling():
    """Verify optional demographic defaults (race, weight) without crashing."""
    data = {
        "age": 30,
        "height_cm": 175,
        "sex": "Male",
        "fev1": 3.8,
        "fvc": 4.5,
    }
    result = interpret_spirometry_dual_engine(data)
    assert result["physiological_engine"]["pattern"] == "Normal Spirometry"
    assert "comparison" in result


def test_dual_engine_backward_compatibility():
    """Verify existing hybrid_interpret function works as expected."""
    legacy_result = hybrid_interpret(
        fev1=2.1,
        fvc=3.5,
        age=55,
        sex="Male",
        height=170,
        weight=75,
        race="White",
    )
    assert legacy_result["final_pattern"] == "Obstructive"
    assert "ml_model" in legacy_result
    assert "rule_based" in legacy_result
    assert "shap_explanation" in legacy_result
