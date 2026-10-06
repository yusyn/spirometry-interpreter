"""
Schema-validation tests for coefficient / spline data files.

All numeric values used here are *synthetic* and have no relation to
official GLI coefficients.  They only exercise structural checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reference.schema import (  # noqa: E402
    SchemaError,
    SplineTable,
    load_coefficients,
    load_spline_csv,
    validate_method_data_dir,
)


def _write_valid_coeffs(path: Path, method: str = "gli_global") -> None:
    # Synthetic numbers only — NOT official GLI coefficients.
    data = {
        "method": method,
        "version": "synthetic-test-v0",
        "source": "unit-test fixture (not official)",
        "licence": "test-only",
        "accessed": "2026-01-01",
        "parameters": {
            "FEV1": {
                "male": {"a": 0.0, "b_height": 0.0, "c_age": 0.0, "L": 1.0},
                "female": {"a": 0.0, "b_height": 0.0, "c_age": 0.0, "L": 1.0},
            },
            "FVC": {
                "male": {"a": 0.0, "b_height": 0.0, "c_age": 0.0, "L": 1.0},
                "female": {"a": 0.0, "b_height": 0.0, "c_age": 0.0, "L": 1.0},
            },
            "FEV1/FVC": {
                "male": {"a": 0.0, "b_height": 0.0, "c_age": 0.0, "L": 1.0},
                "female": {"a": 0.0, "b_height": 0.0, "c_age": 0.0, "L": 1.0},
            },
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_valid_spline(path: Path) -> None:
    lines = ["age,spline", "3.0,0.0", "10.0,0.1", "25.0,0.0", "50.0,-0.1", "95.0,-0.2"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCoefficientsSchema:
    def test_valid_coefficients(self, tmp_path: Path):
        p = tmp_path / "coefficients.json"
        _write_valid_coeffs(p)
        table = load_coefficients(p)
        assert table.method == "gli_global"
        assert table.version == "synthetic-test-v0"
        assert "FEV1" in table.parameters

    def test_missing_key(self, tmp_path: Path):
        p = tmp_path / "coefficients.json"
        p.write_text(json.dumps({"method": "x"}), encoding="utf-8")
        with pytest.raises(SchemaError, match="missing required key"):
            load_coefficients(p)

    def test_missing_index(self, tmp_path: Path):
        p = tmp_path / "coefficients.json"
        data = {
            "method": "gli_global",
            "version": "v",
            "source": "s",
            "licence": "l",
            "accessed": "d",
            "parameters": {"FEV1": {"male": {"a": 0, "b_height": 0, "c_age": 0, "L": 1},
                                    "female": {"a": 0, "b_height": 0, "c_age": 0, "L": 1}}},
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SchemaError, match="missing index"):
            load_coefficients(p)

    def test_non_finite_coefficient(self, tmp_path: Path):
        p = tmp_path / "coefficients.json"
        data = {
            "method": "gli_global",
            "version": "v",
            "source": "s",
            "licence": "l",
            "accessed": "d",
            "parameters": {
                "FEV1": {
                    "male": {"a": float("nan"), "b_height": 0, "c_age": 0, "L": 1},
                    "female": {"a": 0, "b_height": 0, "c_age": 0, "L": 1},
                },
                "FVC": {
                    "male": {"a": 0, "b_height": 0, "c_age": 0, "L": 1},
                    "female": {"a": 0, "b_height": 0, "c_age": 0, "L": 1},
                },
                "FEV1/FVC": {
                    "male": {"a": 0, "b_height": 0, "c_age": 0, "L": 1},
                    "female": {"a": 0, "b_height": 0, "c_age": 0, "L": 1},
                },
            },
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SchemaError, match="finite"):
            load_coefficients(p)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(SchemaError, match="not found"):
            load_coefficients(tmp_path / "nope.json")


class TestSplineSchema:
    def test_valid_spline(self, tmp_path: Path):
        p = tmp_path / "m_spline.csv"
        _write_valid_spline(p)
        table = load_spline_csv(p)
        assert len(table.ages) == 5
        assert table.ages[0] == 3.0

    def test_interpolation(self, tmp_path: Path):
        p = tmp_path / "m_spline.csv"
        _write_valid_spline(p)
        table = load_spline_csv(p)
        mid = table.interpolate(6.5)
        assert abs(mid - 0.05) < 1e-12

    def test_no_extrapolation(self, tmp_path: Path):
        p = tmp_path / "m_spline.csv"
        _write_valid_spline(p)
        table = load_spline_csv(p)
        with pytest.raises(SchemaError, match="outside spline table domain"):
            table.interpolate(2.0)
        with pytest.raises(SchemaError, match="outside spline table domain"):
            table.interpolate(100.0)

    def test_non_increasing_ages(self, tmp_path: Path):
        p = tmp_path / "m_spline.csv"
        p.write_text("age,spline\n10.0,0.0\n5.0,0.1\n", encoding="utf-8")
        with pytest.raises(SchemaError, match="strictly increasing"):
            load_spline_csv(p)

    def test_missing_columns(self, tmp_path: Path):
        p = tmp_path / "m_spline.csv"
        p.write_text("x,y\n1,2\n", encoding="utf-8")
        with pytest.raises(SchemaError, match="age.*spline"):
            load_spline_csv(p)

    def test_too_few_points(self, tmp_path: Path):
        p = tmp_path / "m_spline.csv"
        p.write_text("age,spline\n3.0,0.0\n", encoding="utf-8")
        with pytest.raises(SchemaError, match="at least 2"):
            load_spline_csv(p)


class TestMethodDataDir:
    def test_full_valid_dir(self, tmp_path: Path):
        method = "gli_global"
        d = tmp_path / method
        d.mkdir()
        _write_valid_coeffs(d / "coefficients.json", method=method)
        _write_valid_spline(d / "m_spline.csv")
        _write_valid_spline(d / "s_spline.csv")
        summary = validate_method_data_dir(tmp_path, method)
        assert summary["method"] == method
        assert summary["n_spline_points"] == 5

    def test_mismatched_method_key(self, tmp_path: Path):
        method = "gli_global"
        d = tmp_path / method
        d.mkdir()
        _write_valid_coeffs(d / "coefficients.json", method="wrong_key")
        _write_valid_spline(d / "m_spline.csv")
        _write_valid_spline(d / "s_spline.csv")
        with pytest.raises(SchemaError, match="does not match"):
            validate_method_data_dir(tmp_path, method)

    def test_missing_directory(self, tmp_path: Path):
        with pytest.raises(SchemaError, match="data directory missing"):
            validate_method_data_dir(tmp_path, "gli_global")
