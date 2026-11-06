import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_engine.validator import validate_spirometry_input


def test_valid_input():
    valid_data = {"age": 45, "height_cm": 175, "sex": "Male", "fev1": 3.5, "fvc": 4.5}
    is_valid, msg = validate_spirometry_input(valid_data)
    assert is_valid is True
    assert msg is None


def test_fev1_greater_than_fvc():
    invalid_data = {"age": 45, "height_cm": 175, "sex": "Male", "fev1": 5.0, "fvc": 4.0}
    is_valid, msg = validate_spirometry_input(invalid_data)
    assert is_valid is False
    assert "FEV1 cannot exceed FVC" in msg


def test_age_out_of_bounds():
    invalid_data = {"age": 95, "height_cm": 175, "sex": "Male", "fev1": 2.0, "fvc": 3.0}
    is_valid, msg = validate_spirometry_input(invalid_data)
    assert is_valid is False
    assert "Age must be between 6 and 80" in msg


def test_age_too_young():
    invalid_data = {"age": 4, "height_cm": 110, "sex": "Male", "fev1": 1.0, "fvc": 1.2}
    is_valid, msg = validate_spirometry_input(invalid_data)
    assert is_valid is False
    assert "Age must be between 6 and 80" in msg


def test_height_out_of_bounds():
    # Height below 100 cm
    data_low = {"age": 30, "height_cm": 90, "sex": "Male", "fev1": 3.0, "fvc": 3.5}
    is_valid, msg = validate_spirometry_input(data_low)
    assert is_valid is False
    assert "Height must be between 100 and 220" in msg

    # Height above 220 cm
    data_high = {"age": 30, "height_cm": 230, "sex": "Male", "fev1": 3.0, "fvc": 3.5}
    is_valid, msg = validate_spirometry_input(data_high)
    assert is_valid is False
    assert "Height must be between 100 and 220" in msg


def test_height_normalization_from_height_key():
    data = {"age": 45, "height": 175, "sex": "Female", "fev1": 3.0, "fvc": 3.5}
    is_valid, msg = validate_spirometry_input(data)
    assert is_valid is True
    assert msg is None
    assert data["height_cm"] == 175.0


def test_height_in_meters_normalized():
    data = {"age": 45, "height": 1.75, "sex": "Female", "fev1": 3.0, "fvc": 3.5}
    is_valid, msg = validate_spirometry_input(data)
    assert is_valid is True
    assert msg is None
    assert data["height_cm"] == 175.0


def test_sex_validation():
    for valid_sex in ["Male", "Female", "male", "female", "M", "F", "m", "f"]:
        data = {"age": 30, "height_cm": 170, "sex": valid_sex, "fev1": 3.0, "fvc": 3.5}
        is_valid, msg = validate_spirometry_input(data)
        assert is_valid is True, f"Failed for valid sex: {valid_sex}"
        assert msg is None

    invalid_sex_data = {"age": 30, "height_cm": 170, "sex": "Unknown", "fev1": 3.0, "fvc": 3.5}
    is_valid, msg = validate_spirometry_input(invalid_sex_data)
    assert is_valid is False
    assert "Sex must be" in msg


def test_missing_fields():
    data = {"age": 30, "sex": "Male", "fev1": 3.0}
    is_valid, msg = validate_spirometry_input(data)
    assert is_valid is False
    assert "Missing" in msg


def test_non_numeric_values():
    data = {"age": "thirty", "height_cm": 175, "sex": "Male", "fev1": 3.5, "fvc": 4.5}
    is_valid, msg = validate_spirometry_input(data)
    assert is_valid is False
    assert "numeric" in msg.lower()


def test_negative_or_zero_values():
    data_zero_fev1 = {"age": 30, "height_cm": 175, "sex": "Male", "fev1": 0, "fvc": 4.5}
    is_valid, msg = validate_spirometry_input(data_zero_fev1)
    assert is_valid is False
    assert "positive" in msg.lower() or "greater than zero" in msg.lower()

    data_negative_fvc = {"age": 30, "height_cm": 175, "sex": "Male", "fev1": 2.0, "fvc": -1.0}
    is_valid, msg = validate_spirometry_input(data_negative_fvc)
    assert is_valid is False

    data_zero_age = {"age": 0, "height_cm": 175, "sex": "Male", "fev1": 2.0, "fvc": 3.0}
    is_valid, msg = validate_spirometry_input(data_zero_age)
    assert is_valid is False


def test_fev1_equal_to_fvc():
    # Exactly equal is physiologically boundary, but fev1 <= fvc is allowed
    data = {"age": 45, "height_cm": 175, "sex": "Male", "fev1": 4.0, "fvc": 4.0}
    is_valid, msg = validate_spirometry_input(data)
    assert is_valid is True
    assert msg is None


def test_non_dict_input():
    is_valid, msg = validate_spirometry_input("not a dict")
    assert is_valid is False
    assert "dictionary" in msg.lower()
