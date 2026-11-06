"""
تفسیر هیبریدی و دوگانهٔ اسپیرومتری (Dual-Engine Spirometry Interpreter)

نقش‌ها:
  - موتور مرجع قطعی (Deterministic Reference Engine): محاسبات ATS/ERS، Predicted، LLN، Z-score و طبقه‌بندی قطعی الگو
  - مدل جانشین ML (XGBoost Surrogate): بازنمایی رفتار داده‌آموخته NHANES
  - SHAP: توضیح ویژگی‌های مؤثر روی تصمیم مدل
  - لایه مقایسه و واگرایی (Discrepancy Analyzer): بررسی انطباق دو موتور و گزارش علل بالینی/آماری اختلاف
"""

from pathlib import Path
import sys
from typing import Optional

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import explain_prediction
from src.reference_engine.calculator import ReferenceOutput, calculate_reference_values
from src.reference_engine.pattern_classifier import PatternResult, classify_physiological_pattern
from src.reference_engine.validator import validate_spirometry_input
from src.rule_based.interpreter import interpret_spirometry

MODELS_DIR = PROJECT_ROOT / "models"

_model = None
_le_pattern = None
_le_sex = None
_le_race = None

EDUCATIONAL_DISCLAIMER = (
    "This system is strictly an educational and research tool for assessing "
    "physiological spirometry patterns and surrogate model behavior. It does not provide "
    "a medical diagnosis ('تشخیص قطعی') or clinical treatment advice. Spirometric evaluation "
    "identifies physiological patterns only. Any suspected restrictive pattern is presumptive "
    "and mandatorily requires Total Lung Capacity (TLC) measurement via plethysmography or "
    "gas dilution to confirm true restrictive defect."
)


def _load_artifacts():
    global _model, _le_pattern, _le_sex, _le_race

    if _model is not None:
        return

    required = [
        "xgboost_baseline.joblib",
        "label_encoder_pattern.joblib",
        "label_encoder_sex.joblib",
        "label_encoder_race.joblib",
    ]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "فایل‌های مدل پیدا نشدند:\n  - "
            + "\n  - ".join(missing)
            + "\n\nابتدا این دستور را اجرا کنید:\n"
            "  python src/ml_models/train_xgboost.py"
        )

    _model = joblib.load(MODELS_DIR / "xgboost_baseline.joblib")
    _le_pattern = joblib.load(MODELS_DIR / "label_encoder_pattern.joblib")
    _le_sex = joblib.load(MODELS_DIR / "label_encoder_sex.joblib")
    _le_race = joblib.load(MODELS_DIR / "label_encoder_race.joblib")


def _normalize_pattern(name: str) -> str:
    """Normalize physiological pattern labels for cross-engine comparison."""
    s = str(name).strip().lower()
    if "obstruct" in s and "restrict" in s:
        return "mixed"
    if "obstruct" in s:
        return "obstructive"
    if "restrict" in s:
        return "restrictive"
    if "normal" in s:
        return "normal"
    return s


def interpret_spirometry_dual_engine(data: dict, use_fixed_ratio: bool = False) -> dict:
    """Interpret spirometry using dual-engine architecture (Deterministic ATS/ERS + ML Surrogate).

    Parameters
    ----------
    data : dict
        Patient demographics and spirometry measurements:
        - age: float (years)
        - sex: str ('Male', 'Female', 'M', 'F')
        - height or height_cm: float (cm or m)
        - fev1: float (litres)
        - fvc: float (litres)
        - weight: Optional[float] (kg, defaults to 70.0)
        - race: Optional[str] (defaults to 'White')
        - bmi: Optional[float] (computed if omitted)
    use_fixed_ratio : bool, default=False
        If True, evaluate obstruction using GOLD fixed ratio (FEV1/FVC < 0.70).
        If False (default), use ATS/ERS Lower Limit of Normal (LLN).

    Returns
    -------
    dict
        Dual-engine output structure containing:
        - 'physiological_engine': ATS/ERS reference values, LLN, Z-scores, pattern, severity
        - 'ml_surrogate': Class predictions, class probabilities, SHAP explanations
        - 'comparison': Concordance status, discrepancy flag, clinical/statistical explanation
        - 'disclaimer': Mandatory educational non-clinical disclaimer

    Raises
    ------
    ValueError
        If inputs fail physiological or demographic validation.
    """
    _load_artifacts()

    norm_data = dict(data)
    is_valid, err_msg = validate_spirometry_input(norm_data)
    if not is_valid:
        raise ValueError(err_msg)

    age = float(norm_data["age"])
    height_cm = float(norm_data["height_cm"])
    sex = str(norm_data["sex"])
    fev1 = float(norm_data["fev1"])
    fvc = float(norm_data["fvc"])

    # 1. Deterministic Reference & Physiological Pattern Engine (ATS/ERS)
    ref_output = calculate_reference_values(
        age=age,
        height_cm=height_cm,
        sex=sex,
        fev1=fev1,
        fvc=fvc,
    )
    pattern_res = classify_physiological_pattern(
        fev1=fev1,
        fvc=fvc,
        ref=ref_output,
        use_fixed_ratio=use_fixed_ratio,
    )

    # 2. ML Surrogate Model (XGBoost + SHAP)
    sex_lookup = {str(value).casefold(): value for value in _le_sex.classes_}
    sex_value = sex_lookup.get(sex.casefold())
    if sex_value is None:
        raise ValueError(
            f"Unsupported sex value {sex!r}; expected one of {_le_sex.classes_.tolist()}."
        )
    sex_encoded = int(_le_sex.transform([sex_value])[0])

    race_input = norm_data.get("race") or "White"
    race_lookup = {str(value).casefold(): value for value in _le_race.classes_}
    race_value = race_lookup.get(str(race_input).strip().casefold())
    if race_value is None:
        raise ValueError(
            f"Unsupported race group {race_input!r}; expected one of {_le_race.classes_.tolist()}."
        )
    race_encoded = int(_le_race.transform([race_value])[0])

    weight = float(norm_data.get("weight") or 70.0)
    if "bmi" in norm_data and norm_data["bmi"] is not None:
        bmi = float(norm_data["bmi"])
    else:
        height_m = height_cm / 100.0
        bmi = weight / (height_m ** 2) if height_m > 0 else 25.0

    ratio = fev1 / fvc if fvc > 0 else 0.0

    feature_row = [
        sex_encoded,
        race_encoded,
        age,
        height_cm,
        weight,
        bmi,
        fev1,
        fvc,
        ratio,
    ]
    features = np.array([feature_row])

    ml_pred_encoded = int(_model.predict(features)[0])
    ml_pattern = str(_le_pattern.inverse_transform([ml_pred_encoded])[0])
    probabilities = _model.predict_proba(features)[0]
    prob_dict = {
        cls: round(float(p), 3)
        for cls, p in zip(_le_pattern.classes_, probabilities)
    }

    try:
        shap_top = explain_prediction(_model, feature_row, ml_pred_encoded, top_k=5)
    except Exception:
        shap_top = []

    # 3. Discrepancy & Comparison Analysis
    norm_phys = _normalize_pattern(pattern_res.pattern)
    norm_ml = _normalize_pattern(ml_pattern)
    is_concordant = (norm_phys == norm_ml)
    discrepancy_detected = not is_concordant

    if is_concordant:
        parts = [
            f"Concordant: Both physiological engine and ML surrogate agree on the {pattern_res.pattern} pattern."
        ]
        if "restrict" in norm_phys:
            parts.append(
                "Note: Spirometric restriction is presumptive and Total Lung Capacity (TLC) "
                "measurement is required to confirm a true restrictive defect."
            )
        if pattern_res.is_borderline:
            parts.append(
                "Note: Patient's FEV1/FVC ratio is borderline near the physiological threshold (LLN/0.70)."
            )
        discrepancy_explanation = " ".join(parts)
    else:
        parts = [
            f"Discrepancy detected: Physiological Engine categorized as '{pattern_res.pattern}' "
            f"while ML surrogate predicted '{ml_pattern}'."
        ]
        if pattern_res.is_borderline:
            parts.append(
                "ML boundary divergence near ratio LLN or physiological threshold."
            )
        else:
            parts.append(
                "Divergence observed between deterministic ATS/ERS reference criteria and statistical ML surrogate boundaries."
            )
        if "restrict" in norm_phys or "restrict" in norm_ml:
            parts.append(
                "Note that definitive restrictive defect requires TLC (Total Lung Capacity) measurement."
            )
        discrepancy_explanation = " ".join(parts)

    return {
        "physiological_engine": {
            "predicted": {
                "fev1": round(float(ref_output.fev1_pred), 2),
                "fvc": round(float(ref_output.fvc_pred), 2),
                "ratio": round(float(ref_output.ratio_pred), 2),
            },
            "lln": {
                "fev1": round(float(ref_output.fev1_lln), 2),
                "fvc": round(float(ref_output.fvc_lln), 2),
                "ratio": round(float(ref_output.ratio_lln), 2),
            },
            "zscore": {
                "fev1": round(float(ref_output.fev1_zscore), 2),
                "fvc": round(float(ref_output.fvc_zscore), 2),
                "ratio": round(float(ref_output.ratio_zscore), 2),
            },
            "pct_predicted": {
                "fev1": round(float(ref_output.fev1_pct_pred), 1),
                "fvc": round(float(ref_output.fvc_pct_pred), 1),
            },
            "pattern": pattern_res.pattern,
            "severity": pattern_res.severity,
            "clinical_note": pattern_res.clinical_note,
            "is_borderline": pattern_res.is_borderline,
        },
        "ml_surrogate": {
            "pattern": ml_pattern,
            "probabilities": prob_dict,
            "shap_explanation": shap_top,
        },
        "comparison": {
            "concordant": is_concordant,
            "discrepancy_detected": discrepancy_detected,
            "discrepancy_explanation": discrepancy_explanation,
        },
        "disclaimer": EDUCATIONAL_DISCLAIMER,
    }


def hybrid_interpret(
    fev1,
    fvc,
    age,
    sex,
    height,
    weight=None,
    bmi=None,
    race="Other",
):
    """
    تفسیر هیبریدی اسپیرومتری (نسخه قبلی برای حفظ سازگاری backward compatibility).

    تصمیم نهایی = خروجی مدل ML
    Rule فقط گزارش مکمل بالینی می‌دهد.
    SHAP توضیح می‌دهد کدام ویژگی‌ها روی تصمیم مدل اثر داشته‌اند.
    """
    _load_artifacts()

    rule_result = interpret_spirometry(fev1, fvc, age, sex, height)

    try:
        sex_encoded = int(_le_sex.transform([sex])[0])
    except Exception:
        sex_lookup = {str(value).casefold(): value for value in _le_sex.classes_}
        sex_value = sex_lookup.get(str(sex).strip().casefold())
        if sex_value is not None:
            sex_encoded = int(_le_sex.transform([sex_value])[0])
        else:
            sex_encoded = 0

    try:
        race_encoded = int(_le_race.transform([race])[0])
    except Exception:
        race_lookup = {str(value).casefold(): value for value in _le_race.classes_}
        race_value = race_lookup.get(str(race).strip().casefold())
        if race_value is not None:
            race_encoded = int(_le_race.transform([race_value])[0])
        else:
            race_encoded = 0

    if weight is None:
        weight = 70.0

    if bmi is None:
        bmi = weight / ((height / 100.0) ** 2) if height and height > 0 else 25.0

    ratio = float(fev1) / float(fvc) if fvc and fvc > 0 else 0.0

    feature_row = [
        sex_encoded,
        race_encoded,
        age,
        height,
        weight,
        bmi,
        fev1,
        fvc,
        ratio,
    ]
    features = np.array([feature_row])

    ml_pred_encoded = int(_model.predict(features)[0])
    ml_pattern = str(_le_pattern.inverse_transform([ml_pred_encoded])[0])

    probabilities = _model.predict_proba(features)[0]
    prob_dict = {
        cls: round(float(p), 3)
        for cls, p in zip(_le_pattern.classes_, probabilities)
    }
    ml_confidence = round(float(max(probabilities)), 3)

    # SHAP برای کلاس پیش‌بینی‌شده
    try:
        shap_top = explain_prediction(
            _model, feature_row, ml_pred_encoded, top_k=5
        )
    except Exception as exc:  # noqa: BLE001
        shap_top = []
        shap_error = str(exc)
    else:
        shap_error = None

    rule_pattern = rule_result.get("pattern")
    agree = rule_pattern == ml_pattern

    if agree:
        agreement = "موافق"
        review_flag = False
        confidence_label = "بالا" if ml_confidence >= 0.7 else "متوسط"
    else:
        agreement = "مخالف"
        review_flag = True
        confidence_label = "متوسط - نیاز به بررسی پزشک"

    return {
        "final_pattern": ml_pattern,
        "decision_source": "ml_model",
        "agreement": agreement,
        "needs_physician_review": review_flag,
        "confidence": confidence_label,
        "ml_model": {
            "pattern": ml_pattern,
            "probabilities": prob_dict,
            "top_probability": ml_confidence,
        },
        "shap_explanation": {
            "top_features": shap_top,
            "error": shap_error,
            "note": (
                "مقادیر SHAP نشان می‌دهند هر ویژگی چقدر به سمت کلاس پیش‌بینی‌شده "
                "هل داده (افزاینده) یا از آن دور کرده (کاهنده)."
            ),
        },
        "rule_based": {
            "pattern": rule_pattern,
            "ratio": rule_result.get("fev1_fvc_ratio"),
            "fev1_pct_predicted": rule_result.get("fev1_pct_predicted"),
            "fvc_pct_predicted": rule_result.get("fvc_pct_predicted"),
            "severity": rule_result.get("severity"),
            "confidence": rule_result.get("confidence"),
            "method": rule_result.get("method"),
            "note": rule_result.get("note"),
        },
        "input_values": {
            "fev1": fev1,
            "fvc": fvc,
            "ratio": round(ratio, 3),
            "age": age,
            "sex": sex,
            "height": height,
            "weight": weight,
            "bmi": round(bmi, 2),
            "race": race,
        },
    }


if __name__ == "__main__":
    test_cases = [
        {"fev1": 2.1, "fvc": 3.5, "age": 55, "sex": "Male", "height": 170, "weight": 78, "race": "White"},
        {"fev1": 3.2, "fvc": 3.8, "age": 30, "sex": "Female", "height": 165, "weight": 60, "race": "White"},
        {"fev1": 1.5, "fvc": 2.8, "age": 68, "sex": "Male", "height": 175, "weight": 80, "race": "White"},
    ]

    print("نتایج تست تابع هیبریدی + SHAP\n")
    for i, case in enumerate(test_cases, 1):
        try:
            result = hybrid_interpret(**case)
            print(f"تست {i}:")
            print(f"  نتیجه نهایی  : {result['final_pattern']}")
            print(f"  توافق با Rule: {result['agreement']}")
            print("  SHAP (ویژگی‌های مؤثر):")
            for item in result["shap_explanation"]["top_features"]:
                print(
                    f"    - {item['feature_fa']}: value={item['value']}, "
                    f"SHAP={item['shap_value']} ({item['direction']})"
                )
            if result["shap_explanation"]["error"]:
                print(f"  خطای SHAP: {result['shap_explanation']['error']}")
        except FileNotFoundError as e:
            print(f"تست {i}: خطا — {e}")
            break
        print("-" * 60)
