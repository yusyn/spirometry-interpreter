"""
تفسیر هیبریدی اسپیرومتری

نقش‌ها:
  - مدل XGBoost: تصمیم‌گیرنده نهایی الگوی بیماری
  - Rule-based: لایه بالینی مکمل (GOLD + %predicted تقریبی ECSC)
  - SHAP: توضیح ویژگی‌های مؤثر روی تصمیم مدل

در صورت اختلاف مدل و Rule، نتیجه نهایی همان مدل است و
پرچم «نیاز به بررسی پزشک» فعال می‌شود.
"""
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import explain_prediction
from src.rule_based.interpreter import interpret_spirometry

MODELS_DIR = PROJECT_ROOT / "models"

_model = None
_le_pattern = None
_le_sex = None
_le_race = None


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
    تفسیر هیبریدی اسپیرومتری.

    تصمیم نهایی = خروجی مدل ML
    Rule فقط گزارش مکمل بالینی می‌دهد.
    SHAP توضیح می‌دهد کدام ویژگی‌ها روی تصمیم مدل اثر داشته‌اند.
    """
    _load_artifacts()

    rule_result = interpret_spirometry(fev1, fvc, age, sex, height)

    try:
        sex_encoded = int(_le_sex.transform([sex])[0])
    except Exception:
        sex_encoded = 0

    try:
        race_encoded = int(_le_race.transform([race])[0])
    except Exception:
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
        {"fev1": 2.1, "fvc": 3.5, "age": 55, "sex": "Male", "height": 170},
        {"fev1": 3.2, "fvc": 3.8, "age": 30, "sex": "Female", "height": 165},
        {"fev1": 1.5, "fvc": 2.8, "age": 68, "sex": "Male", "height": 175},
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
