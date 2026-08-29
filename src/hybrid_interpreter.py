"""
تفسیر هیبریدی اسپیرومتری
ترکیب سیستم قاعده‌محور (GOLD) + مدل XGBoost
"""
import sys
from pathlib import Path

import joblib
import numpy as np

# اضافه کردن ریشه پروژه به sys.path تا importها از هر جا کار کنند
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rule_based.interpreter import interpret_spirometry

MODELS_DIR = PROJECT_ROOT / "models"

# بارگذاری تنبل (lazy) مدل‌ها
_model = None
_le_pattern = None
_le_sex = None
_le_race = None


def _load_artifacts():
    """بارگذاری مدل و encoderها فقط یک‌بار"""
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

    پارامترها:
        fev1, fvc: مقادیر به لیتر
        age: سن به سال
        sex: "Male" یا "Female"
        height: قد به سانتی‌متر
        weight: وزن به کیلوگرم (اختیاری)
        bmi: در صورت نبودن از قد و وزن محاسبه می‌شود
        race: برچسب نژاد مطابق دیتاست (اختیاری)

    خروجی:
        dict شامل نتیجه نهایی، توافق دو روش، و جزئیات هر کدام
    """
    _load_artifacts()

    # ۱. سیستم قاعده‌محور
    rule_result = interpret_spirometry(fev1, fvc, age, sex, height)

    # ۲. آماده‌سازی ورودی مدل
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

    features = np.array(
        [
            [
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
        ]
    )

    # ۳. پیش‌بینی مدل
    ml_pred_encoded = int(_model.predict(features)[0])
    ml_pattern = str(_le_pattern.inverse_transform([ml_pred_encoded])[0])

    probabilities = _model.predict_proba(features)[0]
    prob_dict = {
        cls: round(float(p), 3)
        for cls, p in zip(_le_pattern.classes_, probabilities)
    }

    # ۴. ترکیب نتایج
    rule_pattern = rule_result.get("pattern")

    if rule_pattern == ml_pattern:
        final_pattern = rule_pattern
        agreement = "موافق"
        confidence = "بالا"
    else:
        # در نسخه فعلی در صورت اختلاف، مدل را ترجیح می‌دهیم
        # (چون از داده واقعی یاد گرفته؛ بعداً می‌توان وزن‌دهی هوشمندتر کرد)
        final_pattern = ml_pattern
        agreement = "مخالف"
        confidence = "متوسط - نیاز به بررسی پزشک"

    return {
        "final_pattern": final_pattern,
        "agreement": agreement,
        "confidence": confidence,
        "rule_based": {
            "pattern": rule_pattern,
            "ratio": rule_result.get("fev1_fvc_ratio"),
            "method": rule_result.get("method"),
            "confidence": rule_result.get("confidence"),
        },
        "ml_model": {
            "pattern": ml_pattern,
            "probabilities": prob_dict,
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

    print("نتایج تست تابع هیبریدی\n")
    for i, case in enumerate(test_cases, 1):
        try:
            result = hybrid_interpret(**case)
            print(f"تست {i}:")
            print(
                f"  ورودی      : FEV1={case['fev1']}, FVC={case['fvc']}, Age={case['age']}"
            )
            print(
                f"  نتیجه نهایی : {result['final_pattern']} (اطمینان: {result['confidence']})"
            )
            print(f"  توافق       : {result['agreement']}")
            print(f"  قاعده‌محور  : {result['rule_based']['pattern']}")
            print(f"  مدل ML     : {result['ml_model']['pattern']}")
            print(f"  احتمالات   : {result['ml_model']['probabilities']}")
        except FileNotFoundError as e:
            print(f"تست {i}: خطا — {e}")
            break
        print("-" * 60)
