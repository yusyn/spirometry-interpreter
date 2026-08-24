import joblib
import pandas as pd
from src.rule_based.interpreter import interpret_spirometry

# بارگذاری مدل و encoder
model = joblib.load('models/xgboost_baseline.joblib')
le_pattern = joblib.load('models/label_encoder_pattern.joblib')
le_sex = joblib.load('models/label_encoder_sex.joblib')
le_race = joblib.load('models/label_encoder_race.joblib')

def hybrid_interpret(fev1, fvc, age, sex, height, weight=None, bmi=None, race="Other"):
    """
    تفسیر هیبریدی اسپیرومتری
    ترکیب سیستم قاعده‌محور + مدل XGBoost
    """
    
    # ----------------------------------
    # ۱. نتیجه سیستم قاعده‌محور
    # ----------------------------------
    rule_result = interpret_spirometry(fev1, fvc, age, sex, height)
    
    # ----------------------------------
    # ۲. آماده‌سازی ورودی برای مدل ML
    # ----------------------------------
    try:
        sex_encoded = le_sex.transform([sex])[0]
    except:
        sex_encoded = 0  # مقدار پیش‌فرض
        
    try:
        race_encoded = le_race.transform([race])[0]
    except:
        race_encoded = 0  # مقدار پیش‌فرض
    
    if weight is None:
        weight = 70  # مقدار پیش‌فرض تقریبی
    if bmi is None:
        bmi = weight / ((height/100) ** 2) if height else 25
    
    ratio = fev1 / fvc if fvc > 0 else 0
    
    # ترتیب ویژگی‌ها باید دقیقاً مثل زمان آموزش باشد
    features = [[
        sex_encoded,
        race_encoded,
        age,
        height,
        weight,
        bmi,
        fev1,
        fvc,
        ratio
    ]]
    
    # ----------------------------------
    # ۳. پیش‌بینی مدل XGBoost
    # ----------------------------------
    ml_pred_encoded = model.predict(features)[0]
    ml_pattern = le_pattern.inverse_transform([ml_pred_encoded])[0]
    
    # احتمال هر کلاس
    probabilities = model.predict_proba(features)[0]
    prob_dict = dict(zip(le_pattern.classes_, probabilities.round(3)))
    
    # ----------------------------------
    # ۴. ترکیب نتایج (منطق هیبریدی)
    # ----------------------------------
    rule_pattern = rule_result["pattern"]
    
    if rule_pattern == ml_pattern:
        final_pattern = rule_pattern
        agreement = "موافق"
        confidence = "بالا"
    else:
        # در صورت اختلاف، فعلاً نتیجه مدل را ترجیح می‌دهیم
        # (چون مدل از داده واقعی یاد گرفته)
        final_pattern = ml_pattern
        agreement = "مخالف"
        confidence = "متوسط - نیاز به بررسی پزشک"
    
    # ----------------------------------
    # ۵. خروجی نهایی
    # ----------------------------------
    return {
        "final_pattern": final_pattern,
        "agreement": agreement,
        "confidence": confidence,
        "rule_based": {
            "pattern": rule_pattern,
            "ratio": rule_result["fev1_fvc_ratio"],
            "method": rule_result["method"]
        },
        "ml_model": {
            "pattern": ml_pattern,
            "probabilities": prob_dict
        },
        "input_values": {
            "fev1": fev1,
            "fvc": fvc,
            "ratio": round(ratio, 3),
            "age": age,
            "sex": sex,
            "height": height
        }
    }


# -------------------------
# تست تابع هیبریدی
# -------------------------
if __name__ == "__main__":
    test_cases = [
        {"fev1": 2.1, "fvc": 3.5, "age": 55, "sex": "Male", "height": 170},
        {"fev1": 3.2, "fvc": 3.8, "age": 30, "sex": "Female", "height": 165},
        {"fev1": 1.5, "fvc": 2.8, "age": 68, "sex": "Male", "height": 175},
    ]
    
    print("نتایج تست تابع هیبریدی\n")
    for i, case in enumerate(test_cases, 1):
        result = hybrid_interpret(**case)
        print(f"تست {i}:")
        print(f"  ورودی     : FEV1={case['fev1']}, FVC={case['fvc']}, Age={case['age']}")
        print(f"  نتیجه نهایی: {result['final_pattern']} (اطمینان: {result['confidence']})")
        print(f"  توافق     : {result['agreement']}")
        print(f"  قاعده‌محور : {result['rule_based']['pattern']}")
        print(f"  مدل ML    : {result['ml_model']['pattern']}")
        print(f"  احتمالات  : {result['ml_model']['probabilities']}")
        print("-" * 60)