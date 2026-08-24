def interpret_spirometry(fev1, fvc, age=None, sex=None, height=None):
    """
    تفسیر اسپیرومتری - نسخه ۲ (بهبود یافته)
    
    پارامترها:
        fev1 (float): مقدار FEV1 (لیتر)
        fvc (float): مقدار FVC (لیتر)
        age, sex, height: فعلاً اختیاری (برای نسخه‌های بعدی)
    
    خروجی:
        دیکشنری شامل الگوی بیماری و جزئیات
    """
    
    # بررسی ورودی‌ها
    if fev1 is None or fvc is None or fvc <= 0 or fev1 <= 0:
        return {
            "pattern": "Error",
            "message": "مقادیر FEV1 یا FVC نامعتبر است",
            "fev1_fvc_ratio": None,
            "severity": None,
            "method": None
        }
    
    ratio = fev1 / fvc
    ratio = round(ratio, 3)
    
    # -------------------------
    # قواعد تشخیص الگو (نسخه ۲)
    # -------------------------
    
    is_obstructive = ratio < 0.70
    
    # برای تشخیص Restrictive و Mixed فعلاً از یک آستانه ساده استفاده می‌کنیم.
    # این آستانه تقریبی است و بعداً با LLN و GLI جایگزین می‌شود.
    # فرض ساده: اگر FVC کمتر از 3 لیتر باشد، احتمال محدودیت وجود دارد
    # (این فقط برای نسخه آزمایشی است)
    is_low_fvc = fvc < 3.0
    
    if is_obstructive and is_low_fvc:
        pattern = "Mixed"
        confidence = "متوسط (تقریبی)"
    elif is_obstructive:
        pattern = "Obstructive"
        confidence = "بالا (بر اساس GOLD)"
    elif is_low_fvc:
        pattern = "Restrictive"
        confidence = "پایین (تقریبی - نیاز به بررسی بیشتر)"
    else:
        pattern = "Normal"
        confidence = "بالا"
    
    # شدت فعلاً فقط برای انسدادی معنی‌دار است و چون %predicted نداریم، None می‌گذاریم
    severity = None
    if pattern in ["Obstructive", "Mixed"]:
        severity = "نامشخص (نیاز به FEV1 % predicted)"
    
    return {
        "pattern": pattern,
        "fev1_fvc_ratio": ratio,
        "fev1": fev1,
        "fvc": fvc,
        "severity": severity,
        "confidence": confidence,
        "method": "GOLD fixed ratio + simple FVC threshold (نسخه ۲)",
        "note": "تشخیص Restrictive و Mixed تقریبی است. نسخه کامل‌تر با LLN و معادلات GLI بعداً اضافه می‌شود."
    }


# -------------------------
# بخش تست
# -------------------------
if __name__ == "__main__":
    test_cases = [
        {"fev1": 2.1, "fvc": 3.5},   # نسبت 0.60 → Obstructive
        {"fev1": 3.2, "fvc": 3.8},   # نسبت 0.84 → Normal
        {"fev1": 1.5, "fvc": 2.8},   # نسبت 0.54 + FVC پایین → Mixed
        {"fev1": 2.8, "fvc": 2.6},   # نسبت بالا + FVC پایین → Restrictive
        {"fev1": 1.8, "fvc": 4.0},   # نسبت پایین + FVC خوب → Obstructive
    ]
    
    print("نتایج تست سیستم قاعده‌محور - نسخه ۲\n")
    for i, case in enumerate(test_cases, 1):
        result = interpret_spirometry(**case)
        print(f"تست {i}: FEV1={case['fev1']}, FVC={case['fvc']}")
        print(f"  الگو      : {result['pattern']}")
        print(f"  نسبت      : {result['fev1_fvc_ratio']}")
        print(f"  اطمینان   : {result['confidence']}")
        print(f"  شدت       : {result['severity']}")
        print("-" * 55)