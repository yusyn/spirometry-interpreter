"""
سیستم قاعده‌محور تفسیر اسپیرومتری

نقش در پروژه:
  لایه بالینی مکمل (نه تصمیم‌گیرنده نهایی).
  تصمیم نهایی الگوی بیماری با مدل ML است.
  Rule معیار GOLD و مقادیر تقریبی %predicted را گزارش می‌کند.

معادلات predicted:
  فرمول‌های ECSC / Quanjer 1993 (تقریبی و رایج در آموزش).
  برای پروژه لیسانس کافی است؛ نسخه کامل‌تر می‌تواند GLI-2012 باشد.
"""


def _predicted_ecsc(age, sex, height_cm):
    """
    محاسبه FEV1 و FVC پیش‌بینی‌شده با معادلات ECSC/Quanjer 1993.

    height باید به سانتی‌متر باشد؛ داخل تابع به متر تبدیل می‌شود.
    خروجی: (fev1_pred, fvc_pred) به لیتر یا (None, None) اگر ورودی ناقص باشد.
    """
    if age is None or sex is None or height_cm is None:
        return None, None
    if age <= 0 or height_cm <= 0:
        return None, None

    h = height_cm / 100.0  # متر
    sex_norm = str(sex).strip().lower()

    if sex_norm in ("male", "m", "مرد", "آقا"):
        fev1_pred = 4.301 * h - 0.0290 * age - 2.492
        fvc_pred = 5.761 * h - 0.0260 * age - 4.343
    elif sex_norm in ("female", "f", "زن", "خانم"):
        fev1_pred = 3.953 * h - 0.0251 * age - 2.604
        fvc_pred = 4.433 * h - 0.0260 * age - 2.892
    else:
        return None, None

    # مقادیر منفی یا صفر از نظر فیزیولوژیک بی‌معنی‌اند
    if fev1_pred <= 0 or fvc_pred <= 0:
        return None, None

    return round(fev1_pred, 3), round(fvc_pred, 3)


def _percent_predicted(measured, predicted):
    if measured is None or predicted is None or predicted <= 0:
        return None
    return round(100.0 * float(measured) / float(predicted), 1)


def _severity_gold(fev1_pct):
    """
    شدت انسداد بر اساس GOLD (بر حسب FEV1 % predicted).
    فقط وقتی الگوی انسدادی مطرح باشد معنا دارد.
    """
    if fev1_pct is None:
        return None
    if fev1_pct >= 80:
        return "Mild (GOLD 1)"
    if fev1_pct >= 50:
        return "Moderate (GOLD 2)"
    if fev1_pct >= 30:
        return "Severe (GOLD 3)"
    return "Very Severe (GOLD 4)"


def interpret_spirometry(fev1, fvc, age=None, sex=None, height=None):
    """
    تفسیر قاعده‌محور اسپیرومتری (نسخه ۳).

    خروجی برای استفاده به‌عنوان لایه مکمل در کنار مدل ML طراحی شده است.
    """
    if fev1 is None or fvc is None or fvc <= 0 or fev1 <= 0:
        return {
            "pattern": "Error",
            "message": "مقادیر FEV1 یا FVC نامعتبر است",
            "fev1_fvc_ratio": None,
            "fev1_pct_predicted": None,
            "fvc_pct_predicted": None,
            "severity": None,
            "method": None,
            "confidence": None,
            "note": None,
        }

    ratio = round(float(fev1) / float(fvc), 3)

    fev1_pred, fvc_pred = _predicted_ecsc(age, sex, height)
    fev1_pct = _percent_predicted(fev1, fev1_pred)
    fvc_pct = _percent_predicted(fvc, fvc_pred)

    # ----- قواعد -----
    # انسداد: معیار ثابت GOLD
    is_obstructive = ratio < 0.70

    # سوءظن Restrictive: نسبت نرمال/بالا + FVC% پایین
    # آستانه ۸۰٪ رایج و آموزشی است (نه جایگزین TLC)
    is_restrictive_suspect = (
        (not is_obstructive)
        and (fvc_pct is not None)
        and (fvc_pct < 80)
    )

    # Mixed تقریبی: انسداد + FVC% پایین
    is_mixed_suspect = is_obstructive and (fvc_pct is not None) and (fvc_pct < 80)

    if is_mixed_suspect:
        pattern = "Mixed"
        confidence = "متوسط (تقریبی؛ نیاز به بررسی حجم‌های ریوی)"
    elif is_obstructive:
        pattern = "Obstructive"
        confidence = "بالا (معیار GOLD: FEV1/FVC < 0.70)"
    elif is_restrictive_suspect:
        pattern = "Restrictive"
        confidence = "متوسط (سوءظن بر اساس FVC%predicted؛ تأیید با TLC)"
    else:
        pattern = "Normal"
        if fev1_pct is None:
            confidence = "متوسط (بدون %predicted؛ فقط نسبت بررسی شد)"
        else:
            confidence = "بالا (نسبت و %predicted در محدوده قابل قبول)"

    severity = None
    if pattern in ("Obstructive", "Mixed"):
        severity = _severity_gold(fev1_pct)

    note_parts = [
        "Rule لایه مکمل است؛ تصمیم نهایی با مدل ML.",
        "معیار انسداد: GOLD fixed ratio (0.70).",
        "%predicted با معادلات تقریبی ECSC/Quanjer 1993.",
    ]
    if fev1_pred is None:
        note_parts.append("برای %predicted به age, sex, height نیاز است.")

    return {
        "pattern": pattern,
        "fev1_fvc_ratio": ratio,
        "fev1": fev1,
        "fvc": fvc,
        "fev1_predicted": fev1_pred,
        "fvc_predicted": fvc_pred,
        "fev1_pct_predicted": fev1_pct,
        "fvc_pct_predicted": fvc_pct,
        "severity": severity,
        "confidence": confidence,
        "method": "GOLD ratio + ECSC %predicted (نسخه ۳)",
        "note": " ".join(note_parts),
    }


if __name__ == "__main__":
    test_cases = [
        {"fev1": 2.1, "fvc": 3.5, "age": 55, "sex": "Male", "height": 170},
        {"fev1": 3.2, "fvc": 3.8, "age": 30, "sex": "Female", "height": 165},
        {"fev1": 1.5, "fvc": 2.8, "age": 68, "sex": "Male", "height": 175},
        {"fev1": 2.8, "fvc": 2.6, "age": 45, "sex": "Female", "height": 160},
        {"fev1": 1.8, "fvc": 4.0, "age": 60, "sex": "Male", "height": 178},
        # بدون دموگرافیک → فقط ratio
        {"fev1": 2.1, "fvc": 3.5},
    ]

    print("نتایج تست سیستم قاعده‌محور - نسخه ۳ (ECSC + GOLD)\n")
    for i, case in enumerate(test_cases, 1):
        result = interpret_spirometry(**case)
        print(f"تست {i}: {case}")
        print(f"  الگو        : {result['pattern']}")
        print(f"  نسبت        : {result['fev1_fvc_ratio']}")
        print(f"  FEV1%pred   : {result['fev1_pct_predicted']}")
        print(f"  FVC%pred    : {result['fvc_pct_predicted']}")
        print(f"  شدت         : {result['severity']}")
        print(f"  اطمینان     : {result['confidence']}")
        print("-" * 60)
