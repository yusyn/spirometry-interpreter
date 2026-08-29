"""
نقطه ورود ساده برای تفسیر هیبریدی.
پیاده‌سازی اصلی در src/hybrid_interpreter.py است.
"""
from src.hybrid_interpreter import hybrid_interpret

if __name__ == "__main__":
    # اجرای تست‌های تعریف‌شده در ماژول اصلی
    import src.hybrid_interpreter as hi

    # فقط برای اینکه بلوک __main__ ماژول اصلی اجرا شود، مستقیم صدا می‌زنیم
    test_cases = [
        {"fev1": 2.1, "fvc": 3.5, "age": 55, "sex": "Male", "height": 170},
        {"fev1": 3.2, "fvc": 3.8, "age": 30, "sex": "Female", "height": 165},
        {"fev1": 1.5, "fvc": 2.8, "age": 68, "sex": "Male", "height": 175},
    ]

    print("نتایج تست تابع هیبریدی (از ریشه پروژه)\n")
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
            print(str(e))
            break
        print("-" * 60)
