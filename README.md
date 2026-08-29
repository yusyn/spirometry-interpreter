# Spirometry Interpreter

وب‌اپلیکیشن تفسیر اسپیرومتری ریه با ترکیب قواعد بالینی (GOLD / ATS-ERS) و مدل‌های یادگیری ماشین (Random Forest + XGBoost) همراه با تفسیرپذیری SHAP.

## ویژگی‌های اصلی (هدف نهایی)
- تشخیص الگوی بیماری: انسدادی (Obstructive)، محدودیتی (Restrictive)، مختلط (Mixed)، نرمال
- تعیین شدت بیماری
- ورود دستی مقادیر FEV1، FVC و ...
- آپلود تصویر برگه اسپیرومتری + OCR
- تفسیر خروجی مدل با SHAP
- تصمیم‌گیری هیبریدی (Rule-based + ML)

## ساختار پروژه
```
spirometry-interpreter/
├── app/                      # وب‌اپلیکیشن Flask (هنوز خالی)
├── data/
│   ├── raw/                  # دیتاست خام (در گیت نیست)
│   └── processed/            # داده تمیز آماده‌شده
├── docs/                     # مستندات و قواعد بالینی
├── models/                   # مدل‌های آموزش‌دیده (.joblib در گیت نیستند)
├── notebooks/
├── src/
│   ├── rule_based/           # سیستم قاعده‌محور
│   ├── ml_models/            # آموزش مدل‌ها و SHAP
│   ├── hybrid_interpreter.py # هسته هیبریدی
│   ├── ocr/                  # (آینده)
│   └── utils/
├── prepare_data.py
├── hybrid_interpreter.py     # نقطه ورود سریع از ریشه
├── requirements.txt
└── README.md
```

## نصب
```bash
git clone https://github.com/yusyn/spirometry-interpreter.git
cd spirometry-interpreter
python -m venv venv
# ویندوز:
venv\Scripts\activate
# لینوکس/مک:
source venv/bin/activate

pip install -r requirements.txt
```

## اجرای مرحله‌به‌مرحله (وضعیت فعلی)

### ۱. آماده‌سازی داده (یک‌بار)
دیتاست خام را در `data/raw/` قرار دهید، سپس:
```bash
python prepare_data.py
```

### ۲. آموزش مدل XGBoost (یک‌بار یا بعد از تغییر داده)
```bash
python src/ml_models/train_xgboost.py
```
این دستور مدل و Encoderها را در `models/` ذخیره می‌کند.

### ۳. تست سیستم قاعده‌محور
```bash
python src/rule_based/interpreter.py
```

### ۴. تست تفسیر هیبریدی
```bash
python src/hybrid_interpreter.py
# یا
python hybrid_interpreter.py
```

## وضعیت پیشرفت
- [x] ساختار اولیه پروژه
- [x] دانلود و آماده‌سازی دیتاست NHANES (نسخه تمیز)
- [x] سیستم Rule-based نسخه ۲ (GOLD + آستانه ساده FVC)
- [x] آموزش Random Forest و XGBoost (XGBoost بهتر: Macro F1 ≈ 0.83)
- [x] SHAP (نمودار اهمیت ویژگی‌ها)
- [x] هسته هیبریدی اولیه (Rule + XGBoost)
- [ ] بهبود Rule-based با LLN / معادلات GLI و شدت بیماری
- [ ] بهبود منطق هیبریدی و ارزیابی روی داده تست
- [ ] رابط کاربری Flask
- [ ] OCR برگه اسپیرومتری
- [ ] تست و مستندسازی نهایی

## نکات
- فایل‌های `.joblib` مدل در گیت commit نمی‌شوند؛ باید محلی آموزش داده شوند.
- دیتاست خام NHANES هم در گیت نیست.
