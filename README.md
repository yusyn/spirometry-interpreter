# Spirometry Interpreter

وب‌اپلیکیشن تفسیر اسپیرومتری ریه با ترکیب قواعد بالینی (GOLD / ATS-ERS) و مدل‌های یادگیری ماشین (Random Forest + XGBoost) همراه با تفسیرپذیری SHAP.

## ویژگی‌های اصلی
- تشخیص الگوی بیماری: انسدادی (Obstructive)، محدودیتی (Restrictive)، مختلط (Mixed)
- تعیین شدت بیماری
- پشتیبانی از ورود دستی مقادیر FEV1، FVC و نسبت FEV1/FVC
- پشتیبانی از آپلود تصویر برگه اسپیرومتری + OCR
- تفسیر خروجی مدل با SHAP

## ساختار پروژه
```
spirometry-interpreter/
├── app/                  # وب‌اپلیکیشن Flask
├── data/
│   ├── raw/              # دیتاست خام (دانلود شده)
│   └── processed/        # داده پردازش‌شده
├── docs/                 # مستندات تحقیق و قواعد بالینی
├── models/               # مدل‌های آموزش‌دیده ذخیره‌شده
├── notebooks/            # نوت‌بوک‌های اکتشافی
├── src/                  # کد اصلی پروژه
│   ├── rule_based/       # سیستم قاعده‌محور
│   ├── ml_models/        # آموزش و پیش‌بینی مدل‌ها
│   ├── ocr/              # پردازش تصویر و OCR
│   └── utils/            # توابع کمکی
├── tests/                # تست‌ها
├── requirements.txt
└── README.md
```

## نصب و راه‌اندازی
```bash
git clone https://github.com/yusyn/spirometry-interpreter.git
cd spirometry-interpreter
python -m venv venv
# فعال‌سازی محیط مجازی
source venv/bin/activate          # لینوکس/مک
# یا
venv\\Scripts\\activate           # ویندوز

pip install -r requirements.txt
```

## وضعیت فعلی
- [x] ساخت ساختار اولیه پروژه
- [ ] تحقیق و استخراج قواعد بالینی
- [ ] دانلود و آماده‌سازی دیتاست NHANES
- [ ] پیاده‌سازی Rule-based
- [ ] آموزش مدل‌های ML
- [ ] ساخت رابط کاربری
- [ ] OCR
- [ ] تست و مستندسازی
