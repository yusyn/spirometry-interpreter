# Spirometry Interpreter

وب‌اپلیکیشن تفسیر اسپیرومتری ریه با ترکیب قواعد بالینی (GOLD + ECSC) و مدل XGBoost همراه با تفسیرپذیری SHAP.

## ویژگی‌های فعلی
- تشخیص الگو: نرمال / انسدادی / محدودیتی / مختلط
- تصمیم نهایی با مدل ML
- لایه مکمل Rule (GOLD ratio + %predicted تقریبی + شدت)
- توضیح SHAP برای هر پیش‌بینی
- وب‌اپ Flask + API JSON
- هشدار اختلاف مدل و Rule

## ساختار پروژه
```
spirometry-interpreter/
├── app/                      # وب‌اپ Flask
├── data/
│   ├── raw/                  # دیتاست خام (در گیت نیست)
│   └── processed/            # داده تمیز
├── docs/
│   ├── guidelines.md
│   └── results.md            # خلاصه نتایج برای گزارش
├── models/                   # مدل‌های .joblib (محلی)
├── src/
│   ├── rule_based/
│   ├── ml_models/
│   ├── hybrid_interpreter.py
│   ├── explain.py            # SHAP per-prediction
│   ├── evaluate_hybrid.py
│   └── utils/
├── prepare_data.py
├── run.py
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

## اجرا

### آموزش مدل (یک‌بار)
```bash
python src/ml_models/train_xgboost.py
```

### وب‌اپ
```bash
python run.py
```
سپس: http://127.0.0.1:5000

### API
```bash
curl -X POST http://127.0.0.1:5000/api/interpret ^
  -H "Content-Type: application/json" ^
  -d "{\"fev1\":2.1,\"fvc\":3.5,\"age\":55,\"sex\":\"Male\",\"height\":170}"
```

سلامت سرویس:
```bash
curl http://127.0.0.1:5000/api/health
```

### ارزیابی هیبریدی
```bash
python src/evaluate_hybrid.py
```

## وضعیت پیشرفت
- [x] داده NHANES و برچسب‌زنی
- [x] Random Forest و XGBoost (انتخاب نهایی: XGBoost، Macro F1 ≈ 0.83)
- [x] Rule-based v3 (GOLD + ECSC %predicted + شدت)
- [x] هیبریدی با نقش مکمل برای Rule
- [x] SHAP کلی + SHAP برای هر پیش‌بینی در UI
- [x] وب‌اپ Flask
- [x] API JSON
- [x] مستند نتایج (`docs/results.md`)
- [ ] OCR برگه اسپیرومتری
- [ ] تست‌های واحد گسترده‌تر و گزارش نهایی کامل

## نکات
- فایل‌های `.joblib` و دیتاست خام در گیت نیستند.
- جزئیات نتایج در `docs/results.md` آمده است.
