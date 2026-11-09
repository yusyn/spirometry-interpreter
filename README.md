# نمونهٔ پژوهشی تفسیر اسپیرومتری

نمونهٔ پژوهشی وب‌اپلیکیشن برای طبقه‌بندی الگوی اسپیرومتری، با مدل XGBoost، قواعد مکمل GOLD/ECSC و توضیح SHAP. این مخزن ابزار تشخیص یا درمان نیست.

برای وضعیت صادقانهٔ پیشرفت، محدودیت‌ها، سناریوی ارائه و برنامهٔ مرحله‌ای، [بستهٔ جلسهٔ نخست استاد راهنما](docs/first_advisor_meeting.md) را ببینید.

## ویژگی‌های فعلی
- پیش‌بینی برچسب الگو: نرمال / انسدادی / محدودیتی / مختلط
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
  -d "{\"fev1\":2.1,\"fvc\":3.5,\"age\":55.5,\"sex\":\"Male\",\"height\":170,\"weight\":78,\"race\":\"White\"}"
```

سلامت سرویس:
```bash
curl http://127.0.0.1:5000/api/health
```

### ارزیابی هیبریدی
```bash
python src/evaluate_hybrid.py
```

### ممیزی خطا و زیرگروه‌ها
```bash
python src/audit_model.py
```
این گزارش توصیفی است؛ به علت تعداد کم برخی برچسب‌ها در زیرگروه‌ها، برای ادعای عدالت الگوریتمی یا اعتبار بالینی کافی نیست.

### مقایسهٔ روش مدیریت عدم‌توازن
```bash
python src/compare_model_baselines.py
```
مدل فعلی را با baseline OneHotEncoder + وزن‌دهی کلاس روی همان split مقایسه می‌کند؛ نتایج مقدماتی و محدودیت‌ها در `docs/results.md` ثبت شده‌اند.

### مقایسهٔ پنج‌بخشی cross-validation
```bash
python src/cross_validate_models.py
```
در هر fold، encoding و مدیریت عدم‌توازن فقط روی بخش train انجام می‌شود. نتیجه‌ها هنوز مربوط به همان دیتاست و برچسب‌های مشتق‌شدهٔ NHANES هستند، نه اعتبارسنجی بالینی خارجی.

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

### محدودیت‌های مهم نمونهٔ فعلی

- نتایج `docs/results.md` از CSV پردازش‌شده بازتولید شده‌اند؛ استخراج دادهٔ خام NHANES در این مخزن بازتولید نشده است.
- مدل‌ها و Encoderهای `.joblib` در Git نیستند؛ برای نمایش زنده باید ابتدا آموزش اجرا شود. دادهٔ خام NHANES نیز در مخزن نیست و مسیر آماده‌سازی از آن قابل بازتولید کامل نیست.
- برچسب‌ها از شاخص‌های مشتق‌شدهٔ NHANES ساخته شده‌اند و FEV1/FVC نیز در ویژگی‌های مدل حضور دارد؛ بنابراین امتیازهای فعلی، بازسازی برچسب دیتاست‌اند و به‌تنهایی اعتبار تشخیصی بالینی را اثبات نمی‌کنند.
- ماژول GLI در وضعیت حاضر فقط قرارداد/اعتبارسنجی دارد و هنوز محاسبهٔ عددی GLI را انجام نمی‌دهد. هیچ مقدار مرجع جایگزین یا ساختگی تولید نمی‌شود.
- ورودی مدل باید گروه جمعیتی را دقیقاً با یکی از دسته‌های NHANES انتخاب کند؛ ورودی ناشناخته دیگر بی‌صدا به کد دستهٔ دیگری تبدیل نمی‌شود. این فیلد ML است و نباید با انتخاب ethnicity برای GLI اشتباه گرفته شود.
- وزن برای ویژگی‌های مدل لازم است و در فرم/API اجباری است؛ دیگر به‌جای مقدار واردنشده وزن فرضی درج نمی‌شود.
- محدودیت اسپیرومتری به‌تنهایی تأیید نمی‌شود؛ تأیید فیزیولوژیک به TLC نیاز دارد. OCR نیز هنوز پیاده‌سازی نشده است.

## نکات
- فایل‌های `.joblib` و دیتاست خام در گیت نیستند.
- جزئیات نتایج در `docs/results.md` آمده است.
