# قواعد بالینی تفسیر اسپیرومتری

## نقش در این پروژه

سیستم **قاعده‌محور لایه مکمل** است، نه تصمیم‌گیرنده نهایی.

- تصمیم نهایی الگوی بیماری: مدل یادگیری ماشین (XGBoost)
- Rule: گزارش معیار GOLD + مقادیر تقریبی %predicted + شدت تقریبی
- در صورت اختلاف مدل و Rule: پرچم «نیاز به بررسی پزشک»

## منابع

- GOLD (Global Initiative for Chronic Obstructive Lung Disease)
- ATS/ERS interpretive strategies
- معادلات مرجع تقریبی ECSC / Quanjer 1993 (برای %predicted)

> نسخه کامل‌تر آینده می‌تواند از معادلات GLI-2012 استفاده کند.

## معیار انسداد (Obstructive)

- **GOLD (پیاده‌سازی فعلی):** FEV1/FVC < 0.70
- ATS/ERS ترجیح می‌دهد LLN به‌جای نسبت ثابت (هنوز پیاده نشده)

## سوءظن Restrictive

- نسبت FEV1/FVC نرمال یا بالا
- و FVC %predicted < 80%
- تأیید قطعی نیاز به اندازه‌گیری TLC دارد

## Mixed

- نسبت < 0.70 و FVC %predicted < 80% (تقریبی)

## شدت انسداد (GOLD)

بر اساس FEV1 %predicted:

| شدت | FEV1 %predicted |
|-----|-----------------|
| Mild (GOLD 1) | ≥ 80% |
| Moderate (GOLD 2) | 50–79% |
| Severe (GOLD 3) | 30–49% |
| Very Severe (GOLD 4) | < 30% |

## معادلات ECSC/Quanjer 1993 (پیاده‌سازی فعلی)

قد (H) به **متر**، سن (A) به سال:

**مرد:**
- FEV1 = 4.301×H − 0.0290×A − 2.492
- FVC  = 5.761×H − 0.0260×A − 4.343

**زن:**
- FEV1 = 3.953×H − 0.0251×A − 2.604
- FVC  = 4.433×H − 0.0260×A − 2.892

%predicted = (measured / predicted) × 100

## محدودیت‌ها

- ECSC برای همه نژادها و سنین ایده‌آل نیست
- LLN دقیق پیاده نشده
- Restrictive بدون TLC قطعی نیست
- Rule جایگزین مدل نیست
