import pandas as pd
import os

# ۱. خواندن دیتاست
df = pd.read_csv('data/raw/NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv')

print("تعداد کل نمونه‌ها:", len(df))

# ۲. ساخت ستون هدف (pattern)
def create_pattern(row):
    obstructive = row['Obstruction_5th_GAMLSS'] == 'yes'
    restrictive = row['Restrictive_Spirometry_Pattern_5th_GAMLSS'] == 'yes'
    mixed = row['Mixed_5th_GAMLSS'] == 'yes'
    
    if mixed:
        return 'Mixed'
    elif obstructive:
        return 'Obstructive'
    elif restrictive:
        return 'Restrictive'
    else:
        return 'Normal'

df['pattern'] = df.apply(create_pattern, axis=1)

# ۳. نمایش توزیع الگوی جدید
print("\nتوزیع الگوی نهایی:")
print(df['pattern'].value_counts())

# ۴. انتخاب ستون‌های مهم
columns_to_keep = [
    'SEQN', 'Sex', 'Race', 'Age', 'Height', 'Weight', 'BMI',
    'Baseline_FEV1_L', 'Baseline_FVC_L', 'Baseline_FEV1_FVC_Ratio',
    'pattern'
]

df_clean = df[columns_to_keep].copy()

# ۵. ذخیره داده پردازش‌شده
os.makedirs('data/processed', exist_ok=True)
df_clean.to_csv('data/processed/spirometry_clean.csv', index=False)

print("\nداده تمیز با موفقیت در مسیر زیر ذخیره شد:")
print("data/processed/spirometry_clean.csv")
print("تعداد ستون‌های نهایی:", len(df_clean.columns))