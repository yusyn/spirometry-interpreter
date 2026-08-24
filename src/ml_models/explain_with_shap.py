import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("۱. بارگذاری مدل و داده...")
model = joblib.load('models/xgboost_baseline.joblib')
le_pattern = joblib.load('models/label_encoder_pattern.joblib')

df = pd.read_csv('data/processed/spirometry_clean.csv')

# آماده‌سازی داده (مثل قبل)
le_sex = LabelEncoder()
le_race = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Race'] = le_race.fit_transform(df['Race'])

features = ['Sex', 'Race', 'Age', 'Height', 'Weight', 'BMI',
            'Baseline_FEV1_L', 'Baseline_FVC_L', 'Baseline_FEV1_FVC_Ratio']

X = df[features]
y = le_pattern.transform(df['pattern'])

# استفاده از بخشی از داده برای سرعت بیشتر
X_sample = X.sample(n=500, random_state=42)

print("۲. ساخت SHAP Explainer...")
# برای XGBoost از TreeExplainer استفاده می‌کنیم (سریع و دقیق)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("۳. رسم نمودار Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=features, class_names=le_pattern.classes_, show=False)
plt.tight_layout()
plt.savefig('models/shap_summary_plot.png', dpi=150, bbox_inches='tight')
print("نمودار ذخیره شد: models/shap_summary_plot.png")

print("۴. رسم نمودار Bar Plot (اهمیت کلی ویژگی‌ها)...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type="bar", class_names=le_pattern.classes_, show=False)
plt.tight_layout()
plt.savefig('models/shap_importance_plot.png', dpi=150, bbox_inches='tight')
print("نمودار اهمیت ویژگی‌ها ذخیره شد: models/shap_importance_plot.png")

print("\nتوضیح نمودارها:")
print("- در Summary Plot، هر نقطه یک بیمار است.")
print("- رنگ قرمز = مقدار بالای آن ویژگی، آبی = مقدار پایین")
print("- محور افقی نشان می‌دهد آن ویژگی چقدر به سمت یک کلاس خاص هل داده.")
print("- در Bar Plot، میانگین اهمیت هر ویژگی را می‌بینی.")