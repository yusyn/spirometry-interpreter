import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

print("۱. خواندن داده تمیز...")
df = pd.read_csv('data/processed/spirometry_clean.csv')

print("۲. تبدیل ستون‌های متنی به عدد...")
le_sex = LabelEncoder()
le_race = LabelEncoder()
le_pattern = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Race'] = le_race.fit_transform(df['Race'])
df['pattern_encoded'] = le_pattern.fit_transform(df['pattern'])

print("کلاس‌ها:", list(le_pattern.classes_))

print("۳. جدا کردن ویژگی‌ها (X) و برچسب (y)...")
features = ['Sex', 'Race', 'Age', 'Height', 'Weight', 'BMI',
            'Baseline_FEV1_L', 'Baseline_FVC_L', 'Baseline_FEV1_FVC_Ratio']

X = df[features]
y = df['pattern_encoded']

print("۴. تقسیم داده به Train و Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("۵. اعمال SMOTE برای متعادل کردن کلاس‌ها...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("توزیع بعد از SMOTE:")
print(pd.Series(y_train_res).value_counts())

print("۶. آموزش مدل Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

print("۷. پیش‌بینی و ارزیابی...")
y_pred = model.predict(X_test)

print("\nگزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred, target_names=le_pattern.classes_))

macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nMacro F1-Score: {macro_f1:.4f}")

print("۸. ذخیره مدل و Encoderها...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/random_forest_baseline.joblib')
joblib.dump(le_pattern, 'models/label_encoder_pattern.joblib')
joblib.dump(le_sex, 'models/label_encoder_sex.joblib')
joblib.dump(le_race, 'models/label_encoder_race.joblib')

print("\nمدل با موفقیت ذخیره شد در: models/random_forest_baseline.joblib")