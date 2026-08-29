"""
ارزیابی سیستماتیک سیستم هیبریدی روی داده تست

مقایسه می‌کند:
- فقط Rule-based
- فقط XGBoost
- هیبریدی فعلی (در صورت توافق → همان نتیجه؛ در صورت اختلاف → مدل)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rule_based.interpreter import interpret_spirometry

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spirometry_clean.csv"
FEATURES = [
    "Sex",
    "Race",
    "Age",
    "Height",
    "Weight",
    "BMI",
    "Baseline_FEV1_L",
    "Baseline_FVC_L",
    "Baseline_FEV1_FVC_Ratio",
]


def load_artifacts():
    required = [
        "xgboost_baseline.joblib",
        "label_encoder_pattern.joblib",
        "label_encoder_sex.joblib",
        "label_encoder_race.joblib",
    ]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "مدل‌ها پیدا نشدند. اول اجرا کنید:\n  python src/ml_models/train_xgboost.py\n"
            + "\n".join(missing)
        )

    model = joblib.load(MODELS_DIR / "xgboost_baseline.joblib")
    le_pattern = joblib.load(MODELS_DIR / "label_encoder_pattern.joblib")
    le_sex = joblib.load(MODELS_DIR / "label_encoder_sex.joblib")
    le_race = joblib.load(MODELS_DIR / "label_encoder_race.joblib")
    return model, le_pattern, le_sex, le_race


def prepare_test_set(le_sex, le_race, le_pattern):
    """همان تقسیم‌بندی train/test زمان آموزش (random_state=42, stratify)"""
    df = pd.read_csv(DATA_PATH)

    # برای Rule-based به برچسب متنی Sex نیاز داریم؛ قبل از encode نگه می‌داریم
    sex_text = df["Sex"].copy()
    race_text = df["Race"].copy()

    df["Sex_enc"] = le_sex.transform(df["Sex"])
    df["Race_enc"] = le_race.transform(df["Race"])
    y = le_pattern.transform(df["pattern"])

    feature_cols = [
        "Sex_enc",
        "Race_enc",
        "Age",
        "Height",
        "Weight",
        "BMI",
        "Baseline_FEV1_L",
        "Baseline_FVC_L",
        "Baseline_FEV1_FVC_Ratio",
    ]
    X = df[feature_cols]

    indices = np.arange(len(df))
    _, idx_test, _, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )

    test_df = df.iloc[idx_test].copy()
    test_df["Sex_text"] = sex_text.iloc[idx_test].values
    test_df["Race_text"] = race_text.iloc[idx_test].values
    test_df["y_true"] = y_test
    test_df["y_true_label"] = le_pattern.inverse_transform(y_test)

    X_test = test_df[feature_cols]
    return test_df, X_test, y_test


def predict_rule(row):
    result = interpret_spirometry(
        fev1=row["Baseline_FEV1_L"],
        fvc=row["Baseline_FVC_L"],
        age=row["Age"],
        sex=row["Sex_text"],
        height=row["Height"],
    )
    return result["pattern"]


def main():
    print("=" * 60)
    print("ارزیابی سیستماتیک سیستم هیبریدی")
    print("=" * 60)

    model, le_pattern, le_sex, le_race = load_artifacts()
    classes = list(le_pattern.classes_)
    print("کلاس‌ها:", classes)

    test_df, X_test, y_test = prepare_test_set(le_sex, le_race, le_pattern)
    print(f"تعداد نمونه تست: {len(test_df)}")
    print("توزیع واقعی در تست:")
    print(test_df["y_true_label"].value_counts().to_string())
    print()

    # --- پیش‌بینی مدل ---
    print("۱. پیش‌بینی XGBoost...")
    y_pred_ml = model.predict(X_test)
    y_pred_ml_label = le_pattern.inverse_transform(y_pred_ml)

    # --- پیش‌بینی Rule-based ---
    print("۲. پیش‌بینی Rule-based (ممکن است کمی طول بکشد)...")
    y_pred_rule_label = test_df.apply(predict_rule, axis=1).values

    # برچسب‌های Rule که در کلاس‌های مدل نیستند (مثلاً Error) را مدیریت می‌کنیم
    valid_mask = np.isin(y_pred_rule_label, classes)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  هشدار: {n_invalid} پیش‌بینی Rule خارج از کلاس‌های استاندارد بود.")

    y_pred_rule = np.array(
        [
            le_pattern.transform([lab])[0] if lab in classes else -1
            for lab in y_pred_rule_label
        ]
    )

    # --- هیبریدی: موافق → همان؛ مخالف → مدل ---
    print("۳. ساخت پیش‌بینی هیبریدی...")
    agree = y_pred_rule_label == y_pred_ml_label
    y_pred_hybrid_label = np.where(agree, y_pred_ml_label, y_pred_ml_label)
    # منطق فعلی: در اختلاف هم مدل را می‌گیریم → در عمل = خود مدل
    # این را صریح گزارش می‌کنیم؛ بعداً می‌توان منطق بهتری گذاشت
    y_pred_hybrid = y_pred_ml.copy()

    agreement_rate = agree.mean()
    print(f"\nنرخ توافق Rule و ML: {agreement_rate:.1%}")
    print(f"تعداد موافق: {agree.sum()} | تعداد مخالف: {(~agree).sum()}")

    # وقتی مخالف‌اند، کدام درست‌تر است؟
    disagree_idx = np.where(~agree)[0]
    if len(disagree_idx) > 0:
        true_lab = test_df["y_true_label"].values[disagree_idx]
        rule_lab = y_pred_rule_label[disagree_idx]
        ml_lab = y_pred_ml_label[disagree_idx]

        rule_correct = (rule_lab == true_lab).sum()
        ml_correct = (ml_lab == true_lab).sum()
        both_wrong = ((rule_lab != true_lab) & (ml_lab != true_lab)).sum()

        print("\nدر موارد اختلاف:")
        print(f"  Rule درست بوده: {rule_correct}")
        print(f"  ML درست بوده  : {ml_correct}")
        print(f"  هر دو غلط     : {both_wrong}")

        # جدول خلاصه اختلاف‌ها
        print("\nنمونه اختلاف‌ها (حداکثر ۱۵ مورد):")
        sample_n = min(15, len(disagree_idx))
        for i in disagree_idx[:sample_n]:
            print(
                f"  true={test_df['y_true_label'].values[i]:12s} | "
                f"rule={y_pred_rule_label[i]:12s} | "
                f"ml={y_pred_ml_label[i]:12s} | "
                f"FEV1={test_df['Baseline_FEV1_L'].values[i]:.2f} "
                f"FVC={test_df['Baseline_FVC_L'].values[i]:.2f} "
                f"ratio={test_df['Baseline_FEV1_FVC_Ratio'].values[i]:.3f}"
            )

    def report(name, y_true, y_pred, labels_encoded=True):
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        if labels_encoded:
            # فیلتر کردن پیش‌بینی‌های نامعتبر (-1)
            mask = y_pred >= 0
            yt, yp = y_true[mask], y_pred[mask]
            print(classification_report(yt, yp, target_names=classes, digits=3))
            macro = f1_score(yt, yp, average="macro")
        else:
            # y_pred برچسب متنی
            mask = np.isin(y_pred, classes)
            yt = le_pattern.inverse_transform(y_true[mask])
            yp = y_pred[mask]
            print(classification_report(yt, yp, labels=classes, digits=3))
            macro = f1_score(yt, yp, labels=classes, average="macro")
        print(f"Macro F1: {macro:.4f}")
        return macro

    f1_ml = report("فقط XGBoost", y_test, y_pred_ml, labels_encoded=True)
    f1_rule = report("فقط Rule-based", y_test, y_pred_rule_label, labels_encoded=False)
    f1_hybrid = report(
        "هیبریدی فعلی (اختلاف → مدل)", y_test, y_pred_hybrid, labels_encoded=True
    )

    print("\n" + "=" * 60)
    print("جمع‌بندی")
    print("=" * 60)
    print(f"  Macro F1  | XGBoost : {f1_ml:.4f}")
    print(f"  Macro F1  | Rule    : {f1_rule:.4f}")
    print(f"  Macro F1  | Hybrid  : {f1_hybrid:.4f}")
    print(f"  توافق Rule↔ML       : {agreement_rate:.1%}")
    print()
    print(
        "نکته: منطق هیبریدی فعلی در عمل برابر خود مدل است،"
        " چون در اختلاف همیشه مدل را انتخاب می‌کند."
    )
    print(
        "اگر در موارد اختلاف Rule بیشتر درست باشد، باید منطق ترکیب را عوض کنیم."
    )


if __name__ == "__main__":
    main()
