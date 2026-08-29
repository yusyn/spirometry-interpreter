"""
توضیح پیش‌بینی مدل با SHAP برای یک نمونه
"""
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_NAMES = [
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

FEATURE_LABELS_FA = {
    "Sex": "جنس",
    "Race": "نژاد",
    "Age": "سن",
    "Height": "قد",
    "Weight": "وزن",
    "BMI": "BMI",
    "Baseline_FEV1_L": "FEV1",
    "Baseline_FVC_L": "FVC",
    "Baseline_FEV1_FVC_Ratio": "نسبت FEV1/FVC",
}

_explainer = None
_model = None


def _get_explainer(model):
    """TreeExplainer را یک‌بار می‌سازد و نگه می‌دارد."""
    global _explainer, _model
    import shap

    if _explainer is not None and _model is model:
        return _explainer

    _model = model
    _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_prediction(model, features_row, predicted_class_index, top_k=5):
    """
    برای یک ردیف ویژگی، مهم‌ترین عوامل مؤثر روی کلاس پیش‌بینی‌شده را برمی‌گرداند.

    parameters:
        model: مدل XGBoost آموزش‌دیده
        features_row: لیست یا آرایه ۱×n ویژگی‌ها (همان ترتیب آموزش)
        predicted_class_index: ایندکس کلاس پیش‌بینی‌شده
        top_k: تعداد ویژگی‌های برتر

    returns:
        list[dict]: [{feature, feature_fa, value, shap_value, direction}, ...]
    """
    x = np.asarray(features_row, dtype=float).reshape(1, -1)
    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(x)

    # برای چندکلاسه، shap_values معمولاً لیست به ازای هر کلاس است
    if isinstance(shap_values, list):
        class_shap = np.asarray(shap_values[predicted_class_index]).reshape(-1)
    else:
        # بعضی نسخه‌ها آرایه (1, n_features, n_classes) می‌دهند
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            class_shap = arr[0, :, predicted_class_index]
        elif arr.ndim == 2:
            class_shap = arr[0]
        else:
            class_shap = arr.reshape(-1)

    values = x.reshape(-1)
    pairs = []
    for i, name in enumerate(FEATURE_NAMES):
        sv = float(class_shap[i])
        pairs.append(
            {
                "feature": name,
                "feature_fa": FEATURE_LABELS_FA.get(name, name),
                "value": round(float(values[i]), 3),
                "shap_value": round(sv, 4),
                "abs_shap": abs(sv),
                "direction": "افزاینده" if sv >= 0 else "کاهنده",
            }
        )

    pairs.sort(key=lambda d: d["abs_shap"], reverse=True)
    top = pairs[:top_k]
    for item in top:
        item.pop("abs_shap", None)
    return top
