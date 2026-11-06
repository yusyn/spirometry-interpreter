"""مسیرهای وب‌اپ و API تفسیر دوگانه اسپیرومتری (Dual-Engine Architecture)."""

from flask import Blueprint, flash, jsonify, render_template, request

from src.hybrid_interpreter import (
    hybrid_interpret,
    interpret_spirometry_dual_engine,
)
from src.ingest.spirometry import SpirometryReport
from src.reference_engine.validator import validate_spirometry_input
from src.utils.labels import pattern_fa

bp = Blueprint("main", __name__)

MODEL_RACE_GROUPS = (
    "Black",
    "Mexican American",
    "Other hispanic",
    "Other race, including multi-racial",
    "White",
)
MODEL_AGE_MIN = 6.0
MODEL_AGE_MAX = 80.0


def _to_float(value, field_name: str):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"مقدار «{field_name}» عددی نیست.") from exc


def _parse_payload(data: dict) -> dict:
    """استخراج، نرمال‌سازی و اعتبارسنجی ورودی از فرم یا JSON."""
    report = SpirometryReport.from_mapping(data)
    fev1 = report.fev1
    fvc = report.fvc

    if fev1 is None and "fev1" in data:
        fev1 = _to_float(str(data.get("fev1", "")), "FEV1")
    if fvc is None and "fvc" in data:
        fvc = _to_float(str(data.get("fvc", "")), "FVC")

    age = _to_float(str(data.get("age", "")), "سن")

    height_raw = data.get("height", "")
    if not str(height_raw).strip() and "height_cm" in data:
        height_raw = data.get("height_cm", "")
    height = _to_float(str(height_raw), "قد")

    weight_raw = data.get("weight", "")
    weight = _to_float(str(weight_raw), "وزن") if str(weight_raw).strip() else 70.0

    sex_raw = (data.get("sex") or "").strip()
    race_raw = (data.get("race") or "").strip()

    if fev1 is None or fvc is None or age is None or height is None:
        missing = [
            name
            for name, value in (("FEV1", fev1), ("FVC expiratory", fvc), ("سن", age), ("قد", height))
            if value is None
        ]
        hint = " FVCex فقط با پروفایل صریح دستگاه به FVC نگاشت می‌شود؛ FVCin جایگزین نیست." if fvc is None else ""
        raise ValueError("فیلدهای الزامی موجود نیستند: " + "، ".join(missing) + "." + hint)

    if fev1 <= 0 or fvc <= 0 or height <= 0 or age <= 0 or weight <= 0:
        raise ValueError("مقادیر باید بزرگ‌تر از صفر باشند.")

    sex_lookup = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
    sex = sex_lookup.get(sex_raw.casefold())
    if not sex:
        raise ValueError("جنس باید یکی از مقادیر Male یا Female باشد.")

    race = race_raw or "White"
    race_lookup = {r.casefold(): r for r in MODEL_RACE_GROUPS}
    if race.casefold() in race_lookup:
        race = race_lookup[race.casefold()]
    else:
        allowed = "، ".join(MODEL_RACE_GROUPS)
        raise ValueError(f"گروه جمعیتی باید یکی از مقادیر مجموعهٔ آموزش باشد: {allowed}")

    payload = {
        "fev1": fev1,
        "fvc": fvc,
        "age": age,
        "sex": sex,
        "height": height,
        "weight": weight,
        "race": race,
    }

    is_valid, err_msg = validate_spirometry_input(payload)
    if not is_valid:
        raise ValueError(err_msg)

    return payload


def _enrich_result(dual_result: dict, input_values: dict = None) -> dict:
    """افزودن برچسب‌های فارسی و متاداده برای تمپلیت وب‌اپ."""
    result = dict(dual_result)

    if "physiological_engine" in result:
        pe = dict(result["physiological_engine"])
        p_name = pe.get("pattern", "")
        if p_name in ("Normal Spirometry", "Normal"):
            pe["pattern_fa"] = "الگوی اسپیرومتری نرمال"
        elif p_name == "Obstructive":
            pe["pattern_fa"] = "الگوی انسدادی"
        elif "Restrictive" in p_name:
            pe["pattern_fa"] = "مشکوک به الگوی محدودیتی (نیازمند TLC)"
        elif p_name == "Mixed":
            pe["pattern_fa"] = "الگوی مختلط (انسدادی + مشکوک به تحدید)"
        else:
            pe["pattern_fa"] = pattern_fa(p_name)

        sev = pe.get("severity")
        severity_map = {
            "Mild": "خفیف",
            "Moderate": "متوسط",
            "Severe": "شدید",
            "Very Severe": "بسیار شدید",
        }
        pe["severity_fa"] = severity_map.get(sev, sev) if sev else "—"
        result["physiological_engine"] = pe

    if "ml_surrogate" in result:
        ml = dict(result["ml_surrogate"])
        ml_pat = ml.get("pattern", "")
        if ml_pat == "Normal":
            ml["pattern_fa"] = "الگوی نرمال"
        elif ml_pat == "Obstructive":
            ml["pattern_fa"] = "الگوی انسدادی"
        elif ml_pat == "Restrictive":
            ml["pattern_fa"] = "الگوی محدودیتی (Surrogate)"
        else:
            ml["pattern_fa"] = pattern_fa(ml_pat)
        result["ml_surrogate"] = ml

    # سازگاری با کدهای پیشین
    result["final_pattern"] = result.get("physiological_engine", {}).get(
        "pattern", result.get("ml_surrogate", {}).get("pattern", "")
    )
    result["final_pattern_fa"] = result.get("physiological_engine", {}).get(
        "pattern_fa", result.get("ml_surrogate", {}).get("pattern_fa", "")
    )
    result["ml_model"] = result.get("ml_surrogate", {})
    result["rule_based"] = result.get("physiological_engine", {})

    if input_values is not None:
        result["input_values"] = dict(input_values)
        if "ratio" not in result["input_values"]:
            f1 = float(input_values.get("fev1", 0.0))
            fc = float(input_values.get("fvc", 1.0))
            result["input_values"]["ratio"] = round(f1 / fc, 3) if fc > 0 else 0.0

    return result


@bp.route("/", methods=["GET", "POST"])
def index():
    dual_result = None
    form_data = {
        "fev1": "",
        "fvc": "",
        "age": "",
        "sex": "Male",
        "height": "",
        "weight": "",
        "race": "White",
        "use_fixed_ratio": False,
    }

    if request.method == "POST":
        form_data = {
            "fev1": request.form.get("fev1", "").strip(),
            "fvc": request.form.get("fvc", "").strip(),
            "age": request.form.get("age", "").strip(),
            "sex": request.form.get("sex", "Male"),
            "height": request.form.get("height", "").strip(),
            "weight": request.form.get("weight", "").strip(),
            "race": request.form.get("race", "White").strip(),
            "use_fixed_ratio": request.form.get("use_fixed_ratio") in ("on", "true", "1"),
        }

        try:
            payload = _parse_payload(form_data)
            use_fixed_ratio = form_data["use_fixed_ratio"]
            raw_result = interpret_spirometry_dual_engine(
                payload, use_fixed_ratio=use_fixed_ratio
            )
            dual_result = _enrich_result(raw_result, input_values=payload)
        except FileNotFoundError as exc:
            flash(str(exc), "error")
        except ValueError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"خطای غیرمنتظره: {exc}", "error")

    return render_template(
        "index.html",
        result=dual_result,
        dual_result=dual_result,
        form=form_data,
    )


@bp.route("/api/interpret", methods=["POST"])
def api_interpret():
    """
    API تفسیر دوگانه اسپیرومتری (Dual-Engine API).

    ورودی JSON نمونه:
    {
      "fev1": 2.1,
      "fvc": 3.5,
      "age": 55,
      "sex": "Male",
      "height": 170,
      "weight": 75,
      "race": "White",
      "use_fixed_ratio": false
    }
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"ok": False, "error": "بدنه درخواست باید JSON باشد."}), 400

    try:
        payload = _parse_payload(data)
        use_fixed_ratio = bool(
            data.get("use_fixed_ratio", False) or data.get("fixed_ratio", False)
        )
        dual_result = interpret_spirometry_dual_engine(
            payload, use_fixed_ratio=use_fixed_ratio
        )
        disclaimer = dual_result.get("disclaimer", "")

        return (
            jsonify(
                {
                    "ok": True,
                    "physiological_engine": dual_result["physiological_engine"],
                    "ml_surrogate": dual_result["ml_surrogate"],
                    "comparison": dual_result["comparison"],
                    "medical_disclaimer": disclaimer,
                    "disclaimer": disclaimer,
                    "result": dual_result,
                }
            ),
            200,
        )
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"خطای غیرمنتظره: {exc}"}), 500


@bp.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"ok": True, "service": "spirometry-interpreter"})
