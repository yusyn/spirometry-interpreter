"""مسیرهای وب‌اپ تفسیر اسپیرومتری"""
from flask import Blueprint, flash, jsonify, render_template, request

from src.hybrid_interpreter import hybrid_interpret
from src.utils.labels import pattern_fa

bp = Blueprint("main", __name__)


def _to_float(value, field_name):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"مقدار «{field_name}» عددی نیست.") from exc


def _to_int(value, field_name):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"مقدار «{field_name}» عددی نیست.") from exc


def _parse_payload(data):
    """استخراج و اعتبارسنجی ورودی از form یا JSON."""
    fev1 = _to_float(str(data.get("fev1", "")), "FEV1")
    fvc = _to_float(str(data.get("fvc", "")), "FVC")
    age = _to_int(str(data.get("age", "")), "سن")
    height = _to_float(str(data.get("height", "")), "قد")
    weight_raw = data.get("weight", "")
    weight = _to_float(str(weight_raw), "وزن") if str(weight_raw).strip() else None
    sex = (data.get("sex") or "Male").strip()
    race = (data.get("race") or "Other").strip() or "Other"

    if fev1 is None or fvc is None or age is None or height is None:
        raise ValueError("فیلدهای FEV1، FVC، سن و قد الزامی هستند.")
    if fev1 <= 0 or fvc <= 0 or height <= 0 or age <= 0:
        raise ValueError("مقادیر باید بزرگ‌تر از صفر باشند.")

    return {
        "fev1": fev1,
        "fvc": fvc,
        "age": age,
        "sex": sex,
        "height": height,
        "weight": weight,
        "race": race,
    }


def _enrich_result(result):
    """افزودن برچسب فارسی برای نمایش."""
    result = dict(result)
    result["final_pattern_fa"] = pattern_fa(result.get("final_pattern"))
    if "ml_model" in result:
        result["ml_model"] = dict(result["ml_model"])
        result["ml_model"]["pattern_fa"] = pattern_fa(result["ml_model"].get("pattern"))
    if "rule_based" in result:
        result["rule_based"] = dict(result["rule_based"])
        result["rule_based"]["pattern_fa"] = pattern_fa(result["rule_based"].get("pattern"))
    return result


@bp.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {
        "fev1": "",
        "fvc": "",
        "age": "",
        "sex": "Male",
        "height": "",
        "weight": "",
        "race": "Other",
    }

    if request.method == "POST":
        form_data = {
            "fev1": request.form.get("fev1", "").strip(),
            "fvc": request.form.get("fvc", "").strip(),
            "age": request.form.get("age", "").strip(),
            "sex": request.form.get("sex", "Male"),
            "height": request.form.get("height", "").strip(),
            "weight": request.form.get("weight", "").strip(),
            "race": request.form.get("race", "Other").strip() or "Other",
        }

        try:
            payload = _parse_payload(form_data)
            result = _enrich_result(hybrid_interpret(**payload))
        except FileNotFoundError as exc:
            flash(str(exc), "error")
        except ValueError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"خطای غیرمنتظره: {exc}", "error")

    return render_template("index.html", result=result, form=form_data)


@bp.route("/api/interpret", methods=["POST"])
def api_interpret():
    """
    API تفسیر هیبریدی.

    ورودی JSON نمونه:
    {
      "fev1": 2.1,
      "fvc": 3.5,
      "age": 55,
      "sex": "Male",
      "height": 170,
      "weight": 75,
      "race": "Other"
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "بدنه درخواست باید JSON باشد."}), 400

    try:
        payload = _parse_payload(data)
        result = _enrich_result(hybrid_interpret(**payload))
        return jsonify({"ok": True, "result": result})
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"خطای غیرمنتظره: {exc}"}), 500


@bp.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"ok": True, "service": "spirometry-interpreter"})
