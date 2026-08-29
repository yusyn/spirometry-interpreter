"""مسیرهای وب‌اپ تفسیر اسپیرومتری"""
from flask import Blueprint, flash, render_template, request

from src.hybrid_interpreter import hybrid_interpret

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
            fev1 = _to_float(form_data["fev1"], "FEV1")
            fvc = _to_float(form_data["fvc"], "FVC")
            age = _to_int(form_data["age"], "سن")
            height = _to_float(form_data["height"], "قد")
            weight = _to_float(form_data["weight"], "وزن")

            if fev1 is None or fvc is None or age is None or height is None:
                raise ValueError("فیلدهای FEV1، FVC، سن و قد الزامی هستند.")
            if fev1 <= 0 or fvc <= 0 or height <= 0 or age <= 0:
                raise ValueError("مقادیر باید بزرگ‌تر از صفر باشند.")

            result = hybrid_interpret(
                fev1=fev1,
                fvc=fvc,
                age=age,
                sex=form_data["sex"],
                height=height,
                weight=weight,
                race=form_data["race"],
            )
        except FileNotFoundError as exc:
            flash(str(exc), "error")
        except ValueError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"خطای غیرمنتظره: {exc}", "error")

    return render_template("index.html", result=result, form=form_data)
