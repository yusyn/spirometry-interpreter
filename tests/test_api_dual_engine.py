"""Tests for Flask API and Web Application Dual-Engine updates."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_interpret_returns_dual_engine(client):
    payload = {
        "age": 45,
        "height": 175,
        "weight": 80,
        "sex": "Male",
        "fev1": 3.2,
        "fvc": 4.2,
        "race": "White",
    }
    resp = client.post("/api/interpret", json=payload)
    assert resp.status_code == 200
    json_data = resp.get_json()

    assert json_data["ok"] is True
    assert "physiological_engine" in json_data
    assert "ml_surrogate" in json_data
    assert "comparison" in json_data
    assert "medical_disclaimer" in json_data
    assert "result" in json_data

    # Physiological engine contents
    pe = json_data["physiological_engine"]
    assert "predicted" in pe
    assert "lln" in pe
    assert "zscore" in pe
    assert "pct_predicted" in pe
    assert "pattern" in pe
    assert "severity" in pe
    assert "clinical_note" in pe

    # ML surrogate contents
    ml = json_data["ml_surrogate"]
    assert "pattern" in ml
    assert "probabilities" in ml
    assert "shap_explanation" in ml

    # Comparison contents
    comp = json_data["comparison"]
    assert "concordant" in comp
    assert "discrepancy_detected" in comp
    assert "discrepancy_explanation" in comp

    # Educational / safety disclaimer
    disclaimer = json_data["medical_disclaimer"]
    assert "educational" in disclaimer.lower() or "پژوهشی" in disclaimer or "research" in disclaimer.lower()


def test_api_interpret_invalid_payload_returns_400(client):
    resp = client.post("/api/interpret", json={})
    assert resp.status_code == 400
    json_data = resp.get_json()
    assert json_data["ok"] is False
    assert "error" in json_data


def test_api_interpret_fev1_exceeds_fvc_returns_400(client):
    payload = {
        "age": 45,
        "height": 175,
        "weight": 80,
        "sex": "Male",
        "fev1": 5.0,
        "fvc": 3.0,
        "race": "White",
    }
    resp = client.post("/api/interpret", json=payload)
    assert resp.status_code == 400
    json_data = resp.get_json()
    assert json_data["ok"] is False
    assert "cannot exceed" in json_data["error"].lower() or "FEV1" in json_data["error"]


def test_api_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "service": "spirometry-interpreter"}


def test_web_index_get(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "تفسیر اسپیرومتری" in html
    assert "fev1" in html
    assert "fvc" in html


def test_web_index_post_concordant(client):
    form_data = {
        "fev1": "3.5",
        "fvc": "4.5",
        "age": "35",
        "sex": "Male",
        "height": "175",
        "weight": "75",
        "race": "White",
    }
    resp = client.post("/", data=form_data)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Check side-by-side elements in UI
    assert "موتور فیزیولوژیک مرجع" in html or "Physiological" in html or "ATS/ERS" in html
    assert "مدل یادگیری ماشین" in html or "ML Surrogate" in html or "XGBoost" in html
    assert "یاسین زمانی" in html


def test_web_index_post_restrictive_tlc_disclaimer(client):
    # Restrictive: low FVC (e.g. 2.0L), normal/high ratio (FEV1 1.8L -> ratio 0.9)
    form_data = {
        "fev1": "1.8",
        "fvc": "2.0",
        "age": "50",
        "sex": "Male",
        "height": "175",
        "weight": "75",
        "race": "White",
    }
    resp = client.post("/", data=form_data)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Must contain TLC caveat
    assert "TLC" in html or "Total Lung Capacity" in html
