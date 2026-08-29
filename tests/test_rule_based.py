"""تست‌های ساده سیستم قاعده‌محور"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rule_based.interpreter import interpret_spirometry


def test_obstructive_gold():
    r = interpret_spirometry(2.1, 3.5, age=55, sex="Male", height=170)
    assert r["pattern"] == "Obstructive"
    assert r["fev1_fvc_ratio"] == 0.6
    assert r["fev1_pct_predicted"] is not None
    assert r["severity"] is not None


def test_normal_case():
    r = interpret_spirometry(3.2, 3.8, age=30, sex="Female", height=165)
    assert r["pattern"] == "Normal"
    assert r["fev1_fvc_ratio"] > 0.7


def test_missing_demographics_still_works():
    r = interpret_spirometry(2.1, 3.5)
    assert r["pattern"] == "Obstructive"
    assert r["fev1_pct_predicted"] is None


def test_invalid_input():
    r = interpret_spirometry(0, 3.5)
    assert r["pattern"] == "Error"


if __name__ == "__main__":
    test_obstructive_gold()
    test_normal_case()
    test_missing_demographics_still_works()
    test_invalid_input()
    print("همه تست‌های rule-based پاس شدند.")
