"""Unit and integration tests for the Dual-Engine Evaluation Benchmark Runner."""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_dual_engine import run_dual_engine_benchmark


def test_benchmark_runner_subset(tmp_path):
    """Verify evaluation runner executes on subset and returns required schema in tmp_path."""
    summary = run_dual_engine_benchmark(
        sample_size=30,
        random_state=42,
        generate_docs=True,
        output_dir=tmp_path,
    )

    assert isinstance(summary, dict)
    assert "overall_concordance" in summary
    assert isinstance(summary["overall_concordance"], float)
    assert 0.0 <= summary["overall_concordance"] <= 1.0

    assert "discrepancy_matrix" in summary
    assert isinstance(summary["discrepancy_matrix"], dict)

    assert "discrepancy_summary" in summary
    assert isinstance(summary["discrepancy_summary"], dict)

    assert "total_samples" in summary
    assert summary["total_samples"] == 30

    # Verify generated documentation artifacts exist in isolated tmp_path
    eval_doc = tmp_path / "evaluation_results.md"
    thesis_doc = tmp_path / "thesis_defense_guide.md"

    assert eval_doc.exists()
    assert eval_doc.stat().st_size > 500

    assert thesis_doc.exists()
    assert thesis_doc.stat().st_size > 1000


def test_benchmark_deterministic():
    """Verify evaluation benchmark is deterministic for a fixed random seed."""
    run1 = run_dual_engine_benchmark(sample_size=25, random_state=123, generate_docs=False)
    run2 = run_dual_engine_benchmark(sample_size=25, random_state=123, generate_docs=False)

    assert run1["overall_concordance"] == run2["overall_concordance"]
    assert run1["concordant_count"] == run2["concordant_count"]
    assert run1["discrepancy_matrix"] == run2["discrepancy_matrix"]


def test_evaluation_safety_and_clinical_caveats(tmp_path):
    """Verify documentation artifacts enforce medical safety and FVC/TLC caveats."""
    summary = run_dual_engine_benchmark(
        sample_size=20,
        random_state=42,
        generate_docs=True,
        output_dir=tmp_path,
    )
    assert summary is not None

    eval_doc = tmp_path / "evaluation_results.md"
    thesis_doc = tmp_path / "thesis_defense_guide.md"

    eval_content = eval_doc.read_text(encoding="utf-8")
    thesis_content = thesis_doc.read_text(encoding="utf-8")

    # Safety: Must specify physiological pattern rather than definitive medical diagnosis
    assert "physiological pattern" in eval_content.lower() or "الگوی فیزیولوژیک" in eval_content
    assert "tlc" in eval_content.lower() or "total lung capacity" in eval_content.lower()

    # Thesis guide must have 3-Phase narrative, slide outline, and Q&A
    assert "Phase 1" in thesis_content or "فاز ۱" in thesis_content
    assert "Phase 2" in thesis_content or "فاز ۲" in thesis_content
    assert "Phase 3" in thesis_content or "فاز ۳" in thesis_content
    assert "Slide" in thesis_content or "اسلاید" in thesis_content
    assert "Q&A" in thesis_content or "پرسش و پاسخ" in thesis_content
