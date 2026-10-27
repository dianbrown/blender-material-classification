"""Regression checks for the publication's evidence and assessment boundaries."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "project_report", ROOT / "scripts/report/build_project_report.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
OUT = ROOT / "reports/project"


def test_same_test_population_and_grouped_plan():
    evaluations, _, folds, final = REPORT.verify_evidence()
    assert len(folds["inner"]) == 5 and len(final) == 3
    assert evaluations["frozen_selected"]["image_level"]["metrics"]["accuracy"] == pytest.approx(
        505 / 630
    )
    assert evaluations["dinov2_finetuned"]["image_level"]["metrics"]["accuracy"] == pytest.approx(
        547 / 630
    )


@pytest.mark.parametrize("model", REPORT.MODELS)
def test_report_contains_each_saved_metric_and_per_class_row(model):
    text = (OUT / "report.md").read_text(encoding="utf-8")
    level = REPORT.read_json(REPORT.RESULT / "fresh" / model / "evaluation.json")["image_level"]
    metrics = level["metrics"]
    assert f"{metrics['accuracy']:.2%}" in text
    for key in ["macro_precision", "macro_recall", "macro_f1"]:
        assert f"{metrics[key]:.4f}" in text
    for row in level["per_class"]:
        fragment = (
            f"| {row['class']} | 90 | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} |"
        )
        assert fragment in text


def test_examples_retain_saved_predictions_and_deterministic_selection():
    selected = json.loads((OUT / "data/examples.json").read_text())["rows"]
    predictions = REPORT.read_csv(REPORT.RESULT / "fresh/frozen_selected/predictions.csv")
    expected = []
    for correct in [True, False]:
        groups = set()
        for row in sorted(predictions, key=lambda r: r["image_path"]):
            if (row["y_true"] == row["y_pred"]) == correct and row["source_group"] not in groups:
                expected.append(row)
                groups.add(row["source_group"])
                if len(groups) == 3:
                    break
    assert [r["image_path"] for r in selected] == [r["image_path"] for r in expected]
    for saved, actual in zip(selected, expected):
        assert all(saved[k] == v for k, v in actual.items())
        assert (OUT / "figures" / saved["display_file"]).is_file()


def test_primary_and_missing_curve_disclosures_are_preserved():
    text = (OUT / "report.md").read_text(encoding="utf-8")
    assert "80.16%" in text and "66.03-93.65%" in text
    assert "86.83%" in text and "76.98-95.87%" in text
    assert "predeclared primary" in text
    assert "post hoc exploratory" in text
    assert "no epoch-wise validation loss" in text
    assert "no saved per-iteration loss history" in text
    assert "C=0.1 for ConvNeXt and C=1 for fusion" in text
    assert text.count("Student number: ______________") == 3
    for forbidden in [
        "V1",
        "V3",
        "Windows",
        "SHA-256",
        "ZIP",
        "agent",
        "C:/Users",
        "reports/results",
    ]:
        assert forbidden not in text


def test_pdf_layout_and_editable_source_agree():
    pymupdf = pytest.importorskip("pymupdf")
    text = (OUT / "report.md").read_text(encoding="utf-8")
    sections = text.split("<!-- pagebreak -->")
    with pymupdf.open(OUT / "report.pdf") as pdf:
        assert len(pdf) == len(sections) == 19
        for page, source in zip(pdf, sections):
            title = source.strip().splitlines()[0][2:]
            assert title in page.get_text()
            assert "\ufffd" not in page.get_text()
        assert "References" in pdf[14].get_text()
    layout = json.loads((OUT / "qa/layout.json").read_text())
    assert layout["main_pages"] == 14
    assert all(not p["out_of_bounds"] for p in layout["pages"])
