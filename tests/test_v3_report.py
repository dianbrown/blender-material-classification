"""Report contract tests. ALL numeric data below are SYNTHETIC test fixtures."""

import csv
import json

import pytest
from scripts.v3 import build_report as report


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def refresh_hashes(root, manifest):
    for item in manifest["evidence_files"]:
        item["sha256"] = report.sha256(root / item["path"])
    save_json(root / "report_manifest.json", manifest)


@pytest.fixture
def synthetic_results(tmp_path):
    root = tmp_path / "synthetic_inputs"
    root.mkdir()
    # Deliberately non-alphabetic class order tests preservation of evaluator order.
    classes = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]
    matrix = [[0] * 7 for _ in classes]
    rows = []
    for fold in range(5):
        for index, cls in enumerate(classes):
            for view in range(2):
                predicted = index if view == 0 else (index + 1) % 7
                matrix[index][predicted] += 1
                row = {
                    "image_path": f"SYNTHETIC/{cls}/{fold}/{view}.png",
                    "class": cls,
                    "material_id": f"synthetic_{cls}_{fold}",
                    "source_group": f"synthetic_family_{cls}_{fold}",
                    "outer_fold": str(fold),
                    "y_true": index,
                    "y_pred": predicted,
                }
                row.update(
                    {f"p_{c}": 0.7 if i == predicted else 0.05 for i, c in enumerate(classes)}
                )
                rows.append(row)
    with (root / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "class_order": list(classes),
        "n_images": 70,
        "n_source_groups": 35,
        "confusion_matrix": matrix,
        "per_class": [
            {"class": c, "support": 10, "precision": 0.5, "recall": 0.5, "f1": 0.5} for c in classes
        ],
        "metrics": {
            "accuracy": {
                "value": 0.5,
                "ci": {
                    "low": 0.35,
                    "high": 0.65,
                    "level": 0.95,
                    "unit": "source_group",
                    "method": "SYNTHETIC test interval only",
                    "n_resamples": 100,
                    "seed": 17,
                },
            },
            "balanced_accuracy": {"value": 0.5},
            "macro_f1": {"value": 0.5},
            "weighted_f1": {"value": 0.5},
            "auroc_macro_ovr": {"value": None, "reason": "Not computed in synthetic fixture"},
        },
        "calibration_bins": [{"confidence": 0.7, "accuracy": 0.5, "count": 70}],
        "selective_risk": [{"coverage": 0.5, "risk": 0.5}, {"coverage": 1.0, "risk": 0.5}],
    }
    save_json(root / "summary.json", summary)
    (root / "protocol.txt").write_text("SYNTHETIC protocol evidence, not a real experiment.")
    (root / "comparison.csv").write_text(
        "Candidate,Scope,Accuracy\nSYNTHETIC baseline,SYNTHETIC fixture,0.5\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "evidence_kind": "synthetic_fixture",
        "title": "SYNTHETIC v3 report layout fixture",
        "run_id": "synthetic-test-only",
        "created_utc": "2026-10-27T12:00:00Z",
        "code_commit": "a" * 40,
        "class_order": classes,
        "primary_endpoint": "nested_oof",
        "sections": {
            k: f"SYNTHETIC fixture: {title}. No measured v3 result is asserted. "
            "Protocol evidence: protocol.txt. Literal markup <sample> & values are text."
            for k, title in report.SECTIONS.items()
        },
        "evidence_files": [
            {"path": p, "sha256": "", "role": "SYNTHETIC fixture"}
            for p in ("summary.json", "predictions.csv", "protocol.txt", "comparison.csv")
        ],
        "endpoints": [
            {"id": "historical_test", "status": "not_available", "reason": "Not in test fixture"},
            {
                "id": "nested_oof",
                "status": "measured",
                "summary": "summary.json",
                "predictions": "predictions.csv",
            },
            {"id": "fresh_holdout", "status": "not_available", "reason": "Not yet evaluated"},
        ],
        "deployment_fit": {"status": "not_available", "description": "No synthetic fit artifact"},
        "supplements": [
            {
                "section": "comparison_ablations",
                "kind": "table",
                "path": "comparison.csv",
                "caption": "SYNTHETIC comparison table",
            }
        ],
    }
    refresh_hashes(root, manifest)
    return root, manifest, summary


def test_requires_actual_results_without_creating_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(report.ReportError, match="Cannot read JSON"):
        report.build_report(tmp_path, output)
    assert not output.exists()


def test_synthetic_requires_explicit_opt_in(synthetic_results):
    root, _, _ = synthetic_results
    with pytest.raises(report.ReportError, match="--allow-synthetic"):
        report.load_report(root)


def test_valid_contract(synthetic_results):
    root, manifest, _ = synthetic_results
    loaded, endpoints, hashes = report.load_report(root, True)
    assert loaded == manifest
    assert endpoints[0]["data"]["class_order"] == manifest["class_order"]
    assert len(hashes) == 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m: m.update(primary_endpoint="historical_test"), "Historical test cannot"),
        (lambda m: m.update(primary_endpoint="fresh_holdout"), "Primary endpoint must"),
        (lambda m: m.update(schema_version=2), "schema_version"),
        (lambda m: m["class_order"].append("wood"), "seven unique"),
        (lambda m: m["sections"].pop("protocol"), "sections.protocol"),
        (lambda m: m["endpoints"].pop(), "all three"),
        (lambda m: m["endpoints"][2].update(summary="summary.json"), "cannot carry results"),
        (lambda m: m["evidence_files"][0].update(sha256="0" * 64), "hash mismatch"),
        (lambda m: m["deployment_fit"].update(status="fitted"), "Deployment artifact"),
        (lambda m: m["supplements"][0].update(path="untracked.csv"), "hashed evidence"),
    ],
)
def test_manifest_rejections(synthetic_results, mutation, message):
    root, manifest, _ = synthetic_results
    mutation(manifest)
    save_json(root / "report_manifest.json", manifest)
    with pytest.raises(report.ReportError, match=message):
        report.load_report(root, True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda s: s["class_order"].reverse(), "class_order mismatch"),
        (lambda s: s["metrics"]["accuracy"].update(value=0.99), "accuracy disagrees"),
        (lambda s: s["metrics"]["accuracy"].pop("ci"), "accuracy CI"),
        (lambda s: s["metrics"]["accuracy"]["ci"].update(unit="image"), "source groups"),
        (lambda s: s["metrics"]["accuracy"]["ci"].update(low=0.9), "bounds reversed"),
        (lambda s: s["metrics"]["accuracy"]["ci"].pop("seed"), "ci.seed"),
        (lambda s: s["metrics"]["macro_f1"].update(value=float("nan")), "finite number"),
        (lambda s: s["metrics"]["auroc_macro_ovr"].pop("reason"), "reason"),
        (lambda s: s["per_class"][0].update(f1=1.0), "f1 mismatch"),
        (lambda s: s.update(n_source_groups=34), "n_source_groups disagrees"),
        (lambda s: s["calibration_bins"][0].update(count=69), "bin counts mismatch"),
    ],
)
def test_summary_rejections(synthetic_results, mutation, message):
    root, manifest, summary = synthetic_results
    mutation(summary)
    save_json(root / "summary.json", summary)
    refresh_hashes(root, manifest)
    with pytest.raises(report.ReportError, match=message):
        report.load_report(root, True)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("p_wood", "0.2", "sum to one"),
        ("p_wood", "NaN", "finite number"),
        ("class", "metal", "class/y_true mismatch"),
        ("outer_fold", "100", "crosses outer folds"),
        ("image_path", "SYNTHETIC/wood/0/1.png", "Duplicate image"),
        ("y_pred", "1", "argmax"),
    ],
)
def test_prediction_rejections(synthetic_results, column, value, message):
    root, manifest, _ = synthetic_results
    path = root / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0][column] = value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    refresh_hashes(root, manifest)
    with pytest.raises(report.ReportError, match=message):
        report.load_report(root, True)


def test_path_escape_and_duplicate_json_rejected(synthetic_results):
    root, manifest, _ = synthetic_results
    outside = root.parent / "outside.txt"
    outside.write_text("Not inside the supplied results directory")
    manifest["evidence_files"].append(
        {"path": "../outside.txt", "sha256": report.sha256(outside), "role": "escape"}
    )
    save_json(root / "report_manifest.json", manifest)
    with pytest.raises(report.ReportError, match="escapes"):
        report.load_report(root, True)
    (root / "report_manifest.json").write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(report.ReportError, match="Duplicate JSON key"):
        report.load_report(root, True)


def test_fresh_holdout_overlap_rejected(synthetic_results):
    root, manifest, _ = synthetic_results
    manifest["endpoints"][2] = {
        "id": "fresh_holdout",
        "status": "measured",
        "summary": "summary.json",
        "predictions": "predictions.csv",
        "selection_frozen_at": "2026-10-26T00:00:00Z",
        "independence_audit": "protocol.txt",
    }
    save_json(root / "report_manifest.json", manifest)
    with pytest.raises(report.ReportError, match="source-group overlap"):
        report.load_report(root, True)


def test_existing_output_is_preserved(synthetic_results, tmp_path):
    root, _, _ = synthetic_results
    output = tmp_path / "old_report"
    output.mkdir()
    old = output / "report.pdf"
    old.write_bytes(b"Preserve old report")
    with pytest.raises(report.ReportError, match="already exists"):
        report.build_report(root, output, allow_synthetic=True)
    assert old.read_bytes() == b"Preserve old report"


def test_render_failure_does_not_publish_partial_output(synthetic_results, tmp_path, monkeypatch):
    root, _, _ = synthetic_results
    output = tmp_path / "not_published"

    def fail(*args, **kwargs):
        raise RuntimeError("Simulated renderer failure")

    monkeypatch.setattr(report, "make_plots", lambda *args: [])
    monkeypatch.setattr(report, "write_pdf", fail)
    with pytest.raises(RuntimeError, match="renderer failure"):
        report.build_report(root, output, allow_synthetic=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".v3-report-*"))


def test_build_synthetic_pdf_and_markdown(synthetic_results, tmp_path):
    from pypdf import PdfReader

    root, _, _ = synthetic_results
    output = report.build_report(root, tmp_path / "synthetic_report", allow_synthetic=True)
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC FIXTURE - NOT EXPERIMENTAL EVIDENCE" in markdown
    assert "accuracy 50.00% (95% source-group CI 35.00%-65.00%)" in markdown
    assert "Source: summary.json#/metrics" in markdown
    assert "Not yet evaluated" in markdown
    assert "All-data deployment fit: not_available" in markdown
    assert "SYNTHETIC comparison table" in markdown
    assert "Authors:" not in markdown
    assert (output / "report_manifest.json").read_bytes() == (
        root / "report_manifest.json"
    ).read_bytes()
    images = list((output / "figures").glob("*.png"))
    assert len(images) == 4
    reader = PdfReader(output / "report.pdf")
    pdf_text = " ".join(page.extract_text() for page in reader.pages)
    assert "accuracy 50.00%" in pdf_text
    assert "<sample> & values" in pdf_text
    assert "Not yet evaluated" in pdf_text
    for page in reader.pages:
        assert "SYNTHETIC FIXTURE - NOT EXPERIMENTAL EVIDENCE" in page.extract_text()
        assert "Page " in page.extract_text()
    # Rasterize every page and check text bounds, complementing manual visual QA.
    import pymupdf

    with pymupdf.open(output / "report.pdf") as document:
        for index, page in enumerate(document):
            for word in page.get_text("words"):
                assert 20 <= word[0] <= word[2] <= page.rect.width - 20
                assert 15 <= word[1] <= word[3] <= page.rect.height - 15
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1, 1))
            pixmap.save(str(tmp_path / f"synthetic_page_{index + 1}.png"))


def test_cli_requires_explicit_result_directory():
    with pytest.raises(SystemExit) as error:
        report.main([])
    assert error.value.code == 2


def test_shared_uncertainty_prose_preserves_distinct_methods(synthetic_results):
    root, _, _ = synthetic_results
    manifest, endpoints, hashes = report.load_report(root, True)
    manifest["_manifest_sha256"] = "a" * 64
    manifest["sections"]["motivation"] = "First paragraph.\n\nSecond paragraph."
    metrics = endpoints[0]["data"]["metrics"]
    metrics["macro_f1"]["ci"] = dict(metrics["accuracy"]["ci"])
    metrics["weighted_f1"]["ci"] = dict(metrics["accuracy"]["ci"], seed=19)
    blocks = report.report_blocks(manifest, endpoints, hashes, {"nested_oof": []}, {})
    uncertainty = [v for k, v in blocks if k == "p" and v.startswith("Uncertainty for")]
    assert len(uncertainty) == 2
    assert "accuracy, macro f1" in uncertainty[0] and "seed 17" in uncertainty[0]
    assert "weighted f1" in uncertainty[1] and "seed 19" in uncertainty[1]
    assert ("p", "First paragraph.") in blocks
    assert ("p", "Second paragraph.") in blocks


@pytest.mark.parametrize("label", ["", "x" * 66, "preview\nline", "preview\u2011label"])
def test_invalid_footer_status_rejected(synthetic_results, label):
    root, manifest, _ = synthetic_results
    manifest["status_label"] = label
    save_json(root / "report_manifest.json", manifest)
    with pytest.raises(report.ReportError, match="status_label"):
        report.load_report(root, True)


@pytest.mark.parametrize("synthetic", [False, True])
def test_pdf_status_footer_keeps_synthetic_warning(tmp_path, synthetic):
    from pypdf import PdfReader

    path = tmp_path / "layout_only.pdf"
    report.write_pdf(
        [("title", "Layout-only status test; no results")],
        path,
        tmp_path,
        synthetic,
        "INTERNAL PRELIMINARY PREVIEW | v3",
    )
    text = PdfReader(path).pages[0].extract_text()
    expected = (
        "SYNTHETIC FIXTURE - NOT EXPERIMENTAL EVIDENCE"
        if synthetic
        else "INTERNAL PRELIMINARY PREVIEW | v3"
    )
    assert expected in text
