"""Focused tests for the protocol's asymmetric CV design."""

import csv
import json
from pathlib import Path

from scripts.generate_report import _recommendation

MODELS = ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16")
TRANSFER = MODELS[1:]


def _rows(root: Path) -> list[dict[str, object]]:
    for model in MODELS:
        path = root / "reports" / "results" / model / "test" / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "per_class": [{"f1": 0.7 + 0.01 * i} for i in range(7)],
                }
            ),
            encoding="utf-8",
        )
    return [
        {
            "model": model,
            "test_macro_f1": 0.70 + i * 0.02,
            "test_accuracy": 0.72 + i * 0.02,
            "model_size_mb": 10.0 + i,
            "inference_ms_per_image": 1.0 + i * 0.2,
        }
        for i, model in enumerate(MODELS)
    ]


def _cv(root: Path, missing: tuple[str, int] | None = None) -> None:
    path = root / "reports" / "results" / "cross_validation_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "fold", "f1_macro"))
        writer.writeheader()
        for model in TRANSFER:
            for fold in range(1, 6):
                if missing == (model, fold):
                    continue
                writer.writerow({"model": model, "fold": fold, "f1_macro": 0.65 + fold / 100})


def test_recommendation_accepts_customcnn_without_cv(tmp_path: Path):
    rows = _rows(tmp_path)
    _cv(tmp_path)
    payload = _recommendation(tmp_path, rows)

    assert payload["status"] == "completed"
    assert payload["selected_model"] in MODELS
    custom = payload["models"]["custom_cnn"]
    assert custom["raw"]["cv_stability"] is None
    assert custom["cv_folds"] == "not_applicable"
    assert payload["criterion_coverage"]["custom_cnn"]["cv_stability"] == "not_applicable"
    assert len(custom["criteria_used"]) == 5
    assert all(len(payload["models"][model]["criteria_used"]) == 6 for model in TRANSFER)


def test_recommendation_stays_pending_when_transfer_cv_incomplete(tmp_path: Path):
    rows = _rows(tmp_path)
    _cv(tmp_path, ("convnext_tiny", 5))
    payload = _recommendation(tmp_path, rows)

    assert payload["status"] == "pending"
    assert payload["selected_model"] is None
    assert "convnext_tiny" in payload["missing_criteria_models"]
    assert payload["models"]["convnext_tiny"]["cv_folds"] == 4


def test_recommendation_rejects_duplicate_cv_fold(tmp_path: Path):
    rows = _rows(tmp_path)
    _cv(tmp_path)
    with (tmp_path / "reports" / "results" / "cross_validation_results.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("resnet50,1,0.99\n")

    payload = _recommendation(tmp_path, rows)

    assert payload["status"] == "pending"
    assert payload["selected_model"] is None
    assert any(
        "duplicate CV row for resnet50, fold 1" in item for item in payload["cv_gate_errors"]
    )
    assert payload["missing_criteria_models"] == ["convnext_tiny", "resnet50", "vit_b16"]
