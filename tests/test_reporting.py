"""Small fixture-based tests for the result-only reporting layer."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from scripts.generate_report import build_comparison
from scripts.generate_report import generate as generate_report
from scripts.generate_visuals import _cached_explainability_target, _prediction, validate_artifacts
from scripts.generate_visuals import generate as generate_visuals

from material_classifier.plotting import (
    error_gallery,
    material_accuracy_rows,
    plot_confusion_matrix,
    plot_roc_pr,
    plot_training_curves,
    project_embeddings,
)


def test_plot_training_curves_and_confusion_are_result_only(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,train_macro_f1,val_macro_f1,learning_rate\n"
        "1,1.2,1.1,0.3,0.4,0.2,0.3,0.001\n"
        "2,0.8,0.9,0.6,0.5,0.55,0.45,0.0005\n",
        encoding="utf-8",
    )
    outputs = plot_training_curves({"fixture": history}, tmp_path / "figures")
    assert outputs and all(path.exists() for path in outputs)
    confusion = plot_confusion_matrix([[2, 1], [0, 3]], ["a", "b"], tmp_path / "cm.png")
    assert confusion.exists()


def test_prediction_parser_and_missing_artifact_validation(tmp_path):
    prediction = tmp_path / "predictions.csv"
    with prediction.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["true_label", "pred_label", "confidence", "prob_a", "prob_b"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "true_label": "a",
                "pred_label": "a",
                "confidence": "0.9",
                "prob_a": "0.9",
                "prob_b": "0.1",
            }
        )
    rows, payload = _prediction(prediction)
    assert rows[0]["_probability"] == [0.9, 0.1]
    assert payload["true_indices"].tolist() == [0]
    assert len(validate_artifacts(tmp_path)) == 10


def test_embeddings_projection_and_vit_reshape(tmp_path):
    artifact = tmp_path / "embeddings.npz"
    np.savez_compressed(artifact, embeddings=np.eye(4), true_label=np.array(["a", "a", "b", "b"]))
    points, labels, method = project_embeddings(artifact, "pca")
    assert points.shape == (4, 2)
    assert labels.tolist() == ["a", "a", "b", "b"]
    assert method == "pca"
    torch = pytest.importorskip("torch")
    from material_classifier.explainability import vit_reshape_transform

    transformed = vit_reshape_transform(torch.randn(2, 197, 8))
    assert transformed.shape == (2, 8, 14, 14)


def test_report_preserves_missing_metrics(tmp_path):
    results = tmp_path / "reports" / "results" / "custom_cnn" / "test"
    results.mkdir(parents=True)
    (results / "metrics.json").write_text(
        json.dumps({"accuracy": 0.8, "f1_macro": 0.75}), encoding="utf-8"
    )
    rows, missing = build_comparison(tmp_path)
    assert rows[0]["test_accuracy"] == 0.8
    assert rows[0]["test_weighted_f1"] is None
    assert len(missing) == 3


def _write_complete_fixture(root, split="val"):
    models = ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16")
    for model in models:
        result = root / "reports" / "results" / model / split
        result.mkdir(parents=True)
        metrics = {
            "split": split,
            "accuracy": 0.75,
            "f1_macro": 0.70,
            "f1_weighted": 0.71,
            "best_epoch": 2,
            "confusion_matrix": [[1, 0], [0, 1]],
            "per_class": [
                {"class": "wood", "precision": 0.8, "recall": 0.7, "f1": 0.75},
                {"class": "metal", "precision": 0.7, "recall": 0.8, "f1": 0.75},
            ],
        }
        (result / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        with (result / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "true_label",
                    "pred_label",
                    "confidence",
                    "material_id",
                    "image_path",
                    "prob_wood",
                    "prob_metal",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "true_label": "wood",
                        "pred_label": "wood",
                        "confidence": 0.9,
                        "material_id": "w1",
                        "image_path": "sample.png",
                        "prob_wood": 0.9,
                        "prob_metal": 0.1,
                    },
                    {
                        "true_label": "metal",
                        "pred_label": "wood",
                        "confidence": 0.8,
                        "material_id": "m1",
                        "image_path": "sample.png",
                        "prob_wood": 0.8,
                        "prob_metal": 0.2,
                    },
                ]
            )
        checkpoint_dir = root / "artifacts" / "checkpoints" / model
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "history.csv").write_text(
            "epoch,train_loss,val_loss,train_accuracy,val_accuracy,train_macro_f1,val_macro_f1\n"
            "1,1.0,1.0,0.5,0.4,0.4,0.3\n"
            "2,0.5,0.5,0.9,0.75,0.85,0.7\n",
            encoding="utf-8",
        )
    cv = root / "reports" / "results" / "cross_validation_results.csv"
    cv.parent.mkdir(parents=True, exist_ok=True)
    cv.write_text(
        "model,fold,accuracy,f1_macro\n"
        + "\n".join(f"{model},{fold},0.7,0.7" for model in models for fold in range(1, 6)),
        encoding="utf-8",
    )
    (root / "reports" / "results" / "efficiency_metrics.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": model,
                        "checkpoint_size_mb": 2.0,
                        "total_parameters": 1000,
                        "total_training_time_seconds": 20,
                        "benchmark": {
                            "1": {"latency_ms_per_image": 3.0},
                            "32": {"latency_ms_per_image": 1.0},
                        },
                    }
                    for model in models
                ]
            }
        ),
        encoding="utf-8",
    )
    from PIL import Image

    Image.new("RGB", (8, 8), "red").save(root / "sample.png")


def test_global_efficiency_and_same_epoch_overfit_rule(tmp_path):
    _write_complete_fixture(tmp_path)
    rows, _ = build_comparison(tmp_path, "val")
    assert rows[0]["inference_ms_per_image"] == 3.0
    history = tmp_path / "artifacts" / "checkpoints" / "custom_cnn" / "history.csv"
    history.write_text(
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,train_macro_f1,val_macro_f1\n"
        "1,1.0,1.0,0.5,0.4,0.4,0.3\n"
        "2,0.5,0.5,0.9,0.75,0.85,0.7\n"
        "3,0.4,0.8,0.92,0.70,0.88,0.65\n"
        "4,0.3,0.9,0.94,0.70,0.90,0.65\n",
        encoding="utf-8",
    )
    rows, _ = build_comparison(tmp_path, "val")
    assert rows[0]["train_accuracy"] == 0.9
    assert rows[0]["accuracy_gap"] == pytest.approx(0.15)
    assert rows[0]["overfitting_indicator"] == "potential_overfitting"


def test_macro_roc_and_error_material_outputs(tmp_path):
    predictions = {
        "fixture": {
            "true_indices": np.array([0, 0, 1, 1]),
            "probabilities": np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
        }
    }
    paths = plot_roc_pr(predictions, ["wood", "metal"], tmp_path)
    assert any(path.name == "macro_comparison.png" for path in paths)
    from PIL import Image

    Image.new("RGB", (8, 8), "red").save(tmp_path / "sample.png")
    rows = [
        {
            "true_label": "wood",
            "pred_label": "wood",
            "confidence": "0.7",
            "material_id": "w1",
            "image_path": "sample.png",
        },
        {
            "true_label": "metal",
            "pred_label": "wood",
            "confidence": "0.95",
            "material_id": "m1",
            "image_path": "sample.png",
        },
    ]
    assert material_accuracy_rows(rows)[1]["accuracy"] == 0.0
    assert error_gallery(rows, tmp_path / "errors.png", tmp_path, "incorrect")


def test_vit_target_and_corrupt_completed_marker_gate(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from torch import nn

    from material_classifier.explainability import (
        assert_explainability_ready,
        target_layer_for_model,
        vit_reshape_transform,
    )

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(4)

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.class_token = nn.Parameter(torch.zeros(1, 1, 4))
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([Block()])

    model = ViT()
    assert target_layer_for_model(model, "vit_b16") is model.encoder.layers[-1].ln_1
    assert vit_reshape_transform(torch.randn(1, 17, 4)).shape == (1, 4, 4, 4)
    monkeypatch.setattr("material_classifier.evaluate.require_methodology_lock", lambda root: {})
    result = tmp_path / "reports" / "results" / "vit_b16" / "test"
    result.mkdir(parents=True)
    (result / "metrics.json").write_text('{"split":"test"}', encoding="utf-8")
    (result / "predictions.csv").write_text("true_label,pred_label\nwood,wood\n", encoding="utf-8")
    checkpoint = tmp_path / "artifacts" / "checkpoints" / "vit_b16" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    marker = tmp_path / "reports" / "results" / "test_evaluated_vit_b16_bad.json"
    marker.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": "bad",
                "metrics_file": str(result / "metrics.json"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="valid completed TEST marker"):
        assert_explainability_ready(tmp_path, "vit_b16")


def test_explainability_targets_cached_prediction_class():
    classes = ["wood", "metal", "stone"]
    assert (
        _cached_explainability_target(classes, {"true_label": "wood", "pred_label": "metal"}) == 1
    )
    assert (
        _cached_explainability_target(classes, {"true_label": "stone", "pred_label": "stone"}) == 2
    )
    with pytest.raises(ValueError, match="not in class list"):
        _cached_explainability_target(classes, {"true_label": "wood", "pred_label": "fabric"})


def test_complete_val_report_and_visual_generation(tmp_path):
    _write_complete_fixture(tmp_path)
    outputs = generate_report(tmp_path, "val")
    assert outputs["dashboard"].exists()
    figures = generate_visuals(tmp_path, "val")
    assert figures
