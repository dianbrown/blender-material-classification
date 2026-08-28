"""Focused safety and smoke tests for the ML core."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from material_classifier.cross_validate import grouped_folds
from material_classifier.metrics import classification_metrics
from material_classifier.utils import seed_everything


def _records(include_test: bool = False):
    rows = [
        {"class": cls, "material_id": f"{cls}_{mat}", "split": "train"}
        for cls in ("wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather")
        for mat in range(5)
        for _ in range(2)
    ]
    if include_test:
        rows.append({"class": "wood", "material_id": "wood_test", "split": "test"})
    return rows


def test_grouped_cv_materials_are_disjoint():
    for train, valid in grouped_folds(_records(), 5, 42):
        assert {_records()[i]["material_id"] for i in train}.isdisjoint(
            {_records()[i]["material_id"] for i in valid}
        )


def test_grouped_cv_rejects_test_records():
    with pytest.raises(ValueError, match="TEST"):
        grouped_folds(_records(True), 5, 42)


def test_cv_csv_merge_retains_other_models_and_completed_folds(tmp_path):
    from material_classifier.cross_validate import run_grouped_cv

    path = tmp_path / "cv.csv"
    fields = (
        "model,fold,accuracy,balanced_accuracy,precision_macro,recall_macro,"
        "f1_macro,f1_weighted,best_epoch,training_time_seconds\n"
    )
    path.write_text(fields + "resnet50,1,0.5,0.5,0.5,0.5,0.5,0.5,1,1\n", encoding="utf-8")

    def runner(model, fold, train, valid):
        return {
            "accuracy": 0.6,
            "balanced_accuracy": 0.6,
            "precision_macro": 0.6,
            "recall_macro": 0.6,
            "f1_macro": 0.6,
            "f1_weighted": 0.6,
            "best_epoch": 2,
            "training_time_seconds": 2,
        }

    result = run_grouped_cv(_records(), "resnet50", runner, 5, 42, path)
    text = path.read_text(encoding="utf-8")
    assert text.count("resnet50,") == 5
    assert len(result) == 5
    aggregate = tmp_path / "cv_aggregate.csv"
    assert aggregate.exists()
    assert "precision_macro_mean" in aggregate.read_text(encoding="utf-8")
    assert "recall_macro_std" in aggregate.read_text(encoding="utf-8")


def test_metrics_shape_and_values():
    result = classification_metrics(
        [0, 1, 2, 0], [0, 1, 1, 2], np.eye(3)[[0, 1, 1, 2]], ["a", "b", "c"]
    )
    assert result["accuracy"] == pytest.approx(0.5)
    assert len(result["per_class"]) == 3
    assert len(result["confusion_matrix"]) == 3


def test_seed_helper_reproducible():
    seed_everything(42)
    first = (random.random(), np.random.rand())
    seed_everything(42)
    second = (random.random(), np.random.rand())
    assert first == second


def test_oom_retry_allows_only_batch_size_change_when_torch_available():
    pytest.importorskip("torch")
    from material_classifier.train import _safe_oom_config_change

    old = {"name": "resnet50", "training": {"batch_size": 32, "max_epochs": 40}}
    assert _safe_oom_config_change(
        old, {"name": "resnet50", "training": {"batch_size": 16, "max_epochs": 40}}
    )
    assert not _safe_oom_config_change(
        old, {"name": "resnet50", "training": {"batch_size": 16, "max_epochs": 30}}
    )


def test_oom_retry_seeds_factory_and_persists_adjustment(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import material_classifier.train as training

    seen = []
    calls = {"count": 0}

    def fake_train(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CUDA OOM: reduce batch_size")
        return {"best_epoch": 1, "history": []}

    def factory():
        seen.append(float(torch.rand(1)))
        return torch.nn.Linear(1, 1)

    def loaders(batch):
        seen.append(batch)
        return [], []

    monkeypatch.setattr(training, "train_model", fake_train)
    training.train_with_oom_retry(
        factory,
        loaders,
        {"seed": 42, "training": {"batch_size": 8}},
        tmp_path,
        torch.device("cuda"),
    )
    assert seen[0] == 8
    assert seen[2] == 4
    assert seen[1] == seen[3]
    assert (tmp_path / "oom_adjustments.json").exists()
    training.train_with_oom_retry(
        factory,
        loaders,
        {"seed": 42, "training": {"batch_size": 8}},
        tmp_path,
        torch.device("cuda"),
    )
    assert seen[-2:] == [4, seen[1]]


def test_interrupted_oom_chain_appends_and_original_config_reuses(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import json

    import material_classifier.train as training

    calls = {"count": 0}

    (tmp_path / "oom_adjustments.json").write_text(
        '{"adjustments": [{"from_batch_size": 32, "to_batch_size": 16}], "final_batch_size": 16}\n',
        encoding="utf-8",
    )
    (tmp_path / "run_state.json").write_text('{"status": "oom"}\n', encoding="utf-8")

    def fake_train(*args, **kwargs):
        calls["count"] += 1
        state_path = tmp_path / "run_state.json"
        if calls["count"] == 1:
            state_path.write_text('{"status": "oom"}\n', encoding="utf-8")
            raise RuntimeError("CUDA OOM")
        state_path.write_text('{"status": "completed"}\n', encoding="utf-8")
        return {"best_epoch": 1, "history": []}

    monkeypatch.setattr(training, "train_model", fake_train)
    config = {"seed": 42, "training": {"batch_size": 32}}

    def factory():
        return torch.nn.Linear(1, 1)

    def loaders(batch):
        return [], []

    training.train_with_oom_retry(factory, loaders, config, tmp_path, torch.device("cuda"))
    chain = json.loads((tmp_path / "oom_adjustments.json").read_text(encoding="utf-8"))
    assert chain["adjustments"] == [
        {"from_batch_size": 32, "to_batch_size": 16},
        {"from_batch_size": 16, "to_batch_size": 8},
    ]
    assert chain["final_batch_size"] == 8
    training.train_with_oom_retry(factory, loaders, config, tmp_path, torch.device("cuda"))
    assert calls["count"] == 3


def test_oom_before_first_checkpoint_preserves_timing_and_artifacts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for OOM state path")
    import json

    import material_classifier.train as training

    config = {"seed": 42, "name": "custom_cnn", "training": {"batch_size": 32, "max_epochs": 1}}

    def raise_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(training, "_epoch", raise_oom)
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        training.train_model(torch.nn.Linear(3, 7), [], [], config, tmp_path, torch.device("cuda"))
    first_state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    (tmp_path / "oom_adjustments.json").write_text(
        '{"adjustments": [{"from_batch_size": 32, "to_batch_size": 16}], "final_batch_size": 16}\n',
        encoding="utf-8",
    )
    good_metrics = {"accuracy": 0.5, "f1_macro": 0.5}

    def good_epoch(*args, **kwargs):
        return 1.0, good_metrics

    monkeypatch.setattr(training, "_epoch", good_epoch)
    smaller = dict(config, training={"batch_size": 16, "max_epochs": 1})
    training.train_model(torch.nn.Linear(3, 7), [], [], smaller, tmp_path, torch.device("cuda"))
    final_state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    resolved = json.loads((tmp_path / "resolved_config.json").read_text(encoding="utf-8"))
    assert final_state["started_at"] == first_state["started_at"]
    assert final_state["total_training_time_seconds"] >= first_state["total_training_time_seconds"]
    assert resolved["effective_batch_size"] == 16
    assert resolved["oom_adjustments"] == [{"from_batch_size": 32, "to_batch_size": 16}]
    best = training.load_torch_checkpoint(tmp_path / "best.pt")
    assert best["effective_batch_size"] == final_state["effective_batch_size"] == 16
    assert best["oom_adjustments"] == final_state["oom_adjustments"] == resolved["oom_adjustments"]


def test_model_output_shape_when_torch_available():
    torch = pytest.importorskip("torch")
    from material_classifier.models import MODEL_NAMES, build_model, forward_with_embeddings

    for name in MODEL_NAMES:
        model = build_model(name, 7, pretrained=False).eval()
        # Keep this smoke test small; it only verifies the public contract.
        with torch.no_grad():
            output, embedding = forward_with_embeddings(model, torch.randn(2, 3, 224, 224))
        assert tuple(output.shape) == (2, 7)
        assert embedding.shape[0] == 2


def test_forward_embeddings_match_forward_when_torch_available():
    torch = pytest.importorskip("torch")
    from material_classifier.models import MODEL_NAMES, build_model, forward_with_embeddings

    for name in MODEL_NAMES:
        model = build_model(name, 7, pretrained=False).eval()
        inputs = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            direct = model(inputs)
            combined, embedding = forward_with_embeddings(model, inputs)
        assert torch.allclose(direct, combined, atol=1e-5, rtol=1e-5)
        assert embedding.ndim == 2


def test_head_only_mode_freezes_batchnorm_running_stats_when_torch_available():
    pytest.importorskip("torch")
    from torch import nn

    from material_classifier.train import _set_training_mode

    model = nn.Module()
    model.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    model.classifier = nn.Linear(4, 2)
    _set_training_mode(model, True, True)
    assert model.training is False
    assert model.backbone.training is False
    assert model.backbone[1].training is False
    assert model.classifier.training is True
    _set_training_mode(model, True, False)
    assert model.training is True


def test_checkpoint_roundtrip_when_torch_available(tmp_path):
    torch = pytest.importorskip("torch")
    from material_classifier.utils import atomic_torch_save, load_torch_checkpoint

    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"model_state": {"x": torch.tensor([1, 2])}}, path)
    loaded = load_torch_checkpoint(path)
    assert loaded["model_state"]["x"].tolist() == [1, 2]


def test_completed_test_cache_is_reused_without_inference(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import json

    import material_classifier.evaluate as evaluation

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    result_dir = tmp_path / "reports" / "results" / "custom_cnn" / "test"
    result_dir.mkdir(parents=True)
    metrics = result_dir / "metrics.json"
    predictions = result_dir / "predictions.csv"
    embeddings = result_dir / "embeddings.npz"
    metrics.write_text('{"accuracy": 1}\n', encoding="utf-8")
    predictions.write_text("image_path\n", encoding="utf-8")
    embeddings.write_bytes(b"embedding")
    monkeypatch.setattr(evaluation, "require_methodology_lock", lambda root: {})
    monkeypatch.setattr(
        evaluation,
        "sha256_file",
        lambda path: "checkpoint-hash" if Path(path) == checkpoint else "artifact-hash",
    )
    monkeypatch.setattr(
        evaluation,
        "load_torch_checkpoint",
        lambda *args: {"config": {"name": "custom_cnn", "num_classes": 7}},
    )
    marker = tmp_path / "reports" / "results" / "test_evaluated_custom_cnn_checkpoint-hash.json"
    marker.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": "checkpoint-hash",
                "metrics_file": str(metrics),
                "predictions_file": str(predictions),
                "embeddings_file": str(embeddings),
                "metrics_sha256": "artifact-hash",
                "predictions_sha256": "artifact-hash",
                "embeddings_sha256": "artifact-hash",
            }
        ),
        encoding="utf-8",
    )
    cached = evaluation.evaluate_checkpoint(checkpoint, object(), tmp_path, "test", ["a"])
    assert cached == {"accuracy": 1}


def test_methodology_lock_is_one_way(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.lock_methodology import create_lock

    from material_classifier.utils import stable_hash

    (tmp_path / "configs").mkdir()
    for name in ("data", "custom_cnn", "resnet50", "convnext_tiny", "vit_b16"):
        (tmp_path / "configs" / f"{name}.yaml").write_text("name: " + name + "\n", encoding="utf-8")
    (tmp_path / "reports" / "results").mkdir(parents=True)
    (tmp_path / "reports" / "results" / "dataset_validation.json").write_text(
        '{"passed": true}\n', encoding="utf-8"
    )
    rows = [
        "model,fold,accuracy,balanced_accuracy,precision_macro,recall_macro,f1_macro,f1_weighted,best_epoch,training_time_seconds"
    ]
    for model in ("resnet50", "convnext_tiny", "vit_b16"):
        rows.extend(f"{model},{fold},0,0,0,0,0,0,1,1" for fold in range(1, 6))
    (tmp_path / "reports" / "results" / "cross_validation_results.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "reports" / "results" / "cross_validation_results_aggregate.csv").write_text(
        "model,f1_macro_mean,f1_macro_std\nresnet50,0,0\nconvnext_tiny,0,0\nvit_b16,0,0\n",
        encoding="utf-8",
    )
    for relative in (
        "metadata/material_manifest.csv",
        "metadata/material_manifest_split.csv",
        "metadata/render_manifest.csv",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for relative in (
        "src/material_classifier/models.py",
        "src/material_classifier/train.py",
        "src/material_classifier/evaluate.py",
        "src/material_classifier/cross_validate.py",
        "src/material_classifier/data.py",
        "src/material_classifier/augmentations.py",
        "src/material_classifier/metrics.py",
        "src/material_classifier/utils.py",
        "src/material_classifier/inference.py",
        "scripts/evaluate_all.py",
        "scripts/run_group_cv.py",
        "scripts/train_model.py",
        "scripts/lock_methodology.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test source\n", encoding="utf-8")
    for model in ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16"):
        path = tmp_path / "artifacts" / "checkpoints" / model
        path.mkdir(parents=True)
        model_config = {"name": model}
        config_hash = stable_hash(model_config)
        torch.save(
            {
                "config": model_config,
                "config_hash": config_hash,
                "resolved_config": {"config_hash": config_hash},
            },
            path / "best.pt",
        )
        (path / "run_state.json").write_text(
            '{"status": "completed", "config_hash": "'
            + config_hash
            + '", "model": "'
            + model
            + '", "resolved_config": {"config_hash": "'
            + config_hash
            + '"}}\n',
            encoding="utf-8",
        )
        (path / "resolved_config.json").write_text(
            '{"config_hash": "' + config_hash + '"}\n', encoding="utf-8"
        )
    lock = create_lock(tmp_path)
    assert '"test_evaluation_allowed": true' in lock.read_text(encoding="utf-8")
    from material_classifier.evaluate import require_methodology_lock

    require_methodology_lock(tmp_path)
    (tmp_path / "metadata" / "render_manifest.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalidated"):
        require_methodology_lock(tmp_path)
    with pytest.raises(FileExistsError):
        create_lock(tmp_path)
