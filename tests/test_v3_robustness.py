"""Synthetic CPU-only robustness/diagnostic tests; no benchmark results or training."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3 import features as ft
from material_classifier.v3 import robustness as rb

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v3/robustness.json"


class TinyFrozen(torch.nn.Module):
    def forward_features(self, batch):
        return batch[:, :2]

    def forward_head(self, values, pre_logits=True):
        return values.mean(dim=(2, 3))


def factory(spec, size, **kwargs):
    return TinyFrozen().eval().requires_grad_(False)


def fixture_pipeline():
    # Populate a tiny known fitted state directly. Never call fit, even in tests.
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(2)
    scaler.scale_ = scaler.var_ = np.ones(2)
    scaler.n_features_in_ = 2
    scaler.n_samples_seen_ = 7
    classifier = LogisticRegression()
    classifier.classes_ = np.arange(7)[::-1]
    classifier.coef_ = np.arange(14).reshape(7, 2) / 20
    classifier.intercept_ = np.arange(7) / 10
    classifier.n_features_in_ = 2
    classifier.n_iter_ = np.array([1])
    return Pipeline([("scaler", scaler), ("classifier", classifier)])


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        image = np.random.default_rng(i).integers(0, 256, (19, 23, 3), dtype=np.uint8)
        Image.fromarray(image).save(root / f"{name}.png")
        rows.append(
            {
                "image_path": f"{name}.png",
                "class": name,
                "material_id": f"m{i}",
                "source_group": f"g{i}",
                "original_split": "fresh_holdout",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    protocol = ft.read_json(ROOT / "configs/v3/protocol.json")
    protocol["models"] = {
        "convnext": {**protocol["models"]["convnext"], "width": 2, "resolutions": [8]}
    }
    protocol["candidates"] = [{"name": "tiny", "parts": ["convnext_8_pooled"]}]
    protocol["control"] = "tiny"
    return manifest, root, tmp_path / "stage", protocol


def prepare_model_and_clean(sample):
    manifest, root, output, protocol = sample
    metadata = {
        "recipe": protocol["candidates"][0],
        "protocol": protocol,
        "protocol_sha256": ft.json_hash(protocol),
        "class_order": list(CLASS_NAMES),
        "feature_implementation_sha256": ft.implementation_hash(),
        "feature_environment": ft.environment(),
        "input_features": 2,
        "training_source": {
            "source_groups": ["training-only"],
            "sources": [{"sha256": "not-a-test-image"}],
            "image_paths": ["training-only.png"],
        },
    }
    model = output.parent / "model.joblib"
    joblib.dump({"pipeline": fixture_pipeline(), "metadata": metadata}, model)
    clean = pd.read_csv(manifest, dtype=str)
    clean["outer_fold"] = "holdout"
    clean["y_true"] = clean["class"]
    clean["y_pred"] = CLASS_NAMES[0]
    for index, name in enumerate(CLASS_NAMES):
        clean[f"p_{name}"] = float(index == 0)
    predictions = output.parent / "clean.csv"
    clean.iloc[::-1].to_csv(predictions, index=False)
    cache = output.parent / "clean_features"
    record = ft.extract_features(
        manifest, root, cache, protocol, device="cpu", batch_size=7, model_factory=factory
    )
    run = output.parent / "prediction_run.json"
    ft.write_json(
        run,
        {
            "status": "complete",
            "model_sha256": ft.file_hash(model),
            "manifest_sha256": ft.file_hash(manifest),
            "protocol_sha256": ft.json_hash(protocol),
            "predictions_sha256": ft.file_hash(predictions),
            "feature_cache": str(cache),
            "feature_cache_identity_sha256": record["identity_sha256"],
        },
    )
    return model, predictions, run


@pytest.mark.parametrize("condition", rb.CONDITIONS)
def test_deterministic_original_rgb_transforms(condition):
    pixels = np.random.default_rng(17).integers(0, 256, (160, 240, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    a, b = rb.corrupt_image(image, condition), rb.corrupt_image(image, condition)
    assert a.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(image), pixels)
    assert a.tobytes() != image.tobytes()
    if condition["operation"] == "brightness":
        expected = ImageEnhance.Brightness(image).enhance(condition["factor"])
        np.testing.assert_array_equal(np.asarray(a), np.asarray(expected))
    if condition["operation"] == "gaussian_blur":
        expected = image.filter(ImageFilter.GaussianBlur(1.5))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(expected))
    if condition["operation"] == "downsample":
        assert a.size == (192, 128)
        assert rb.corrupt_image(Image.new("RGB", (32, 16)), condition).size == (32, 16)


def test_exif_orientation_is_applied_before_corruption():
    image = Image.new("RGB", (12, 20))
    image.getexif()[274] = 6
    assert rb.corrupt_image(image, rb.CONDITIONS[0]).size == (20, 12)


def test_preregistered_config_rejects_tuning(tmp_path):
    config = rb.load_config(CONFIG)
    config["conditions"][0]["factor"] = 0.7
    path = tmp_path / "changed.json"
    ft.write_json(path, config)
    with pytest.raises(ValueError, match="preregistered"):
        rb.load_config(path)
    with pytest.raises(ValueError, match="preregistered"):
        rb.corrupt_image(Image.new("RGB", (10, 10)), config["conditions"][0])


@pytest.mark.parametrize(
    "path",
    [
        "../outside.png",
        "/absolute.png",
        "C:/outside.png",
        "C:foo",
        "\\\\server\\share\\image.png",
        "a/../../bad.png",
        "",
    ],
)
def test_source_path_containment(path):
    with pytest.raises(ValueError, match="relative"):
        rb.relative_path(path)


def test_stage_provenance_and_verified_reuse(sample):
    manifest, root, output, protocol = sample
    record = rb.stage_corruptions(
        manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout"
    )
    assert len(record["manifests"]) == 5
    for name, path in record["manifests"].items():
        rows = ft.read_manifest(output / path)
        assert rows.original_image_path.tolist() == ft.read_manifest(manifest).image_path.tolist()
        assert rows.outer_fold.eq("holdout").all()
        assert rows.condition.eq(name).all()
        assert rows.image_path.str.startswith(name + "/images/").all()
    assert record == rb.stage_corruptions(
        manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout", resume=True
    )
    target = next((output / "brightness_low/images").glob("*.png"))
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        rb.stage_corruptions(
            manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout", resume=True
        )


def test_source_content_and_root_identity_change(sample, tmp_path):
    manifest, root, output, protocol = sample
    rb.stage_corruptions(manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout")
    Image.new("RGB", (19, 23)).save(root / "wood.png")
    with pytest.raises(ValueError, match="Stale staging"):
        rb.stage_corruptions(
            manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout", resume=True
        )


def test_stage_rejects_changed_manifest_config_and_extractor(sample, tmp_path, monkeypatch):
    manifest, root, output, protocol = sample
    rb.stage_corruptions(manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout")
    changed_config = tmp_path / "config.json"
    changed_config.write_text(CONFIG.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale staging"):
        rb.stage_corruptions(
            manifest,
            root,
            output,
            changed_config,
            protocol,
            estimate_type="fresh_holdout",
            resume=True,
        )
    monkeypatch.setattr(ft, "implementation_hash", lambda: "changed-extractor")
    with pytest.raises(ValueError, match="Stale staging"):
        rb.stage_corruptions(
            manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout", resume=True
        )


def test_missing_and_escaped_source_files(sample):
    manifest, root, _, _ = sample
    frame = ft.read_manifest(manifest)
    frame.loc[0, "image_path"] = "missing.png"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="missing or outside"):
        rb.source_frame(manifest, root, "fresh_holdout")
    frame.loc[0, "image_path"] = "../outside.png"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="relative"):
        rb.source_frame(manifest, root, "fresh_holdout")


def test_old_test_data_and_aliased_paths_rejected(sample):
    manifest, root, output, protocol = sample
    frame = ft.read_manifest(manifest)
    frame["original_split"] = "test"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="old test"):
        rb.source_frame(manifest, root, "fresh_holdout")
    rb.source_frame(manifest, root, "development_diagnostic")
    frame.loc[1, "image_path"] = "./wood.png"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="aliased"):
        rb.source_frame(manifest, root, "development_diagnostic")
    with pytest.raises(ValueError, match="Explicit"):
        rb.source_frame(manifest, root, None)


def test_identifiable_training_overlap(sample):
    manifest, root, _, _ = sample
    frame = rb.source_frame(manifest, root, "fresh_holdout")
    source = ft.source_identity(manifest, frame, root)
    for training in (
        {"source_groups": ["g0"], "sources": [{"sha256": "unseen"}]},
        {"source_groups": ["other"], "sources": [source["sources"][0]]},
        {
            "source_groups": ["other"],
            "sources": [{"sha256": "unseen"}],
            "image_paths": ["wood.png"],
        },
    ):
        with pytest.raises(ValueError, match="overlap"):
            rb.check_training_overlap(frame, source, {"training_source": training}, "fresh_holdout")
        assert any(
            rb.check_training_overlap(
                frame, source, {"training_source": training}, "development_diagnostic"
            ).values()
        )
    with pytest.raises(ValueError, match="requires training"):
        rb.check_training_overlap(frame, source, {}, "fresh_holdout")


def test_probability_mapping_and_never_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen pipeline must never fit")

    for cls in (Pipeline, StandardScaler, LogisticRegression):
        monkeypatch.setattr(cls, "fit", forbidden)
    pipeline = fixture_pipeline()
    matrix = np.array([[1, 2], [3, 4]])
    before = joblib.hash(pipeline)
    actual = rb.frozen_probabilities(pipeline, matrix, list(CLASS_NAMES))
    np.testing.assert_allclose(actual, pipeline.predict_proba(matrix)[:, ::-1])
    assert before == joblib.hash(pipeline)
    pipeline.named_steps["classifier"].classes_ = np.asarray(CLASS_NAMES)[::-1]
    np.testing.assert_allclose(rb.frozen_probabilities(pipeline, matrix, list(CLASS_NAMES)), actual)
    pipeline.named_steps["classifier"].classes_ = np.arange(7) + 1
    with pytest.raises(ValueError, match="zero-based"):
        rb.frozen_probabilities(pipeline, matrix, list(CLASS_NAMES))


def test_invalid_probability_matrix_rejected(monkeypatch):
    pipeline = fixture_pipeline()
    monkeypatch.setattr(pipeline, "predict_proba", lambda matrix: np.ones((len(matrix), 7)))
    with pytest.raises(ValueError, match="probabilities"):
        rb.frozen_probabilities(pipeline, np.ones((2, 2)), list(CLASS_NAMES))


def test_model_metadata_mismatch_and_unfitted_head_rejected(sample):
    model, _, _ = prepare_model_and_clean(sample)
    protocol = sample[-1]
    bundle = joblib.load(model)
    bundle["metadata"]["feature_implementation_sha256"] = "wrong"
    joblib.dump(bundle, model)
    with pytest.raises(ValueError, match="extractor identity"):
        rb.load_model(model, protocol)
    bundle["metadata"]["feature_implementation_sha256"] = ft.implementation_hash()
    bundle["pipeline"] = Pipeline([("scaler", StandardScaler()), ("head", LogisticRegression())])
    joblib.dump(bundle, model)
    with pytest.raises(ValueError, match="not fitted"):
        rb.load_model(model, protocol)


def test_clean_cache_must_match_original_pixels(sample):
    manifest, root, output, protocol = sample
    model, predictions, run = prepare_model_and_clean(sample)
    Image.new("RGB", (19, 23), "blue").save(root / "wood.png")
    rb.stage_corruptions(manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout")
    with pytest.raises(ValueError, match="source image content"):
        rb.evaluate_staged(
            manifest,
            root,
            output,
            CONFIG,
            protocol,
            model=model,
            clean_predictions=predictions,
            clean_run=run,
            estimate_type="fresh_holdout",
            endpoint_locked=True,
        )


def test_full_sequential_cpu_evaluation_and_strict_pairing(sample, monkeypatch):
    manifest, root, output, protocol = sample
    model, clean_path, run = prepare_model_and_clean(sample)
    rb.stage_corruptions(manifest, root, output, CONFIG, protocol, estimate_type="fresh_holdout")
    extract = ft.extract_features
    calls = []

    def tiny_extract(*args, **kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["parts"] == ["convnext_8_pooled"]
        calls.append(Path(args[0]).parent.name)
        return extract(*args, **kwargs, model_factory=factory)

    def forbidden(*args, **kwargs):
        raise AssertionError("No fitting allowed")

    for cls in (Pipeline, StandardScaler, LogisticRegression):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(ft, "extract_features", tiny_extract)
    kwargs = dict(
        model=model,
        clean_predictions=clean_path,
        clean_run=run,
        estimate_type="fresh_holdout",
        endpoint_locked=True,
        device="cpu",
        batch_size=7,
    )
    record = rb.evaluate_staged(manifest, root, output, CONFIG, protocol, **kwargs)
    assert calls == [condition["name"] for condition in rb.CONDITIONS]
    assert record["status"] == "complete"
    assert record["identity"]["model_metadata_sha256"]
    assert (output / "comparison.png").stat().st_size > 1000
    assert len(pd.read_csv(output / "per_class_recall.csv")) == 35
    assert (
        rb.evaluate_staged(manifest, root, output, CONFIG, protocol, resume=True, **kwargs)
        == record
    )
    assert len(calls) == 5
    clean = pd.read_csv(clean_path)
    altered = pd.read_csv(output / "brightness_low/predictions.csv")
    a, b = rb.paired_frames(clean, altered.iloc[::-1], list(CLASS_NAMES))
    assert a.index.equals(b.index)
    metrics = rb.paired_metrics(clean, clean.iloc[::-1], list(CLASS_NAMES))
    assert metrics["accuracy_drop_from_clean"] == 0
    assert metrics["macro_f1_drop_from_clean"] == 0
    with pytest.raises(ValueError, match="identical original"):
        rb.paired_frames(clean, altered.iloc[:-1], list(CLASS_NAMES))
    altered.loc[0, "material_id"] = "wrong"
    with pytest.raises(ValueError, match="material_id"):
        rb.paired_frames(clean, altered, list(CLASS_NAMES))
    with pytest.raises(ValueError, match="endpoint-locked"):
        rb.evaluate_staged(
            manifest, root, output, CONFIG, protocol, **{**kwargs, "endpoint_locked": False}
        )
    lock = ft.read_json(run)
    lock["model_sha256"] = "wrong"
    ft.write_json(run, lock)
    with pytest.raises(ValueError, match="locked model"):
        rb.evaluate_staged(manifest, root, output, CONFIG, protocol, resume=True, **kwargs)


def load_diagnostics():
    spec = importlib.util.spec_from_file_location("diagnostics", ROOT / "scripts/v3/diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_assets_provenance_and_descriptive_pca(sample):
    manifest, root, output, _ = sample
    _, predictions, _ = prepare_model_and_clean(sample)
    diagnostics = load_diagnostics()
    target = output.parent / "diagnostics"
    record = diagnostics.create_diagnostics(
        predictions,
        root,
        list(CLASS_NAMES),
        target,
        per_class=1,
        feature_cache=output.parent / "clean_features",
        feature_manifest=manifest,
        parts=["convnext_8_pooled"],
    )
    assert record["status"] == "complete"
    assert "not evaluation" in record["pca"]["interpretation"]
    chosen = pd.read_csv(target / "chosen_images.csv")
    assert set(chosen.gallery) == {"representative", "correct", "incorrect"}
    assert {"image_sha256", "confidence", "y_true", "y_pred", "source_group"} <= set(chosen)
    for path in target.glob("*.png"):
        with Image.open(path) as image:
            assert image.width > 500 and image.height > 400
    families = pd.read_csv(target / "family_failures.csv")
    assert families.n_errors.sum() == 6
    with pytest.raises(FileExistsError):
        diagnostics.create_diagnostics(predictions, root, list(CLASS_NAMES), target)
    with pytest.raises(ValueError, match="together"):
        diagnostics.create_diagnostics(
            predictions, root, list(CLASS_NAMES), target, feature_manifest=manifest
        )


def test_cli_requires_explicit_estimate_and_endpoint(tmp_path):
    script = ROOT / "scripts/v3/robustness.py"
    command = [
        sys.executable,
        str(script),
        "--manifest",
        "missing.csv",
        "--data-root",
        ".",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ))
    assert result.returncode != 0 and "--estimate-type" in result.stderr
    result = subprocess.run(
        command + ["--estimate-type", "fresh_holdout", "--evaluate"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert result.returncode != 0 and "--endpoint-locked" in result.stderr
