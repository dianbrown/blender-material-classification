"""CPU stubs only: no model construction, weights, fresh inference, or dataset assets."""

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3 import features as ft
from material_classifier.v3 import neural_robustness as nr
from material_classifier.v3 import robustness as rb
from material_classifier.v3.learning import prediction_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["resnet", "dinov2"])
def case(tmp_path, request, monkeypatch):
    family = request.param
    backend = nr._backend(family)
    root = tmp_path / "source"
    root.mkdir()
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (9, 11, 3), dtype=np.uint8)).save(
            root / f"{name}.png"
        )
        rows.append(
            dict(
                image_path=f"{name}.png",
                **{"class": name},
                material_id=f"m{i}",
                source_group=f"g{i}",
                original_split="fresh_holdout",
            )
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    frame = ft.read_manifest(manifest)
    staging = tmp_path / "stage"
    rb.stage_corruptions(
        manifest,
        root,
        staging,
        nr.CONFIG_PATH,
        ft.load_protocol(nr.PROTOCOL_PATH),
        estimate_type="fresh_holdout",
    )
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "model.pt"
    artifact.write_bytes(b"synthetic deployment bytes; never loaded")
    config = backend.load_config() if family == "resnet" else {"input_size": 14}
    metadata = dict(
        model_sha256=ft.file_hash(artifact),
        config=config,
        config_sha256=ft.json_hash(config),
        class_order=list(CLASS_NAMES),
        training_source={
            "source_groups": ["train-group"],
            "material_ids": ["train-material"],
            "image_paths": ["train.png"],
            "sources": [{"sha256": "train-bytes"}],
            "decoded_rgb_sha256": ["train-pixels"],
        },
    )
    if family == "resnet":
        metadata.update(
            architecture="torchvision.models.resnet50",
            training_stage="final",
            preprocessing=config["preprocessing"],
            implementation=backend.implementation_identity(),
        )
        metadata["metadata_sha256"] = ft.json_hash(metadata)
    else:
        metadata.update(kind="exploratory-partial-dinov2", source_hashes=backend.source_hashes())
    ft.write_json(artifact.with_suffix(".json"), metadata)
    entry = dict(
        artifact_sha256=ft.file_hash(artifact),
        metadata_sha256=ft.file_hash(artifact.with_suffix(".json")),
        artifact_path="models/model.pt",
        metadata_path="models/model.json",
        role="assessment control",
    )
    lock = tmp_path / "endpoint_lock.json"
    ft.write_json(
        lock,
        dict(
            schema_version=1,
            frozen_at_utc=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
            primary_model="other-primary",
            models={"control": entry, "other-primary": entry},
            holdout=dict(manifest_sha256=ft.file_hash(manifest), n_images=7, source_group_count=7),
            class_order=list(CLASS_NAMES),
            selection_rule="Synthetic pre-score decision",
        ),
    )
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_path = clean_dir / "predictions.csv"
    prediction_frame(frame, np.eye(7), "holdout").iloc[::-1].to_csv(clean_path, index=False)
    clean_run = clean_dir / "prediction_run.json"

    def write_run(manifest_path, data_root, predictions_path, run_path, status="complete"):
        current_frame = ft.read_manifest(manifest_path)
        run = dict(
            status=status,
            model_sha256=entry["artifact_sha256"],
            evaluation_requested=True,
            source=nr._source(backend, family, manifest_path, current_frame, data_root),
            rows=len(current_frame),
            predictions_sha256=ft.file_hash(predictions_path),
        )
        if family == "resnet":
            run["config_sha256"] = metadata["config_sha256"]
        else:
            run["metadata_sha256"] = entry["metadata_sha256"]
        ft.write_json(run_path, run)

    write_run(manifest, root, clean_path, clean_run)
    calls = []

    def predict(**kwargs):
        assert kwargs["evaluate"] is True
        assert kwargs["device"] == "cpu" and kwargs["cpu_threads"] == 4
        assert kwargs["artifact"] == artifact
        if family == "dinov2":
            assert kwargs["expected_sha256"] == entry["artifact_sha256"]
            assert kwargs["metadata_sha256"] == entry["metadata_sha256"]
        calls.append(Path(kwargs["manifest"]).parent.name)
        destination = Path(kwargs["output"])
        destination.mkdir()
        source_frame = ft.read_manifest(kwargs["manifest"])
        # One known error, shuffled independently of both source and clean rows.
        probabilities = np.eye(7)
        probabilities[0] = probabilities[1]
        prediction_frame(source_frame, probabilities, "holdout").iloc[[3, 1, 6, 0, 4, 2, 5]].to_csv(
            destination / "predictions.csv", index=False
        )
        ft.write_json(destination / "metrics.json", {"stub": True})
        write_run(
            kwargs["manifest"],
            kwargs["data_root"],
            destination / "predictions.csv",
            destination / "prediction_run.json",
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("No training, weight loading, or fresh inference allowed")

    for name in ("fit_fixed", "fit_one", "train_run", "load_model", "load_deployed"):
        if hasattr(backend, name):
            monkeypatch.setattr(backend, name, forbidden)
    monkeypatch.setattr(
        backend, "predict_manifest" if family == "resnet" else "predict_finetuned", predict
    )
    return dict(
        kwargs=dict(
            manifest=manifest,
            data_root=root,
            staging_dir=staging,
            output_dir=tmp_path / "result",
            family=family,
            artifact=artifact,
            clean_predictions=clean_path,
            clean_run=clean_run,
            endpoint_lock=lock,
            model_name="control",
            device="cpu",
            cpu_threads=4,
        ),
        calls=calls,
        write_run=write_run,
        backend=backend,
        predict=predict,
        entry=entry,
    )


def rewrite(path, mutate, *, preserve_time=False):
    stat = path.stat()
    value = ft.read_json(path)
    mutate(value)
    ft.write_json(path, value)
    if preserve_time:
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_sequential_joined_metrics_no_fitting_and_immutable_inputs(case):
    args = case["kwargs"]
    stage = args["staging_dir"]
    before = {str(p): ft.file_hash(p) for p in stage.rglob("*") if p.is_file()}
    record = nr.evaluate_staged(**args)
    assert record["status"] == "complete"
    assert case["calls"] == [c["name"] for c in rb.CONDITIONS]
    assert before == {str(p): ft.file_hash(p) for p in stage.rglob("*") if p.is_file()}
    output = args["output_dir"]
    comparison = pd.read_csv(output / "comparison.csv")
    assert comparison.clean_accuracy.eq(1).all()
    assert comparison.accuracy.to_numpy() == pytest.approx([6 / 7] * 5)
    assert comparison.accuracy_drop_from_clean.to_numpy() == pytest.approx([1 / 7] * 5)
    assert comparison.macro_f1.to_numpy() == pytest.approx([(5 + 2 / 3) / 7] * 5)
    assert comparison.macro_f1_drop_from_clean.to_numpy() == pytest.approx([4 / 21] * 5)
    recalls = pd.read_csv(output / "per_class_recall.csv")
    assert len(recalls) == 35
    assert recalls.loc[recalls["class"] == CLASS_NAMES[0], "recall_drop_from_clean"].eq(1).all()
    for name in case["calls"]:
        directory = output / name
        result = pd.read_csv(directory / "predictions.csv")
        expected = ft.read_manifest(stage / name / "manifest.csv")
        joined = result.merge(expected, on="image_path", validate="one_to_one")
        assert joined.original_image_path_x.equals(joined.original_image_path_y)
        raw = directory / "inference"
        assert ft.read_json(raw / "prediction_run.json")["predictions_sha256"] == ft.file_hash(
            raw / "predictions.csv"
        )
    assert (output / "comparison.png").stat().st_size > 1000
    rb._verify_files(output, record["files"])
    with pytest.raises(FileExistsError, match="NEW"):
        nr.evaluate_staged(**args)
    assert len(case["calls"]) == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("status", "failed"),
        ("evaluation_requested", False),
        ("model_sha256", "another model"),
        ("predictions_sha256", "tampered"),
        ("rows", 6),
        ("source", {}),
    ],
)
def test_completed_matching_clean_run_required(case, field, value):
    args = case["kwargs"]
    rewrite(args["clean_run"], lambda run: run.update({field: value}))
    with pytest.raises(ValueError, match="Completed prediction run"):
        nr.evaluate_staged(**args)
    assert not case["calls"] and not args["output_dir"].exists()


@pytest.mark.parametrize(
    "target",
    [
        "artifact",
        "metadata",
        "clean_predictions",
        "manifest",
        "staged_image",
        "source_image",
    ],
)
def test_tampered_bytes_rejected_before_inference(case, target):
    args = case["kwargs"]
    paths = dict(
        artifact=args["artifact"],
        metadata=args["artifact"].with_suffix(".json"),
        clean_predictions=args["clean_predictions"],
        manifest=args["manifest"],
        staged_image=next(args["staging_dir"].rglob("*.png")),
        source_image=next(args["data_root"].glob("*.png")),
    )
    with paths[target].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        nr.evaluate_staged(**args)
    assert not case["calls"]


@pytest.mark.parametrize(
    "change",
    [
        "class_order",
        "holdout",
        "model_name",
        "model_path",
        "metadata_hash",
        "schema",
        "late_lock",
    ],
)
def test_wrong_lock_rejected(case, change):
    args = case["kwargs"]

    def mutate(lock):
        if change == "class_order":
            lock["class_order"].reverse()
        elif change == "holdout":
            lock["holdout"]["manifest_sha256"] = "wrong"
        elif change == "model_name":
            del lock["models"]["control"]
        elif change == "model_path":
            lock["models"]["control"]["artifact_path"] = "another/model.pt"
        elif change == "metadata_hash":
            lock["models"]["control"]["metadata_sha256"] = "wrong"
        elif change == "schema":
            lock["schema_version"] = 2
        else:
            lock["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()

    rewrite(args["endpoint_lock"], mutate, preserve_time=True)
    with pytest.raises(ValueError, match="lock"):
        nr.evaluate_staged(**args)
    assert not case["calls"]


@pytest.mark.parametrize("column", ["image_path", "material_id", "original_image_path"])
def test_clean_coverage_and_identity_even_with_updated_prediction_hash(case, column):
    args = case["kwargs"]
    frame = pd.read_csv(args["clean_predictions"])
    if column == "image_path":
        frame = frame.iloc[:-1]
    elif column == "original_image_path":
        frame[column] = "wrong.png"
    else:
        frame.loc[0, column] = "wrong-id"
    frame.to_csv(args["clean_predictions"], index=False)
    rewrite(
        args["clean_run"],
        lambda run: run.update(predictions_sha256=ft.file_hash(args["clean_predictions"])),
    )
    with pytest.raises(ValueError, match="identity|coverage"):
        nr.evaluate_staged(**args)
    assert not case["calls"]


def test_wrong_family_rejected(case):
    args = {
        **case["kwargs"],
        "family": "dinov2" if case["kwargs"]["family"] == "resnet" else "resnet",
    }
    with pytest.raises(ValueError, match="family"):
        nr.evaluate_staged(**args)


def test_original_training_content_overlap_rejected(case):
    args = case["kwargs"]
    sidecar = args["artifact"].with_suffix(".json")

    def mutate(metadata):
        metadata["training_source"]["sources"] = [
            {"sha256": ft.file_hash(next(args["data_root"].glob("*.png")))}
        ]
        if args["family"] == "resnet":
            metadata["metadata_sha256"] = ft.json_hash(
                {k: v for k, v in metadata.items() if k != "metadata_sha256"}
            )

    rewrite(sidecar, mutate)
    rewrite(
        args["endpoint_lock"],
        lambda lock: lock["models"]["control"].update(metadata_sha256=ft.file_hash(sidecar)),
        preserve_time=True,
    )
    with pytest.raises(ValueError, match="overlap"):
        nr.evaluate_staged(**args)
    assert not case["calls"]


@pytest.mark.parametrize("failure", ["incomplete", "missing_row", "model_mutated"])
def test_failed_condition_never_silently_skipped_or_resumed(case, monkeypatch, failure):
    args = case["kwargs"]

    def predict(**kwargs):
        case["predict"](**kwargs)
        output = Path(kwargs["output"])
        if failure == "incomplete":
            rewrite(output / "prediction_run.json", lambda run: run.update(status="running"))
        elif failure == "missing_row":
            path = output / "predictions.csv"
            pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
            rewrite(
                output / "prediction_run.json",
                lambda run: run.update(predictions_sha256=ft.file_hash(path)),
            )
        else:
            args["artifact"].write_bytes(b"changed during prediction")

    monkeypatch.setattr(
        case["backend"],
        "predict_manifest" if args["family"] == "resnet" else "predict_finetuned",
        predict,
    )
    with pytest.raises(ValueError):
        nr.evaluate_staged(**args)
    assert len(case["calls"]) == 1
    assert ft.read_json(args["output_dir"] / "evaluation_run.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        nr.evaluate_staged(**args)


@pytest.mark.parametrize("change", ["inventory", "status", "config", "source_mapping"])
def test_stage_incomplete_inventory_config_and_join_rejected(case, change):
    args = case["kwargs"]
    path = args["staging_dir"] / "staging.json"

    def mutate(record):
        if change == "inventory":
            record["files"].pop(next(iter(record["files"])))
        elif change == "status":
            record["status"] = "running"
        elif change == "config":
            record["identity"]["config_sha256"] = "wrong"
            record["identity_sha256"] = ft.json_hash(record["identity"])
        else:
            relative = record["manifests"]["brightness_low"]
            manifest = args["staging_dir"] / relative
            frame = ft.read_manifest(manifest)
            frame["original_image_path"] = frame.original_image_path.tolist()[::-1]
            frame.to_csv(manifest, index=False)
            record["files"][relative] = ft.file_hash(manifest)

    rewrite(path, mutate)
    with pytest.raises(ValueError, match="Staging"):
        nr.evaluate_staged(**args)
    assert not case["calls"]


def test_family_specific_clean_provenance_required(case):
    args = case["kwargs"]
    field = "config_sha256" if args["family"] == "resnet" else "metadata_sha256"
    rewrite(args["clean_run"], lambda run: run.update({field: "wrong"}))
    with pytest.raises(ValueError, match="Completed prediction run"):
        nr.evaluate_staged(**args)
    assert not case["calls"]


def test_model_config_checksum_required_even_if_lock_matches_sidecar(case):
    args = case["kwargs"]
    sidecar = args["artifact"].with_suffix(".json")
    rewrite(sidecar, lambda metadata: metadata.update(config_sha256="wrong"))
    rewrite(
        args["endpoint_lock"],
        lambda lock: lock["models"]["control"].update(metadata_sha256=ft.file_hash(sidecar)),
        preserve_time=True,
    )
    with pytest.raises(ValueError, match="config checksum"):
        nr.evaluate_staged(**args)
    assert not case["calls"]


def test_cli_help_and_required_paths():
    command = [sys.executable, str(ROOT / "scripts/v3/neural_robustness.py")]
    result = subprocess.run(command + ["--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for option in ("--endpoint-lock", "--model-name", "--clean-run", "--staging-dir"):
        assert option in result.stdout
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "required" in result.stderr
