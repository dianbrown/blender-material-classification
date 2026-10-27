"""Read-only staged sensitivity evaluation of locked neural deployment artifacts.

The parent writes schema-v1 endpoint_lock BEFORE any fresh predictions: UTC
frozen_at_utc, primary_model, models keyed by friendly name (artifact_sha256,
metadata_sha256, artifact_path, metadata_path, role), holdout (manifest_sha256,
n_images, source_group_count), class_order, selection_rule. Paths are absolute
or relative to the lock file. Controls need only be listed, not selected primary.
metadata_sha256 is the sidecar FILE hash, not ResNet's internal JSON digest.

Existing predictors have no start timestamp or lock field. We enforce the lock's
UTC timestamp and file mtime precede both completed clean files; these local
filesystem checks are not signed historical proof. Parent provenance remains
necessary. No training, staging, resumption, or modification of clean inputs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import CLASS_NAMES
from . import features as ft
from . import robustness as rb
from .evaluation import validate_predictions

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/v3/robustness.json"
PROTOCOL_PATH = ROOT / "configs/v3/protocol.json"


def _backend(family):
    if family == "resnet":
        from . import resnet_baseline

        return resnet_baseline
    if family == "dinov2":
        from . import finetune

        return finetune
    raise ValueError("family must be resnet or dinov2")


def _lock_entry(path, model_name, artifact, manifest, frame, clean_paths):
    lock = ft.read_json(path)
    if (
        lock.get("schema_version") != 1
        or lock.get("class_order") != list(CLASS_NAMES)
        or not isinstance(lock.get("models"), dict)
        or not isinstance(lock.get("primary_model"), str)
        or lock["primary_model"] not in lock["models"]
        or not isinstance(lock.get("selection_rule"), str)
        or not lock["selection_rule"].strip()
    ):
        raise ValueError("Invalid endpoint lock schema/class order/selection")
    try:
        frozen = datetime.fromisoformat(lock["frozen_at_utc"].replace("Z", "+00:00"))
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Endpoint lock requires ISO frozen_at_utc") from exc
    if frozen.utcoffset() != timedelta(0) or frozen > datetime.now(timezone.utc):
        raise ValueError("Endpoint lock requires a past UTC frozen_at_utc")
    if any(
        max(frozen.timestamp(), Path(path).stat().st_mtime) > Path(p).stat().st_mtime
        for p in clean_paths
    ):
        raise ValueError("Endpoint lock must precede clean predictions and completed run")
    expected = {
        "manifest_sha256": ft.file_hash(manifest),
        "n_images": len(frame),
        "source_group_count": int(frame.source_group.nunique()),
    }
    if lock.get("holdout") != expected:
        raise ValueError("Endpoint lock holdout manifest/counts mismatch")
    entry = lock["models"].get(model_name)
    required = {"artifact_sha256", "metadata_sha256", "artifact_path", "metadata_path", "role"}
    if (
        not isinstance(entry, dict)
        or not required <= entry.keys()
        or any(not isinstance(entry[k], str) or not entry[k].strip() for k in required)
    ):
        raise ValueError("Endpoint lock model-name requires complete model entry")
    for key, actual in (("artifact", artifact), ("metadata", artifact.with_suffix(".json"))):
        declared = Path(entry[key + "_path"])
        if not declared.is_absolute():
            declared = Path(path).resolve().parent / declared
        if declared.resolve() != actual.resolve():
            raise ValueError(f"Endpoint lock {key} path mismatch")
        if ft.file_hash(actual) != entry[key + "_sha256"]:
            raise ValueError(f"Endpoint lock {key} checksum mismatch")
    return lock, entry


def _verify_stage(staging, manifest, data_root, config_path, protocol_path, frame):
    """Verify stage_corruptions' schema without invoking it or taking its write lock."""
    record = ft.read_json(staging / "staging.json")
    identity = record.get("identity", {})
    config = rb.load_config(config_path)
    expected = {
        "manifest": str(Path(manifest).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "source": ft.source_identity(manifest, frame, data_root),
        "source_records": frame.to_dict("records"),
        "config": config,
        "config_sha256": ft.file_hash(config_path),
        "protocol_sha256": ft.json_hash(ft.load_protocol(protocol_path)),
        "estimate_type": "fresh_holdout",
        "implementation_sha256": rb._implementation(),
        "extractor_sha256": ft.implementation_hash(),
        "environment": ft.environment(),
        "transform_semantics": rb.corrupt_image.__doc__,
    }
    if (
        record.get("status") != "complete"
        or identity != expected
        or record.get("identity_sha256") != ft.json_hash(identity)
    ):
        raise ValueError("Staging identity/content/config mismatch or incomplete staging")
    manifests = {c["name"]: f"{c['name']}/manifest.csv" for c in rb.CONDITIONS}
    if record.get("manifests") != manifests:
        raise ValueError("Staging must contain exactly the five preregistered conditions")
    required_files = set(manifests.values())
    for name, relative in manifests.items():
        condition = ft.read_manifest(ft.image_file(staging, relative))
        if not {"original_image_path", "condition", "outer_fold"} <= set(condition):
            raise ValueError("Staging lacks original image identity")
        original = condition.original_image_path.map(rb.relative_path)
        if original.duplicated().any() or not condition.condition.eq(name).all():
            raise ValueError("Staging original identity/condition mismatch")
        expected_rows = frame.set_index("image_path").sort_index()
        actual_rows = condition.assign(image_path=original).set_index("image_path").sort_index()
        if not expected_rows.equals(actual_rows[expected_rows.columns]):
            raise ValueError("Staging source coverage/identity mismatch")
        if not condition.image_path.str.startswith(name + "/images/").all():
            raise ValueError("Staging image path outside condition")
        required_files.update(condition.image_path)
    if set(record.get("files", {})) != required_files:
        raise ValueError("Staging checksum inventory incomplete or unexpected")
    rb._verify_files(staging, record["files"])
    return record


def _model_metadata(artifact, entry, family, backend):
    metadata = ft.read_json(artifact.with_suffix(".json"))
    if (
        metadata.get("model_sha256") != entry["artifact_sha256"]
        or metadata.get("class_order") != list(CLASS_NAMES)
        or metadata.get("config_sha256") != ft.json_hash(metadata.get("config"))
    ):
        raise ValueError("Model metadata model/class/config checksum mismatch")
    if family == "resnet":
        if (
            metadata.get("architecture") != "torchvision.models.resnet50"
            or metadata.get("training_stage") != "final"
            or metadata.get("implementation") != backend.implementation_identity()
            or metadata.get("metadata_sha256")
            != ft.json_hash({k: v for k, v in metadata.items() if k != "metadata_sha256"})
            or metadata.get("preprocessing") != metadata["config"].get("preprocessing")
        ):
            raise ValueError("ResNet family/final model/implementation metadata mismatch")
        backend.validate_config(metadata["config"])
    elif (
        metadata.get("kind") != "exploratory-partial-dinov2"
        or metadata.get("source_hashes") != backend.source_hashes()
    ):
        raise ValueError("DINO family/implementation metadata mismatch")
    return metadata


def _source(backend, family, manifest, frame, data_root):
    function = backend.training_identity if family == "resnet" else backend.neural_source_identity
    return function(manifest, frame, data_root)


def _read_predictions(path, run_path, source, metadata, entry, family):
    run = ft.read_json(run_path)
    expected = {
        "status": "complete",
        "evaluation_requested": True,
        "model_sha256": entry["artifact_sha256"],
        "source": source,
        "rows": source["rows"],
        "predictions_sha256": ft.file_hash(path),
    }
    if family == "resnet":
        expected["config_sha256"] = metadata["config_sha256"]
    else:
        expected["metadata_sha256"] = entry["metadata_sha256"]
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed prediction run model/source/config/predictions mismatch")
    predictions = validate_predictions(path, list(CLASS_NAMES))
    if not predictions.outer_fold.isin(["holdout", "-1"]).all():
        raise ValueError("Clean/corrupt predictions must be final holdout evaluation")
    predictions["outer_fold"] = "holdout"
    return predictions


def _join_identity(predictions, manifest_frame):
    """One-to-one checked join; predictors deliberately omit original_image_path."""
    columns = ["class", "material_id", "source_group"]
    expected = manifest_frame.set_index("image_path").sort_index()
    actual = predictions.set_index("image_path").sort_index()
    if not expected.index.equals(actual.index) or not expected[columns].equals(actual[columns]):
        raise ValueError("Prediction source coverage/row identity mismatch")
    mapping = manifest_frame[["image_path"]].copy()
    mapping["original_image_path"] = (
        manifest_frame.original_image_path
        if "original_image_path" in manifest_frame
        else manifest_frame.image_path
    )
    if "original_image_path" in predictions:
        supplied = predictions.set_index("image_path").original_image_path.sort_index()
        if not supplied.equals(mapping.set_index("image_path").original_image_path.sort_index()):
            raise ValueError("Prediction original image identity mismatch")
        predictions = predictions.drop(columns="original_image_path")
    return predictions.merge(mapping, on="image_path", how="left", validate="one_to_one")


def evaluate_staged(
    manifest,
    data_root,
    staging_dir,
    output_dir,
    *,
    family,
    artifact,
    clean_predictions,
    clean_run,
    endpoint_lock,
    model_name,
    device="cpu",
    cpu_threads=4,
    config_path=CONFIG_PATH,
    protocol_path=PROTOCOL_PATH,
):
    """Evaluate five conditions sequentially in a NEW directory; never resume.

    Raw predictor outputs live in <condition>/inference; joined predictions and
    paired metrics live in <condition> so prediction_run checksums remain valid.
    A failure leaves status=failed; retry requires a different output directory.
    """
    output, staging, artifact = Path(output_dir), Path(staging_dir), Path(artifact)
    if output.exists():
        raise FileExistsError(f"Output exists; choose a NEW output directory: {output}")
    if cpu_threads < 1 or device not in ("cpu", "cuda"):
        raise ValueError("Positive cpu_threads and device cpu|cuda required")
    # Never create even a new subdirectory inside immutable staging or clean inputs.
    protected = [staging, Path(clean_run).parent, Path(clean_predictions).parent, Path(data_root)]
    if any(output.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Output must be separate from staging, clean inputs, and source data")
    backend = _backend(family)
    watched = [
        artifact,
        artifact.with_suffix(".json"),
        Path(endpoint_lock),
        Path(clean_predictions),
        Path(clean_run),
        staging / "staging.json",
        Path(manifest),
        Path(config_path),
        Path(protocol_path),
        Path(__file__),
        Path(rb.__file__),
    ]
    hashes = {str(p.resolve()): ft.file_hash(p) for p in watched}
    frame = rb.source_frame(manifest, data_root, "fresh_holdout")
    lock, entry = _lock_entry(
        endpoint_lock, model_name, artifact, manifest, frame, (clean_predictions, clean_run)
    )
    metadata = _model_metadata(artifact, entry, family, backend)
    staged = _verify_stage(staging, manifest, data_root, config_path, protocol_path, frame)
    raw_frame = ft.read_manifest(manifest)
    source = _source(backend, family, manifest, raw_frame, data_root)
    check_fresh = backend.check_fresh if family == "resnet" else backend.verify_fresh
    check_fresh(raw_frame, source, metadata.get("training_source", {}))
    clean = _join_identity(
        _read_predictions(clean_predictions, clean_run, source, metadata, entry, family), frame
    )

    def verify_inputs():
        if any(ft.file_hash(p) != digest for p, digest in hashes.items()):
            raise ValueError("Locked input checksum changed during evaluation")
        _model_metadata(artifact, entry, family, backend)
        if _source(backend, family, manifest, raw_frame, data_root) != source:
            raise ValueError("Original source content changed during evaluation")
        rb._verify_files(staging, staged["files"])

    identity = {
        "schema_version": 1,
        "family": family,
        "model_name": model_name,
        "endpoint_lock": lock,
        "input_sha256": hashes,
        "staging_identity_sha256": staged["identity_sha256"],
        "device": device,
        "cpu_threads": cpu_threads,
        "fitting": "none",
    }
    record = {
        "identity": identity,
        "identity_sha256": ft.json_hash(identity),
        "status": "running",
        "estimate_type": "fresh_holdout",
        "limitations": staged["identity"]["config"]["limitations"],
    }
    output.mkdir(parents=True, exist_ok=False)
    with ft.directory_lock(output):
        run_path = output / "evaluation_run.json"
        ft.write_json(run_path, record)
        try:
            clean.to_csv(output / "clean_predictions.csv", index=False)
            summaries, per_class = [], []
            for condition in rb.CONDITIONS:
                verify_inputs()
                name = condition["name"]
                condition_manifest = staging / staged["manifests"][name]
                directory = output / name
                directory.mkdir()
                inference = directory / "inference"
                kwargs = dict(
                    artifact=artifact,
                    manifest=condition_manifest,
                    data_root=staging,
                    output=inference,
                    evaluate=True,
                    device=device,
                    cpu_threads=cpu_threads,
                )
                if family == "resnet":
                    backend.predict_manifest(**kwargs)
                else:
                    backend.predict_finetuned(
                        **kwargs,
                        expected_sha256=entry["artifact_sha256"],
                        metadata_sha256=entry["metadata_sha256"],
                    )
                condition_frame = ft.read_manifest(condition_manifest)
                condition_source = _source(
                    backend, family, condition_manifest, condition_frame, staging
                )
                predictions = _join_identity(
                    _read_predictions(
                        inference / "predictions.csv",
                        inference / "prediction_run.json",
                        condition_source,
                        metadata,
                        entry,
                        family,
                    ),
                    condition_frame,
                )
                metrics = rb.paired_metrics(clean, predictions, list(CLASS_NAMES))
                metrics.update(condition=name, evaluation_identity_sha256=record["identity_sha256"])
                predictions.to_csv(directory / "predictions.csv", index=False)
                ft.write_json(directory / "metrics.json", metrics)
                summaries.append(
                    {
                        "condition": name,
                        **{k: v for k, v in metrics.items() if isinstance(v, (float, int))},
                        "clean_accuracy": metrics["clean"]["accuracy"],
                        "clean_macro_f1": metrics["clean"]["macro_f1"],
                    }
                )
                per_class.extend({"condition": name, **row} for row in metrics["per_class"])
            comparison = pd.DataFrame(summaries)
            comparison.to_csv(output / "comparison.csv", index=False)
            pd.DataFrame(per_class).to_csv(output / "per_class_recall.csv", index=False)
            rb._plot_comparison(comparison, output / "comparison.png", "fresh_holdout")
            verify_inputs()
            files = {
                p.relative_to(output).as_posix(): ft.file_hash(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p not in (run_path, output / ".writer.lock")
            }
            record.update(status="complete", files=files)
            ft.write_json(run_path, record)
            return record
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            ft.write_json(run_path, record)
            raise
