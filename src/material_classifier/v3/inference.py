"""Apply a saved final recipe and fitted head to actual images or a new manifest.

No function here fits or updates a scaler/classifier. Load only trusted joblib
artifacts: joblib is executable Python serialization, not an untrusted exchange format.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

from . import CLASS_NAMES
from .features import (
    assemble_features,
    configure_torch,
    environment,
    extract_batch,
    extract_features,
    file_hash,
    implementation_hash,
    jobs_for_protocol,
    json_hash,
    load_feature_cache,
    load_frozen_model,
    read_json,
    read_manifest,
    transform_image,
    write_json,
)
from .learning import class_probabilities, labels_for, prediction_frame, probability_metrics


def load_artifact(path):
    path = Path(path)
    metadata = read_json(path.with_name(path.stem + "_metadata.json"))
    if file_hash(path) != metadata["model_sha256"]:
        raise ValueError("Final model checksum mismatch")
    current = environment()["packages"]
    fitted = metadata["learning_environment"]["packages"]
    changed = [
        name
        for name in ("scikit-learn", "numpy", "scipy", "joblib")
        if current[name] != fitted[name]
    ]
    if changed:
        raise ValueError(
            f"Saved classifier runtime differs for {changed}; use its training environment"
        )
    bundle = joblib.load(path)
    expected = {key: value for key, value in metadata.items() if key != "model_sha256"}
    if bundle["metadata"] != expected:
        raise ValueError("Final model embedded metadata disagrees with its sidecar")
    if metadata["class_order"] != list(CLASS_NAMES):
        raise ValueError("Final model class order differs from v3")
    if json_hash(metadata["protocol"]) != metadata["protocol_sha256"]:
        raise ValueError("Final model frozen protocol is corrupt")
    if metadata["feature_implementation_sha256"] != implementation_hash():
        raise ValueError("Feature implementation differs from the final model recipe")
    protocol_recipe = next(
        (
            candidate
            for candidate in metadata["protocol"]["candidates"]
            if candidate["name"] == metadata["recipe"]["name"]
        ),
        None,
    )
    if protocol_recipe != metadata["recipe"]:
        raise ValueError("Final recipe is not one of the frozen candidates")
    return bundle


def _check_feature_runtime(metadata, runtime=None):
    current = (runtime if runtime is not None else environment())["packages"]
    reference = metadata["feature_environment"]["packages"]
    changed = [
        name
        for name in ("torch", "torchvision", "timm", "numpy", "Pillow", "safetensors")
        if current[name] != reference[name]
    ]
    if changed:
        raise ValueError(
            f"Frozen feature runtime differs for {changed}; use the recorded environment"
        )


def predict_image(
    artifact, image_path, *, device="cpu", cpu_threads=4, offline=False, model_factory=None
):
    """Return class-order probabilities for a real image, extracting only the saved recipe."""
    bundle = load_artifact(artifact)
    metadata = bundle["metadata"]
    _check_feature_runtime(metadata)
    device = configure_torch(device, cpu_threads)
    factory = model_factory or load_frozen_model
    base = {}
    for prefix, spec, size in jobs_for_protocol(metadata["protocol"], metadata["recipe"]["parts"]):
        model = factory(spec, size, device=device, offline=offline)
        model.eval().requires_grad_(False)
        with Image.open(image_path) as image:
            batch = transform_image(image, spec, size)[None].to(device)
        base.update(
            {
                prefix + "_" + key: value
                for key, value in extract_batch(model, batch, spec, size).items()
            }
        )
        del model, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    X = assemble_features(base, metadata["recipe"]["parts"])
    if X.shape[1] != metadata["input_features"]:
        raise ValueError("Extracted image feature width differs from final head")
    with threadpool_limits(limits=min(cpu_threads, os.cpu_count() or 1, 8)):
        probabilities = class_probabilities(bundle["pipeline"], X)[0]
    return {
        "image_path": str(image_path),
        "image_sha256": file_hash(image_path),
        "y_pred": int(probabilities.argmax()),
        "predicted_class": CLASS_NAMES[int(probabilities.argmax())],
        "probabilities": {
            name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "model_sha256": file_hash(artifact),
        "protocol_sha256": metadata["protocol_sha256"],
        "recipe": metadata["recipe"],
    }


def predict_manifest(
    artifact,
    manifest,
    data_root,
    output,
    *,
    feature_cache=None,
    device="cpu",
    batch_size=16,
    cpu_threads=4,
    offline=False,
    evaluate=False,
    model_factory=None,
):
    """Predict a new manifest without fitting. An existing cache must match its content.

    If no cache exists, extract all predeclared base arrays, allowing the selected
    and control artifacts to share exactly the same cache. Evaluation is explicit
    and requires groups and image contents disjoint from all development data.
    """
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    bundle = load_artifact(artifact)
    metadata = bundle["metadata"]
    _check_feature_runtime(metadata)
    frame = read_manifest(manifest)
    training_source = metadata["training_source"]
    # Older custom/test artifacts can omit training group inventory; those may
    # predict, but must not claim a fresh independent evaluation.
    if evaluate:
        if "source_groups" not in training_source:
            raise ValueError(
                "Artifact lacks training source groups needed for holdout verification"
            )
        if set(frame["source_group"]) & set(training_source["source_groups"]):
            raise ValueError("Fresh evaluation source groups overlap development data")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "model_sha256": file_hash(artifact),
        "protocol_sha256": metadata["protocol_sha256"],
        "manifest_sha256": file_hash(manifest),
        "recipe": metadata["recipe"],
        "evaluation_requested": evaluate,
        "environment": environment(),
        "fitting": "none; frozen saved scaler and classifier only",
    }
    write_json(output / "prediction_run.json", record)
    try:
        cache_path = Path(feature_cache) if feature_cache is not None else output / "features"
        if not cache_path.exists():
            extract_features(
                manifest,
                data_root,
                cache_path,
                metadata["protocol"],
                batch_size=batch_size,
                device=device,
                threads=cpu_threads,
                offline=offline,
                model_factory=model_factory,
            )
        base, cache = load_feature_cache(
            cache_path, manifest=manifest, data_root=data_root, protocol=metadata["protocol"]
        )
        _check_feature_runtime(metadata, cache["identity"]["environment"])
        if evaluate:
            training_hashes = {source["sha256"] for source in training_source["sources"]}
            if any(
                source["sha256"] in training_hashes
                for source in cache["identity"]["source"]["sources"]
            ):
                raise ValueError("Fresh evaluation contains image content seen in development")
        X = assemble_features(base, metadata["recipe"]["parts"])
        if X.shape[1] != metadata["input_features"]:
            raise ValueError("Cached feature width differs from final head")
        with threadpool_limits(limits=min(cpu_threads, os.cpu_count() or 1, 8)):
            probabilities = class_probabilities(bundle["pipeline"], X)
        predictions = prediction_frame(frame, probabilities)
        predictions["outer_fold"] = "holdout" if evaluate else "prediction_only"
        predictions.to_csv(output / "predictions.csv", index=False)
        if evaluate:
            write_json(
                output / "metrics.json",
                {
                    **probability_metrics(labels_for(frame), probabilities),
                    "interpretation": "Fresh manifest evaluated with fixed development artifact; "
                    "no scaler, head, calibration, or model selection fitted on these data.",
                },
            )
        record.update(
            {
                "status": "complete",
                "rows": len(frame),
                "feature_cache": str(cache_path.resolve()),
                "feature_cache_identity_sha256": cache["identity_sha256"],
                "predictions_sha256": file_hash(output / "predictions.csv"),
            }
        )
        write_json(output / "prediction_run.json", record)
        return predictions
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(output / "prediction_run.json", record)
        raise
