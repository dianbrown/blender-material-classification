"""Preregistered paired sensitivity checks; never fit a scaler or classifier.

Staging is CPU-only. Evaluation requires an explicit locked clean prediction run.
All generated paths are relative to the staging directory; original paths remain
available for paired joins. Only trusted local joblib artifacts may be loaded.
"""

from __future__ import annotations

import io
from pathlib import Path, PurePosixPath, PureWindowsPath

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from . import CLASS_NAMES
from . import features as ft
from .evaluation import validate_predictions

CONDITIONS = [
    {"name": "brightness_low", "operation": "brightness", "factor": 0.6},
    {"name": "brightness_high", "operation": "brightness", "factor": 1.4},
    {"name": "gaussian_blur", "operation": "gaussian_blur", "radius": 1.5},
    {"name": "jpeg_quality30", "operation": "jpeg", "quality": 30},
    {"name": "resolution128", "operation": "downsample", "size": 128},
]
ESTIMATES = ("fresh_holdout", "development_diagnostic")


def relative_path(value):
    """Reject absolute paths and traversal, including Windows syntax on Unix."""
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if (
        not value.strip()
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in path.parts
        or str(path) == "."
    ):
        raise ValueError(f"Expected a contained relative image path: {value}")
    return path.as_posix()


def load_config(path):
    config = ft.read_json(path)
    if (
        config.get("schema_version") != 1
        or config.get("declared_before_v3_results") is not True
        or config.get("conditions") != CONDITIONS
        or config.get("seed") != 62643
    ):
        raise ValueError("Robustness config differs from the five preregistered conditions")
    return config


def corrupt_image(image, condition):
    """Transform EXIF-oriented original RGB pixels, before frozen preprocessing.

    JPEG uses Pillow quality=30, subsampling=2, optimize=False, progressive=False;
    decoded pixels are stored losslessly. Resolution128 reduces the shorter edge
    to 128 with bicubic resampling, preserves aspect ratio (floor longer edge),
    and never upsamples an already smaller image. Frozen preprocessing resizes it.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown or modified preregistered corruption")
    image = ImageOps.exif_transpose(image).convert("RGB")
    operation = condition["operation"]
    if operation == "brightness":
        return ImageEnhance.Brightness(image).enhance(condition["factor"])
    if operation == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(condition["radius"]))
    if operation == "jpeg":
        buffer = io.BytesIO()
        image.save(
            buffer, format="JPEG", quality=30, subsampling=2, optimize=False, progressive=False
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    width, height = image.size
    scale = min(1.0, condition["size"] / min(width, height))
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC
    )


def source_frame(manifest, data_root, estimate_type):
    if estimate_type not in ESTIMATES:
        raise ValueError(f"Explicit estimate_type required: {ESTIMATES}")
    frame = ft.read_manifest(manifest)
    frame["image_path"] = frame.image_path.map(relative_path)
    resolved = [ft.image_file(data_root, path) for path in frame.image_path]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Source manifest contains aliased image paths")
    if frame.groupby("material_id").source_group.nunique().gt(1).any():
        raise ValueError("A material_id belongs to multiple source groups")
    if (
        estimate_type == "fresh_holdout"
        and not frame.original_split.isin(["holdout", "fresh_holdout", "independent_holdout"]).all()
    ):
        raise ValueError(
            "Fresh holdout requires explicit holdout original_split, not old test data"
        )
    if "outer_fold" in frame and not frame.outer_fold.isin(["holdout", "-1"]).all():
        raise ValueError("Fixed final-model sensitivity requires holdout outer_fold, not OOF")
    frame["outer_fold"] = "holdout"
    return frame


def _implementation():
    return ft.json_hash(Path(__file__).read_text(encoding="utf-8"))


def _verify_files(directory, files):
    for name, expected in files.items():
        path = ft.image_file(directory, relative_path(name))
        if ft.file_hash(path) != expected:
            raise ValueError(f"Staged/output checksum mismatch: {name}")


def stage_corruptions(
    manifest, data_root, output, config_path, protocol, *, estimate_type, resume=False
):
    """Materialize all five conditions without loading models or starting CUDA."""
    config = load_config(config_path)
    frame = source_frame(manifest, data_root, estimate_type)
    source = ft.source_identity(manifest, frame, data_root)
    identity = {
        "manifest": str(Path(manifest).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "source": source,
        "source_records": frame.to_dict(orient="records"),
        "config": config,
        "config_sha256": ft.file_hash(config_path),
        "protocol_sha256": ft.json_hash(protocol),
        "estimate_type": estimate_type,
        "implementation_sha256": _implementation(),
        "extractor_sha256": ft.implementation_hash(),
        "environment": ft.environment(),
        "transform_semantics": corrupt_image.__doc__,
    }
    output = Path(output)
    existed = output.exists()
    if existed and not resume:
        raise FileExistsError(f"Output exists: {output}; use --resume for exact verified reuse")
    output.mkdir(parents=True, exist_ok=True)
    with ft.directory_lock(output):
        record_path = output / "staging.json"
        if existed:
            if not record_path.is_file():
                raise ValueError("Incomplete staging directory; choose a new output")
            record = ft.read_json(record_path)
            if record.get("identity") != identity or record.get("identity_sha256") != ft.json_hash(
                identity
            ):
                raise ValueError("Stale staging identity; choose a new output")
            _verify_files(output, record["files"])
            return record
        files, manifests = {}, {}
        for condition in config["conditions"]:
            name = condition["name"]
            directory = output / name
            (directory / "images").mkdir(parents=True)
            transformed = frame.copy()
            transformed["original_image_path"] = frame.image_path
            transformed["condition"] = name
            paths = []
            for index, row in frame.iterrows():
                target = directory / "images" / f"{index:08d}.png"
                with Image.open(ft.image_file(data_root, row.image_path)) as image:
                    corrupt_image(image, condition).save(target, format="PNG")
                relative = target.relative_to(output).as_posix()
                paths.append(relative)
                files[relative] = ft.file_hash(target)
            transformed["image_path"] = paths
            target = directory / "manifest.csv"
            transformed.to_csv(target, index=False)
            relative = target.relative_to(output).as_posix()
            manifests[name] = relative
            files[relative] = ft.file_hash(target)
        if ft.source_identity(manifest, frame, data_root) != source:
            raise ValueError("Source changed during staging; discard this staging directory")
        record = {
            "identity": identity,
            "identity_sha256": ft.json_hash(identity),
            "manifests": manifests,
            "files": files,
            "status": "complete",
        }
        ft.write_json(record_path, record)
        return record


def load_model(path, protocol):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"pipeline", "metadata"} <= bundle.keys():
        raise ValueError("Expected final {'pipeline', 'metadata'} model artifact")
    metadata, pipeline = bundle["metadata"], bundle["pipeline"]
    if metadata.get("protocol") != protocol or metadata.get("class_order") != list(CLASS_NAMES):
        raise ValueError("Model protocol/class order differs from frozen protocol")
    if metadata.get("recipe") not in protocol["candidates"]:
        raise ValueError("Model recipe is not a frozen candidate")
    if metadata.get("protocol_sha256", ft.json_hash(protocol)) != ft.json_hash(protocol):
        raise ValueError("Model protocol checksum mismatch")
    if metadata.get("feature_implementation_sha256") != ft.implementation_hash():
        raise ValueError("Model extractor identity is missing or differs")
    reference = metadata.get("feature_environment", {}).get("packages")
    current = ft.environment()["packages"]
    if reference is None or any(
        reference.get(key) != current[key]
        for key in ("torch", "torchvision", "timm", "numpy", "Pillow", "safetensors")
    ):
        raise ValueError("Model feature runtime is missing or differs")
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) < 2:
        raise ValueError("Final model must contain a fitted scaler/head Pipeline")
    for _, step in pipeline.steps:
        check_is_fitted(step)
    sidecar = Path(path).with_name(Path(path).stem + "_metadata.json")
    if sidecar.exists():
        saved = ft.read_json(sidecar)
        if saved != {**metadata, "model_sha256": ft.file_hash(path)}:
            raise ValueError("Final model metadata sidecar/checksum mismatch")
    return bundle


def check_training_overlap(frame, source, metadata, estimate_type):
    training = metadata.get("training_source", {})
    if estimate_type == "fresh_holdout" and not (
        training.get("source_groups") and training.get("sources")
    ):
        raise ValueError("Fresh holdout requires training source groups and content hashes")
    overlap = {
        "source_groups": sorted(set(frame.source_group) & set(training.get("source_groups", []))),
        "material_ids": sorted(set(frame.material_id) & set(training.get("material_ids", []))),
        "image_paths": sorted(
            set(frame.image_path) & {relative_path(p) for p in training.get("image_paths", [])}
        ),
        "content_sha256": sorted(
            {r["sha256"] for r in source["sources"]}
            & {r["sha256"] for r in training.get("sources", [])}
        ),
    }
    if estimate_type == "fresh_holdout" and any(overlap.values()):
        raise ValueError(f"Fresh holdout has identifiable training overlap: {overlap}")
    return overlap


def frozen_probabilities(pipeline, matrix, classes):
    """Map sklearn probability columns to protocol order without fitting anything."""
    learned = np.asarray(pipeline.classes_)
    if learned.ndim != 1 or len(learned) != len(classes):
        raise ValueError("Fitted head must cover every protocol class exactly once")
    if all(isinstance(value, (str, np.str_)) for value in learned):
        names = learned.tolist()
    else:
        if not np.issubdtype(learned.dtype, np.integer) or set(learned) != set(range(len(classes))):
            raise ValueError("Fitted class labels must be names or zero-based integer indices")
        names = [classes[int(value)] for value in learned]
    if set(names) != set(classes):
        raise ValueError("Fitted head class mapping differs from protocol")
    probabilities = np.asarray(pipeline.predict_proba(matrix), dtype=float)
    if probabilities.shape != (len(matrix), len(classes)):
        raise ValueError("Invalid probability matrix shape")
    probabilities = probabilities[:, [names.index(name) for name in classes]]
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid frozen-head probabilities")
    return probabilities


def paired_frames(clean, corrupt, classes):
    clean = validate_predictions(clean, classes)
    corrupt = validate_predictions(corrupt, classes)
    for frame in (clean, corrupt):
        column = "original_image_path" if "original_image_path" in frame else "image_path"
        frame["pair_path"] = frame[column].map(relative_path)
        if frame.pair_path.duplicated().any():
            raise ValueError("Duplicate original image paths in paired predictions")
    if set(clean.pair_path) != set(corrupt.pair_path):
        raise ValueError("Paired predictions must contain identical original image paths")
    clean = clean.set_index("pair_path").sort_index()
    corrupt = corrupt.set_index("pair_path").sort_index()
    for column in ("class", "material_id", "source_group", "y_true", "outer_fold"):
        if not clean[column].equals(corrupt[column]):
            raise ValueError(f"Paired predictions differ in {column}")
    return clean, corrupt


def paired_metrics(clean, corrupt, classes):
    clean, corrupt = paired_frames(clean, corrupt, classes)

    def scores(frame):
        return {
            "accuracy": float(accuracy_score(frame.y_true, frame.y_pred)),
            "macro_f1": float(
                f1_score(
                    frame.y_true, frame.y_pred, labels=classes, average="macro", zero_division=0
                )
            ),
        }

    reference, altered = scores(clean), scores(corrupt)
    recalls = recall_score(
        corrupt.y_true, corrupt.y_pred, labels=classes, average=None, zero_division=0
    )
    clean_recalls = recall_score(
        clean.y_true, clean.y_pred, labels=classes, average=None, zero_division=0
    )
    correct_clean = clean.y_true == clean.y_pred
    correct_corrupt = corrupt.y_true == corrupt.y_pred
    return {
        "n_pairs": len(clean),
        **altered,
        "clean": reference,
        "accuracy_drop_from_clean": reference["accuracy"] - altered["accuracy"],
        "macro_f1_drop_from_clean": reference["macro_f1"] - altered["macro_f1"],
        "correct_to_incorrect": int((correct_clean & ~correct_corrupt).sum()),
        "incorrect_to_correct": int((~correct_clean & correct_corrupt).sum()),
        "per_class": [
            {
                "class": name,
                "support": int((clean.y_true == name).sum()),
                "recall": float(recalls[i]),
                "clean_recall": float(clean_recalls[i]),
                "recall_drop_from_clean": float(clean_recalls[i] - recalls[i]),
            }
            for i, name in enumerate(classes)
        ],
        "missing_class_convention": "Fixed class order; undefined recall/F1 set to zero",
    }


def evaluate_staged(
    manifest,
    data_root,
    output,
    config_path,
    protocol,
    *,
    model,
    clean_predictions,
    clean_run,
    estimate_type,
    endpoint_locked=False,
    device="cpu",
    batch_size=16,
    cpu_threads=4,
    offline=False,
    resume=False,
):
    """Evaluate sequentially only after caller explicitly acknowledges endpoint lock."""
    if not endpoint_locked:
        raise ValueError("Evaluation requires explicit --endpoint-locked")
    if cpu_threads < 1 or batch_size < 1:
        raise ValueError("Thread count and batch size must be positive")
    output = Path(output)
    # Verify staging before touching the model or any scores.
    staged = stage_corruptions(
        manifest, data_root, output, config_path, protocol, estimate_type=estimate_type, resume=True
    )
    bundle = load_model(model, protocol)
    metadata = bundle["metadata"]
    frame = source_frame(manifest, data_root, estimate_type)
    overlap = check_training_overlap(frame, staged["identity"]["source"], metadata, estimate_type)
    lock = ft.read_json(clean_run)
    expected = {
        "status": "complete",
        "model_sha256": ft.file_hash(model),
        "manifest_sha256": ft.file_hash(manifest),
        "protocol_sha256": ft.json_hash(protocol),
        "predictions_sha256": ft.file_hash(clean_predictions),
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise ValueError("Clean prediction run does not match locked model/manifest/predictions")
    if not lock.get("feature_cache") or not lock.get("feature_cache_identity_sha256"):
        raise ValueError("Clean prediction run lacks content-verified feature cache provenance")
    _, clean_cache = ft.load_feature_cache(
        lock["feature_cache"], manifest=manifest, data_root=data_root, protocol=protocol
    )
    if clean_cache["identity_sha256"] != lock["feature_cache_identity_sha256"]:
        raise ValueError("Clean prediction feature cache identity mismatch")
    classes = metadata["class_order"]
    clean = validate_predictions(clean_predictions, classes)
    if not clean.outer_fold.isin(["holdout", "-1"]).all():
        raise ValueError("Clean reference must use the final fitted head, not OOF predictions")
    clean["outer_fold"] = "holdout"
    # Check full source identity, not just common rows or row order.
    expected_rows = frame.set_index("image_path").sort_index()
    clean["image_path"] = clean.image_path.map(relative_path)
    actual = clean.set_index("image_path").sort_index()
    if not expected_rows.index.equals(actual.index) or any(
        not expected_rows[c].equals(actual[c]) for c in ("class", "material_id", "source_group")
    ):
        raise ValueError("Clean reference differs from explicit source manifest")
    clean["original_image_path"] = clean.image_path
    identity = {
        "staging_sha256": staged["identity_sha256"],
        **expected,
        "model": str(Path(model).resolve()),
        "clean_predictions": str(Path(clean_predictions).resolve()),
        "clean_run": str(Path(clean_run).resolve()),
        "model_metadata_sha256": ft.json_hash(metadata),
        "clean_run_sha256": ft.file_hash(clean_run),
        "endpoint_locked": True,
        "estimate_type": estimate_type,
        "device": device,
        "batch_size": batch_size,
        "cpu_threads": cpu_threads,
        "offline": offline,
    }
    with ft.directory_lock(output):
        record_path = output / "evaluation_run.json"
        if record_path.exists():
            old = ft.read_json(record_path)
            if (
                not resume
                or old["identity"] != identity
                or old.get("identity_sha256") != ft.json_hash(identity)
            ):
                raise ValueError(
                    "Evaluation exists or identity differs; explicit exact resume required"
                )
            if old["status"] == "complete":
                _verify_files(output, old["files"])
                return old
        record = {
            "identity": identity,
            "identity_sha256": ft.json_hash(identity),
            "status": "running",
            "training_overlap": overlap,
            "interpretation": (
                "Fresh holdout paired sensitivity; external provenance is "
                "still required to establish independence."
                if estimate_type == "fresh_holdout"
                else "Development-only sensitivity of a fixed final head; "
                "may be in-sample, not OOF or unseen-test evaluation."
            ),
            "limitations": staged["identity"]["config"]["limitations"],
        }
        ft.write_json(record_path, record)
        summaries, per_class, files = [], [], {}
        clean.to_csv(output / "clean_predictions.csv", index=False)
        files["clean_predictions.csv"] = ft.file_hash(output / "clean_predictions.csv")
        before = joblib.hash(bundle["pipeline"])
        try:
            for name, relative in staged["manifests"].items():
                condition_manifest = output / relative
                directory = condition_manifest.parent
                cache_path = directory / "features"
                ft.extract_features(
                    condition_manifest,
                    output,
                    cache_path,
                    protocol,
                    parts=metadata["recipe"]["parts"],
                    device=device,
                    batch_size=batch_size,
                    threads=cpu_threads,
                    offline=offline,
                    resume=resume and cache_path.exists(),
                )
                base, cache = ft.load_feature_cache(
                    cache_path, manifest=condition_manifest, data_root=output, protocol=protocol
                )
                matrix = ft.assemble_features(base, metadata["recipe"]["parts"])
                with threadpool_limits(limits=min(cpu_threads, 8)):
                    probabilities = frozen_probabilities(bundle["pipeline"], matrix, classes)
                if joblib.hash(bundle["pipeline"]) != before:
                    raise ValueError("Frozen pipeline mutated during prediction")
                predictions = ft.read_manifest(condition_manifest)
                predictions["y_true"] = predictions["class"]
                predictions["y_pred"] = np.asarray(classes)[probabilities.argmax(axis=1)]
                for i, label in enumerate(classes):
                    predictions[f"p_{label}"] = probabilities[:, i]
                metrics = paired_metrics(clean, predictions, classes)
                metrics.update(
                    {
                        "condition": name,
                        "estimate_type": estimate_type,
                        "feature_cache_identity_sha256": cache["identity_sha256"],
                        "evaluation_identity_sha256": record["identity_sha256"],
                    }
                )
                predictions.to_csv(directory / "predictions.csv", index=False)
                ft.write_json(directory / "metrics.json", metrics)
                summaries.append(
                    {
                        "condition": name,
                        "estimate_type": estimate_type,
                        **{k: v for k, v in metrics.items() if isinstance(v, (float, int))},
                        "clean_accuracy": metrics["clean"]["accuracy"],
                        "clean_macro_f1": metrics["clean"]["macro_f1"],
                    }
                )
                per_class.extend(
                    {"condition": name, "estimate_type": estimate_type, **row}
                    for row in metrics["per_class"]
                )
                for filename in ("predictions.csv", "metrics.json"):
                    path = directory / filename
                    files[path.relative_to(output).as_posix()] = ft.file_hash(path)
            comparison = pd.DataFrame(summaries)
            comparison.to_csv(output / "comparison.csv", index=False)
            pd.DataFrame(per_class).to_csv(output / "per_class_recall.csv", index=False)
            _plot_comparison(comparison, output / "comparison.png", estimate_type)
            for filename in ("comparison.csv", "per_class_recall.csv", "comparison.png"):
                files[filename] = ft.file_hash(output / filename)
            # Refuse to complete a run whose inputs changed during extraction.
            if (
                ft.file_hash(model) != expected["model_sha256"]
                or ft.file_hash(clean_predictions) != expected["predictions_sha256"]
            ):
                raise ValueError("Locked model/clean predictions changed during evaluation")
            if ft.source_identity(manifest, frame, data_root) != staged["identity"]["source"]:
                raise ValueError("Original source content changed during evaluation")
            _verify_files(output, staged["files"])
            record.update(status="complete", files=files)
            ft.write_json(record_path, record)
            return record
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            ft.write_json(record_path, record)
            raise


def _plot_comparison(frame, path, estimate_type):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    y = np.arange(len(frame))
    for ax, metric, color in zip(axes, ("accuracy", "macro_f1"), ("#28658A", "#A76334")):
        ax.barh(y, frame[metric], color=color)
        ax.axvline(
            frame["clean_" + metric].iloc[0],
            color="#333333",
            ls="--",
            label="Locked clean reference",
        )
        ax.set(yticks=y, yticklabels=frame.condition, xlim=(0, 1), xlabel=metric.replace("_", " "))
        ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired corruption sensitivity | " + estimate_type.replace("_", " "))
    fig.savefig(path, dpi=180)
    plt.close(fig)
