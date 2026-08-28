"""Generate figures from cached artifacts without running evaluation.

This command is intentionally result-driven: it reads existing histories,
metrics, predictions and embeddings.  It never constructs a DataLoader or
calls a model.  Missing artifacts are reported by ``--dry-run`` and skipped
with warnings during normal resumable generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from material_classifier.plotting import (
    error_gallery,
    material_accuracy_rows,
    plot_calibration,
    plot_confusion_matrix,
    plot_cv,
    plot_efficiency,
    plot_embeddings,
    plot_grouped_f1,
    plot_material_accuracy,
    plot_per_class_metrics,
    plot_roc_pr,
    plot_training_curves,
    project_embeddings,
)

LOGGER = logging.getLogger("generate_visuals")
MODELS = ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16")
TRANSFER_MODELS = ("resnet50", "convnext_tiny", "vit_b16")


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Cannot read %s: %s", path, exc)
        return None


def _prediction(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probability_columns: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        probability_columns = [key for key in (reader.fieldnames or []) if key.startswith("prob_")]
        for row in reader:
            row["confidence"] = float(row["confidence"]) if row.get("confidence") else None
            row["_probability"] = [float(row[key]) for key in probability_columns]
            rows.append(row)
    labels = [row.get("true_label", "") for row in rows]
    classes = [column.removeprefix("prob_") for column in probability_columns]
    if not classes:
        classes = list(dict.fromkeys(labels))
    indices = np.asarray([classes.index(label) for label in labels], dtype=int)
    probabilities = np.asarray([row["_probability"] for row in rows], dtype=float)
    return rows, {"true_indices": indices, "probabilities": probabilities, "classes": classes}


def validate_artifacts(root: Path, split: str = "test") -> list[str]:
    """Return human-readable missing-input diagnostics; no data are loaded."""
    missing: list[str] = []
    results = root / "reports" / "results"
    for model in MODELS:
        for relative in ((model, split, "metrics.json"), (model, split, "predictions.csv")):
            path = results.joinpath(*relative)
            if not path.exists():
                missing.append(str(path))
    if not (results / "cross_validation_results.csv").exists():
        missing.append(str(results / "cross_validation_results.csv"))
    if split.lower() == "test" and not (results / "methodology_lock.json").exists():
        missing.append(str(results / "methodology_lock.json"))
    return missing


def _cached_explainability_target(classes: list[str], row: dict[str, Any]) -> int:
    """Return the cached predicted-class index used by CAM/saliency."""
    predicted = str(row.get("pred_label", ""))
    if predicted not in classes:
        raise ValueError(f"Cached prediction is not in class list: {predicted}")
    return classes.index(predicted)


def _generate_explainability(
    root: Path, classes: list[str], prediction_rows: dict[str, list[dict[str, Any]]], figures: Path
) -> list[Path]:
    """Load transfer checkpoints and explain only selected cached predictions."""
    from PIL import Image
    from torchvision import transforms

    from material_classifier.explainability import (
        assert_explainability_ready,
        generate_curated_gallery,
    )
    from material_classifier.inference import load_model

    generated: list[Path] = []
    for model_name in TRANSFER_MODELS:
        rows = prediction_rows.get(model_name, [])
        checkpoint = root / "artifacts" / "checkpoints" / model_name / "best.pt"
        try:
            metrics_path = assert_explainability_ready(root, model_name)
            model, payload, device = load_model(checkpoint)
            preprocessing = payload.get("resolved_config", {}).get(
                "input_preprocessing", payload.get("config", {}).get("input_preprocessing", {})
            )
            mean = preprocessing.get("normalization_mean", [0.485, 0.456, 0.406])
            std = preprocessing.get("normalization_std", [0.229, 0.224, 0.225])
            crop = int(payload.get("config", {}).get("input_size", 224))
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(crop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            display_transform = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(crop), transforms.ToTensor()]
            )
            selected: list[dict[str, Any]] = []
            for class_name in classes:
                candidates = [
                    row
                    for row in rows
                    if row.get("true_label") == class_name
                    and row.get("true_label") == row.get("pred_label")
                ]
                if candidates:
                    selected.append(max(candidates, key=lambda row: row.get("confidence") or 0))
            errors = [row for row in rows if row.get("true_label") != row.get("pred_label")]
            selected.extend(
                sorted(errors, key=lambda row: row.get("confidence") or 0, reverse=True)[:6]
            )
            records: list[dict[str, Any]] = []
            for row in selected:
                image_path = Path(str(row.get("image_path", "")))
                if not image_path.is_absolute():
                    image_path = root / image_path
                with Image.open(image_path) as handle:
                    raw = handle.convert("RGB")
                    records.append(
                        {
                            **row,
                            "image": transform(raw),
                            "display_image": display_transform(raw),
                            # Explain the model's cached decision.  For
                            # misclassified examples this is pred_label, not
                            # the ground-truth class shown in the title.
                            "target": _cached_explainability_target(classes, row),
                        }
                    )
            output = figures / "explainability" / f"{model_name}_curated.png"
            generated.append(
                generate_curated_gallery(root, model_name, model, records, output, device)
            )
            LOGGER.info("Generated gated explainability gallery from %s", metrics_path)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            LOGGER.warning("Skipping explainability for %s: %s", model_name, exc)
    return generated


def generate(
    root: Path, split: str = "test", dry_run: bool = False, explainability: bool = False
) -> list[Path]:
    missing = validate_artifacts(root, split)
    if dry_run:
        if missing:
            print("Missing optional/prerequisite artifacts:")
            print("\n".join(f"- {path}" for path in missing))
        else:
            print("All reporting prerequisites are present.")
        return []
    results = root / "reports" / "results"
    figures = root / "reports" / "figures"
    classes: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    histories: dict[str, Path] = {}
    generated: list[Path] = []
    for model in MODELS:
        metric_path = results / model / split / "metrics.json"
        prediction_path = results / model / split / "predictions.csv"
        payload = _json(metric_path) if metric_path.exists() else None
        if payload:
            metrics[model] = payload
            if not classes:
                classes = [str(row.get("class")) for row in payload.get("per_class", [])]
            matrix = payload.get("confusion_matrix")
            if matrix and classes:
                generated.append(
                    plot_confusion_matrix(
                        matrix,
                        classes,
                        figures / "confusion_matrices" / f"{model}_raw.png",
                        False,
                        f"{model} — raw confusion",
                    )
                )
                array = np.asarray(matrix, dtype=float)
                array /= np.maximum(array.sum(axis=1, keepdims=True), 1)
                generated.append(
                    plot_confusion_matrix(
                        array,
                        classes,
                        figures / "confusion_matrices" / f"{model}_normalized.png",
                        True,
                        f"{model} — row-normalized confusion",
                    )
                )
            per_class = payload.get("per_class", [])
            if per_class:
                generated.append(
                    plot_per_class_metrics(
                        per_class,
                        figures / "per_class" / f"{model}.png",
                        f"{model} — precision, recall and F1",
                    )
                )
        if prediction_path.exists():
            try:
                rows, cache = _prediction(prediction_path)
                prediction_rows[model], predictions[model] = rows, cache
                if not classes:
                    classes = cache["classes"]
                for mode in ("correct", "incorrect", "lowest_confidence"):
                    result = error_gallery(
                        rows, figures / "errors" / f"{model}_{mode}.png", root, mode
                    )
                    if result:
                        generated.append(result)
                matrix = np.asarray(payload.get("confusion_matrix", []), dtype=float)
                if matrix.ndim == 2 and matrix.shape[0] == len(classes):
                    pairs = sorted(
                        (
                            (float(matrix[i, j]), classes[i], classes[j])
                            for i in range(len(classes))
                            for j in range(len(classes))
                            if i != j
                        ),
                        reverse=True,
                    )
                    for _, true_name, pred_name in pairs[:3]:
                        result = error_gallery(
                            rows,
                            figures / "errors" / f"{model}_pair_{true_name}_vs_{pred_name}.png",
                            root,
                            "incorrect",
                            pair=(true_name, pred_name),
                        )
                        if result:
                            generated.append(result)
                material_path = figures / "materials" / f"{model}_accuracy.png"
                result = plot_material_accuracy(rows, material_path)
                generated.append(result)
                material_rows = material_accuracy_rows(rows)
                table_path = figures / "materials" / f"{model}_accuracy.csv"
                table_path.parent.mkdir(parents=True, exist_ok=True)
                with table_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=("material_id", "n_images", "correct", "accuracy")
                    )
                    writer.writeheader()
                    writer.writerows(sorted(material_rows, key=lambda item: item["accuracy"]))
                generated.append(table_path)
            except (OSError, ValueError) as exc:
                LOGGER.warning("Skipping predictions for %s: %s", model, exc)
        history_path = root / "artifacts" / "checkpoints" / model / "history.csv"
        if history_path.exists():
            histories[model] = history_path
    if histories:
        generated.extend(plot_training_curves(histories, figures))
    if predictions and classes:
        generated.extend(plot_roc_pr(predictions, classes, figures))
        generated.extend(plot_calibration(predictions, figures))
        grouped = {
            name: metrics[name].get("per_class", [])
            for name in metrics
            if metrics[name].get("per_class")
        }
        if grouped:
            generated.append(plot_grouped_f1(grouped, figures / "per_class" / "grouped_f1.png"))
    cv_path = results / "cross_validation_results.csv"
    if cv_path.exists():
        generated.extend(plot_cv(cv_path, figures))
    comparison_path = results / "model_comparison.csv"
    if comparison_path.exists():
        try:
            with comparison_path.open(newline="", encoding="utf-8") as handle:
                comparison_rows = list(csv.DictReader(handle))
            generated.append(
                plot_efficiency(comparison_rows, figures / "efficiency" / "tradeoffs.png")
            )
        except OSError as exc:
            LOGGER.warning("Skipping efficiency figure: %s", exc)
    for model in MODELS:
        embedding_path = results / model / split / "embeddings.npz"
        if not embedding_path.exists():
            continue
        try:
            for method in ("pca", "umap"):
                points, labels, actual = project_embeddings(embedding_path, method)
                generated.append(
                    plot_embeddings(
                        points,
                        labels,
                        figures / "embeddings" / f"{model}_{actual}.png",
                        f"{model} — {actual.upper()} embeddings",
                    )
                )
                if method == "umap" and actual != "umap":
                    break
        except (OSError, ValueError, ImportError) as exc:
            LOGGER.warning("Skipping embedding projection for %s: %s", model, exc)
    if missing:
        LOGGER.warning(
            "Some prerequisites are absent; generated available figures only: %s",
            ", ".join(missing),
        )
    if explainability and split == "test" and classes:
        generated.extend(_generate_explainability(root, classes, prediction_rows, figures))
    elif explainability:
        LOGGER.warning(
            "Explainability requires TEST cached predictions and a valid methodology lock"
        )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate artifacts without writing figures"
    )
    parser.add_argument(
        "--explainability",
        action="store_true",
        help="Generate gated curated CAM/saliency galleries from cached predictions",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = generate(args.root.resolve(), args.split, args.dry_run, args.explainability)
    print(f"Generated {len(paths)} figure(s).")


if __name__ == "__main__":
    main()
