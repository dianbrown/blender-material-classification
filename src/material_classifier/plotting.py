"""Publication-quality, result-driven figures for the material study.

The functions in this module intentionally operate on plain mappings, CSVs and
NPZ files.  They do not know how to train or evaluate a model, which makes the
visualisation stage safe to rerun and prevents accidental TEST evaluation.
Missing optional fields are skipped with an explicit warning rather than being
filled with invented values.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_COLORS = ("#2f6690", "#3a7d44", "#f4a261", "#e76f51", "#7b2cbf", "#008c95", "#bc6c25")
DEFAULT_CLASS_NAMES = ("wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save a figure atomically enough for resumable report generation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def _rows(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        with Path(source).open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return [dict(row) for row in source]


def _number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() in {"", "none", "nan", "null"}:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _history_rows(history: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = _rows(history)
    rows.sort(key=lambda row: _number(row.get("epoch")) or 0)
    return rows


def plot_training_curves(
    histories: Mapping[str, str | Path | Iterable[Mapping[str, Any]]],
    output_dir: str | Path,
) -> list[Path]:
    """Write per-model and combined loss/accuracy/F1/learning-rate curves."""
    _style()
    out = Path(output_dir)
    paths: list[Path] = []
    metrics = (
        ("loss", "Loss", "loss"),
        ("accuracy", "Accuracy", "accuracy"),
        ("macro_f1", "Macro F1", "f1"),
    )
    parsed = {name: _history_rows(value) for name, value in histories.items()}
    for model, rows in parsed.items():
        if not rows:
            LOGGER.warning("No history rows for %s", model)
            continue
        epochs = [_number(row.get("epoch")) or i + 1 for i, row in enumerate(rows)]
        for key, title, suffix in metrics:
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            train_key, val_key = f"train_{key}", f"val_{key}"
            if key == "macro_f1":
                train_key, val_key = "train_macro_f1", "val_macro_f1"
            train = [_number(row.get(train_key)) for row in rows]
            valid = [_number(row.get(val_key)) for row in rows]
            if not any(value is not None for value in train + valid):
                plt.close(fig)
                continue
            if any(value is not None for value in train):
                ax.plot(epochs, train, marker="o", ms=2.5, label="Train", color=DEFAULT_COLORS[0])
            if any(value is not None for value in valid):
                ax.plot(
                    epochs, valid, marker="o", ms=2.5, label="Validation", color=DEFAULT_COLORS[3]
                )
            ax.set(
                title=f"{model.replace('_', ' ').title()} — {title}", xlabel="Epoch", ylabel=title
            )
            ax.legend()
            paths.append(save_figure(fig, out / "training" / f"{model}_{suffix}.png"))
        learning = [_number(row.get("learning_rate")) for row in rows]
        if any(value is not None for value in learning):
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.plot(epochs, learning, marker="o", ms=2.5, color=DEFAULT_COLORS[4])
            ax.set(
                title=f"{model.replace('_', ' ').title()} — Learning rate",
                xlabel="Epoch",
                ylabel="Learning rate",
            )
            ax.set_yscale("log")
            paths.append(save_figure(fig, out / "training" / f"{model}_learning_rate.png"))
    for key, title, suffix in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        plotted = False
        for index, (model, rows) in enumerate(parsed.items()):
            epochs = [_number(row.get("epoch")) or i + 1 for i, row in enumerate(rows)]
            column = f"val_{key}" if key != "macro_f1" else "val_macro_f1"
            values = [_number(row.get(column)) for row in rows]
            if not rows or not any(value is not None for value in values):
                continue
            ax.plot(
                epochs,
                values,
                label=model.replace("_", " ").title(),
                color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
            )
            plotted = True
        if plotted:
            ax.set(title=f"Validation {title} comparison", xlabel="Epoch", ylabel=title)
            ax.legend(ncol=2)
            paths.append(save_figure(fig, out / "training" / f"combined_val_{suffix}.png"))
        else:
            plt.close(fig)
    return paths


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]],
    class_names: Sequence[str],
    path: str | Path,
    normalized: bool = False,
    title: str | None = None,
) -> Path:
    """Render raw or row-normalised confusion counts."""
    _style()
    data = np.asarray(matrix, dtype=float)
    if data.shape != (len(class_names), len(class_names)):
        raise ValueError("confusion matrix shape does not match class_names")
    if normalized and data.size and np.nanmax(data) > 1.0:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=ax, fraction=0.046)
    threshold = data.max() * 0.55 if data.size else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.0%}" if normalized else f"{data[i, j]:.0f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if data[i, j] > threshold else "#16324f",
                fontsize=8,
            )
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set(
        xlabel="Predicted label",
        ylabel="True label",
        title=title or ("Normalized confusion matrix" if normalized else "Confusion matrix"),
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return save_figure(fig, path)


def plot_per_class_metrics(
    per_class: Sequence[Mapping[str, Any]], path: str | Path, title: str = "Per-class metrics"
) -> Path:
    """Render precision, recall and F1 together, preserving supplied values."""
    _style()
    classes = [str(row.get("class", row.get("label", i))) for i, row in enumerate(per_class)]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.25
    for offset, metric in enumerate(("precision", "recall", "f1")):
        values = [_number(row.get(metric)) or 0.0 for row in per_class]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=metric.upper(),
            color=DEFAULT_COLORS[offset + 1],
        )
    ax.set(title=title, ylabel="Score", ylim=(0, 1.05), xticks=x, xticklabels=classes)
    ax.legend(ncol=3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return save_figure(fig, path)


def plot_grouped_f1(per_model: Mapping[str, Sequence[Mapping[str, Any]]], path: str | Path) -> Path:
    """Compare per-class F1 across models."""
    _style()
    labels = [
        str(row.get("class", row.get("label", i)))
        for i, row in enumerate(next(iter(per_model.values()), []))
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    models = list(per_model)
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        mapping = {
            str(row.get("class", row.get("label", j))): _number(row.get("f1")) or 0.0
            for j, row in enumerate(per_model[model])
        }
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            [mapping.get(label, np.nan) for label in labels],
            width,
            label=model.replace("_", " ").title(),
            color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
        )
    ax.set(
        title="Per-class F1 comparison by material class",
        ylabel="F1",
        ylim=(0, 1.05),
        xticks=x,
        xticklabels=labels,
    )
    ax.legend(ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return save_figure(fig, path)


def plot_roc_pr(
    predictions: Mapping[str, Mapping[str, Any]],
    class_names: Sequence[str],
    output_dir: str | Path,
) -> list[Path]:
    """Create macro ROC and per-class ROC/PR plots from cached probabilities."""
    _style()
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    out = Path(output_dir)
    paths: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for index, (model, payload) in enumerate(predictions.items()):
        y = np.asarray(payload.get("true_indices", payload.get("true_label", [])), dtype=int)
        probabilities = np.asarray(payload.get("probabilities", []), dtype=float)
        if probabilities.ndim != 2 or not len(y):
            continue
        if (
            probabilities.shape[1] != len(class_names)
            or int(y.min()) < 0
            or int(y.max()) >= len(class_names)
        ):
            LOGGER.warning("Skipping malformed probability artifact for %s", model)
            continue
        onehot = np.eye(len(class_names))[y]
        grid = np.linspace(0.0, 1.0, 201)
        roc_values = []
        pr_values = []
        for class_index in range(len(class_names)):
            target = onehot[:, class_index]
            if len(np.unique(target)) < 2:
                continue
            fpr_class, tpr_class, _ = roc_curve(target, probabilities[:, class_index])
            roc_values.append(np.interp(grid, fpr_class, tpr_class))
            precision_class, recall_class, _ = precision_recall_curve(
                target, probabilities[:, class_index]
            )
            order = np.argsort(recall_class)
            pr_values.append(np.interp(grid, recall_class[order], precision_class[order]))
        if not roc_values:
            continue
        fpr, tpr = grid, np.mean(roc_values, axis=0)
        axes[0].plot(
            fpr,
            tpr,
            label=f"{model.replace('_', ' ').title()} ({auc(fpr, tpr):.3f})",
            color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
        )
        recall, precision = grid, np.mean(pr_values, axis=0)
        axes[1].plot(
            recall,
            precision,
            label=model.replace("_", " ").title(),
            color=DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.45)
    axes[0].set(
        title="Macro ROC comparison", xlabel="False positive rate", ylabel="True positive rate"
    )
    axes[1].set(
        title="Macro precision–recall comparison",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for ax in axes:
        ax.legend(fontsize=8)
    paths.append(save_figure(fig, out / "roc_pr" / "macro_comparison.png"))
    for model, payload in predictions.items():
        y = np.asarray(payload.get("true_indices", payload.get("true_label", [])), dtype=int)
        probabilities = np.asarray(payload.get("probabilities", []), dtype=float)
        if probabilities.ndim != 2 or not len(y):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        for i, name in enumerate(class_names):
            target = (y == i).astype(int)
            if len(np.unique(target)) < 2:
                continue
            fpr, tpr, _ = roc_curve(target, probabilities[:, i])
            precision, recall, _ = precision_recall_curve(target, probabilities[:, i])
            axes[0].plot(fpr, tpr, label=name, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            axes[1].plot(
                recall, precision, label=name, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            )
        axes[0].set(
            title=f"{model} — one-vs-rest ROC",
            xlabel="False positive rate",
            ylabel="True positive rate",
        )
        axes[1].set(
            title=f"{model} — one-vs-rest PR",
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        for ax in axes:
            ax.legend(fontsize=7, ncol=2)
        paths.append(save_figure(fig, out / "roc_pr" / f"{model}_per_class.png"))
    return paths


def plot_calibration(
    predictions: Mapping[str, Mapping[str, Any]], output_dir: str | Path, bins: int = 10
) -> list[Path]:
    """Render reliability and confidence distributions from cached probabilities."""
    _style()
    out = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for index, (model, payload) in enumerate(predictions.items()):
        y = np.asarray(payload.get("true_indices", []), dtype=int)
        p = np.asarray(payload.get("probabilities", []), dtype=float)
        if p.ndim != 2 or not len(y):
            continue
        confidence = p.max(1)
        correct = (p.argmax(1) == y).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        means, accuracies = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (confidence > lo) & (confidence <= hi)
            if mask.any():
                means.append(float(confidence[mask].mean()))
                accuracies.append(float(correct[mask].mean()))
        color = DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        axes[0].plot(
            means, accuracies, marker="o", label=model.replace("_", " ").title(), color=color
        )
        axes[1].hist(
            confidence,
            bins=edges,
            histtype="step",
            linewidth=1.8,
            label=model.replace("_", " ").title(),
            color=color,
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(
        title="Reliability diagram",
        xlabel="Mean confidence",
        ylabel="Accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set(title="Confidence distribution", xlabel="Confidence", ylabel="Examples")
    for ax in axes:
        ax.legend(fontsize=8)
    return [save_figure(fig, out / "calibration" / "reliability_and_confidence.png")]


def plot_cv(rows: str | Path | Iterable[Mapping[str, Any]], output_dir: str | Path) -> list[Path]:
    """Plot grouped-CV fold distributions, means ± standard deviation and heatmaps."""
    _style()
    data = _rows(rows)
    out = Path(output_dir)
    paths: list[Path] = []
    models = sorted({str(row.get("model", "model")) for row in data})
    if not models:
        return paths
    for metric, label in (("accuracy", "Accuracy"), ("f1_macro", "Macro F1")):
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        values = [
            [
                _number(row.get(metric))
                for row in data
                if str(row.get("model")) == model and _number(row.get(metric)) is not None
            ]
            for model in models
        ]
        ax.boxplot(values, patch_artist=True)
        ax.set_xticks(np.arange(1, len(models) + 1))
        ax.set_xticklabels([m.replace("_", " ").title() for m in models])
        ax.set(title=f"Grouped CV {label} distribution", ylabel=label)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        paths.append(save_figure(fig, out / "cv" / f"{metric}_distribution.png"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    means, stds = [], []
    for model in models:
        values = [
            v
            for row in data
            if str(row.get("model")) == model
            for v in [_number(row.get("f1_macro"))]
            if v is not None
        ]
        means.append(float(np.mean(values)) if values else np.nan)
        stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    ax.errorbar(
        np.arange(len(models)), means, yerr=stds, fmt="o", capsize=4, color=DEFAULT_COLORS[0]
    )
    ax.set(
        title="Grouped CV Macro F1 (mean ± sample SD)",
        ylabel="Macro F1",
        xticks=np.arange(len(models)),
        xticklabels=[m.replace("_", " ").title() for m in models],
        ylim=(0, 1),
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    paths.append(save_figure(fig, out / "cv" / "macro_f1_mean_std.png"))
    max_fold = max((int(_number(row.get("fold")) or 0) for row in data), default=0)
    matrix = np.full((len(models), max_fold), np.nan)
    for i, model in enumerate(models):
        for row in data:
            if str(row.get("model")) == model and _number(row.get("fold")):
                matrix[i, int(_number(row["fold"])) - 1] = _number(row.get("f1_macro")) or np.nan
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Macro F1")
    ax.set(
        title="Fold-by-model Macro F1",
        xlabel="Fold",
        ylabel="Model",
        xticks=np.arange(max_fold),
        xticklabels=[str(fold) for fold in range(1, max_fold + 1)],
        yticks=np.arange(len(models)),
        yticklabels=models,
    )
    paths.append(save_figure(fig, out / "cv" / "fold_heatmap.png"))
    return paths


def plot_efficiency(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Plot accuracy/F1 against model size, parameters, latency and training time."""
    _style()
    metrics = (
        ("parameters", "Parameters", "test_accuracy"),
        ("model_size_mb", "Checkpoint size (MB)", "test_macro_f1"),
        ("inference_ms_per_image", "Inference latency (ms/image)", "test_macro_f1"),
        ("training_time_seconds", "Training time (s)", "test_macro_f1"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (x_key, x_label, y_key) in zip(axes.flat, metrics):
        for i, row in enumerate(rows):
            x, y = _number(row.get(x_key)), _number(row.get(y_key))
            if x is not None and y is not None:
                ax.scatter(
                    x,
                    y,
                    s=52,
                    color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                    label=str(row.get("model", "model")),
                )
                ax.annotate(
                    str(row.get("model", "model")),
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.set(xlabel=x_label, ylabel=y_key.replace("_", " ").title())
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Accuracy/quality versus efficiency")
    return save_figure(fig, path)


def plot_embeddings(
    points: np.ndarray, labels: Sequence[str], path: str | Path, title: str = "Embedding projection"
) -> Path:
    """Render a 2-D PCA/UMAP/t-SNE result supplied by the caller."""
    _style()
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) != len(labels):
        raise ValueError("points must have shape [n_samples, 2] and match labels")
    fig, ax = plt.subplots(figsize=(7, 5.4))
    names = list(dict.fromkeys(map(str, labels)))
    for i, name in enumerate(names):
        mask = np.asarray([str(label) == name for label in labels])
        ax.scatter(
            values[mask, 0],
            values[mask, 1],
            s=12,
            alpha=0.7,
            label=name,
            color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
        )
    ax.set(title=title, xlabel="Component 1", ylabel="Component 2")
    ax.legend(fontsize=8, ncol=2)
    return save_figure(fig, path)


def _read_image(path: Path, size: tuple[int, int] = (128, 128)) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB").resize(size))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read gallery image %s: %s", path, exc)
        return None


def error_gallery(
    prediction_rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    root: str | Path | None = None,
    mode: str = "incorrect",
    limit: int = 20,
    pair: tuple[str, str] | None = None,
) -> Path | None:
    """Create a cached-prediction gallery (never performs inference)."""
    _style()
    rows = list(prediction_rows)
    if pair is not None:
        pair_labels = {str(pair[0]), str(pair[1])}
        rows = [
            row
            for row in rows
            if str(row.get("true_label")) in pair_labels
            and str(row.get("pred_label")) in pair_labels
            and str(row.get("true_label")) != str(row.get("pred_label"))
        ]
    if mode == "incorrect":
        rows = [row for row in rows if str(row.get("true_label")) != str(row.get("pred_label"))]
        rows.sort(key=lambda row: _number(row.get("confidence")) or 0, reverse=True)
    elif mode == "correct":
        rows = [row for row in rows if str(row.get("true_label")) == str(row.get("pred_label"))]
        rows.sort(key=lambda row: -(_number(row.get("confidence")) or 0))
    elif mode == "lowest_confidence":
        rows.sort(key=lambda row: _number(row.get("confidence")) or 0)
    else:
        raise ValueError("mode must be incorrect, correct or lowest_confidence")
    root_path = Path(root) if root else None
    tiles: list[tuple[np.ndarray, Mapping[str, Any]]] = []
    for row in rows[:limit]:
        image_path = Path(str(row.get("image_path", "")))
        if root_path and not image_path.is_absolute():
            image_path = root_path / image_path
        image = _read_image(image_path)
        if image is not None:
            tiles.append((image, row))
    if not tiles:
        LOGGER.warning("No readable rows available for %s gallery", mode)
        return None
    cols, rows_n = 4, int(np.ceil(len(tiles) / 4))
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, rows_n * 3.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (image, row) in zip(axes.flat, tiles):
        ax.imshow(image)
        confidence = _number(row.get("confidence"))
        score = "?" if confidence is None else f"{confidence:.2f}"
        ax.set_title(
            f"T: {row.get('true_label', '?')}\nP: {row.get('pred_label', '?')} ({score})",
            fontsize=8,
        )
    return save_figure(fig, path)


def plot_material_accuracy(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Plot per-material accuracy from already-computed prediction rows."""
    _style()
    aggregates = material_accuracy_rows(rows)
    ordered = sorted(aggregates, key=lambda item: item["accuracy"])
    values = [item["accuracy"] for item in ordered]
    labels = [item["material_id"] for item in ordered]
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(ordered) * 0.16)))
    ax.barh(
        np.arange(len(ordered)),
        values,
        color=[DEFAULT_COLORS[3] if value < 0.5 else DEFAULT_COLORS[1] for value in values],
    )
    ax.set(
        title="Held-out material accuracy",
        xlabel="Accuracy",
        xlim=(0, 1),
        yticks=np.arange(len(ordered)),
        yticklabels=labels,
    )
    return save_figure(fig, path)


def material_accuracy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate cached predictions by material identity for CSV/table output."""
    aggregates: dict[str, list[int]] = {}
    for row in rows:
        material = str(row.get("material_id", "unknown"))
        aggregates.setdefault(material, []).append(
            int(str(row.get("true_label")) == str(row.get("pred_label")))
        )
    return [
        {
            "material_id": material,
            "n_images": len(values),
            "correct": int(sum(values)),
            "accuracy": float(np.mean(values)),
        }
        for material, values in aggregates.items()
    ]


def project_embeddings(
    embedding_file: str | Path,
    method: str = "pca",
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Project cached embeddings using PCA, UMAP, or a documented t-SNE fallback."""
    file = Path(embedding_file)
    with np.load(file, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"])
        labels = np.asarray(data["true_label"])
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError(f"Invalid embeddings artifact {file}: expected embeddings and true_label")
    try:
        numeric = np.asarray(labels, dtype=int)
        if np.all((numeric >= 0) & (numeric < len(class_names))):
            labels = np.asarray([class_names[index] for index in numeric], dtype=str)
    except (TypeError, ValueError, OverflowError):
        labels = np.asarray(labels).astype(str)
    if len(embeddings) < 2 or embeddings.shape[1] < 1:
        return np.zeros((len(embeddings), 2), dtype=float), labels, "pca"
    requested = method.lower()
    if requested == "umap":
        try:
            import umap

            points = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
            return points, labels, "umap"
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            LOGGER.warning("UMAP unavailable (%s); falling back to t-SNE", exc)
            requested = "tsne"
    if requested == "tsne":
        if len(embeddings) < 3:
            requested = "pca"
        else:
            from sklearn.manifold import TSNE

            perplexity = min(30.0, max(1.0, (len(embeddings) - 1) / 3))
            points = TSNE(
                n_components=2, random_state=42, init="pca", perplexity=perplexity
            ).fit_transform(embeddings)
            return points, labels, "tsne"
    from sklearn.decomposition import PCA

    components = min(2, embeddings.shape[1], len(embeddings))
    points = PCA(n_components=components, random_state=42).fit_transform(embeddings)
    if components == 1:
        points = np.column_stack((points, np.zeros(len(points))))
    return points, labels, "pca"
