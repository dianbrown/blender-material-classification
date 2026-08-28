"""Development-only, material-grouped cross-validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _columns(records: Any) -> tuple[list[Any], list[Any], list[Any]]:
    if hasattr(records, "__getitem__") and hasattr(records, "columns"):
        return (
            list(records["class"]),
            list(records["material_id"]),
            list(records.get("split", ["train"] * len(records))),
        )
    rows = list(records)
    return (
        [r["class"] for r in rows],
        [r["material_id"] for r in rows],
        [r.get("split", "train") for r in rows],
    )


def grouped_folds(
    records: Any, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return folds over TRAIN+VAL and assert group disjointness explicitly."""
    classes, groups, splits = _columns(records)
    if any(str(split).lower() == "test" for split in splits):
        raise ValueError("TEST records are forbidden in grouped development CV")
    labels = np.asarray(classes)
    group_ids = np.asarray(groups)
    # Class labels are material-level labels in this dataset.  Using image rows
    # is valid because groups force all 30 renders to stay on one side.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels, group_ids):
        train_groups, val_groups = set(group_ids[train_idx]), set(group_ids[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            raise AssertionError(f"Material leakage in grouped fold: {sorted(overlap)}")
        folds.append((train_idx, val_idx))
    return folds


def run_grouped_cv(
    records: Any,
    model_name: str,
    fold_runner: Callable[[str, int, Sequence[int], Sequence[int]], dict[str, Any]],
    n_splits: int = 5,
    seed: int = 42,
    output_csv: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run caller-owned fold training and persist fold metrics."""
    folds = grouped_folds(records, n_splits, seed)
    rows: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    prior_model_rows: list[dict[str, Any]] = []
    if output_csv and output_csv.exists() and not force:
        with output_csv.open(newline="", encoding="utf-8") as handle:
            all_existing = list(csv.DictReader(handle))
            existing = [row for row in all_existing if row.get("model") != model_name]
            prior_model_rows = [row for row in all_existing if row.get("model") == model_name]
    elif output_csv and output_csv.exists() and force:
        with output_csv.open(newline="", encoding="utf-8") as handle:
            existing = [row for row in csv.DictReader(handle) if row.get("model") != model_name]
    prior_folds = {int(row["fold"]) for row in prior_model_rows}
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        if fold in prior_folds:
            continue
        metrics = dict(fold_runner(model_name, fold, train_idx.tolist(), val_idx.tolist()))
        metrics.update({"model": model_name, "fold": fold})
        rows.append(metrics)
    combined = existing + prior_model_rows + rows
    if output_csv and combined:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "model",
            "fold",
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "best_epoch",
            "training_time_seconds",
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in sorted(
                combined, key=lambda item: (str(item.get("model", "")), int(item.get("fold", 0)))
            ):
                writer.writerow({field: row.get(field) for field in fields})
        aggregate = aggregate_cv(combined)
        aggregate_path = output_csv.with_name(f"{output_csv.stem}_aggregate{output_csv.suffix}")
        with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
            fields = list(aggregate[0]) if aggregate else ["model"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(aggregate)
    return combined


def aggregate_cv(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str] = (
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ),
) -> list[dict[str, Any]]:
    """Summarise fold scores as mean and sample standard deviation."""
    out: list[dict[str, Any]] = []
    models = sorted({str(row.get("model", "model")) for row in rows})
    for model in models:
        selected = [row for row in rows if str(row.get("model")) == model]
        record: dict[str, Any] = {"model": model, "folds": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            record[f"{metric}_mean"] = float(np.mean(values)) if values else None
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            )
        out.append(record)
    return out
