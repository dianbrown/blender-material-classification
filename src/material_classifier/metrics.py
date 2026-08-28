"""Leakage-agnostic classification and probability metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: Sequence[int], probabilities: np.ndarray, bins: int = 15
) -> float:
    """Compute confidence calibration error using equal-width bins."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or len(y) != len(p) or not len(y):
        return float("nan")
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi if hi < 1 else confidence <= hi)
        if mask.any():
            out += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(out)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    loss: float | None = None,
) -> dict[str, Any]:
    """Return all required scalar, per-class, and confusion metrics.

    Probability metrics are omitted when probabilities are unavailable. ROC
    AUC is guarded because a fold or custom caller may not contain every class.
    """
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    n_classes = (
        len(class_names)
        if class_names is not None
        else int(max(y.max(initial=0), pred.max(initial=0)) + 1)
    )
    labels = np.arange(n_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, average=None, zero_division=0
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(
            precision_recall_fscore_support(
                y, pred, labels=labels, average="macro", zero_division=0
            )[0]
        ),
        "recall_macro": float(
            precision_recall_fscore_support(
                y, pred, labels=labels, average="macro", zero_division=0
            )[1]
        ),
        "f1_macro": float(
            precision_recall_fscore_support(
                y, pred, labels=labels, average="macro", zero_division=0
            )[2]
        ),
        "precision_weighted": float(
            precision_recall_fscore_support(
                y, pred, labels=labels, average="weighted", zero_division=0
            )[0]
        ),
        "recall_weighted": float(
            precision_recall_fscore_support(
                y, pred, labels=labels, average="weighted", zero_division=0
            )[1]
        ),
        "f1_weighted": float(
            precision_recall_fscore_support(
                y, pred, labels=labels, average="weighted", zero_division=0
            )[2]
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_class": [],
    }
    if loss is not None:
        result["loss"] = float(loss)
    names = list(class_names) if class_names is not None else [str(i) for i in labels]
    for i, name in enumerate(names):
        result["per_class"].append(
            {
                "class": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )

    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        result["log_loss"] = float(log_loss(y, probs, labels=labels))
        result["ece"] = expected_calibration_error(y, probs)
        try:
            onehot = np.eye(n_classes, dtype=float)[y]
            result["roc_auc_macro"] = float(
                roc_auc_score(onehot, probs, average="macro", multi_class="ovr")
            )
            result["roc_auc_weighted"] = float(
                roc_auc_score(onehot, probs, average="weighted", multi_class="ovr")
            )
            result["roc_auc_per_class"] = {
                names[i]: float(roc_auc_score(onehot[:, i], probs[:, i]))
                for i in range(n_classes)
                if len(np.unique(onehot[:, i])) > 1
            }
        except ValueError:
            result["roc_auc_macro"] = None
            result["roc_auc_weighted"] = None
            result["roc_auc_per_class"] = {}
    return result


compute_metrics = classification_metrics
