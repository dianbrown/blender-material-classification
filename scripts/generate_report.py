"""Build machine-readable and visual model comparisons from completed artifacts.

No model is loaded and no split is evaluated.  Values are copied only from
existing metrics, run-state, history and grouped-CV artifacts; unavailable
values remain blank in the comparison table and are called out in the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from material_classifier.plotting import plot_efficiency

LOGGER = logging.getLogger("generate_report")
MODELS = ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16")
CV_MODELS = ("resnet50", "convnext_tiny", "vit_b16")
FIELDS = (
    "family",
    "model",
    "pretrained",
    "parameters",
    "best_epoch",
    "val_accuracy",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "train_accuracy",
    "train_macro_f1",
    "accuracy_gap",
    "macro_f1_gap",
    "inference_ms_per_image",
    "model_size_mb",
    "training_time_seconds",
    "overfitting_indicator",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip().lower() in {"", "none", "nan", "null"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _history_at(path: Path, epoch: Any) -> dict[str, Any]:
    if not path.exists() or epoch is None:
        return {}
    try:
        target = int(epoch)
    except (TypeError, ValueError):
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                if int(row.get("epoch", -1)) == target:
                    return row
            except (TypeError, ValueError):
                continue
    return {}


def _cv(path: Path) -> dict[str, tuple[float | None, float | None]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, tuple[float | None, float | None]] = {}
    for model in sorted({str(row.get("model", "")) for row in rows if row.get("model")}):
        values = [_float(row.get("f1_macro")) for row in rows if row.get("model") == model]
        values = [value for value in values if value is not None]
        result[model] = (
            float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        )
    return result


def _paired_cv(path: Path) -> list[dict[str, Any]]:
    """Return descriptive paired fold differences only for common five-fold CV."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_model: dict[str, dict[int, float]] = {}
    for row in rows:
        value = _float(row.get("f1_macro"))
        fold = row.get("fold")
        if value is None or fold is None:
            continue
        try:
            by_model.setdefault(str(row.get("model")), {})[int(fold)] = value
        except ValueError:
            continue
    output: list[dict[str, Any]] = []
    models = sorted(by_model)
    for i, left in enumerate(models):
        for right in models[i + 1 :]:
            common = sorted(set(by_model[left]) & set(by_model[right]))
            if len(common) != 5:
                continue
            differences = np.asarray(
                [by_model[left][fold] - by_model[right][fold] for fold in common]
            )
            output.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "folds": 5,
                    "f1_macro_difference_mean": float(differences.mean()),
                    "f1_macro_difference_std": float(differences.std(ddof=1)),
                }
            )
    return output


def _efficiency(payload: dict[str, Any], run_state: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (run_state, payload, payload.get("efficiency", {})):
        if isinstance(source, dict):
            merged.update(source)
    benchmark = merged.get("benchmark")
    if isinstance(benchmark, dict):
        batch_one = benchmark.get("1", benchmark.get(1, {}))
        if isinstance(batch_one, dict):
            merged.update(batch_one)
    aliases = {
        "parameters": ("total_parameters", "parameters", "parameter_count"),
        "inference_ms_per_image": ("inference_ms_per_image", "latency_ms_per_image"),
        "model_size_mb": ("model_size_mb", "checkpoint_size_mb"),
        "training_time_seconds": ("total_training_time_seconds", "training_time_seconds"),
    }
    return {
        key: next((merged.get(alias) for alias in names if merged.get(alias) is not None), None)
        for key, names in aliases.items()
    }


def build_comparison(root: Path, split: str = "test") -> tuple[list[dict[str, Any]], list[str]]:
    results = root / "reports" / "results"
    cv = _cv(results / "cross_validation_results.csv")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for model in MODELS:
        metrics_path = results / model / split / "metrics.json"
        metrics = _json(metrics_path)
        if not metrics:
            missing.append(str(metrics_path))
        run_state = _json(root / "artifacts" / "checkpoints" / model / "run_state.json")
        history_path = root / "artifacts" / "checkpoints" / model / "history.csv"
        history_rows = _history_rows(history_path)
        config = _json(root / "artifacts" / "checkpoints" / model / "resolved_config.json")
        if not config:
            config = _json(root / "artifacts" / "checkpoints" / model / "best.json")
        efficiency = _efficiency(metrics, run_state)
        global_efficiency = _json(results / "efficiency_metrics.json")
        model_entries = global_efficiency.get("models", {})
        if isinstance(model_entries, list):
            model_efficiency = next(
                (entry for entry in model_entries if entry.get("model") == model), {}
            )
        else:
            model_efficiency = model_entries.get(model, {})
        if isinstance(model_efficiency, dict):
            efficiency.update(
                {
                    key: value
                    for key, value in _efficiency(model_efficiency, {}).items()
                    if value is not None
                }
            )
        global_csv = results / "efficiency_metrics.csv"
        if global_csv.exists():
            with global_csv.open(newline="", encoding="utf-8") as handle:
                for benchmark_row in csv.DictReader(handle):
                    if benchmark_row.get("model") == model:
                        csv_efficiency = {
                            "checkpoint_size_mb": benchmark_row.get("checkpoint_size_mb"),
                            "total_parameters": benchmark_row.get("total_parameters"),
                            "total_training_time_seconds": benchmark_row.get(
                                "total_training_time_seconds"
                            ),
                            "latency_ms_per_image": benchmark_row.get(
                                "batch_1_latency_ms_per_image"
                            ),
                        }
                        efficiency.update(
                            {
                                key: value
                                for key, value in _efficiency(csv_efficiency, {}).items()
                                if value is not None
                            }
                        )
                        break
        checkpoint_path = root / "artifacts" / "checkpoints" / model / "best.pt"
        if efficiency["model_size_mb"] is None and checkpoint_path.exists():
            efficiency["model_size_mb"] = checkpoint_path.stat().st_size / (1024**2)
        val = _json(results / model / "val" / "metrics.json")
        best_epoch = val.get("best_epoch", run_state.get("best", {}).get("best_epoch"))
        best_history = _history_at(
            root / "artifacts" / "checkpoints" / model / "history.csv", best_epoch
        )
        train_accuracy = _float(best_history.get("train_accuracy"))
        train_macro_f1 = _float(best_history.get("train_macro_f1"))
        val_accuracy = _float(val.get("accuracy"))
        if val_accuracy is None:
            val_accuracy = _float(best_history.get("val_accuracy"))
        val_macro_f1 = _float(val.get("f1_macro"))
        if val_macro_f1 is None:
            val_macro_f1 = _float(best_history.get("val_macro_f1"))
        accuracy_gap = (
            train_accuracy - val_accuracy
            if train_accuracy is not None and val_accuracy is not None
            else None
        )
        macro_f1_gap = (
            train_macro_f1 - val_macro_f1
            if train_macro_f1 is not None and val_macro_f1 is not None
            else None
        )
        later = []
        for item in history_rows:
            try:
                if best_epoch is not None and int(item.get("epoch", -1)) > int(best_epoch):
                    later.append(item)
            except (TypeError, ValueError):
                continue
        later_val_losses = [_float(item.get("val_loss")) for item in later]
        later_train_losses = [_float(item.get("train_loss")) for item in later]
        later_val_losses = [value for value in later_val_losses if value is not None]
        later_train_losses = [value for value in later_train_losses if value is not None]
        loss_rises = (
            len(later_val_losses) >= 2
            and len(later_train_losses) >= 2
            and later_val_losses[-1] > later_val_losses[0]
            and later_train_losses[-1] < later_train_losses[0]
        )
        large_gap = any(gap is not None and gap >= 0.10 for gap in (accuracy_gap, macro_f1_gap))
        indicator = (
            "potential_overfitting"
            if large_gap and loss_rises
            else "not_flagged"
            if accuracy_gap is not None or macro_f1_gap is not None
            else None
        )
        cv_mean, cv_std = cv.get(model, (None, None))
        row = {
            "family": "baseline CNN"
            if model == "custom_cnn"
            else "residual CNN"
            if model == "resnet50"
            else "modern CNN"
            if model == "convnext_tiny"
            else "vision transformer",
            "model": model,
            "pretrained": config.get("pretrained", model != "custom_cnn"),
            "parameters": efficiency["parameters"],
            "best_epoch": best_epoch,
            "val_accuracy": val_accuracy,
            "test_accuracy": _float(metrics.get("accuracy")),
            "test_macro_f1": _float(metrics.get("f1_macro")),
            "test_weighted_f1": _float(metrics.get("f1_weighted")),
            "cv_macro_f1_mean": cv_mean,
            "cv_macro_f1_std": cv_std,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "accuracy_gap": accuracy_gap,
            "macro_f1_gap": macro_f1_gap,
            "inference_ms_per_image": _float(efficiency["inference_ms_per_image"]),
            "model_size_mb": _float(efficiency["model_size_mb"]),
            "training_time_seconds": _float(efficiency["training_time_seconds"]),
            "overfitting_indicator": indicator,
        }
        rows.append(row)
    return rows, missing


def _fmt(value: Any) -> str:
    if value is None or str(value).lower() in {"none", "nan", ""}:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _dashboard(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit widths and a wide canvas keep long family/metric labels
    # readable at publication scale.
    fig, ax = plt.subplots(figsize=(24, max(3.4, 0.85 * len(rows) + 2.0)))
    ax.axis("off")
    columns = [
        "Family",
        "Model",
        "Pretrained",
        "Params",
        "Best epoch",
        "Val acc",
        "Test acc",
        "Test F1",
        "Test weighted F1",
        "CV F1 mean ± SD",
        "Train acc",
        "Acc gap",
        "F1 gap",
        "Latency ms",
        "Size MB",
        "Training s",
    ]
    table_rows = []
    for row in rows:
        cv = (
            f"{_fmt(row['cv_macro_f1_mean'])} ± {_fmt(row['cv_macro_f1_std'])}"
            if row["cv_macro_f1_mean"] is not None
            else "—"
        )
        table_rows.append(
            [
                row["family"],
                row["model"],
                _fmt(row["pretrained"]),
                _fmt(row["parameters"]),
                _fmt(row["best_epoch"]),
                _fmt(row["val_accuracy"]),
                _fmt(row["test_accuracy"]),
                _fmt(row["test_macro_f1"]),
                _fmt(row["test_weighted_f1"]),
                cv,
                _fmt(row["train_accuracy"]),
                _fmt(row["accuracy_gap"]),
                _fmt(row["macro_f1_gap"]),
                _fmt(row["inference_ms_per_image"]),
                _fmt(row["model_size_mb"]),
                _fmt(row["training_time_seconds"]),
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        colWidths=[
            0.14,
            0.12,
            0.075,
            0.075,
            0.07,
            0.07,
            0.07,
            0.085,
            0.105,
            0.11,
            0.08,
            0.07,
            0.075,
            0.09,
            0.10,
            0.09,
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.9)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#d7e3ea")
    for (row_index, _column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#16324f")
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f2f6f8")
    ax.set_title(
        "Material classifier benchmark — cached results", pad=18, weight="bold", color="#16324f"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _rankings(rows: list[dict[str, Any]], path: Path) -> Path:
    valid = [row for row in rows if row.get("test_macro_f1") is not None]
    if not valid:
        return path
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    def efficiency_score(row: dict[str, Any]) -> float | None:
        values = []
        for field in ("inference_ms_per_image", "model_size_mb", "training_time_seconds"):
            value = _float(row.get(field))
            if value is not None and value > 0:
                values.append(value)
        return float(1 / (1 + np.mean(values))) if values else None

    specs = (
        ("test_macro_f1", "Test Macro F1"),
        ("test_accuracy", "Test accuracy"),
        ("_efficiency", "Efficiency score"),
    )
    for ax, (field, title) in zip(axes, specs):
        ordered = sorted(
            valid,
            key=lambda row: (
                (efficiency_score(row) if field == "_efficiency" else row.get(field))
                if (efficiency_score(row) if field == "_efficiency" else row.get(field)) is not None
                else -1
            ),
        )
        labels = [row["model"] for row in ordered]
        values = [
            efficiency_score(row) if field == "_efficiency" else row.get(field) for row in ordered
        ]
        ax.barh(labels, values, color="#2f6690")
        ax.set(title=title, xlim=(0, 1))
    fig.suptitle("Model rankings (quality first; efficiency reported separately)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _test_gate_diagnostics(root: Path) -> list[str]:
    """Validate lock integrity and one completed marker for every TEST model."""
    diagnostics: list[str] = []
    try:
        from material_classifier.evaluate import require_methodology_lock

        require_methodology_lock(root)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        diagnostics.append(f"methodology lock invalid: {exc}")
    results = root / "reports" / "results"
    for model in MODELS:
        metrics = results / model / "test" / "metrics.json"
        predictions = results / model / "test" / "predictions.csv"
        checkpoint = root / "artifacts" / "checkpoints" / model / "best.pt"
        if not metrics.exists() or not predictions.exists() or not checkpoint.exists():
            diagnostics.append(f"completed TEST artifacts missing for {model}")
            continue
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        valid = False
        for marker in results.glob(f"test_evaluated_{model}_*.json"):
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                valid = (
                    payload.get("status") == "completed"
                    and payload.get("checkpoint_sha256") == digest
                    and Path(payload.get("metrics_file", "")).resolve() == metrics.resolve()
                    and marker.name.endswith(f"{digest[:16]}.json")
                )
            except (OSError, json.JSONDecodeError):
                valid = False
            if valid:
                break
        if not valid:
            diagnostics.append(f"valid completed TEST marker missing or corrupt for {model}")
    return diagnostics


def _scale(values: dict[str, float | None], higher_is_better: bool) -> dict[str, float | None]:
    """Min-max scale one recommendation criterion, preserving unavailable values."""
    present = [value for value in values.values() if value is not None]
    if not present:
        return {key: None for key in values}
    low, high = min(present), max(present)
    if high == low:
        return {key: 1.0 if value is not None else None for key, value in values.items()}
    return {
        key: ((value - low) / (high - low) if higher_is_better else (high - value) / (high - low))
        if value is not None
        else None
        for key, value in values.items()
    }


def _recommendation(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a model using the locked study's quality, stability and efficiency criteria."""
    raw: dict[str, dict[str, float | None]] = {}
    cv_path = root / "reports" / "results" / "cross_validation_results.csv"
    cv_values: dict[str, dict[int, float]] = {model: {} for model in CV_MODELS}
    cv_gate_errors: list[str] = []
    if cv_path.exists():
        with cv_path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                model = str(item.get("model", ""))
                fold = item.get("fold")
                value = _float(item.get("f1_macro"))
                try:
                    fold_number = int(fold) if fold is not None else -1
                except (TypeError, ValueError):
                    fold_number = -1
                if model not in cv_values:
                    cv_gate_errors.append(f"unexpected CV model: {model}")
                elif fold_number not in {1, 2, 3, 4, 5}:
                    cv_gate_errors.append(f"invalid CV fold for {model}: {fold}")
                elif value is None or not math.isfinite(value):
                    cv_gate_errors.append(f"non-finite CV Macro F1 for {model}, fold {fold}")
                elif fold_number in cv_values[model]:
                    cv_gate_errors.append(f"duplicate CV row for {model}, fold {fold}")
                else:
                    cv_values[model][fold_number] = value
    else:
        cv_gate_errors.append(f"missing grouped-CV artifact: {cv_path}")
    for row in rows:
        model = str(row["model"])
        metrics = _json(root / "reports" / "results" / model / "test" / "metrics.json")
        per_class = metrics.get("per_class", [])
        f1_values = []
        for item in per_class:
            value = _float(item.get("f1"))
            if value is not None:
                f1_values.append(value)
        f1_mean = float(np.mean(f1_values)) if f1_values else None
        if len(f1_values) > 1:
            f1_std = float(np.std(f1_values, ddof=1))
        elif f1_values:
            f1_std = 0.0
        else:
            f1_std = None
        model_cv = list(cv_values.get(model, {}).values())
        cv_complete = not cv_gate_errors and set(cv_values.get(model, {})) == {1, 2, 3, 4, 5}
        cv_mean = float(np.mean(model_cv)) if cv_complete else None
        cv_std = float(np.std(model_cv, ddof=1)) if cv_complete else None
        raw[model] = {
            "test_macro_f1": _float(row.get("test_macro_f1")),
            "test_accuracy": _float(row.get("test_accuracy")),
            # CustomCNN CV is intentionally not run by the protocol; it is not scored as 0 or 1.
            "cv_stability": (
                None
                if model == "custom_cnn"
                else (cv_mean - cv_std if cv_mean is not None and cv_std is not None else None)
            ),
            "per_class_consistency": (
                f1_mean - f1_std if f1_mean is not None and f1_std is not None else None
            ),
            "model_size_mb": _float(row.get("model_size_mb")),
            "inference_ms_per_image": _float(row.get("inference_ms_per_image")),
        }
    directions = {
        "test_macro_f1": True,
        "test_accuracy": True,
        "cv_stability": True,
        "per_class_consistency": True,
        "model_size_mb": False,
        "inference_ms_per_image": False,
    }
    scaled = {
        criterion: _scale({model: values[criterion] for model, values in raw.items()}, higher)
        for criterion, higher in directions.items()
    }
    model_scores: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for model, values in raw.items():
        criterion_scores = {criterion: scaled[criterion][model] for criterion in directions}
        unavailable = [criterion for criterion, value in criterion_scores.items() if value is None]
        if unavailable != ["cv_stability"] or model != "custom_cnn":
            if unavailable:
                missing.append(model)
        if model in CV_MODELS and not cv_complete:
            missing.append(model)
        if model == "custom_cnn" and unavailable != ["cv_stability"]:
            missing.append(model)
        score_values = [value for value in criterion_scores.values() if value is not None]
        included = [criterion for criterion, value in criterion_scores.items() if value is not None]
        weights = {criterion: 1.0 / len(included) for criterion in included} if included else {}
        coverage[model] = {
            criterion: (
                "not_applicable"
                if criterion == "cv_stability" and model == "custom_cnn"
                else "used"
                if value is not None
                else "missing"
            )
            for criterion, value in criterion_scores.items()
        }
        model_scores[model] = {
            "raw": values,
            "scaled": criterion_scores,
            "criteria_used": included,
            "weights": weights,
            "cv_folds": (
                "not_applicable" if model == "custom_cnn" else len(cv_values.get(model, {}))
            ),
            "composite_score": float(np.mean(score_values)) if included else None,
        }
    complete = not missing and len(model_scores) == len(MODELS)
    selected = (
        max(
            model_scores,
            key=lambda model: (
                model_scores[model]["composite_score"],
                raw[model]["test_macro_f1"],
                raw[model]["test_accuracy"],
            ),
        )
        if complete
        else None
    )
    return {
        "schema_version": 1,
        "status": "completed" if complete else "pending",
        "selected_model": selected,
        "selection_rule": (
            "Equal-weight min-max composite of test Macro F1, test accuracy, grouped-CV "
            "stability (mean minus SD), per-class F1 consistency (mean minus SD), inverse "
            "checkpoint size, and inverse batch-1 inference latency."
        ),
        "missing_criteria_models": sorted(set(missing)),
        "cv_gate_errors": cv_gate_errors,
        "criterion_coverage": coverage,
        "models": model_scores,
        "source": {
            "comparison_csv": "reports/results/model_comparison.csv",
            "dashboard": "reports/figures/comparison/model_comparison_dashboard.png",
            "test_gate": (
                "reports/results/methodology_lock.json and "
                "test_evaluated_<model>_<checkpoint_hash>.json"
            ),
        },
    }


def generate(root: Path, split: str = "test", dry_run: bool = False) -> dict[str, Path]:
    rows, missing = build_comparison(root, split)
    gate_diagnostics = _test_gate_diagnostics(root) if split.lower() == "test" else []
    missing.extend(gate_diagnostics)
    paired = _paired_cv(root / "reports" / "results" / "cross_validation_results.csv")
    if dry_run:
        print(
            "Report prerequisites are complete."
            if not missing
            else "Missing metrics artifacts:\n" + "\n".join(f"- {item}" for item in missing)
        )
        return {}
    if gate_diagnostics:
        raise RuntimeError(
            "Refusing to write a TEST report until lock and completed markers validate:\n- "
            + "\n- ".join(gate_diagnostics)
        )
    results = root / "reports" / "results"
    figures = root / "reports" / "figures" / "comparison"
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)
    markdown = results / "model_comparison.md"
    lines = [
        "# Model comparison",
        "",
        "Values are copied from completed artifacts; `—` means the source metric was unavailable.",
        "",
        "| " + " | ".join(FIELDS) + " |",
        "|" + "---|" * len(FIELDS),
    ]
    lines.extend("| " + " | ".join(_fmt(row.get(field)) for field in FIELDS) + " |" for row in rows)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dashboard = _dashboard(rows, figures / "model_comparison_dashboard.png")
    ranking = _rankings(rows, figures / "model_rankings.png")
    efficiency_figure = plot_efficiency(rows, figures / "efficiency_tradeoffs.png")
    recommendation_path: Path | None = None
    if split.lower() == "test":
        recommendation_path = results / "recommended_model.json"
        recommendation_path.write_text(
            json.dumps(_recommendation(root, rows), indent=2) + "\n", encoding="utf-8"
        )
    report = results / "generated_report.md"
    report_lines = [
        "# Generated evaluation report",
        "",
        (
            "This report is generated exclusively from cached validation/TEST metrics, run "
            "metadata, histories and development grouped-CV results. It does not evaluate a "
            "dataset."
        ),
        "",
        "## Overfitting indicator",
        "",
        (
            "A model is marked `potential_overfitting` only when the selected-epoch train-accuracy "
            "minus validation-accuracy exceeds 0.10 **and** validation loss rises above an "
            "earlier minimum while training continues. This is a screening indicator, not a "
            "statistical conclusion."
        ),
        "",
        "## Statistical caution",
        "",
        (
            "Grouped-CV values are reported as mean +/- sample standard deviation. Paired fold "
            "comparisons are only defensible when every compared model has the same five fold IDs; "
            "n=5 is small and should be interpreted descriptively."
        ),
        "",
        "## Artifacts",
        "",
        f"- Comparison CSV: `{csv_path.relative_to(root)}`",
        f"- Comparison Markdown: `{markdown.relative_to(root)}`",
        f"- Dashboard: `{dashboard.relative_to(root)}`",
        f"- Rankings: `{ranking.relative_to(root)}`",
    ]
    if paired:
        report_lines.extend(
            [
                "",
                "### Descriptive paired fold differences",
                "",
                "| Model A | Model B | Folds | Mean F1 difference (A−B) | SD |",
                "|---|---|---:|---:|---:",
            ]
        )
        report_lines.extend(
            (
                f"| {item['model_a']} | {item['model_b']} | 5 | "
                f"{item['f1_macro_difference_mean']:.3f} | "
                f"{item['f1_macro_difference_std']:.3f} |"
            )
            for item in paired
        )
    if missing:
        report_lines.extend(
            ["", "## Missing source artifacts", "", *[f"- `{item}`" for item in missing]]
        )
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    outputs = {
        "csv": csv_path,
        "markdown": markdown,
        "report": report,
        "dashboard": dashboard,
        "rankings": ranking,
        "efficiency": efficiency_figure,
    }
    if recommendation_path is not None:
        outputs["recommendation"] = recommendation_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outputs = generate(args.root.resolve(), args.split, args.dry_run)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
