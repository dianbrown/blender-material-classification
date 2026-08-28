# ruff: noqa: E501

"""Render README.md and MODEL_CARD.md from completed, cached project artifacts.

The documentation stage is deliberately read-only with respect to models and data:
it never evaluates a checkpoint and never falls back to validation scores for the
headline table.  Until every locked TEST metrics file exists, the generated docs
show an explicit pending status instead of freezing interim numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

MODELS = ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16")
CLASSES = ("wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather")
FAMILIES = {
    "custom_cnn": "Baseline CNN",
    "resnet50": "Residual CNN",
    "convnext_tiny": "Modern CNN",
    "vit_b16": "Vision transformer",
}
DISPLAY_NAMES = {
    "custom_cnn": "CustomCNN",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b16": "ViT-B/16",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip().lower() in {"", "none", "nan", "null"}:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = _number(value)
    if number is not None:
        return f"{number:.{digits}f}"
    return "-"


def _cv_summary(path: Path) -> dict[str, tuple[float | None, float | None]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, tuple[float | None, float | None]] = {}
    for model in MODELS:
        values = [
            value
            for row in rows
            if row.get("model") == model
            for value in [_number(row.get("f1_macro"))]
            if value is not None
        ]
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            result[model] = (mean, variance**0.5 if len(values) > 1 else 0.0)
    return result


def _history_at(root: Path, model: str, epoch: Any) -> dict[str, Any]:
    path = root / "artifacts" / "checkpoints" / model / "history.csv"
    try:
        target = int(epoch)
    except (TypeError, ValueError):
        return {}
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                if int(row.get("epoch", -1)) == target:
                    return row
            except (TypeError, ValueError):
                continue
    return {}


def _prediction_example(root: Path, model: str) -> dict[str, str]:
    """Read one cached prediction for the model-card reproduction example.

    Docs generation never performs inference; the snippet below calls the
    public API while its illustrative output is sourced from the sealed table.
    """
    path = root / "reports" / "results" / model / "test" / "predictions.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
    except (OSError, StopIteration):
        return {}
    image = str(row.get("image_path", "")).strip()
    predicted = str(row.get("pred_label", "")).strip()
    confidence = _number(row.get("confidence"))
    if not image or not predicted or confidence is None:
        return {}
    return {"image": image, "predicted": predicted, "confidence": f"{confidence:.3f}"}


def collect_rows(root: Path) -> list[dict[str, Any]]:
    """Collect comparison rows using TEST metrics only for headline scores."""
    cv = _cv_summary(root / "reports" / "results" / "cross_validation_results.csv")
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        metrics = _read_json(root / "reports" / "results" / model / "test" / "metrics.json")
        state = _read_json(root / "artifacts" / "checkpoints" / model / "run_state.json")
        config = _read_json(root / "artifacts" / "checkpoints" / model / "resolved_config.json")
        best = state.get("best", {}) if isinstance(state.get("best"), dict) else {}
        epoch = metrics.get("best_epoch", best.get("best_epoch"))
        history = _history_at(root, model, epoch)
        cv_mean, cv_std = cv.get(model, (None, None))
        rows.append(
            {
                "model": model,
                "family": FAMILIES[model],
                "pretrained": config.get("pretrained", model != "custom_cnn"),
                "parameters": state.get("total_parameters", metrics.get("total_parameters")),
                "best_epoch": epoch,
                "test_accuracy": metrics.get("accuracy"),
                "test_macro_f1": metrics.get("f1_macro"),
                "test_weighted_f1": metrics.get("f1_weighted"),
                "cv_macro_f1_mean": cv_mean,
                "cv_macro_f1_std": cv_std,
                "train_accuracy": history.get("train_accuracy"),
                "train_macro_f1": history.get("train_macro_f1"),
                "metrics": metrics,
                "config": config,
            }
        )
    return rows


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Family | Pretrained | Best epoch | Test accuracy | Test macro F1 | Test weighted F1 | CV macro F1 (mean +/- SD) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cv = (
            f"{_fmt(row['cv_macro_f1_mean'])} +/- {_fmt(row['cv_macro_f1_std'])}"
            if row["cv_macro_f1_mean"] is not None
            else "N/A (not run)"
        )
        lines.append(
            "| {model} | {family} | {pretrained} | {epoch} | {accuracy} | {f1} | {weighted} | {cv} |".format(
                model=DISPLAY_NAMES[row["model"]],
                family=row["family"],
                pretrained="yes" if row["pretrained"] else "no",
                epoch=_fmt(row["best_epoch"], 0),
                accuracy=_fmt(row["test_accuracy"]),
                f1=_fmt(row["test_macro_f1"]),
                weighted=_fmt(row["test_weighted_f1"]),
                cv=cv,
            )
        )
    return "\n".join(lines)


def _complete(rows: list[dict[str, Any]]) -> bool:
    required = ("test_accuracy", "test_macro_f1", "test_weighted_f1")
    return all(all(_number(row.get(field)) is not None for field in required) for row in rows)


def _gate_diagnostics(root: Path) -> list[str]:
    """Reuse the report generator's lock/hash/TEST-marker validation."""
    try:
        try:
            from scripts.generate_report import _test_gate_diagnostics
        except ModuleNotFoundError:
            from generate_report import _test_gate_diagnostics

        return _test_gate_diagnostics(root)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return [f"cannot validate TEST gate: {exc}"]


def _recommendation(root: Path) -> tuple[dict[str, Any], list[str]]:
    path = root / "reports" / "results" / "recommended_model.json"
    payload = _read_json(path)
    diagnostics: list[str] = []
    if not payload:
        return {}, [f"missing recommendation artifact: {path}"]
    if payload.get("status") != "completed":
        diagnostics.append("recommendation artifact is not completed")
    if payload.get("cv_gate_errors"):
        diagnostics.append("recommendation artifact contains grouped-CV gate errors")
    if payload.get("selected_model") not in MODELS:
        diagnostics.append("recommendation artifact has no valid selected_model")
    models = payload.get("models")
    criteria = (
        "test_macro_f1",
        "test_accuracy",
        "cv_stability",
        "per_class_consistency",
        "model_size_mb",
        "inference_ms_per_image",
    )
    if not isinstance(models, dict) or set(MODELS) - set(models):
        diagnostics.append("recommendation artifact does not cover all headline models")
    else:
        for model in MODELS:
            entry = models[model] if isinstance(models[model], dict) else {}
            raw = entry.get("raw", {}) if isinstance(entry.get("raw"), dict) else {}
            required = (
                criteria
                if model != "custom_cnn"
                else tuple(criterion for criterion in criteria if criterion != "cv_stability")
            )
            if any(_number(raw.get(criterion)) is None for criterion in required):
                diagnostics.append(f"recommendation criteria incomplete for {model}")
            coverage = payload.get("criterion_coverage", {}).get(model, {})
            if model == "custom_cnn":
                if (
                    raw.get("cv_stability") is not None
                    or coverage.get("cv_stability") != "not_applicable"
                ):
                    diagnostics.append("CustomCNN CV must be explicitly marked not_applicable")
            elif entry.get("cv_folds") != 5 or coverage.get("cv_stability") != "used":
                diagnostics.append(f"transfer-model CV must contain exactly five folds for {model}")
    return payload, diagnostics


def render_readme(root: Path, rows: list[dict[str, Any]], gate_ready: bool = False) -> str:
    validation = _read_json(root / "reports" / "results" / "dataset_validation.json")
    expected = validation.get("expected", {})
    summary = validation.get("summary", {})
    status = (
        "Completed locked TEST evaluation."
        if gate_ready and _complete(rows)
        else "Headline TEST evaluation is pending; dashes below are intentional and are not validation scores."
    )
    recommendation, _ = _recommendation(root)
    selected = recommendation.get("selected_model") if recommendation else None
    selected_row = next((row for row in rows if row.get("model") == selected), None)
    if selected_row and gate_ready:
        recommendation_line = (
            f"The canonical recommendation is **{DISPLAY_NAMES.get(selected, selected)}**: TEST Macro F1 "
            f"{_fmt(selected_row.get('test_macro_f1'))}, accuracy "
            f"{_fmt(selected_row.get('test_accuracy'))}, and grouped-CV Macro F1 "
            f"{_fmt(selected_row.get('cv_macro_f1_mean'))} +/- {_fmt(selected_row.get('cv_macro_f1_std'))}."
        )
    else:
        recommendation_line = (
            "The canonical recommendation is pending the complete locked TEST gate."
        )
    figure = "reports/figures/dataset/representative_images_all_classes.png"
    return f"""# Blender Material Classification

Leakage-safe classification of seven synthetic PBR material classes rendered in Blender EEVEE. The study compares a small CNN baseline with ImageNet-pretrained ResNet-50, ConvNeXt-Tiny, and ViT-B/16 models.

## Overview and motivation

This design-engineering study tests whether texture, reflectance, and shading cues learned from procedural Blender materials transfer to unseen base materials. The engineering motivation is to measure accuracy, robustness, and inference efficiency before a material recognizer is considered for downstream inspection tooling. Grouping by base-material identity makes the comparison meaningful: near-duplicate renders cannot leak between development and held-out splits.

![Representative synthetic renders]({figure})

## Dataset

The dataset contains **{expected.get("total_images", summary.get("rows", "-"))}** RGB PNG renders at 512 x 512, generated with Blender and RenderSynth from **{expected.get("materials_total", summary.get("materials", "-"))}** independent base materials (30 renders per material). Classes are wood, metal, stone, concrete, ceramic, fabric, and leather. The canonical split is grouped by `material_id`: train {expected.get("images", {}).get("train", "-")} images / {expected.get("materials_by_split", {}).get("train", "-")} materials, validation {expected.get("images", {}).get("val", "-")} / {expected.get("materials_by_split", {}).get("val", "-")}, and test {expected.get("images", {}).get("test", "-")} / {expected.get("materials_by_split", {}).get("test", "-")}.

Raw renders and PBR assets remain local. See [DATASET.md](DATASET.md) for regeneration, provenance, licensing notes, and placement instructions. For an existing checkout, place files exactly as `dataset/train/{{class}}/*.png`, `dataset/val/{{class}}/*.png`, and `dataset/test/{{class}}/*.png`; place source PBR material folders at `materials/{{class}}/{{material_id}}/`. Run `python scripts/verify_dataset.py` before training.

Classes: `wood`, `metal`, `stone`, `concrete`, `ceramic`, `fabric`, `leather`.

### Procedural generation pipeline

Blender builds the scene and EEVEE render settings; RenderSynth assigns each PBR texture set to a base material; deterministic camera/light variations produce 30 renders per material; manifest validation records class, `material_id`, and split. The generation seed is 62642. Place local data at `dataset/{{train,val,test}}/{{class}}/*.png` and source textures at `materials/{{class}}/{{material_id}}/`; raw data is intentionally gitignored.

## Method

All models use 224 x 224 inputs, ImageNet normalization, conservative training augmentation, AdamW, label smoothing, warmup/cosine scheduling, AMP, and best-validation-macro-F1 checkpointing. Grouped five-fold `StratifiedGroupKFold` uses development data only; the locked test split is evaluated once after methodology lock.

Compared architectures are intentionally distinct: a from-scratch CustomCNN baseline (capacity floor), ImageNet-pretrained ResNet-50 (residual CNN), ConvNeXt-Tiny (modern convolutional design), and ViT-B/16 (patch-token transformer). Transfer models train a frozen head stage followed by whole-model fine-tuning; all runs use the fixed 40-epoch ceiling and early stopping (patience 8). The baseline is not assigned a fabricated CV score: its table entry is `N/A (not run)`.

Training histories, TensorBoard logs, best/last checkpoints, effective batch sizes, runtime, peak VRAM, environment metadata, and configuration hashes are retained under `artifacts/checkpoints/` and `reports/results/`. Five-fold grouped CV is reported as mean +/- sample SD for the three transfer models; it is computed only on the development materials.

## Results

{status}

{_table(rows)}

{recommendation_line}

The machine-readable source is [reports/results/model_comparison.csv](reports/results/model_comparison.csv), generated from cached artifacts. The recommendation is read from [reports/results/recommended_model.json](reports/results/recommended_model.json) and is not inferred in this document. Supporting tables and figures are under `reports/results/` and `reports/figures/`; see the [report asset index](reports/REPORT_ASSETS.md) for figure links and recommended report usage.

![Recommended ResNet-50 normalized confusion matrix](reports/figures/confusion_matrices/resnet50_normalized.png)

![Model comparison dashboard](reports/figures/comparison/model_comparison_dashboard.png)

![Training curves](reports/figures/training/combined_val_f1.png)

## Repository structure

```text
src/material_classifier/   reusable data, models, training, metrics, and plotting code
configs/                   fixed data and model configurations
scripts/                   verification, training, CV, evaluation, reporting, and docs
metadata/                  manifests and split definitions
reports/results/           cached metrics, predictions, embeddings, and tables
reports/figures/           publication-ready figures and galleries
notebooks/                 executed EDA, results, comparison, and error-analysis views
```

## Setup and reproducibility

Use Python 3.13 with the pinned environment. From the repository root run `uv sync --extra dev --extra explainability --extra embeddings` (or install the pinned base, development, explainability, and embeddings requirement sets), then verify CUDA/GPU details with `python scripts/collect_environment.py`. The project seed is 42; configs, environment metadata, hashes, histories, and timestamps are recorded under `reports/results/` and `artifacts/checkpoints/`.

## Reproduce

```powershell
uv sync --extra dev --extra explainability --extra embeddings
python scripts/verify_dataset.py
python scripts/train_all.py
python scripts/run_group_cv.py --folds 5
python scripts/lock_methodology.py
python scripts/evaluate_all.py --split test
python scripts/generate_report.py
python scripts/generate_visuals.py --explainability
python scripts/update_docs.py
```

Run notebooks from the repository root after artifacts are available: `jupyter lab notebooks/`. Notebooks 02-04 read histories, metrics, predictions, embeddings, and figures; they never train or evaluate models. The `--split test` command is allowed only after the methodology lock; its hashed completion markers make subsequent calls reuse the sealed TEST artifacts. Do not tune after TEST evaluation.

## Limitations

This is a synthetic Blender-render study. Synthetic-to-real generalisation has not been established, and the model should not be treated as a validated arbitrary real-world material recogniser.

## Future work

Synthetic-to-real evaluation, Hugging Face deployment, an interactive web demo, and a Kaggle dataset release are intentionally out of scope for this repository stage.

See [MODEL_CARD.md](MODEL_CARD.md) for intended and inappropriate uses.
"""


def render_model_card(
    root: Path,
    rows: list[dict[str, Any]],
    recommendation: dict[str, Any] | None = None,
    gate_ready: bool = False,
) -> str:
    complete = gate_ready and _complete(rows) and recommendation is not None
    selected_model = recommendation.get("selected_model") if recommendation else None
    chosen = next((row for row in rows if row.get("model") == selected_model), None)
    if not complete or chosen is None:
        chosen = None
    metrics = chosen.get("metrics", {}) if chosen else {}
    config = chosen.get("config", {}) if chosen else {}
    cv_mean = chosen.get("cv_macro_f1_mean") if chosen else None
    cv_std = chosen.get("cv_macro_f1_std") if chosen else None
    per_class = metrics.get("per_class", []) if isinstance(metrics, dict) else []
    if per_class:
        class_lines = "\n".join(
            f"| {item.get('class', '-')} | {_fmt(item.get('precision'))} | {_fmt(item.get('recall'))} | {_fmt(item.get('f1'))} | {_fmt(item.get('support'), 0)} |"
            for item in per_class
        )
    else:
        class_lines = "| - | - | - | - | - |"
    recommendation = (
        f"**{DISPLAY_NAMES.get(chosen['model'], chosen['model'])}** is the canonical artifact-derived recommendation. It uses the reporting stack's composite of TEST Macro F1, TEST accuracy, grouped-CV stability, per-class consistency, checkpoint size, and inference latency."
        if chosen
        else "The recommendation is pending completion of locked TEST evaluation for all four headline models; validation results are intentionally excluded."
    )
    checkpoint = (
        f"`artifacts/checkpoints/{chosen['model']}/best.pt`"
        if chosen
        else "Pending final checkpoint selection"
    )
    example = _prediction_example(root, selected_model) if chosen else {}
    inference_block = (
        "```python\n"
        "from pathlib import Path\n"
        "from material_classifier.inference import predict_image\n"
        f'\nprediction = predict_image(Path("artifacts/checkpoints/{chosen["model"]}/best.pt"), Path("{example.get("image", "dataset/val/wood/example.png")}"))\n'
        "print(f\"{prediction['class']}: {prediction['confidence']:.3f}\")\n"
        "```\n\n"
        f"Example cached result: `{example.get('predicted', 'pending')}: {example.get('confidence', 'pending')}`"
        if chosen
        else "Inference code is withheld until the canonical recommendation and checkpoint path are available."
    )
    return f"""# Model Card - Blender Material Classifier

## Summary

{recommendation}

The model classifies synthetic Blender EEVEE renders into wood, metal, stone, concrete, ceramic, fabric, and leather. This card is generated by `scripts/update_docs.py` from cached result artifacts.

## Model and data

- Architecture: `{config.get("architecture", "Pending final selection")}`
- Pretrained: {_fmt(config.get("pretrained"))}
- Input: 224 x 224 RGB, resize 256 then center crop 224
- Normalization: ImageNet mean/std
- Training source: 3,150 synthetic renders from 105 independent PBR base materials
- Split policy: grouped by `material_id`; no material crosses train/validation/test
- Checkpoint: {checkpoint}

## Performance

| Metric | Value |
|---|---:|
| Test accuracy | {_fmt(metrics.get("accuracy"))} |
| Test balanced accuracy | {_fmt(metrics.get("balanced_accuracy"))} |
| Test macro F1 | {_fmt(metrics.get("f1_macro"))} |
| Test weighted F1 | {_fmt(metrics.get("f1_weighted"))} |
| Test ROC-AUC (macro) | {_fmt(metrics.get("roc_auc_macro"))} |
| Test log loss | {_fmt(metrics.get("log_loss"))} |
| Grouped-CV macro F1 (mean +/- SD) | {f"{_fmt(cv_mean)} +/- {_fmt(cv_std)}" if cv_mean is not None else "N/A (baseline)"} |

### Per-class behaviour

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{class_lines}

## Intended use

Use for research analysis of this synthetic material-render benchmark, reproducibility experiments, and model-comparison teaching. It may assist inspection of renders produced with the same Blender/RenderSynth setup.

## Inappropriate use and limitations

Do not use this model as a validated detector of arbitrary materials in photographs, production safety systems, or high-stakes decisions. The model is trained only on synthetic Blender PBR renders; synthetic-to-real generalisation, robustness to real lighting/viewpoints, and fairness across real-world material sources have not been established.

## Inference

{inference_block}

The class order is `wood`, `metal`, `stone`, `concrete`, `ceramic`, `fabric`, `leather`; exact preprocessing and class order are also stored in each checkpoint's resolved configuration. Re-run `python scripts/update_docs.py` after final cached artifacts change; do not hand-edit reported metrics.
"""


def update(root: Path) -> tuple[Path, Path]:
    rows = collect_rows(root)
    diagnostics = _gate_diagnostics(root)
    results = root / "reports" / "results"
    for required in (
        results / "model_comparison.csv",
        root / "reports" / "figures" / "comparison" / "model_comparison_dashboard.png",
        root / "reports" / "REPORT_ASSETS.md",
    ):
        if not required.exists():
            diagnostics.append(f"missing canonical comparison artifact: {required}")
    recommendation, recommendation_diagnostics = _recommendation(root)
    diagnostics.extend(recommendation_diagnostics)
    gate_ready = not diagnostics
    readme = root / "README.md"
    card = root / "MODEL_CARD.md"
    readme.write_text(render_readme(root, rows, gate_ready), encoding="utf-8")
    card.write_text(render_model_card(root, rows, recommendation, gate_ready), encoding="utf-8")
    if diagnostics:
        print("Documentation remains pending because canonical final artifacts are incomplete:")
        for diagnostic in diagnostics:
            print(f"- {diagnostic}")
    return readme, card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for path in update(args.root.resolve()):
        print(path)


if __name__ == "__main__":
    main()
