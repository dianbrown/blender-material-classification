"""Create the one-way gate required before held-out TEST evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from material_classifier.utils import (
    atomic_json_dump,
    git_commit,
    now_utc,
    sha256_file,
    stable_hash,
)

MODELS = ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16")
CV_MODELS = ("resnet50", "convnext_tiny", "vit_b16")
SOURCE_FILES = (
    "src/material_classifier/models.py",
    "src/material_classifier/train.py",
    "src/material_classifier/evaluate.py",
    "src/material_classifier/cross_validate.py",
    "src/material_classifier/data.py",
    "src/material_classifier/augmentations.py",
    "src/material_classifier/metrics.py",
    "src/material_classifier/utils.py",
    "src/material_classifier/inference.py",
    "scripts/evaluate_all.py",
    "scripts/run_group_cv.py",
    "scripts/train_model.py",
    "scripts/lock_methodology.py",
)


def _require(path: Path, description: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Cannot lock methodology: missing {description}: {path}")


def create_lock(root: Path) -> Path:
    root = Path(root).resolve()
    target = root / "reports" / "results" / "methodology_lock.json"
    if target.exists():
        raise FileExistsError(f"Methodology is already locked at {target}; it is immutable")
    validation_path = root / "reports" / "results" / "dataset_validation.json"
    _require(validation_path, "dataset validation report")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise RuntimeError("Cannot lock methodology: dataset validation has not passed")
    config_hashes = {}
    for path in sorted((root / "configs").glob("*.yaml")):
        config_hashes[path.name] = sha256_file(path)
    checkpoints: dict[str, str] = {}
    for model in MODELS:
        checkpoint = root / "artifacts" / "checkpoints" / model / "best.pt"
        state_path = root / "artifacts" / "checkpoints" / model / "run_state.json"
        _require(checkpoint, f"{model} best checkpoint")
        _require(state_path, f"{model} run state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        resolved_path = root / "artifacts" / "checkpoints" / model / "resolved_config.json"
        if (
            state.get("status") != "completed"
            or not state.get("config_hash")
            or state.get("model") not in {None, model}
        ):
            raise RuntimeError(f"Cannot lock methodology: {model} run is not completed")
        _require(resolved_path, f"{model} resolved config")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if (
            state.get("config_hash") != resolved.get("config_hash")
            or state.get("resolved_config") != resolved
        ):
            raise RuntimeError(f"Cannot lock methodology: {model} run/config hash mismatch")
        try:
            import torch

            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu")
        recomputed_hash = stable_hash(checkpoint_payload.get("config", {}))
        if (
            checkpoint_payload.get("config_hash") != recomputed_hash
            or checkpoint_payload.get("config_hash") != state.get("config_hash")
            or checkpoint_payload.get("resolved_config") != resolved
        ):
            raise RuntimeError(f"Cannot lock methodology: {model} checkpoint/config hash mismatch")
        checkpoints[f"artifacts/checkpoints/{model}/best.pt"] = sha256_file(checkpoint)
        checkpoints[f"artifacts/checkpoints/{model}/run_state.json"] = sha256_file(state_path)
        checkpoints[f"artifacts/checkpoints/{model}/resolved_config.json"] = sha256_file(
            resolved_path
        )
    cv_path = root / "reports" / "results" / "cross_validation_results.csv"
    _require(cv_path, "grouped CV results")
    with cv_path.open(newline="", encoding="utf-8") as handle:
        cv_rows = list(csv.DictReader(handle))
    required_cv = {
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "best_epoch",
        "training_time_seconds",
    }
    for model in CV_MODELS:
        model_rows = [row for row in cv_rows if row.get("model") == model]
        folds = [int(row["fold"]) for row in model_rows if row.get("fold", "").strip()]
        if (
            len(model_rows) != 5
            or sorted(folds) != [1, 2, 3, 4, 5]
            or any(not required_cv.issubset(row) for row in model_rows)
        ):
            raise RuntimeError(
                f"Cannot lock methodology: {model} requires exactly one complete row for folds 1..5"
            )
        for row in model_rows:
            if any(
                not row.get(field, "").strip() or not math.isfinite(float(row[field]))
                for field in required_cv
            ):
                raise RuntimeError(
                    f"Cannot lock methodology: non-finite CV value for {model} fold {row['fold']}"
                )
    aggregate_path = cv_path.with_name(f"{cv_path.stem}_aggregate{cv_path.suffix}")
    _require(aggregate_path, "grouped CV aggregate results")
    artifacts = {
        "reports/results/dataset_validation.json": sha256_file(validation_path),
        "reports/results/cross_validation_results.csv": sha256_file(cv_path),
        "reports/results/cross_validation_results_aggregate.csv": sha256_file(aggregate_path),
        **checkpoints,
    }
    for relative in SOURCE_FILES:
        path = root / relative
        _require(path, f"methodology source {relative}")
        artifacts[relative] = sha256_file(path)
    for name in (
        "metadata/material_manifest.csv",
        "metadata/material_manifest_split.csv",
        "metadata/render_manifest.csv",
        "configs/data.yaml",
    ):
        path = root / name
        _require(path, f"canonical artifact {name}")
        artifacts[name] = sha256_file(path)
    payload = {
        "locked": True,
        "test_evaluation_allowed": True,
        "locked_at": now_utc(),
        "git_commit": git_commit(root),
        "config_sha256": config_hashes,
        "artifacts_sha256": artifacts,
        "protocol": (
            "Original train/val headline configuration; grouped CV development only; "
            "test evaluated once per model after lock."
        ),
    }
    payload["lock_payload_hash"] = stable_hash(payload)
    atomic_json_dump(payload, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(create_lock(args.root))


if __name__ == "__main__":
    main()
