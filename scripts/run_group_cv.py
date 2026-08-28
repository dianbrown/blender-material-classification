"""Grouped development-CV CLI; TEST is never read."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from material_classifier.augmentations import build_eval_transform, build_train_transform
from material_classifier.cross_validate import aggregate_cv, run_grouped_cv
from material_classifier.data import MaterialRenderDataset
from material_classifier.evaluate import evaluate_loader
from material_classifier.models import build_model
from material_classifier.train import train_with_oom_retry
from material_classifier.utils import load_torch_checkpoint, seed_worker


def run_one(
    root: Path, rows: list[dict[str, str]], model_name: str, folds: int, force: bool
) -> list[dict[str, object]]:
    config = yaml.safe_load((root / "configs" / f"{model_name}.yaml").read_text(encoding="utf-8"))
    data_config = yaml.safe_load((root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    classes, dataset_root = data_config["classes"], root / "dataset"
    config["data_contract"] = data_config
    config["classes"] = classes
    config["input_preprocessing"] = {
        "resize": 256,
        "center_crop": data_config.get("image", {}).get("input_size", 224),
        "normalization_mean": data_config.get("image", {}).get("normalization", {}).get("mean"),
        "normalization_std": data_config.get("image", {}).get("normalization", {}).get("std"),
    }

    def fold_runner(name, fold, train_idx, val_idx):
        directory = root / "artifacts" / "cv" / name / f"fold_{fold}"

        def loaders(batch):
            train_ds = MaterialRenderDataset(
                [rows[i] for i in train_idx], dataset_root, build_train_transform(224), classes
            )
            val_ds = MaterialRenderDataset(
                [rows[i] for i in val_idx], dataset_root, build_eval_transform(224), classes
            )
            generator = torch.Generator().manual_seed(42 + fold)
            kwargs = {
                "num_workers": 2,
                "pin_memory": torch.cuda.is_available(),
                "worker_init_fn": seed_worker,
            }
            return DataLoader(
                train_ds, batch_size=batch, shuffle=True, generator=generator, **kwargs
            ), DataLoader(val_ds, batch_size=batch, shuffle=False, **kwargs)

        result = train_with_oom_retry(
            lambda: build_model(
                name, len(classes), pretrained=bool(config.get("pretrained", True))
            ),
            loaders,
            config,
            directory,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            {label: i for i, label in enumerate(classes)},
            force,
        )
        _, val_loader = loaders(1)
        model = build_model(name, len(classes), pretrained=False)
        model.load_state_dict(load_torch_checkpoint(directory / "best.pt")["model_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        metrics = evaluate_loader(model, val_loader, device, classes)
        run_state = json.loads((directory / "run_state.json").read_text(encoding="utf-8"))
        metrics.update(
            {
                "best_epoch": result["best_epoch"],
                "training_time_seconds": run_state["total_training_time_seconds"],
            }
        )
        return metrics

    return run_grouped_cv(
        rows,
        model_name,
        fold_runner,
        folds,
        42,
        root / "reports" / "results" / "cross_validation_results.csv",
        force,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--model",
        choices=("resnet50", "convnext_tiny", "vit_b16"),
        default=None,
        help="Omit to run all three transfer models",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    with (args.root / "metadata" / "render_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in {"train", "val"}]
    for model_name in [args.model] if args.model else ["resnet50", "convnext_tiny", "vit_b16"]:
        result = run_one(args.root.resolve(), rows, model_name, args.folds, args.force)
        print(aggregate_cv(result))


if __name__ == "__main__":
    main()
