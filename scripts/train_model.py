"""Thin CLI wrapper for one fixed train/validation headline run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from material_classifier.models import build_model
from material_classifier.train import train_with_oom_retry


def _load_loaders(config: dict):
    from material_classifier import data

    for name in ("build_dataloaders", "create_dataloaders", "make_dataloaders"):
        fn = getattr(data, name, None)
        if fn is None:
            continue
        try:
            out = fn(config, splits=("train", "val"))
        except TypeError:
            out = fn(config)
        if isinstance(out, dict):
            return out["train"], out["val"]
        return out
    # The manifest/data module intentionally does not own a global loader
    # policy, so the CLI supplies the canonical transforms and split filters.
    from torch.utils.data import DataLoader

    from material_classifier.augmentations import build_eval_transform, build_train_transform

    root = Path(__file__).resolve().parents[1]
    rows = data.load_render_manifest(root / "metadata" / "render_manifest.csv")
    common = dict(dataset_root=root / "dataset", classes=config.get("classes", data.CLASS_ORDER))
    train = data.MaterialRenderDataset(
        data.records_for_split(rows, "train"),
        transform=build_train_transform(int(config.get("input_size", 224))),
        **common,
    )
    val = data.MaterialRenderDataset(
        data.records_for_split(rows, "val"),
        transform=build_eval_transform(int(config.get("input_size", 224))),
        **common,
    )
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    kwargs = {
        "batch_size": batch_size,
        "num_workers": min(4, (os.cpu_count() or 2) // 2),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": __import__(
            "material_classifier.utils", fromlist=["seed_worker"]
        ).seed_worker,
    }
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(val, shuffle=False, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch in artifacts/smoke; never a headline namespace",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "data.yaml").read_text(encoding="utf-8")
    )
    # Keep the exact data contract in every experiment artifact.
    config["data_contract"] = data_config
    config.setdefault("classes", data_config.get("classes", []))
    image = data_config.get("image", {})
    config["input_preprocessing"] = {
        "resize": 256,
        "center_crop": int(image.get("input_size", config.get("input_size", 224))),
        "normalization_mean": image.get("normalization", {}).get("mean", [0.485, 0.456, 0.406]),
        "normalization_std": image.get("normalization", {}).get("std", [0.229, 0.224, 0.225]),
    }
    if args.smoke:
        config["run_kind"] = "smoke"
        config["training"] = dict(
            config.get("training", {}), max_epochs=1, early_stopping_patience=1
        )
    name = config["name"]
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / (
        f"artifacts/smoke/{name}"
        if args.smoke
        else config.get("checkpoint", {}).get("directory", f"artifacts/checkpoints/{name}")
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = config.get(
        "classes", ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]
    )

    def loaders(batch_size):
        adjusted = dict(config)
        adjusted["training"] = dict(config.get("training", {}), batch_size=batch_size)
        return _load_loaders(adjusted)

    result = train_with_oom_retry(
        lambda: build_model(
            name,
            int(config.get("num_classes", 7)),
            pretrained=bool(config.get("pretrained", False)),
        ),
        loaders,
        config,
        checkpoint_dir,
        device,
        {c: i for i, c in enumerate(classes)},
        args.force,
    )
    print(result)


if __name__ == "__main__":
    main()
