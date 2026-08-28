"""Locked final evaluation entry point (never evaluates TEST before the gate)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    if args.split == "test":
        from material_classifier.evaluate import require_methodology_lock

        require_methodology_lock(args.root)
    from material_classifier.evaluate import evaluate_checkpoint

    data_cfg = yaml.safe_load((args.root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    from torch.utils.data import DataLoader

    from material_classifier import data
    from material_classifier.augmentations import build_eval_transform

    rows = data.load_render_manifest(args.root / "metadata" / "render_manifest.csv")
    records = data.records_for_split(rows, args.split)
    dataset = data.MaterialRenderDataset(
        records,
        args.root / "dataset",
        build_eval_transform(int(data_cfg.get("image", {}).get("input_size", 224))),
        classes=data_cfg.get("classes", data.CLASS_ORDER),
        return_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=__import__("torch").cuda.is_available(),
    )
    classes = data_cfg.get("classes", [])
    for name in ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16"):
        checkpoint = args.root / "artifacts" / "checkpoints" / name / "best.pt"
        if not checkpoint.exists():
            if args.split == "test":
                raise FileNotFoundError(
                    f"Required locked headline checkpoint missing: {checkpoint}"
                )
            print(f"skip {name}: {checkpoint} not found")
            continue
        metrics = evaluate_checkpoint(
            checkpoint,
            loader,
            args.root,
            args.split,
            classes,
            output_dir=args.root / "reports" / "results",
        )
        print(name, metrics)


if __name__ == "__main__":
    main()
