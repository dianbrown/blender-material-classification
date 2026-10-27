"""Evaluate explicit v3 predictions; no dataset or prediction autodiscovery.

Example (from repository root)::

    python scripts/v3/evaluate.py --predictions artifacts/v3/oof.csv \
        --metadata artifacts/v3/metadata.json --output-dir reports/results/v3/model

Metadata must contain an ordered ``classes``, ``class_names`` or ``class_order`` JSON array.
Alternatively use ``--classes wood metal stone concrete ceramic fabric leather``.
Labels may be names or zero-based indices; --label-format resolves ambiguity.
Default estimate type is nested_oof. Use --estimate-type fresh_holdout for
independent new sources, with outer_fold='holdout' or -1. Bootstrap resamples
whole sources within class by default; --n-bootstrap 10000 increases CI draws.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct invocation also works without an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from material_classifier.v3.evaluation import evaluate_predictions, write_evaluation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Explicit OOF CSV path")
    parser.add_argument("--classes", nargs="+", help="Class names in probability-column order")
    parser.add_argument("--metadata", type=Path, help="JSON with ordered class names")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-format", choices=("auto", "names", "indices"), default="auto")
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000, help="Source-group draws; 0 disables CIs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15, help="Equal-width ECE bins")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--estimate-type",
        default="nested_oof",
        choices=("nested_oof", "fresh_holdout", "independent_holdout"),
        help="Explicit evaluation provenance; holdout accepts outer_fold=holdout or -1",
    )
    parser.add_argument(
        "--unstratified-bootstrap",
        action="store_true",
        help="Disable default class-stratified source-group resampling",
    )
    args = parser.parse_args(argv)
    try:
        classes = args.classes
        if args.metadata is not None:
            metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON object")
            metadata_classes = metadata.get(
                "classes", metadata.get("class_names", metadata.get("class_order"))
            )
            if not isinstance(metadata_classes, list):
                raise ValueError("Metadata must contain classes, class_names or class_order")
            if classes is not None and classes != metadata_classes:
                raise ValueError("--classes and metadata class order disagree")
            classes = metadata_classes
        if classes is None:
            raise ValueError("Supply --classes or --metadata; class order is never inferred")
        result = evaluate_predictions(
            args.predictions,
            classes,
            label_format=args.label_format,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            n_bins=args.n_bins,
            estimate_type=args.estimate_type,
            stratify_by_class=not args.unstratified_bootstrap,
        )
        result["input_predictions"] = str(args.predictions.resolve())
        result["input_metadata"] = str(args.metadata.resolve()) if args.metadata else None
        paths = write_evaluation(result, args.output_dir, figures=not args.no_figures)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(result["estimate_type"])
    metrics = result["image_level"]["metrics"]
    print(f"Images: {metrics['n_images']}; source groups: {metrics['n_source_groups']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}")
    print(f"Evaluation: {paths['evaluation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
