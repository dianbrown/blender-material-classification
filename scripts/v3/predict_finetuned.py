"""Predict with exported fine-tuned weights; --evaluate explicitly checks fresh overlap."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.finetune import predict_finetuned  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, required=True, help="Exported model.pt + model.json"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--image", help="Image path relative to --data-root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--expected-sha256", help="Additional expected model.pt checksum")
    parser.add_argument("--metadata-sha256", help="Expected model.json checksum from run exports")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.evaluate and not args.manifest:
        parser.error("--evaluate requires --manifest")
    result = predict_finetuned(**vars(args))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
