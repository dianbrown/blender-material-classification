"""Apply saved ResNet state only; --evaluate checks a fresh labeled holdout once.

Use <final-run>/final/model.pt and its adjacent model.json. No pretrained download
or training occurs. Labels/identities use the standard v3 manifest schema.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.resnet_baseline import predict_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Final model.pt + model.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New immutable output directory")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    predictions = predict_manifest(
        args.artifact,
        args.manifest,
        args.data_root,
        args.output,
        evaluate=args.evaluate,
        device=args.device,
        cpu_threads=args.cpu_threads,
        batch_size=args.batch_size,
    )
    print(json.dumps({"rows": len(predictions), "output": str(args.output)}))


if __name__ == "__main__":
    main()
