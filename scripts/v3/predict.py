"""Apply an immutable final model to an image or manifest; never fit on new data."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.features import image_file, write_json  # noqa: E402
from material_classifier.v3.inference import predict_image, predict_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Trusted model.joblib")
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--image", help="Image path relative to --data-root")
    parser.add_argument(
        "--output", type=Path, required=True, help="New prediction output directory"
    )
    parser.add_argument("--feature-cache", type=Path, help="Manifest cache: reuse or create")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--evaluate", action="store_true", help="Score an independent labeled manifest"
    )
    args = parser.parse_args()
    if args.image:
        if args.evaluate or args.feature_cache:
            parser.error("--evaluate and --feature-cache apply only to --manifest")
        args.output.mkdir(parents=True, exist_ok=False)
        result = predict_image(
            args.artifact,
            image_file(args.data_root, args.image),
            device=args.device,
            cpu_threads=args.cpu_threads,
            offline=args.offline,
        )
        write_json(args.output / "prediction.json", result)
        print(json.dumps(result))
    else:
        predictions = predict_manifest(
            args.artifact,
            args.manifest,
            args.data_root,
            args.output,
            feature_cache=args.feature_cache,
            device=args.device,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
            offline=args.offline,
            evaluate=args.evaluate,
        )
        print(json.dumps({"rows": len(predictions), "output": str(args.output)}))


if __name__ == "__main__":
    main()
