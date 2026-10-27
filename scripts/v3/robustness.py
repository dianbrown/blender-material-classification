"""CPU-stage preregistered corruptions; explicitly evaluate only after endpoint lock."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.features import load_protocol  # noqa: E402
from material_classifier.v3.robustness import (  # noqa: E402
    ESTIMATES,
    evaluate_staged,
    stage_corruptions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Owned staging directory under ignored artifacts/v3/robustness",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/robustness.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v3/protocol.json")
    parser.add_argument("--estimate-type", required=True, choices=ESTIMATES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--endpoint-locked", action="store_true")
    parser.add_argument("--model", type=Path, help="Trusted final joblib artifact")
    parser.add_argument("--clean-predictions", type=Path)
    parser.add_argument("--clean-run", type=Path, help="Completed clean prediction_run.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.evaluate and not all(
        (args.endpoint_locked, args.model, args.clean_predictions, args.clean_run)
    ):
        parser.error(
            "--evaluate requires --endpoint-locked, --model, --clean-predictions, --clean-run"
        )
    protocol = load_protocol(args.protocol)
    common = (args.manifest, args.data_root, args.output_dir, args.config, protocol)
    if args.evaluate:
        record = evaluate_staged(
            *common,
            estimate_type=args.estimate_type,
            resume=args.resume,
            model=args.model,
            clean_predictions=args.clean_predictions,
            clean_run=args.clean_run,
            endpoint_locked=args.endpoint_locked,
            device=args.device,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
            offline=args.offline,
        )
    else:
        record = stage_corruptions(*common, estimate_type=args.estimate_type, resume=args.resume)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "status": record["status"],
                "evaluated": args.evaluate,
                "identity": record["identity_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
