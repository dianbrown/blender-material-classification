"""Bounded exploratory development selection, then a separately frozen final fit.

All pretrained weights must already be in the local Hugging Face cache. This
script never installs packages or downloads weights. --resume reuses identical
completed folds and replays only an uncommitted interrupted epoch.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.features import load_protocol, read_json  # noqa: E402
from material_classifier.v3.finetune import run_finetune  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-run",
        type=Path,
        required=True,
        help="Original run containing final_folds.json and model artifacts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v3/protocol.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/finetune.json")
    parser.add_argument("--stage", choices=("select", "final"), default="select")
    parser.add_argument("--selection-sha256", help="Required for final: exact selection.json hash")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "final" and (not args.resume or not args.selection_sha256):
        parser.error("--stage final requires --resume and --selection-sha256")
    result = run_finetune(
        args.manifest,
        args.data_root,
        args.frozen_run,
        args.output,
        load_protocol(args.protocol),
        read_json(args.config),
        stage=args.stage,
        resume=args.resume,
        selection_sha256=args.selection_sha256,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
