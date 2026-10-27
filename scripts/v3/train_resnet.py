"""Run the fixed supplemental ResNet baseline; OOF and final are separate jobs.

No feature cache, primary selector, early stopping, or outer checkpoint selection.
The config must already be committed. Output directories must not exist.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.features import load_protocol  # noqa: E402
from material_classifier.v3.resnet_baseline import load_config, train_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("oof", "final"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v3/resnet_baseline.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v3/protocol.json")
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        help="Optional primary fold_assignments.csv; validated, never scored",
    )
    parser.add_argument(
        "--physical-batch-size",
        type=int,
        choices=(16, 8),
        default=16,
        help="Explicit pre-run memory choice; effective batch remains 32",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="Require cached pinned weights")
    args = parser.parse_args()
    record = train_run(
        args.manifest,
        args.data_root,
        args.output,
        load_config(args.config, physical_batch_size=args.physical_batch_size),
        load_protocol(args.protocol),
        stage=args.stage,
        device=args.device,
        cpu_threads=args.cpu_threads,
        offline=args.offline,
        assignments_path=args.fold_assignments,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": record["status"],
                "run_identity_sha256": record["run_identity_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
