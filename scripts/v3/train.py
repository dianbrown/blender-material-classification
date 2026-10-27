"""Run five-by-three nested source-group CV, then fit final selected/control artifacts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.features import load_protocol  # noqa: E402
from material_classifier.v3.learning import train_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New immutable run directory")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v3/protocol.json")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    record = train_run(
        args.manifest,
        args.data_root,
        args.feature_cache,
        args.output,
        load_protocol(args.protocol),
        cpu_threads=args.cpu_threads,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "run_identity": record["run_identity_sha256"],
                "final_selection": record["final_selection"],
            }
        )
    )


if __name__ == "__main__":
    main()
