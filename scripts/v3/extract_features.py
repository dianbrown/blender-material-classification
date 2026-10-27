"""Extract immutable, reusable frozen feature arrays (one GPU process at a time)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.features import extract_features, load_protocol  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New cache directory")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v3/protocol.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached pinned weights only")
    args = parser.parse_args()
    record = extract_features(
        args.manifest,
        args.data_root,
        args.output,
        load_protocol(args.protocol),
        batch_size=args.batch_size,
        device=args.device,
        threads=args.cpu_threads,
        resume=args.resume,
        offline=args.offline,
    )
    print(
        json.dumps(
            {
                "cache": str(args.output),
                "identity": record["identity_sha256"],
                "complete": record["complete"],
            }
        )
    )


if __name__ == "__main__":
    main()
