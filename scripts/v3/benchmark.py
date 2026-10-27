"""Benchmark a locked artifact on its exact development corpus, without downloads.

Run separately for the frozen selected joblib, frozen control joblib, ResNet final
.pt and fine-tuned DINO .pt when available. Use identical arguments/new output
directories for comparison, only after training and artifact selection are locked.
Warm timings exclude disk I/O and decode. --cold-repetitions 3 adds process-warm
reload plus one predecoded image, NOT fresh-process/OS-cold CLI startup.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.benchmark import DEFAULT_SEED, benchmark  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("frozen", "resnet", "dinov2"))
    parser.add_argument(
        "--artifact", type=Path, required=True, help="Trusted final artifact + sidecar"
    )
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--device", required=True, help="Explicit cpu, cuda or cuda:N; no fallback")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cold-repetitions", type=int, choices=(0, 3), default=0)
    args = parser.parse_args(argv)
    record = benchmark(**vars(args))
    print(json.dumps({"status": record["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
