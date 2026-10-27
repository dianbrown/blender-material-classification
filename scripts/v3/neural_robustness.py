"""Evaluate the five staged corruptions after endpoint lock; NEW output directory only."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3.neural_robustness import (  # noqa: E402
    CONFIG_PATH,
    PROTOCOL_PATH,
    evaluate_staged,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("resnet", "dinov2"))
    for name, help_text in (
        ("artifact", "Trusted fixed model.pt; adjacent model.json is required"),
        ("clean-predictions", "Completed clean predictions.csv"),
        ("clean-run", "Matching completed clean prediction_run.json"),
        ("staging-dir", "Existing complete stage; never modified"),
        ("output-dir", "NEW directory; existing/incomplete outputs are rejected"),
        ("manifest", "Original holdout classification manifest"),
        ("data-root", "Original holdout image root"),
        ("endpoint-lock", "Schema-v1 lock written BEFORE fresh predictions"),
    ):
        parser.add_argument("--" + name, type=Path, required=True, help=help_text)
    parser.add_argument("--model-name", required=True, help="Friendly key in endpoint-lock models")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--config", dest="config_path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    record = evaluate_staged(**vars(args))
    print(json.dumps({"status": record["status"], "identity": record["identity_sha256"]}))


if __name__ == "__main__":
    main()
