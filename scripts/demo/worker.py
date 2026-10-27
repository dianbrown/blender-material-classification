"""Persistent local JSONL worker; independent of the caller's current directory."""

import argparse
import os
import sys
from pathlib import Path

# Import this checkout/release's reviewed implementation, never an editable install elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from material_classifier.demo.runtime import ModelRuntime  # noqa: E402
from material_classifier.demo.service import run_service  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    # Reserve a dedicated protocol descriptor. Even native library stdout is redirected
    # to diagnostics before any ML import, protecting the one-response-per-line framing.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        run_service(ModelRuntime(args.root, args.device), sys.stdin.buffer, protocol)


if __name__ == "__main__":
    main()
