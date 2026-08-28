"""Run all four headline model configs; completed runs resume/skip safely."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16"):
        cmd = [
            sys.executable,
            str(root / "scripts" / "train_model.py"),
            "--config",
            str(root / "configs" / f"{name}.yaml"),
        ]
        if args.force:
            cmd.append("--force")
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
