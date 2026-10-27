"""Generate portable static assets from already-saved study results and verified renders."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from material_classifier.demo.assets import prepare_assets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = prepare_assets(args.root, args.data_root, args.output or args.root / "app/assets")
    print(
        f"Prepared {len(manifest['models'])} saved model summaries and "
        f"{len(manifest['samples'])} verified gallery images."
    )


if __name__ == "__main__":
    main()
