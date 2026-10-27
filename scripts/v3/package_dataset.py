"""Stage a local ZIP only; source licenses block release unless --audit-only is explicit."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from material_classifier.v3.data import package_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, default=REPO / "metadata/v3")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Archive outside this isolated checkout, e.g. parent artifacts/v3/",
    )
    parser.add_argument("--license-registry", type=Path)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Allow unresolved licenses/conflicts; mark archive NOT PUBLISHABLE",
    )
    parser.add_argument("--kaggle-id", default="local-audit/material-classification-v3")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(REPO):
        parser.error("stage outside this worktree; do not copy the corpus into the checkout")
    try:
        result = package_dataset(
            args.data_root,
            args.audit_dir,
            args.output,
            audit_only=args.audit_only,
            registry_path=args.license_registry,
            kaggle_id=args.kaggle_id,
        )
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
