"""Audit parent data in place, without copying images/textures into this checkout."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from material_classifier.v3.data import (  # noqa: E402
    audit_data,
    family_registry_sha256,
    json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--render-manifest", type=Path, default=REPO / "metadata/render_manifest.csv"
    )
    parser.add_argument(
        "--material-manifest", type=Path, default=REPO / "metadata/material_manifest_split.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "metadata/v3")
    parser.add_argument(
        "--family-overrides", type=Path, default=REPO / "metadata/v3/source_family_overrides.json"
    )
    parser.add_argument("--report", type=Path, default=REPO / "reports/results/v3/data_audit.json")
    parser.add_argument("--expected-images", type=int, default=3150)
    parser.add_argument("--expected-materials", type=int, default=105)
    args = parser.parse_args()
    family_registry = json.loads(args.family_overrides.read_text(encoding="utf-8"))
    overrides = family_registry["overrides"]
    frozen_hash = family_registry.get("frozen_registry_sha256")
    if frozen_hash and frozen_hash != family_registry_sha256(overrides):
        parser.error("family registry differs from frozen decisions; re-register before evaluation")
    result = audit_data(
        args.data_root,
        args.render_manifest,
        args.material_manifest,
        args.output_dir,
        expected_images=args.expected_images,
        expected_materials=args.expected_materials,
        family_overrides=overrides,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(json_bytes(result))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "image_count",
                    "material_count",
                    "source_group_count",
                    "readable_render_count",
                    "integrity_errors",
                ]
            },
            indent=2,
        )
    )
    if result["integrity_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
