"""Derive a classifier view of a QC-approved renderer manifest without changing it."""

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3 import CLASS_NAMES  # noqa: E402
from material_classifier.v3.features import file_hash, image_file, json_hash  # noqa: E402


def derive_manifest(render_manifest, registry_path, qc_path, data_root, output):
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    qc = json.loads(Path(qc_path).read_text(encoding="utf-8"))
    if qc.get("passed") is not True or qc.get("data_only") is not True or qc.get("failures"):
        raise ValueError("Complete model-free QC must pass before deriving classifier inputs")
    with Path(render_manifest).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assets = {asset["material_id"]: asset for asset in registry["assets"]}
    if len(assets) != len(registry["assets"]):
        raise ValueError("Registry contains duplicate material identities")
    expected = len(assets) * registry["views_per_material"]
    if len(rows) != expected or qc["images"] != expected or qc["groups"] != len(assets):
        raise ValueError("Renderer/QC population does not match the sealed registry")
    if Counter(row["material_id"] for row in rows) != dict.fromkeys(
        assets, registry["views_per_material"]
    ):
        raise ValueError("Every sealed material must have its complete render quota")
    if Counter(asset["class"] for asset in assets.values()) != dict.fromkeys(
        CLASS_NAMES, registry["groups_per_class"]
    ):
        raise ValueError("Sealed source quota is not balanced across the seven classes")
    result, seen = [], set()
    for row in rows:
        asset = assets[row["material_id"]]
        path = row["image_path"]
        if path.casefold() in seen:
            raise ValueError("Duplicate image identity")
        seen.add(path.casefold())
        if (
            row["class"] != asset["class"]
            or row["source_asset_id"] != asset["asset_id"]
            or row["provider"] != asset["provider"]
            or row["original_split"] != "holdout"
            or row["selection_sha256"] != registry["selection_sha256"]
            or row["run_sha256"] != qc["run_sha256"]
        ):
            raise ValueError("Render identity/selection differs from the sealed registry or QC")
        if file_hash(image_file(data_root, path)) != row["image_sha256"]:
            raise ValueError("Render content changed after QC")
        result.append(
            {
                "image_path": path,
                "class": asset["class"],
                "material_id": asset["material_id"],
                "source_group": f"{asset['provider']}:{asset['asset_id']}",
                "original_split": "holdout",
                "sha256": row["image_sha256"],
            }
        )
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Classifier manifest output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(result[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(result)
    output.write_bytes(buffer.getvalue().encode("utf-8"))
    provenance = {
        "schema_version": 1,
        "images": len(result),
        "source_groups": len(assets),
        "manifest_sha256": file_hash(output),
        "render_records_sha256": json_hash(rows),
        "registry_sha256": json_hash(registry),
        "qc_sha256": json_hash(qc),
        "selection_sha256": registry["selection_sha256"],
        "run_sha256": qc["run_sha256"],
        "source_group_rule": (
            "provider:asset_id; source-family exclusion is established by the frozen "
            "source identity audit, not by identifier spelling alone"
        ),
        "authoritative_input_hashes": (
            "Canonical JSON records; invariant to CSV/JSON LF versus CRLF"
        ),
        "model_predictions_used": False,
    }
    output.with_suffix(".json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode())
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("render-manifest", "registry", "qc", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            derive_manifest(
                args.render_manifest, args.registry, args.qc, args.data_root, args.output
            )
        )
    )


if __name__ == "__main__":
    main()
