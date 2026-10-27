"""Data-only, pre-score QC for a COMPLETE frozen fresh holdout. No model imports."""

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_common import (  # noqa: E402
    absolute_path,
    digest,
    fingerprint,
    inside,
    matches_text_digest,
    read_json,
    require_suffix,
    validate_identity_audit,
    validate_png,
    validate_receipt,
    validate_registry,
    write_json,
)
from render_holdout import FIELDS, parameters, validate_config  # noqa: E402


def main(argv=None):
    from PIL import Image, ImageStat

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "registry",
        "config",
        "manifest",
        "dataset-root",
        "report",
        "receipt",
        "sources-root",
        "identity-audit",
        "original-manifest",
    ):
        parser.add_argument("--" + name, required=True, type=absolute_path)
    args = parser.parse_args(argv)
    require_suffix(args.dataset_root, ("artifacts", "v3", "holdout_dataset"))
    require_suffix(args.manifest, ("metadata", "v3_holdout", "render_manifest.csv"))
    if args.report != args.dataset_root / "qc.json":
        raise ValueError("QC report must be dataset-root/qc.json")
    registry, config, receipt = (
        read_json(args.registry),
        read_json(args.config),
        read_json(args.receipt),
    )
    validate_registry(registry, args.original_manifest)
    validate_config(config)
    validate_identity_audit(registry, read_json(args.identity_audit))
    validate_receipt(registry, receipt, args.sources_root)
    if config["selection_sha256"] != registry["selection_sha256"] or not matches_text_digest(
        args.identity_audit, config["identity_audit_sha256"]
    ):
        raise ValueError("selection or identity audit differs from frozen protocol")
    run = read_json(args.dataset_root / "run_fingerprint.json")
    run_hash = run.pop("run_sha256")
    if (
        fingerprint(run) != run_hash
        or run["config_sha256"] != digest(args.config)
        or run["selection_sha256"] != registry["selection_sha256"]
        or run["source_sha256"] != fingerprint(receipt)
        or run["runtime"]["mode"] != "holdout"
        or run["runtime"]["engine"] != "BLENDER_EEVEE"
    ):
        raise ValueError("invalid or non-production run fingerprint")
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("unsupported manifest schema")
        rows = list(reader)
    assets = {a["material_id"]: a for a in registry["assets"]}
    expected = {(mid, i) for mid in assets for i in range(1, 31)}
    keys = [(r["material_id"], int(r["render_index"])) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("QC requires exactly 30 unique images for every frozen source")
    names = {r["image_path"] for r in rows}
    if names != {
        p.relative_to(args.dataset_root).as_posix() for p in args.dataset_root.rglob("*.png")
    }:
        raise ValueError("unmanifested or missing images")
    limits = config["qc_policy"]["pixel_limits"]
    failures = []
    for row in rows:
        asset = assets[row["material_id"]]
        index = int(row["render_index"])
        state = parameters(config, asset["class"], asset["material_id"], index)
        expected_path = f"{asset['class']}/{asset['material_id']}_r{index:02d}.png"
        if (
            row["image_path"] != expected_path
            or row["class"] != asset["class"]
            or row["source_asset_id"] != asset["asset_id"]
            or row["provider"] != asset["provider"]
            or row["split"] != "holdout"
            or row["original_split"] != "holdout"
            or row["run_sha256"] != run_hash
            or row["parameters_sha256"] != fingerprint(state)
            or any(row[k] != str(v) for k, v in state.items())
            or any(row[k] != run[k] for k in run if k != "runtime")
            or row["render_engine"] != "BLENDER_EEVEE"
            or row["blender_version"] != "5.1.1"
            or row["resolution_width"] != "512"
            or row["resolution_height"] != "512"
        ):
            raise ValueError(f"stale or invalid render row: {row['image_path']}")
        path = inside(args.dataset_root, row["image_path"])
        if validate_png(path, 512, 512) != row["image_sha256"]:
            raise ValueError("PNG changed after rendering")
        # Analytic silhouette of a unit sphere aimed at its centre in a square frame.
        fraction = (
            2
            * config["camera_lens_mm"]
            / config["camera_sensor_width_mm"]
            / math.sqrt(state["camera_distance"] ** 2 - 1)
        )
        lo, hi = config["framing_height_fraction"]
        if not lo <= fraction <= hi:
            failures.append([row["image_path"], "framing", fraction])
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError("production output must be RGB")
            pixels = list(image.getdata())
            total = len(pixels)
            black = sum(max(p) < 6 for p in pixels) / total
            white = sum(min(p) > 249 for p in pixels) / total
            magenta = sum(r > 229 and g < 26 and b > 229 for r, g, b in pixels) / total
            deviation = ImageStat.Stat(image.convert("L")).stddev[0] / 255
        for name, actual, maximum in (
            ("near_black", black, limits["max_near_black_fraction"]),
            ("near_white", white, limits["max_near_white_fraction"]),
            ("magenta", magenta, limits["max_magenta_fraction"]),
        ):
            if actual > maximum:
                failures.append([row["image_path"], name, actual])
        if deviation < limits["min_luminance_std"]:
            failures.append([row["image_path"], "blank", deviation])
    result = {
        "data_only": True,
        "passed": not failures,
        "images": len(rows),
        "groups": len(assets),
        "class_counts": dict(Counter(r["class"] for r in rows)),
        "run_sha256": run_hash,
        "failures": failures,
        "action": "No replacements or appearance tuning; quarantine failed runs for investigation.",
    }
    write_json(args.report, result)
    if failures:
        raise SystemExit(f"Data-only QC failed for {len(failures)} checks; see {args.report}")
    print(f"DATA_ONLY_QC_PASS images={len(rows)} groups={len(assets)}")


if __name__ == "__main__":
    main()
