"""Build the offline UI manifest from saved results, never rerun evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from . import CLASS_NAMES, LOCK_PATH, MODEL_NAMES

DEFINITIONS = {
    "wood": "Wood grain and wood-derived surface appearances.",
    "metal": "Metallic surface appearances under the study's shader convention.",
    "stone": "Natural stone and rock surface appearances.",
    "concrete": "Concrete, plaster and painted-wall-like archival sources.",
    "ceramic": "Ceramic, tile, mosaic and terrazzo-like archival sources.",
    "fabric": "Woven and other textile surface appearances.",
    "leather": "Leather-like surface appearances, including related generator families.",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def counts(rows):
    return {
        "images": len(rows),
        "materials": len({r["material_id"] for r in rows}),
        "source_groups": len({r["source_group"] for r in rows}),
    }


def compact_ci(bootstrap, name):
    interval = bootstrap["intervals"][name]
    return {
        "low": interval["lower"],
        "high": interval["upper"],
        "level": bootstrap["confidence_level"],
        "unit": bootstrap["resampling_unit"],
    }


def prepare_assets(root, data_root, output):
    """Select three source-distinct development examples/class without using scores."""
    from PIL import Image

    root, data_root, output = Path(root), Path(data_root), Path(output)
    lock = read_json(root / LOCK_PATH)
    if lock["class_order"] != list(CLASS_NAMES):
        raise ValueError("Unexpected study class order")
    development = read_csv(root / "metadata/v3/manifest.csv")
    holdout = read_csv(root / "metadata/v3_holdout/classification_manifest.csv")
    sources = {
        row["material_id"]: row
        for row in read_json(root / "metadata/v3/license_registry.json")["sources"]
    }
    models = []
    for model_id, display_name in MODEL_NAMES.items():
        base = f"reports/results/v3/fresh/{model_id}"
        # Evaluation JSON is the saved raw scoring output, including cluster CIs.
        evaluation = read_json(root / base / "evaluation.json")
        summary = evaluation["image_level"]
        benchmark_path = f"reports/results/v3/benchmarks/{model_id}/benchmark.json"
        benchmark = read_json(root / benchmark_path)
        warm = next(
            r for r in benchmark["results"] if r["scope"] == "warm" and r["batch_size"] == 1
        )
        deployment = benchmark["deployment"]
        if evaluation["classes"] != list(CLASS_NAMES):
            raise ValueError("Unexpected result class order")
        metrics = summary["metrics"]
        models.append(
            {
                "id": model_id,
                "display_name": display_name,
                "role": lock["models"][model_id]["role"],
                "family": lock["models"][model_id]["family"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "accuracy_ci": compact_ci(summary["bootstrap"], "accuracy"),
                "macro_f1_ci": compact_ci(summary["bootstrap"], "macro_f1"),
                "confusion_matrix": summary["confusion_matrix"],
                "per_class": summary["per_class"],
                "latency": {
                    "median_ms": warm["median_seconds"] * 1000,
                    "p95_ms": warm["p95_seconds"] * 1000,
                    "device": benchmark["hardware"]
                    .get("cuda", {})
                    .get("name", benchmark["hardware"]["device"]),
                    "scope": benchmark["scopes"]["warm"],
                    "batch_size": 1,
                    "repetitions": warm["repetitions"],
                },
                "size": {
                    "artifact_bytes": deployment["artifact"]["bytes"],
                    "total_deployment_bytes": deployment["total_deployment_bytes"],
                },
                "sources": [f"{base}/evaluation.json", benchmark_path, LOCK_PATH],
            }
        )
    samples = []
    (output / "samples").mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        seen = set()
        candidates = sorted(
            (r for r in development if r["class"] == class_name),
            key=lambda r: (r["material_id"], r["image_path"]),
        )
        for row in candidates:
            if row["source_group"] in seen:
                continue
            source = sources[row["material_id"]]
            if source["verification_status"] != "verified" or not source["redistribution_allowed"]:
                raise ValueError("Sample source must have verified redistribution evidence")
            path = (data_root / row["image_path"]).resolve()
            if not path.is_relative_to(data_root.resolve()):
                raise ValueError("Sample escaped dataset root")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Sample no longer matches audited dataset")
            sample_id = f"{class_name}_{len(seen) + 1:02d}"
            relative = f"samples/{sample_id}.png"
            with Image.open(path) as image:
                image.load()
                # Fresh object drops all PNG text, EXIF, ICC and other metadata.
                clean = Image.frombytes(image.mode, image.size, image.tobytes())
                clean.save(output / relative)
                with Image.open(output / relative) as saved:
                    if saved.tobytes() != image.tobytes() or saved.info:
                        raise ValueError("Sample pixels/metadata verification failed")
            samples.append(
                {
                    "id": sample_id,
                    "label": (
                        f"{class_name.title()} · {row['source_material_name'].replace('_', ' ')}"
                    ),
                    "class_name": class_name,
                    "path": relative,
                    "source_group": row["source_group"],
                    "material_id": row["material_id"],
                    "split": "development",
                    "attribution": source["attribution"],
                    "source_url": source["source_url"],
                    "license": source["license_spdx"],
                }
            )
            seen.add(row["source_group"])
            if len(seen) == 3:
                break
        if len(seen) != 3:
            raise ValueError("Three distinct sample sources per class required")
    manifest = {
        "schema_version": 1,
        "class_names": list(CLASS_NAMES),
        "primary_model": lock["primary_model"],
        "default_model": "dinov2_finetuned",
        "results_label": "Stored study results",
        "live_label": "Live local inference",
        "results_scope": "Previously evaluated 630-image, 21-source holdout; no new evaluation.",
        "ci_method": "95% class-stratified source-group cluster bootstrap, 10,000 resamples.",
        "confusion_axes": "Rows: true class; columns: predicted class; class_names order.",
        "sample_note": "Development examples selected without prediction scores; gallery labels "
        "are archived ground truth. Classify runs live inference.",
        "dataset": {
            "development": counts(development),
            "holdout": counts(holdout),
            "classes": [
                {
                    "name": name,
                    "definition": DEFINITIONS[name],
                    "development_images": sum(r["class"] == name for r in development),
                    "holdout_images": sum(r["class"] == name for r in holdout),
                }
                for name in CLASS_NAMES
            ],
            "scope": lock["scope"],
        },
        "models": models,
        "samples": samples,
    }
    (output / "study.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
