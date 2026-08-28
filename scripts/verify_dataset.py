"""Strict, leakage-aware verification of the local rendered dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather")
SPLITS = ("train", "val", "test")
EXPECTED = {
    "total_images": 3150,
    "images_per_class": 450,
    "images_per_material": 30,
    "materials_total": 105,
    "materials_per_class": 15,
    "images": {"train": 2310, "val": 420, "test": 420},
    "images_per_class_by_split": {"train": 330, "val": 60, "test": 60},
    "materials_by_split": {"train": 77, "val": 14, "test": 14},
}
ID_RE = re.compile(r"^(?P<class>[a-z]+)_(?P<number>\d{2})$")
FILENAME_RE = re.compile(r"^(?P<class>[a-z]+)_(?P<number>\d{2})_r(?P<render>\d{2})\.png$")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _check(checks: dict[str, dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def verify(
    project_root: str | Path = PROJECT_ROOT,
    *,
    render_manifest: str | Path | None = None,
    split_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run all structural, inheritance, and full image-decode checks."""

    root = Path(project_root).resolve()
    render_path = root / (render_manifest or "metadata/render_manifest.csv")
    split_path = root / (split_manifest or "metadata/material_manifest_split.csv")
    checks: dict[str, dict[str, Any]] = {}
    result: dict[str, Any] = {"schema_version": 1, "expected": EXPECTED, "checks": checks}

    try:
        rows = _read_csv(render_path)
        split_rows = _read_csv(split_path)
    except Exception as exc:
        _check(checks, "manifests_readable", False, str(exc))
        result["passed"] = False
        return result

    required_render = {"image_path", "filename", "class", "material_id", "split", "render_index"}
    required_split = {"class", "material_id", "split"}
    schema_ok = required_render.issubset(rows[0] if rows else set()) and required_split.issubset(
        split_rows[0] if split_rows else set()
    )
    _check(
        checks,
        "manifest_schema",
        schema_ok,
        {
            "render_columns": sorted(rows[0]) if rows else [],
            "split_columns": sorted(split_rows[0]) if split_rows else [],
        },
    )

    manifest_paths = [row.get("image_path", "").replace("\\", "/") for row in rows]
    duplicate_paths = sorted(path for path, count in Counter(manifest_paths).items() if count > 1)
    duplicate_keys = sorted(
        key
        for key, count in Counter(
            (row.get("material_id", ""), row.get("render_index", "")) for row in rows
        ).items()
        if count > 1
    )
    _check(
        checks,
        "no_duplicate_manifest_entries",
        not duplicate_paths and not duplicate_keys,
        {
            "duplicate_image_paths": duplicate_paths,
            "duplicate_material_render_keys": [list(x) for x in duplicate_keys],
        },
    )

    dataset = root / "dataset"
    disk_paths = (
        sorted(
            path.relative_to(root).as_posix()
            for path in dataset.rglob("*")
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if dataset.is_dir()
        else []
    )
    _check(
        checks,
        "manifest_disk_set_equality",
        set(manifest_paths) == set(disk_paths),
        {
            "missing": sorted(set(manifest_paths) - set(disk_paths)),
            "unmanifested": sorted(set(disk_paths) - set(manifest_paths)),
            "disk_png_count": len(disk_paths),
            "manifest_count": len(rows),
        },
    )
    _check(checks, "expected_total_rows", len(rows) == EXPECTED["total_images"], len(rows))

    folder_issues: list[list[str]] = []
    if dataset.is_dir():
        for split_dir in dataset.iterdir():
            if split_dir.is_dir() and split_dir.name not in SPLITS:
                folder_issues.append([split_dir.name])
            if split_dir.is_dir() and split_dir.name in SPLITS:
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir() and class_dir.name not in CLASSES:
                        folder_issues.append([split_dir.name, class_dir.name])
    _check(checks, "no_unexpected_class_folders", not folder_issues, folder_issues)

    split_counts = Counter(row.get("split", "") for row in rows)
    class_counts = Counter(row.get("class", "") for row in rows)
    class_split_counts = Counter((row.get("class", ""), row.get("split", "")) for row in rows)
    _check(checks, "split_counts", dict(split_counts) == EXPECTED["images"], dict(split_counts))
    _check(
        checks, "class_counts", dict(class_counts) == {c: 450 for c in CLASSES}, dict(class_counts)
    )
    expected_cs = {
        (c, s): n for c in CLASSES for s, n in (("train", 330), ("val", 60), ("test", 60))
    }
    _check(
        checks,
        "class_split_counts",
        dict(class_split_counts) == expected_cs,
        {f"{c}/{s}": class_split_counts[(c, s)] for c in CLASSES for s in SPLITS},
    )

    by_material: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_material[row.get("material_id", "")].append(row)
    material_counts = {mid: len(items) for mid, items in by_material.items()}
    _check(
        checks,
        "material_counts",
        len(by_material) == 105 and all(count == 30 for count in material_counts.values()),
        {"n_materials": len(by_material), "counts": dict(sorted(material_counts.items()))},
    )
    material_class = {
        mid: {row.get("class", "") for row in items} for mid, items in by_material.items()
    }
    material_splits = {
        mid: {row.get("split", "") for row in items} for mid, items in by_material.items()
    }
    leakage = sorted(mid for mid, values in material_splits.items() if len(values) != 1)
    _check(checks, "no_material_split_leakage", not leakage, leakage)
    _check(
        checks,
        "material_classes_consistent",
        all(len(values) == 1 for values in material_class.values()),
        sorted(mid for mid, values in material_class.items() if len(values) != 1),
    )

    split_lookup: dict[str, dict[str, str]] = {}
    split_duplicate_ids = []
    for row in split_rows:
        mid = row.get("material_id", "")
        if mid in split_lookup:
            split_duplicate_ids.append(mid)
        split_lookup[mid] = row
    inherited = []
    for row in rows:
        source = split_lookup.get(row.get("material_id", ""))
        if (
            source is None
            or source.get("split") != row.get("split")
            or source.get("class") != row.get("class")
        ):
            inherited.append(
                {
                    "filename": row.get("filename"),
                    "manifest": [row.get("class"), row.get("split")],
                    "split_manifest": None
                    if source is None
                    else [source.get("class"), source.get("split")],
                }
            )
    split_manifest_ok = (
        len(split_rows) == 105
        and len(split_lookup) == 105
        and not split_duplicate_ids
        and set(split_lookup) == set(by_material)
    )
    _check(
        checks,
        "split_manifest_unique_and_complete",
        split_manifest_ok,
        {
            "rows": len(split_rows),
            "duplicate_ids": sorted(set(split_duplicate_ids)),
            "missing_ids": sorted(set(by_material) - set(split_lookup)),
            "extra_ids": sorted(set(split_lookup) - set(by_material)),
        },
    )
    _check(checks, "split_inheritance", not inherited, inherited[:20] if inherited else [])
    split_material_counts = Counter(row.get("split", "") for row in split_rows)
    _check(
        checks,
        "material_split_counts",
        dict(split_material_counts) == EXPECTED["materials_by_split"],
        dict(split_material_counts),
    )

    id_issues = []
    filename_issues = []
    index_issues = []
    path_issues = []
    for row in rows:
        cls = row.get("class", "")
        mid = row.get("material_id", "")
        filename = row.get("filename", "")
        split = row.get("split", "")
        match = ID_RE.fullmatch(mid)
        if not match or match.group("class") != cls:
            id_issues.append([mid, cls])
        file_match = FILENAME_RE.fullmatch(filename)
        if (
            not file_match
            or file_match.group("class") != cls
            or f"{cls}_{file_match.group('number')}" != mid
        ):
            filename_issues.append(filename)
        try:
            if int(row.get("render_index", "0")) not in range(1, 31):
                index_issues.append(filename)
        except ValueError:
            index_issues.append(filename)
        if row.get("image_path", "").replace("\\", "/") != f"dataset/{split}/{cls}/{filename}":
            path_issues.append(row.get("image_path", ""))
    _check(checks, "material_id_format", not id_issues, id_issues)
    _check(
        checks,
        "filename_and_render_indices",
        not filename_issues and not index_issues,
        {"filename_issues": filename_issues, "index_issues": index_issues},
    )
    _check(checks, "manifest_path_format", not path_issues, path_issues)

    decode_errors: list[dict[str, str]] = []
    properties = Counter()
    for row in rows:
        path = root / row.get("image_path", "")
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                properties[(image.format, image.size, image.mode)] += 1
                if image.format != "PNG" or image.size != (512, 512) or image.mode != "RGB":
                    decode_errors.append(
                        {
                            "path": row.get("image_path", ""),
                            "format": str(image.format),
                            "size": str(image.size),
                            "mode": image.mode,
                        }
                    )
        except Exception as exc:
            decode_errors.append(
                {
                    "path": row.get("image_path", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    _check(
        checks,
        "all_images_decode_as_rgb_png_512",
        not decode_errors,
        {"properties": {str(k): v for k, v in properties.items()}, "errors": decode_errors[:20]},
    )

    result["summary"] = {
        "rows": len(rows),
        "disk_png": len(disk_paths),
        "materials": len(by_material),
        "classes": dict(class_counts),
        "splits": dict(split_counts),
    }
    result["passed"] = all(item["passed"] for item in checks.values())
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/results/dataset_validation.json")
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = verify(args.project_root)
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"passed": result["passed"], "checks": len(result["checks"]), "output": str(output)}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
