"""Build the combined, LOCAL audit-only V3 dataset; never upload or select a license.

The CLI has the full frozen population hard-coded. Small Population values are
available only to Python callers testing the validation/archive primitives.
Only Pillow and the data-only holdout integrity helpers are imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "blender/scripts/v3"))
from holdout_common import (  # noqa: E402
    CLASSES,
    fingerprint,
    validate_identity_audit,
    validate_receipt,
    validate_registry,
)

STAMP = "AUDIT ONLY / NOT YET PUBLISHED"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
CHUNK = 1024 * 1024
PRIVATE = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[/\\]|\\{2,}[A-Za-z0-9_.-]+\\+[A-Za-z0-9]"
    r"|(?<![:/])/(?:Users|home|root|tmp|mnt|Volumes)/"
)
FIELDS = (
    "image_path",
    "class",
    "material_id",
    "source_group",
    "split",
    "original_split",
    "sha256",
    "source_sha256",
    "image_transformation",
    "source_provider",
    "lineage_image_path",
    "pixel_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stream_hash(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(CHUNK), b""):
        result.update(block)
    return result.hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return stream_hash(stream)


def canonical_png(path, expected_source):
    """Stream PNG chunks, retaining pixels/color and removing private ancillary data.

    No re-encoding: IHDR/PLTE/IDAT/IEND and fixed-size color interpretation chunks
    retain their exact original bytes. Text, EXIF and other ancillary metadata are
    omitted. ICC profiles fail closed rather than silently changing color rendering.
    Every original chunk CRC and the complete original source hash are checked.
    """
    keep = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"cHRM", b"gAMA", b"sRGB", b"sBIT"}
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        signature = source.read(8)
        require(signature == b"\x89PNG\r\n\x1a\n", "invalid PNG signature")
        digest.update(signature)
        yield signature
        first, ended = True, False
        while not ended:
            header = source.read(8)
            require(len(header) == 8, "truncated PNG chunk")
            size, tag = struct.unpack(">I4s", header)
            require(not first or tag == b"IHDR", "PNG must begin with IHDR")
            first = False
            require(tag != b"iCCP", "ICC profile requires explicit color-preserving review")
            require(tag in keep or tag[0] & 32, "unknown critical PNG chunk")
            fixed_sizes = {
                b"IHDR": {13},
                b"IEND": {0},
                b"cHRM": {32},
                b"gAMA": {4},
                b"sRGB": {1},
                b"sBIT": {3},
            }
            require(tag not in fixed_sizes or size in fixed_sizes[tag], "invalid PNG chunk size")
            digest.update(header)
            crc = zlib.crc32(tag)
            retained = tag in keep
            if retained:
                yield header
            remaining = size
            while remaining:
                block = source.read(min(CHUNK, remaining))
                require(bool(block), "truncated PNG payload")
                remaining -= len(block)
                digest.update(block)
                crc = zlib.crc32(block, crc)
                if retained:
                    yield block
            checksum = source.read(4)
            require(
                len(checksum) == 4 and struct.unpack(">I", checksum)[0] == crc & 0xFFFFFFFF,
                "PNG chunk CRC mismatch",
            )
            digest.update(checksum)
            if retained:
                yield checksum
            ended = tag == b"IEND"
        require(not source.read(1), "trailing PNG bytes")
    require(
        digest.hexdigest() == expected_source,
        "image source hash mismatch (changed while packaging)",
    )


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def csv_bytes(rows, fields=None):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def relative(value):
    require(isinstance(value, str) and bool(value), "empty or non-string path")
    parts = value.split("/")
    require(
        not PurePosixPath(value).is_absolute()
        and not any(c in value for c in "\\:\x00")
        and all(p not in ("", ".", "..") and p == p.rstrip(" .") for p in parts)
        and not any(ord(c) < 32 for c in value),
        "unsafe relative path",
    )
    return value


def inside(root, value):
    root = Path(root).resolve()
    path = (root / relative(value)).resolve()
    require(path.is_relative_to(root), "path escapes input root (possibly a symlink)")
    return path


def private_check(data):
    text = data.decode("utf-8")
    require(not PRIVATE.search(text), "private absolute path in publication text")
    require("\r" not in text, "publication text must use LF")


def portable_cell(value):
    """Only known historical materials paths can be rewritten; unknown paths fail."""
    if PRIVATE.search(value):
        normalized = value.replace("\\", "/")
        require("/materials/" in normalized, "unrecognized private path in input")
        value = "materials/" + normalized.split("/materials/", 1)[1]
        relative(value.rstrip("/"))
    return value


class Inputs:
    """Snapshot allowlisted text, preserving original and canonical hash identities."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.raw = {}
        self.pins = defaultdict(set)
        self.binary = {}

    def read(self, name):
        if name not in self.raw:
            self.raw[name] = inside(self.root, name).read_bytes()
        return self.raw[name]

    def document(self, name):
        return json.loads(self.read(name).decode("utf-8"))

    def rows(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode("utf-8"), newline="")))

    def pin(self, name, expected):
        raw = self.read(name)
        raw.decode("utf-8")
        lf = raw.replace(b"\r\n", b"\n")
        require(
            expected in {sha256(x) for x in (raw, lf, lf.replace(b"\n", b"\r\n"))},
            f"text hash mismatch: {name}",
        )
        self.pins[name].add(expected)

    def binary_pin(self, name, expected, size=None):
        path = inside(self.root, name)
        if name not in self.binary:
            self.binary[name] = {"sha256": file_hash(path), "size_bytes": path.stat().st_size}
        record = self.binary[name]
        require(record["sha256"] == expected, f"binary hash mismatch: {name}")
        require(size is None or record["size_bytes"] == int(size), f"size mismatch: {name}")

    def publication(self):
        entries, index = {}, []
        for name, raw in sorted(self.raw.items()):
            require(
                Path(name).suffix in {".json", ".csv", ".py", ".md"},
                "non-text provenance is not publishable",
            )
            lf = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
            content = lf
            if name.endswith(".csv"):
                rows = self.rows(name)
                content = csv_bytes([{k: portable_cell(v) for k, v in row.items()} for row in rows])
            private_check(content)
            target = "provenance/" + relative(name)
            entries[target] = content
            index.append(
                {
                    "input_path": name,
                    "package_path": target,
                    "source_sha256": sha256(raw),
                    "source_lf_sha256": sha256(lf),
                    "canonical_sha256": sha256(content),
                    "canonical_size_bytes": len(content),
                    "verified_frozen_sha256": sorted(self.pins[name]),
                    "transformation": "UTF-8/LF; portable historical materials paths in CSV",
                }
            )
        entries["metadata/input_hashes.json"] = json_bytes(
            {
                "text_inputs": index,
                "verified_binary_inputs_not_distributed": [
                    {"input_path": p, **v} for p, v in sorted(self.binary.items())
                ],
                "hash_policy": (
                    "source_sha256 hashes actual input bytes; source_lf_sha256 normalizes "
                    "only CRLF. canonical_sha256 hashes the exact packaged UTF-8/LF, "
                    "privacy-normalized bytes. Frozen embedded hashes are preserved, never "
                    "silently rewritten. Resolve them through verified_frozen_sha256 and "
                    "canonical_sha256 here. Canonical JSON record fingerprints in renderer "
                    "evidence use sorted compact JSON, a distinct identity."
                ),
            }
        )
        return entries


@dataclass(frozen=True)
class Population:
    classes: tuple[str, ...] = tuple(sorted(CLASSES))
    development_materials_per_class: int = 15
    test_materials_per_class: int = 3
    views_per_material: int = 30
    development_groups: int = 101
    image_size: int = 512


def validate_population(old, fresh, population=Population()):
    require(bool(old) and bool(fresh), "both populations are required")
    paths, hashes, materials, groups = set(), set(), {}, {}
    for split, rows, quota in (
        ("development", old, population.development_materials_per_class),
        ("test", fresh, population.test_materials_per_class),
    ):
        expected = dict.fromkeys(population.classes, quota * population.views_per_material)
        require(Counter(r["class"] for r in rows) == expected, "class/population count mismatch")
        require(
            set(Counter(r["material_id"] for r in rows).values())
            == {population.views_per_material},
            "material render quota mismatch",
        )
        for row in rows:
            path = relative(row["image_path"])
            require(path.casefold() not in paths, "duplicate image path")
            paths.add(path.casefold())
            require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "invalid SHA256")
            require(row["sha256"] not in hashes, "duplicate image content")
            hashes.add(row["sha256"])
            require(
                row["original_split"]
                in ({"train", "val", "test"} if split == "development" else {"holdout"}),
                "invalid historical split lineage",
            )
            identity = (row["class"], row["source_group"], split, row["original_split"])
            require(bool(row["source_group"]) and bool(row["material_id"]), "missing identity")
            require(
                materials.setdefault(row["material_id"], identity) == identity,
                "material identity/split conflict",
            )
            group_identity = (row["class"], split)
            require(
                groups.setdefault(row["source_group"], group_identity) == group_identity,
                "source family crosses classes or development/test splits",
            )
    require(
        sum(v[1] == "development" for v in groups.values()) == population.development_groups,
        "development source-family count mismatch",
    )
    require(
        sum(v[1] == "test" for v in groups.values())
        == len(population.classes) * population.test_materials_per_class,
        "fresh source-family count mismatch",
    )


def validate_original(book, rows):
    registry = book.document("metadata/v3/license_registry.json")
    sources = registry["sources"]
    by_mid = {s["material_id"]: s for s in sources}
    require(
        len(by_mid) == len(sources) and set(by_mid) == {r["material_id"] for r in rows},
        "original license registry population mismatch",
    )
    audit = book.document("metadata/v3/data_audit.json")
    for key in (
        "integrity_errors",
        "missing_images",
        "unmanifested_images",
        "cross_class_conflicts",
        "duplicate_renders",
        "duplicate_render_pixels",
    ):
        require(audit[key] == [], f"original audit unresolved: {key}")
    book.pin("metadata/v3/manifest.csv", audit["manifest_sha256"])
    book.pin("metadata/v3/source_maps.csv", audit["source_maps_sha256"])
    maps = book.rows("metadata/v3/source_maps.csv")
    map_index = {m["path"]: m for m in maps}
    require(len(map_index) == len(maps), "duplicate source-map path")
    for row in maps:
        book.binary_pin(row["path"], row["sha256"], row["size_bytes"])
    groups = book.document("metadata/v3/source_groups.json")["groups"]
    expected = {}
    for group in groups:
        require(
            not group["cross_class_conflict"] and len(group["classes"]) == 1,
            "source-family class conflict",
        )
        for mid in group["material_ids"]:
            require(mid not in expected, "duplicate family material")
            expected[mid] = (group["source_group"], group["classes"][0])
    require(set(expected) == set(by_mid), "source-family inventory mismatch")
    for row in rows:
        require(
            expected[row["material_id"]] == (row["source_group"], row["class"]),
            "frozen source-family mismatch",
        )
    for source in sources:
        mid = source["material_id"]
        require(
            source["verification_status"] == "verified"
            and source["redistribution_allowed"] is True
            and source["license_spdx"] == "CC0-1.0"
            and source["attribution"]
            and source["source_url"].startswith("https://"),
            "unverified original license/attribution",
        )
        records = sorted((m["path"], m["sha256"]) for m in maps if m["material_id"] == mid)
        require(
            bool(records) and sha256(json_bytes(records)) == source["source_fingerprint"],
            "original source-map fingerprint mismatch",
        )
        name = "metadata/v3/" + relative(source["evidence_path"])
        book.pin(name, source["evidence_sha256"])
        evidence = book.document(name)
        for key in (
            "material_id",
            "provider",
            "provider_asset_id",
            "source_fingerprint",
            "license_spdx",
            "source_url",
        ):
            require(evidence[key] == source[key], f"license evidence identity mismatch: {key}")
        require(
            {m["path"] for m in evidence["maps"]} == {p for p, _ in records},
            "license evidence maps incomplete",
        )
        for m in evidence["maps"]:
            require(
                m["sha256"] == map_index[m["path"]]["sha256"]
                and m["verification_status"]
                in {"official_md5_match", "official_archive_sha256_match"},
                "unverified source-map evidence",
            )
        book.pin("metadata/material_manifest.csv", evidence["original_material_manifest_sha256"])
        book.pin(evidence["verification_report"], evidence["verification_report_sha256"])
    for item in registry["supplemental_evidence"]:
        book.pin("metadata/v3/" + relative(item["evidence_path"]), item["evidence_sha256"])
    provenance = book.document("metadata/v3/license_evidence/render_provenance.json")
    require(provenance["unresolved_external_input_issues"] == [], "unresolved render provenance")
    book.pin("metadata/v3/manifest.csv", provenance["manifest_sha256"])
    book.pin("metadata/v3/source_maps.csv", provenance["source_maps_sha256"])
    for item in provenance["input_files"]:
        if Path(item["path"]).suffix == ".blend":
            book.binary_pin(item["path"], item["sha256"], item["size_bytes"])
        else:
            book.pin(item["path"], item["sha256"])
    archive = provenance["original_archive"]
    book.binary_pin(archive["path"], archive["sha256"])
    require(
        all(
            provenance["image_checks"][k] == len(rows)
            for k in (
                "manifest_images",
                "local_sha256_matches",
                "material_assignment_matches",
                "historical_render_rows_matched",
                "original_archive_sha256_matches",
            )
        ),
        "original render provenance is incomplete",
    )
    history = book.rows("metadata/render_manifest.csv")
    historical = {r["image_path"]: r for r in history}
    require(len(historical) == len(history) == len(rows), "historical render inventory mismatch")
    for row in rows:
        h = historical[row["image_path"]]
        require(
            (h["class"], h["material_id"], h["split"])
            == (row["class"], row["material_id"], row["original_split"]),
            "historical split/material lineage mismatch",
        )
    families = book.document("metadata/v3/source_family_overrides.json")
    require(
        sha256(json_bytes(families["overrides"]))
        == families["frozen_registry_sha256"]
        == audit["family_registry_sha256"],
        "frozen family registry hash mismatch",
    )
    for family in families["overrides"]:
        require(
            len({expected[mid][0] for mid in family["material_ids"]}) == 1,
            "frozen family override not preserved",
        )
    return by_mid


def validate_fresh_handoff(registry, receipt, qc, run, derived, renders, rows):
    """Bind the classifier view to complete data-only QC and the sealed run."""
    require(
        qc.get("passed") is True and qc.get("data_only") is True and qc["failures"] == [],
        "complete data-only QC required",
    )
    selection = registry["selection_sha256"]
    require(
        fingerprint({k: v for k, v in run.items() if k != "run_sha256"}) == run["run_sha256"],
        "run fingerprint mismatch",
    )
    require(
        run["selection_sha256"] == selection == receipt["selection_sha256"]
        and run["source_sha256"] == fingerprint(receipt),
        "receipt/run identity mismatch",
    )
    require(
        run["runtime"]["mode"] == "holdout" and run["runtime"]["engine"] == "BLENDER_EEVEE",
        "non-production render run",
    )
    require(
        qc["run_sha256"] == derived["run_sha256"] == run["run_sha256"]
        and derived["selection_sha256"] == selection
        and derived["model_predictions_used"] is False,
        "classifier/QC identity mismatch",
    )
    for key, value in (
        ("render_records_sha256", renders),
        ("registry_sha256", registry),
        ("qc_sha256", qc),
    ):
        require(derived[key] == fingerprint(value), f"classifier provenance mismatch: {key}")
    assets = {a["material_id"]: a for a in registry["assets"]}
    require(len(assets) == len(registry["assets"]), "duplicate fresh material")
    require(
        len(rows)
        == len(renders)
        == qc["images"]
        == derived["images"]
        == len(assets) * registry["views_per_material"],
        "QC image count mismatch",
    )
    require(len(assets) == qc["groups"] == derived["source_groups"], "QC group count mismatch")
    require(Counter(r["class"] for r in rows) == qc["class_counts"], "QC class counts mismatch")
    rendered = {r["image_path"]: r for r in renders}
    require(
        len(rendered) == len(renders) and set(rendered) == {r["image_path"] for r in rows},
        "render/classifier inventory mismatch",
    )
    for row in rows:
        a, r = assets[row["material_id"]], rendered[row["image_path"]]
        require(
            row["class"] == r["class"] == a["class"]
            and r["material_id"] == a["material_id"]
            and r["source_asset_id"] == a["asset_id"]
            and r["provider"] == a["provider"]
            and row["source_group"] == f"{a['provider']}:{a['asset_id']}"
            and r["original_split"] == r["split"] == row["original_split"] == "holdout"
            and r["image_sha256"] == row["sha256"],
            "fresh render identity mismatch",
        )
        require(
            all(r[k] == v for k, v in run.items() if k != "runtime"),
            "fresh renderer fingerprint mismatch",
        )
    return assets


def validate_fresh(book, rows):
    base = "metadata/v3_holdout/"
    registry = book.document(base + "source_registry.json")
    validate_registry(registry, inside(book.root, "metadata/material_manifest.csv"))
    identity = book.document(base + "source_identity_audit.json")
    validate_identity_audit(registry, identity)
    receipt = book.document(base + "download_receipt.json")
    validate_receipt(registry, receipt, book.root / "artifacts/v3/holdout_sources")
    receipts = {item["path"]: item for item in receipt["files"]}
    old_map_hashes = {m["sha256"] for m in book.rows("metadata/v3/source_maps.csv")}
    for item in receipt["files"]:
        book.binary_pin(
            "artifacts/v3/holdout_sources/" + item["path"], item["sha256"], item["size"]
        )
        require(item["sha256"] not in old_map_hashes, "fresh source map reuses original content")
    for asset in registry["assets"]:
        for spec in asset["maps"].values():
            receipt_map = receipts[spec["path"]]
            require(
                receipt_map.get("archive_member") == spec.get("archive_member"),
                "receipt archive member identity mismatch",
            )
            if "provider_md5" in spec:
                path = inside(book.root / "artifacts/v3/holdout_sources", spec["path"])
                with path.open("rb") as stream:
                    require(
                        hashlib.file_digest(stream, "md5").hexdigest() == spec["provider_md5"],
                        "official source-map MD5 mismatch",
                    )
    config = book.document("configs/v3/holdout.json")
    require(
        config["protocol_sha256"]
        == fingerprint({k: v for k, v in config.items() if k != "protocol_sha256"}),
        "holdout protocol fingerprint mismatch",
    )
    require(
        config["selection_sha256"] == registry["selection_sha256"], "protocol selection mismatch"
    )
    book.pin(base + "source_identity_audit.json", config["identity_audit_sha256"])
    qc = book.document("artifacts/v3/holdout_dataset/qc.json")
    run = book.document("artifacts/v3/holdout_dataset/run_fingerprint.json")
    book.pin("configs/v3/holdout.json", run["config_sha256"])
    derived = book.document(base + "classification_manifest.json")
    book.pin(base + "classification_manifest.csv", derived["manifest_sha256"])
    renders = book.rows(base + "render_manifest.csv")
    assets = validate_fresh_handoff(registry, receipt, qc, run, derived, renders, rows)
    # The renderer hashes exactly these three source files, using source checkout bytes.
    renderer = {
        name: book.read("blender/scripts/v3/" + name)
        for name in ("render_holdout.py", "holdout_common.py", "qc_holdout.py")
    }
    variants = []
    for style in ("raw", "lf", "crlf"):
        values = {}
        for name, raw in renderer.items():
            lf = raw.replace(b"\r\n", b"\n")
            values[name] = sha256(
                raw if style == "raw" else lf if style == "lf" else lf.replace(b"\n", b"\r\n")
            )
        variants.append(fingerprint(values))
    require(run["renderer_sha256"] in variants, "renderer source fingerprint mismatch")
    book.read(base + "leather_lineage_audit.json")
    baseline = config["baseline_evidence"]
    book.binary_pin(baseline["scene_path"], baseline["scene_sha256"])
    book.pin(baseline["generation_report"], baseline["generation_report_sha256"])
    book.pin(baseline["shader_script"], baseline["shader_script_sha256"])
    book.pin(
        "blender/configs/rendersynth_material_pipeline.json",
        baseline["randomization_config_sha256"],
    )
    return assets


def prepare_images(root, old, fresh, old_sources, fresh_sources, population=Population()):
    validate_population(old, fresh, population)
    images, result, seen_pixels = {}, [], set()
    for split, rows, base, sources in (
        ("development", old, Path(root), old_sources),
        ("test", fresh, Path(root) / "artifacts/v3/holdout_dataset", fresh_sources),
    ):
        for row in rows:
            path = inside(base, row["image_path"])
            require(file_hash(path) == row["sha256"], "input image hash mismatch")
            with Image.open(path) as im:
                im.load()
                require(
                    im.format == "PNG"
                    and im.mode == "RGB"
                    and im.size == (population.image_size, population.image_size),
                    "invalid image format/mode/dimensions",
                )
                pixel = sha256(
                    f"RGBA:{im.width}:{im.height}:".encode() + im.convert("RGBA").tobytes()
                )
            canonical = b"".join(canonical_png(path, row["sha256"]))
            with Image.open(io.BytesIO(canonical)) as clean:
                clean.load()
                clean_pixel = sha256(
                    f"RGBA:{clean.width}:{clean.height}:".encode() + clean.convert("RGBA").tobytes()
                )
                require(clean_pixel == pixel, "PNG metadata removal changed pixels")
                for value in clean.info.values():
                    if isinstance(value, str):
                        private_check(value.encode())
            require(
                not row.get("pixel_sha256") or row["pixel_sha256"] == pixel,
                "decoded pixel hash mismatch",
            )
            require(pixel not in seen_pixels, "duplicate decoded image content")
            seen_pixels.add(pixel)
            target = relative(f"images/{split}/{row['class']}/{path.name}")
            require(
                target.casefold() not in {p.casefold() for p in images},
                "archive image path conflict",
            )
            canonical_hash = sha256(canonical)
            images[target] = (path, row["sha256"], canonical_hash)
            result.append(
                {
                    "image_path": target,
                    "class": row["class"],
                    "material_id": row["material_id"],
                    "source_group": row["source_group"],
                    "split": split,
                    "original_split": row["original_split"],
                    "sha256": canonical_hash,
                    "source_sha256": row["sha256"],
                    "image_transformation": "PNG metadata stripped; pixel stream unchanged",
                    "source_provider": sources[row["material_id"]]["provider"],
                    "lineage_image_path": row["image_path"],
                    "pixel_sha256": pixel,
                }
            )
    return images, sorted(result, key=lambda r: r["image_path"])


def dataset_card(rows):
    counts = Counter(r["split"] for r in rows)
    return f"""# V3 material classification study dataset

**{STAMP}**

Dataset license: UNSET. release_approved=false. No Kaggle upload has been made.
The destination and a choice between CC0 and CC-BY-4.0 require the user's decision.
The metadata skeleton is deliberately incomplete and is not upload-ready.

This package contains {len(rows)} synthetic RGB Blender renders: {counts["development"]}
development images and {counts["test"]} fresh source-holdout test images. The source
author states that the original renders are their own work; third-party surface
texture maps are from Poly Haven and ambientCG under CC0-1.0. Source-map licenses
do not choose the rendered dataset's license. Attribution distinguishes verified
asset authors from publisher/operator credits. No real photographs are included.

The original corpus, including its historical train/val/test images, is development
data in V3. It is NOT an untouched test set. metadata/manifest.csv preserves
original_split, lineage_image_path, material IDs and frozen source groups. The
fresh holdout alone is split=test, with original_split=holdout. Related materials
and correlated views must stay within their known source families during tuning.
The complete study has 540 images/class, 126 material IDs and 122 known groups:
101 development families and 21 fresh groups. Small synthetic test fixtures are
not the full study release. Source-ID spelling alone is not evidence of independence.

The frozen identity audit checks exclusions, normalized families, procedural
ancestry and official map identities. Physical lineage is inferred, not universally
proven: undisclosed common captures, crops or recolors may escape those checks.
The fresh endpoint uses a restricted semantic taxonomy: ceramic is fired-clay,
quarry or glazed tile; concrete is cementitious, excluding plaster/paint. Old
ambiguous labels remain as frozen. Metal denotes substrate, including coatings/rust;
scanned suede may resemble fabric. Only three fresh sources per class, each with
30 correlated renders, limit effective sample size and generalization claims.

All images are synthetic. Fresh renders use Blender 5.1.1 EEVEE and the frozen
shader/lighting/camera protocol. The historical engine version was not recorded;
matching retained settings does not prove identical rendering domains. Provider
resolution and shader approximations also differ. Do not infer real-world accuracy.

The fresh test set was frozen before model scoring. Packaging is data-only and
does not inspect scores. Once scored, and especially after publication, this test
set is a public benchmark: repeated feedback or tuning on it must be disclosed and
requires a new independently frozen holdout for future confirmatory evaluation.
Never describe later tuning results on this public test as untouched evaluation.

The archive includes frozen source-attribution and compact license verification
records, renderer provenance, data-only QC, generation configuration, source-map
hashes and lineage manifests. Historical audit statements retain their original
scope/time; later license/render verification supersedes earlier unresolved status.
Raw source textures, Blender scene binaries, model weights and logs are excluded.
Private historical absolute materials paths are made relative in packaged copies.
PNG ancillary metadata (including Blender's private File path) is removed from
packaged image copies. Pixel compression streams and fixed color chunks are copied
unchanged; decoded RGBA hashes are verified identical before packaging. sha256 in
the combined manifest identifies the packaged PNG, while source_sha256 identifies
the untouched original PNG. Frozen lineage manifests retain their source hashes.

metadata/input_hashes.json separates original source-byte hashes from exact packaged
UTF-8/LF canonical hashes. Embedded frozen hashes are retained as historical
identities, not falsely presented as checksums of privacy-normalized copies.
metadata/archive_inventory.json pins every other entry; it excludes itself to
avoid a circular hash. The external SHA256 sidecar pins the complete ZIP.
""".encode("utf-8")


def verify_archive(path, expected, manifest_bytes):
    """Streaming read of every entry verifies ZIP CRC and SHA256, including images."""
    with ZipFile(path) as archive:
        names = archive.namelist()
        require(
            names == sorted(expected) and len(names) == len(set(n.casefold() for n in names)),
            "archive inventory/order conflict",
        )
        for entry in archive.infolist():
            relative(entry.filename)
            require(
                entry.date_time == FIXED_TIME and entry.compress_type == ZIP_STORED,
                "non-deterministic ZIP metadata",
            )
            with archive.open(entry) as stream:
                require(
                    stream_hash(stream) == expected[entry.filename], "archive entry hash mismatch"
                )
        require(
            archive.read("metadata/manifest.csv") == manifest_bytes, "archive manifest mismatch"
        )
    return {
        "crc_verified": True,
        "all_entry_hashes_verified": True,
        "manifest_verified": True,
        "entries": len(expected),
    }


def write_archive(output, images, rows, text_entries):
    """Exclusive outputs; no existing archive/sidecar is ever overwritten."""
    output = Path(output)
    sidecar = Path(str(output) + ".sha256")
    validation = output.with_suffix(".validation.json")
    require(output.suffix == ".zip", "output must be a ZIP")
    for path in (output, sidecar, validation):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path.name}")
    entries = dict(text_entries)
    generated = {
        "metadata/manifest.csv": csv_bytes(rows, FIELDS),
        "DATASET_CARD.md": dataset_card(rows),
        "dataset-metadata.json": json_bytes(
            {
                "title": "V3 Synthetic Material Classification Study — AUDIT ONLY",
                "id": None,
                "licenses": [],
                "isPrivate": True,
                "description": STAMP + "; destination and dataset license pending user decision.",
            }
        ),
        "metadata/release_status.json": json_bytes(
            {
                "status": STAMP,
                "dataset_license": "",
                "release_approved": False,
                "kaggle_destination": None,
                "upload_performed": False,
                "model_scores_accessed": False,
            }
        ),
    }
    require(not set(entries) & set(generated), "generated metadata path conflict")
    entries.update(generated)
    all_names = [*entries, *images, "metadata/archive_inventory.json"]
    require(len(all_names) == len(set(n.casefold() for n in all_names)), "archive path conflict")
    for name, content in entries.items():
        relative(name)
        require(Path(name).suffix in {".json", ".csv", ".md", ".py"}, "forbidden archive file")
        private_check(content)
    require(
        {r["image_path"] for r in rows} == set(images) and len(rows) == len(images),
        "image/manifest inventory mismatch",
    )
    expected = {name: sha256(data) for name, data in entries.items()}
    for name, (_, _, digest) in images.items():
        relative(name)
        require(
            name.startswith(("images/development/", "images/test/")) and name.endswith(".png"),
            "forbidden image entry",
        )
        expected[name] = digest
    entries["metadata/archive_inventory.json"] = json_bytes(expected)
    expected["metadata/archive_inventory.json"] = sha256(entries["metadata/archive_inventory.json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        # Reserve all output names before writing. Cleanup touches only files created here.
        for path in (output, sidecar, validation):
            with path.open("xb"):
                pass
            created.append(path)
        with ZipFile(output, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for name in sorted(expected):
                info = ZipInfo(name, FIXED_TIME)
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = ZIP_STORED
                with archive.open(info, "w", force_zip64=True) as target:
                    if name in entries:
                        target.write(entries[name])
                    else:
                        digest = hashlib.sha256()
                        for block in canonical_png(images[name][0], images[name][1]):
                            digest.update(block)
                            target.write(block)
                        require(
                            digest.hexdigest() == expected[name], "image changed while packaging"
                        )
        checked = verify_archive(output, expected, entries["metadata/manifest.csv"])
        checksum = file_hash(output)
        result = {
            "schema_version": 1,
            "status": STAMP,
            "release_approved": False,
            "dataset_license": "",
            "archive": output.name,
            "sha256": checksum,
            "size_bytes": output.stat().st_size,
            "images": len(rows),
            "class_counts": dict(Counter(r["class"] for r in rows)),
            "split_counts": dict(Counter(r["split"] for r in rows)),
            "material_ids": len({r["material_id"] for r in rows}),
            "source_groups": len({r["source_group"] for r in rows}),
            "split_source_groups": {
                split: len({r["source_group"] for r in rows if r["split"] == split})
                for split in ("development", "test")
            },
            "manifest_sha256": expected["metadata/manifest.csv"],
            "source_and_pixel_hashes_verified": True,
            "png_bytes_changed_by_metadata_removal": sum(
                r["sha256"] != r["source_sha256"] for r in rows
            ),
            "private_path_scan_passed": True,
            "model_scores_accessed": False,
            "upload_performed": False,
            **checked,
        }
        sidecar.write_bytes(f"{checksum}  {output.name}\n".encode())
        validation.write_bytes(json_bytes(result))
        return result
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def package_study(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(
        output.parent == root / "artifacts/v3/release-staging",
        "actual package must be in parent artifacts/v3/release-staging",
    )
    for p in (output, Path(str(output) + ".sha256"), output.with_suffix(".validation.json")):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p.name}")
    book = Inputs(root)
    old = book.rows("metadata/v3/manifest.csv")
    fresh = book.rows("metadata/v3_holdout/classification_manifest.csv")
    validate_population(old, fresh)  # CLI always requires the full 3,780-image population.
    print("Validating frozen original sources and renderer provenance...", flush=True)
    original_sources = validate_original(book, old)
    print("Validating fresh registry, source maps, receipt and data-only QC...", flush=True)
    fresh_sources = validate_fresh(book, fresh)
    print("Validating all image bytes, decoded content and split lineage...", flush=True)
    images, rows = prepare_images(root, old, fresh, original_sources, fresh_sources)
    print("Writing deterministic local ZIP and reading every entry for CRC/SHA256...", flush=True)
    return write_archive(output, images, rows, book.publication())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = package_study(args.parent_root, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Packaging refused: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
