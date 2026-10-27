"""Portable corpus audit and fail-closed, upload-free release staging.

Only Pillow is needed. Source paths are resolved against an explicit data root;
historical absolute folder columns are never followed or exported. Group IDs
are connected components, not labels, and never change the class annotation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import quote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MANIFEST_COLUMNS = [
    "image_path",
    "class",
    "material_id",
    "source_group",
    "original_split",
    "sha256",
    "pixel_sha256",
    "width",
    "height",
    "mode",
    "readable",
    "source_material_name",
]
MAP_COLUMNS = [
    "path",
    "class",
    "material_id",
    "role",
    "sha256",
    "size_bytes",
    "readability",
    "width",
    "height",
    "pixel_sha256",
]
KAGGLE_LICENSES = {"CC0-1.0": "CC0-1.0", "CC-BY-4.0": "CC-BY-4.0"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value: str) -> str:
    """Reject traversal, drives, UNC, ADS, alternate separators and aliases."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("expected a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError("absolute paths are forbidden")
    if any(p in {"", ".", ".."} or p.rstrip(" .") != p for p in value.split("/")):
        raise ValueError("noncanonical or traversing path")
    for part in path.parts:
        if any(ord(c) < 32 or c in '<>"|?*' for c in part):
            raise ValueError("unsafe path component")
        if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", part.split(".")[0]):
            raise ValueError("reserved path component")
    return path.as_posix()


def resolve_data_path(root: Path, value: str) -> Path:
    value = relative_path(value)
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("path escapes data root (including symlinks)")
    return path


def _component(value: str) -> str:
    relative_path(value)
    if "/" in value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid material/class identifier")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("missing or duplicate CSV columns")
        return list(reader)


def csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    return stream.getvalue().encode("utf-8")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _inspect_image(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            rgba = image.convert("RGBA")
            digest = hashlib.sha256()
            digest.update(f"RGBA:{image.width}:{image.height}:".encode())
            digest.update(rgba.tobytes())
            return dict(
                readable=True,
                width=image.width,
                height=image.height,
                mode=image.mode,
                pixel_sha256=digest.hexdigest(),
            )
    except (OSError, ValueError, Image.DecompressionBombError):
        return dict(readable=False, width="", height="", mode="", pixel_sha256="")


def normalized_source_name(name: str) -> str:
    # Preserve digits/variant suffixes: Leather034B is not automatically Leather034A.
    if not name or not re.fullmatch(r"[A-Za-z0-9_ .-]+", name):
        raise ValueError("invalid or missing source material name")
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def source_groups(
    materials: list[dict], maps: list[dict], family_overrides: list[dict] | None = None
) -> tuple[dict, list[dict]]:
    """Union identity aliases, base-color bytes/pixels, and explicit families.

    Other map duplicates (especially constant roughness) are evidence only.
    Families require a rationale/evidence URL; they do not claim exact duplicates.
    """
    ids = {row["material_id"] for row in materials}
    if len(ids) != len(materials):
        raise ValueError("duplicate material IDs")
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    edges = []

    def unite(members, reason, evidence):
        members = sorted(set(members))
        if len(members) < 2:
            return
        for member in members[1:]:
            parent[find(member)] = find(members[0])
        edges.append(dict(material_ids=members, reason=reason, evidence=evidence))

    for column, reason in [
        ("source_material_name", "source_name_alias"),
        ("source_asset_id", "declared_source_asset_identity"),
    ]:
        index = defaultdict(list)
        for row in materials:
            value = row.get(column, "")
            if value:
                key = normalized_source_name(value) if column == "source_material_name" else value
                index[key].append(row["material_id"])
        for key, members in sorted(index.items()):
            unite(members, reason, key)
    for column in ("sha256", "pixel_sha256"):
        index = defaultdict(list)
        for row in maps:
            if row["role"] == "base_color" and row.get(column):
                index[row[column]].append(row["material_id"])
        for key, members in sorted(index.items()):
            unite(members, f"base_color_{column}", key)
    assigned = {}
    family_members = defaultdict(list)
    family_evidence = defaultdict(list)
    for override in family_overrides or []:
        family = override["source_family"]
        if not family or not override.get("rationale") or not _https(override.get("evidence_url")):
            raise ValueError("family override requires family, rationale and HTTPS evidence URL")
        members = override["material_ids"]
        if len(set(members)) < 2 or not set(members) <= ids:
            raise ValueError("family override has unknown or insufficient members")
        for member in members:
            if member in assigned and assigned[member] != family:
                raise ValueError("material assigned to conflicting families")
            assigned[member] = family
        if override.get("source_material_names"):
            actual_names = {
                r["source_material_name"] for r in materials if r["material_id"] in members
            }
            if actual_names != set(override["source_material_names"]):
                raise ValueError("family override source names do not match material manifest")
        family_members[family].extend(members)
        family_evidence[family].append(override)
    for family in sorted(family_members):
        unite(family_members[family], "source_family_override", family_evidence[family])
    components = defaultdict(list)
    for key in sorted(ids):
        components[find(key)].append(key)
    groups = {}
    for members in components.values():
        group = "sg_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        groups.update({member: group for member in members})
    return groups, edges


def _duplicates(rows: list[dict], hash_column: str, path_column: str) -> list[dict]:
    index = defaultdict(list)
    for row in rows:
        if row.get(hash_column):
            index[row[hash_column]].append(row[path_column])
    return [
        {hash_column: key, "paths": sorted(paths)}
        for key, paths in sorted(index.items())
        if len(paths) > 1
    ]


def validate_class_pure_groups(manifest: list[dict]) -> None:
    """Required pre-fold check. Raise on conflicts; never drop or relabel rows."""
    labels = defaultdict(set)
    for row in manifest:
        if not row.get("source_group") or not row.get("class"):
            raise ValueError("manifest requires nonempty source_group and class")
        labels[row["source_group"]].add(row["class"])
    conflicts = {group: sorted(values) for group, values in labels.items() if len(values) > 1}
    if conflicts:
        raise ValueError(
            "cross-class source groups require explicit resolution: "
            + json.dumps(conflicts, sort_keys=True)
        )


def family_registry_sha256(overrides: list[dict]) -> str:
    """Digest the exact family decisions; freeze it before any held-out scores."""
    return hashlib.sha256(json_bytes(overrides)).hexdigest()


def _fingerprint(maps: list[dict], material_id: str) -> str:
    records = sorted(
        (row["path"], row["sha256"]) for row in maps if row["material_id"] == material_id
    )
    return hashlib.sha256(json_bytes(records)).hexdigest()


def _candidate(name: str) -> dict:
    if re.fullmatch(r"[A-Z][A-Za-z]+\d+[A-Z]?", name):
        return {
            "candidate_provider": "ambientCG",
            "candidate_source_url": "https://ambientcg.com/view?id=" + quote(name),
        }
    return {
        "candidate_provider": "Poly Haven",
        "candidate_source_url": "https://polyhaven.com/a/" + quote(name),
    }


def audit_data(
    data_root: Path,
    render_manifest: Path,
    material_manifest: Path,
    output_dir: Path,
    *,
    expected_images: int = 3150,
    expected_materials: int = 105,
    family_overrides: list[dict] | None = None,
) -> dict:
    """Write deterministic audit artifacts and return summary, retaining all labels.

    Invalid/untrusted paths raise before reading data. Missing/corrupt data and
    count/dimension disagreements are recorded in summary.integrity_errors.
    An existing license registry is preserved, never silently overwritten.
    """
    root = data_root.resolve()
    if expected_images <= 0 or expected_materials <= 0:
        raise ValueError("expected counts must be positive")
    materials = read_csv(material_manifest)
    renders = read_csv(render_manifest)
    errors = []
    materials_by_id = {}
    for row in materials:
        mid, label = _component(row["material_id"]), _component(row["class"])
        normalized_source_name(row["source_material_name"])
        if mid in materials_by_id:
            raise ValueError("duplicate material ID in material manifest")
        materials_by_id[mid] = row
        row["source_folder"] = f"materials/{label}/{mid}"
    if len(materials) != expected_materials:
        errors.append(f"material count: expected {expected_materials}, found {len(materials)}")
    if len(renders) != expected_images:
        errors.append(f"render count: expected {expected_images}, found {len(renders)}")
    maps = []
    for row in sorted(materials, key=lambda r: r["material_id"]):
        folder = resolve_data_path(root, row["source_folder"])
        roles = {}
        for role in ("base_color", "roughness", "normal", "metallic"):
            filename = row.get(role + "_file", "")
            if filename:
                # Map references must be directly inside their declared material folder.
                relative_path(filename)
                if "/" in filename:
                    raise ValueError("map reference must be a filename")
                roles[filename] = role
                if not resolve_data_path(root, row["source_folder"] + "/" + filename).is_file():
                    errors.append(f"missing {role}: {row['source_folder']}/{filename}")
        if not row.get("base_color_file"):
            errors.append(f"missing base color declaration: {row['material_id']}")
        if not folder.is_dir():
            errors.append(f"missing material folder: {row['source_folder']}")
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            path = resolve_data_path(root, rel)
            info = _inspect_image(path) if path.suffix.lower() in IMAGE_SUFFIXES else {}
            readability = "readable" if info.get("readable") else "not_decoded"
            if info and not info["readable"]:
                readability = "unreadable"
                errors.append(f"unreadable source image: {rel}")
            maps.append(
                dict(
                    path=rel,
                    material_id=row["material_id"],
                    **{"class": row["class"]},
                    role=roles.get(path.relative_to(folder).as_posix(), "unreferenced"),
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                    readability=readability,
                    width=info.get("width", ""),
                    height=info.get("height", ""),
                    pixel_sha256=info.get("pixel_sha256", ""),
                )
            )
    groups, edges = source_groups(materials, maps, family_overrides)
    manifest = []
    seen = set()
    for row in sorted(renders, key=lambda r: r["image_path"]):
        rel = relative_path(row["image_path"])
        if rel.casefold() in seen:
            raise ValueError("duplicate or case-aliased image path in render manifest")
        seen.add(rel.casefold())
        mid, label = _component(row["material_id"]), _component(row["class"])
        split = _component(row["split"])
        if mid not in materials_by_id or materials_by_id[mid]["class"] != label:
            raise ValueError("render/material manifest class or identity mismatch")
        parts = PurePosixPath(rel).parts
        if len(parts) != 4 or parts[:3] != ("dataset", split, label):
            raise ValueError("render path does not match dataset/split/class")
        if PurePosixPath(rel).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("render must be an image file")
        if row.get("filename") and (
            row["filename"] != parts[-1] or not row["filename"].startswith(mid + "_r")
        ):
            errors.append(f"render filename/material mismatch: {rel}")
        material = materials_by_id[mid]
        if material.get("split") and material["split"] != split:
            errors.append(f"render/material split mismatch: {rel}")
        path = resolve_data_path(root, rel)
        info = _inspect_image(path)
        digest = sha256_file(path) if path.is_file() else ""
        if not info["readable"]:
            errors.append(f"missing/unreadable render: {rel}")
        elif any(
            row.get(key) and int(row[key]) != info[dim]
            for key, dim in [("resolution_width", "width"), ("resolution_height", "height")]
        ):
            errors.append(f"render dimension mismatch: {rel}")
        manifest.append(
            dict(
                image_path=rel,
                **{"class": label},
                material_id=mid,
                source_group=groups[mid],
                original_split=split,
                sha256=digest,
                source_material_name=material["source_material_name"],
                **info,
            )
        )
    actual_images = set()
    for path in (root / "dataset").rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            rel = path.relative_to(root).as_posix()
            resolve_data_path(root, rel)
            actual_images.add(rel)
    listed_images = {r["image_path"] for r in manifest}
    extra_images = sorted(actual_images - listed_images)
    missing_images = sorted(listed_images - actual_images)
    if extra_images:
        errors.append(f"{len(extra_images)} unmanifested render images")
    material_folders = {
        p.parent.relative_to(root).as_posix()
        for p in (root / "materials").glob("*/*/*")
        if p.is_file()
    }
    extra_folders = sorted(material_folders - {r["source_folder"] for r in materials})
    if extra_folders:
        errors.append(f"{len(extra_folders)} unmanifested material folders")
    unused = sorted(set(materials_by_id) - {r["material_id"] for r in manifest})
    if unused:
        errors.append(f"materials without renders: {', '.join(unused)}")
    if expected_images % expected_materials == 0:
        expected_per_material = expected_images // expected_materials
        counts = Counter(r["material_id"] for r in manifest)
        for mid in sorted(materials_by_id):
            if counts[mid] != expected_per_material:
                errors.append(
                    f"render count for {mid}: expected {expected_per_material}, found {counts[mid]}"
                )
    archives = []
    for name in ("dataset.zip", "materials.zip"):
        archive_path = resolve_data_path(root, name)
        if not archive_path.is_file():
            continue
        with ZipFile(archive_path) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            for item in archive.infolist():
                relative_path(item.filename.rstrip("/"))
            archives.append(
                dict(
                    path=name,
                    sha256=sha256_file(archive_path),
                    size_bytes=archive_path.stat().st_size,
                    file_count=len(files),
                    uncompressed_bytes=sum(item.file_size for item in files),
                    content_verified_against_extraction=False,
                )
            )
    components = []
    for group in sorted(set(groups.values())):
        members = sorted(mid for mid, gid in groups.items() if gid == group)
        labels = sorted({materials_by_id[mid]["class"] for mid in members})
        splits = sorted({r["original_split"] for r in manifest if r["material_id"] in members})
        components.append(
            dict(
                source_group=group,
                material_ids=members,
                classes=labels,
                original_splits=splits,
                cross_class_conflict=len(labels) > 1,
                crosses_original_splits=len(splits) > 1,
            )
        )
    summary = dict(
        schema_version=1,
        family_registry_sha256=family_registry_sha256(family_overrides or []),
        class_pure_source_groups=not any(c["cross_class_conflict"] for c in components),
        fold_ready=not errors and not any(c["cross_class_conflict"] for c in components),
        expected_images=expected_images,
        expected_materials=expected_materials,
        archives=archives,
        image_count=len(manifest),
        physical_image_count=len(actual_images),
        material_count=len(materials),
        source_group_count=len(components),
        source_map_count=len(maps),
        classes=dict(sorted(Counter(r["class"] for r in manifest).items())),
        original_splits=dict(sorted(Counter(r["original_split"] for r in manifest).items())),
        render_dimensions=dict(
            sorted(Counter(f"{r['width']}x{r['height']}" for r in manifest).items())
        ),
        renders_per_material=dict(sorted(Counter(r["material_id"] for r in manifest).items())),
        readable_render_count=sum(r["readable"] for r in manifest),
        integrity_errors=sorted(errors),
        unmanifested_images=extra_images,
        missing_images=missing_images,
        unmanifested_material_folders=extra_folders,
        duplicate_renders=_duplicates(manifest, "sha256", "image_path"),
        duplicate_render_pixels=_duplicates(manifest, "pixel_sha256", "image_path"),
        duplicate_source_files=_duplicates(maps, "sha256", "path"),
        source_groups=components,
        source_group_edges=edges,
        cross_class_conflicts=[c for c in components if c["cross_class_conflict"]],
        original_split_leakage=[c for c in components if c["crosses_original_splits"]],
        map_decode_status=dict(sorted(Counter(r["readability"] for r in maps).items())),
        limitations=[
            "EXR and other unsupported source formats are hashed, not decoded.",
            "Name/catalog matches do not establish local-file provenance or license.",
            "Family overrides conservatively group related assets, not exact duplicates.",
            "Archive hashes/inventory recorded; archive members not compared to extraction.",
            "Historical train/val/test membership is descriptive, not a v3 evaluation split.",
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.csv").write_bytes(csv_bytes(manifest, MANIFEST_COLUMNS))
    (output_dir / "source_maps.csv").write_bytes(csv_bytes(maps, MAP_COLUMNS))
    (output_dir / "source_groups.json").write_bytes(
        json_bytes(dict(groups=components, edges=edges))
    )
    summary["manifest_sha256"] = sha256_file(output_dir / "manifest.csv")
    summary["source_maps_sha256"] = sha256_file(output_dir / "source_maps.csv")
    (output_dir / "data_audit.json").write_bytes(json_bytes(summary))
    registry = dict(
        schema_version=1,
        dataset_license="",
        release_approved=False,
        sources=[
            dict(
                material_id=row["material_id"],
                source_material_name=row["source_material_name"],
                source_fingerprint=_fingerprint(maps, row["material_id"]),
                **_candidate(row["source_material_name"]),
                verification_status="unverified",
                source_url="",
                license_spdx="",
                redistribution_allowed=False,
                evidence_url="",
                evidence_path="",
                evidence_sha256="",
                retrieved_at="",
            )
            for row in sorted(materials, key=lambda r: r["material_id"])
        ],
    )
    (output_dir / "provenance_candidates.json").write_bytes(json_bytes(registry))
    if not (output_dir / "license_registry.json").exists():
        (output_dir / "license_registry.json").write_bytes(json_bytes(registry))
    return summary


def _https(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and not parsed.username


def license_blockers(
    registry: dict, manifest: list[dict], maps: list[dict], registry_root: Path
) -> list[str]:
    """Require evidence bound to the audited local source content for every material.

    This validates an explicit human evidence attestation, not the truth of a
    license fetched from the internet. A name match cannot satisfy this gate.
    """
    issues = []
    if registry.get("schema_version") != 1:
        issues.append("unsupported license registry schema")
    if registry.get("release_approved") is not True:
        issues.append("release has not been explicitly approved in the registry")
    license_id = registry.get("dataset_license")
    if license_id not in KAGGLE_LICENSES:
        issues.append("dataset license unresolved/unsupported")
    entries = registry.get("sources", [])
    by_id = {entry["material_id"]: entry for entry in entries}
    expected = {row["material_id"] for row in manifest}
    if len(by_id) != len(entries) or set(by_id) != expected:
        issues.append("license registry must contain each material exactly once")
    for mid in sorted(expected):
        entry = by_id.get(mid, {})
        valid = (
            entry.get("verification_status") == "verified"
            and entry.get("redistribution_allowed") is True
            and entry.get("license_spdx") == license_id
            and entry.get("source_fingerprint") == _fingerprint(maps, mid)
            and _https(entry.get("source_url"))
            and _https(entry.get("evidence_url"))
            and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry.get("retrieved_at", "")))
        )
        evidence = entry.get("evidence_path", "")
        if evidence:
            path = resolve_data_path(registry_root, evidence)
            valid = valid and path.is_file() and sha256_file(path) == entry.get("evidence_sha256")
        else:
            valid = False
        if not valid:
            issues.append(f"unresolved source license/provenance: {mid}")
    return issues


def _no_private_paths(value: object) -> None:
    serialized = value if isinstance(value, str) else json.dumps(value)
    if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]|\\\\|/(?:Users|home|home2)/", serialized):
        raise ValueError("metadata contains an absolute/private path")


def package_dataset(
    data_root: Path,
    audit_dir: Path,
    output_zip: Path,
    *,
    audit_only: bool = False,
    registry_path: Path | None = None,
    kaggle_id: str = "local-audit/material-classification-v3",
) -> dict:
    """Stage a deterministic ZIP locally; never upload. Audit-only is not publishable.

    Rehash images and source files before staging to reject stale audit artifacts.
    Only manifest-listed images and whitelisted metadata enter the ZIP.
    """
    root, audit_dir, output_zip = data_root.resolve(), audit_dir.resolve(), output_zip.resolve()
    if output_zip.exists():
        raise FileExistsError("refusing to overwrite existing staged archive")
    if any(output_zip.is_relative_to(root / folder) for folder in ("dataset", "materials")):
        raise ValueError("staging output cannot be inside the source corpus")
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", kaggle_id):
        raise ValueError("invalid Kaggle owner/slug")
    manifest = read_csv(audit_dir / "manifest.csv")
    maps = read_csv(audit_dir / "source_maps.csv")
    summary = json.loads((audit_dir / "data_audit.json").read_text(encoding="utf-8"))
    if (
        summary.get("integrity_errors")
        or not manifest
        or len(manifest) != summary.get("image_count")
        or sha256_file(audit_dir / "manifest.csv") != summary.get("manifest_sha256")
        or sha256_file(audit_dir / "source_maps.csv") != summary.get("source_maps_sha256")
    ):
        raise ValueError("audit integrity errors, empty manifest or changed audit files")
    seen = set()
    for row in manifest:
        rel = relative_path(row["image_path"])
        parts = PurePosixPath(rel).parts
        if (
            len(parts) != 4
            or parts[:3] != ("dataset", row["original_split"], row["class"])
            or PurePosixPath(rel).suffix.lower() not in IMAGE_SUFFIXES
            or rel.casefold() in seen
            or row["readable"] != "True"
        ):
            raise ValueError("unsafe or invalid staged render entry")
        seen.add(rel.casefold())
    for row in maps:
        parts = PurePosixPath(relative_path(row["path"])).parts
        if len(parts) < 4 or parts[:3] != ("materials", row["class"], row["material_id"]):
            raise ValueError("unsafe source map entry")
        if sha256_file(resolve_data_path(root, row["path"])) != row["sha256"]:
            raise ValueError("source content changed after audit")
    registry_path = registry_path or audit_dir / "license_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    blockers = license_blockers(registry, manifest, maps, registry_path.parent)
    if summary.get("cross_class_conflicts"):
        blockers.append("unresolved cross-class source conflicts; labels retained")
    if blockers and not audit_only:
        raise ValueError("release blocked: " + "; ".join(blockers))
    status = dict(
        schema_version=1,
        audit_only=audit_only,
        publishable=not audit_only,
        uploaded=False,
        blockers=blockers,
        image_count=len(manifest),
        manifest_sha256=summary["manifest_sha256"],
    )
    kaggle = dict(
        title="Blender Material Classification V3"
        + (" - AUDIT ONLY NOT PUBLISHABLE" if audit_only else ""),
        id=kaggle_id,
        licenses=[] if audit_only else [{"name": KAGGLE_LICENSES[registry["dataset_license"]]}],
    )
    card = (
        "# Blender material classification v3\n\n"
        + (
            "**AUDIT ONLY — NOT PUBLISHABLE.** Source licenses/provenance and class conflicts "
            "must be resolved before redistribution.\n\n"
            if audit_only
            else "Local release staging only. No upload has occurred.\n\n"
        )
        + f"{len(manifest)} synthetic Blender renders, {summary['material_count']} materials, "
        + f"{summary['source_group_count']} conservative source groups.\n\n"
        + "The manifest preserves original class labels and historical splits. Those splits "
        "can share source assets/families and are not an untouched v3 test set. Use "
        "source_group for grouped evaluation; inspect cross-class conflicts in data_audit.json. "
        "No silent exclusion or relabeling was applied. Synthetic performance does not "
        "establish performance on real photographs.\n\n"
        "Source names/candidate URLs are discovery hints, not proof of file origin or license. "
        "The registry records evidence status and source-content fingerprints. Generator "
        "family links are conservative dependence assumptions, not exact-byte duplicates. "
        "Raw textures and model weights are excluded. EXR maps were hashed, not decoded.\n"
    )
    metadata = {
        "metadata/manifest.csv": csv_bytes(
            sorted(manifest, key=lambda r: r["image_path"]), MANIFEST_COLUMNS
        ),
        "metadata/source_maps.csv": csv_bytes(maps, MAP_COLUMNS),
        "metadata/data_audit.json": json_bytes(summary),
        "metadata/license_registry.json": json_bytes(registry),
        "STAGING_STATUS.json": json_bytes(status),
        "dataset-metadata.json": json_bytes(kaggle),
        "DATASET_CARD.md": card.encode("utf-8"),
        "image_hashes.sha256": "".join(
            f"{r['sha256']}  {r['image_path']}\n"
            for r in sorted(manifest, key=lambda r: r["image_path"])
        ).encode(),
    }
    for payload in metadata.values():
        _no_private_paths(payload.decode("utf-8"))
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_zip.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with ZipFile(tmp_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for row in sorted(manifest, key=lambda r: r["image_path"]):
                path = resolve_data_path(root, row["image_path"])
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    raise ValueError("render content changed after audit")
                _zip_write(archive, row["image_path"], payload)
            for name, payload in sorted(metadata.items()):
                _zip_write(archive, name, payload)
        # rename (not replace) on Windows; existence already checked above.
        if output_zip.exists():
            raise FileExistsError("staged archive appeared during packaging")
        tmp_path.rename(output_zip)
    finally:
        tmp_path.unlink(missing_ok=True)
    checksum = sha256_file(output_zip)
    output_zip.with_suffix(output_zip.suffix + ".sha256").write_text(
        f"{checksum}  {output_zip.name}\n", encoding="utf-8"
    )
    return dict(status, sha256=checksum)


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(relative_path(name), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compresslevel=6)
