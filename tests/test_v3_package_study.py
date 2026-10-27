"""Synthetic, CPU-only tests of the local study package and its fail-closed gates."""

import copy
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image, PngImagePlugin
from scripts.v3 import package_study as pkg


@pytest.fixture
def corpus(tmp_path):
    population = pkg.Population(("wood", "metal"), 2, 1, 2, 4, 8)
    old, fresh, old_sources, fresh_sources = [], [], {}, {}
    for split, rows, sources, quota in (
        ("development", old, old_sources, 2),
        ("test", fresh, fresh_sources, 1),
    ):
        for label in population.classes:
            for number in range(quota):
                mid = f"{split}_{label}_{number}"
                sources[mid] = {"provider": "synthetic"}
                for index in range(2):
                    filename = f"{mid}_r{index:02d}.png"
                    historical = "train" if number == 0 else "test"
                    relative = (
                        f"dataset/{historical}/{label}/{filename}"
                        if split == "development"
                        else f"{label}/{filename}"
                    )
                    root = (
                        tmp_path
                        if split == "development"
                        else (tmp_path / "artifacts/v3/holdout_dataset")
                    )
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    salt = len(old) + len(fresh) + 1
                    Image.new("RGB", (8, 8), (salt * 13, salt * 7, salt * 3)).save(path)
                    rows.append(
                        {
                            "image_path": relative,
                            "class": label,
                            "material_id": mid,
                            "source_group": f"family_{mid}",
                            "original_split": historical if split == "development" else "holdout",
                            "sha256": pkg.file_hash(path),
                        }
                    )
    return tmp_path, old, fresh, old_sources, fresh_sources, population


def prepare(corpus):
    return pkg.prepare_images(*corpus)


def test_deterministic_archive_counts_lineage_exclusions_and_checksums(corpus):
    root, old, fresh, *_ = corpus
    for name in ("weights.pt", "raw_texture.exr", "model_scores.json", "training.log"):
        (root / name).write_text("MUST NEVER ENTER ARCHIVE", encoding="utf-8")
    images, rows = prepare(corpus)
    outputs = [root / "first.zip", root / "second.zip"]
    results = [pkg.write_archive(p, images, rows, {}) for p in outputs]
    assert results[0]["sha256"] == results[1]["sha256"]
    assert results[0]["images"] == 12
    assert results[0]["class_counts"] == {"metal": 6, "wood": 6}
    assert results[0]["split_counts"] == {"development": 8, "test": 4}
    assert results[0]["material_ids"] == results[0]["source_groups"] == 6
    assert results[0]["crc_verified"] and results[0]["manifest_verified"]
    assert results[0]["size_bytes"] == outputs[0].stat().st_size
    assert Path(str(outputs[0]) + ".sha256").read_text().startswith(pkg.file_hash(outputs[0]))
    with ZipFile(outputs[0]) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == sorted(archive.namelist())
        assert all(i.date_time == pkg.FIXED_TIME for i in archive.infolist())
        assert not any(
            x in " ".join(archive.namelist()) for x in ("weights", "texture.exr", "scores", ".log")
        )
        manifest = list(csv.DictReader(io.StringIO(archive.read("metadata/manifest.csv").decode())))
        old_by_path = {r["image_path"]: r for r in old}
        for r in manifest:
            if r["split"] == "development":
                assert r["original_split"] == old_by_path[r["lineage_image_path"]]["original_split"]
            else:
                assert r["original_split"] == "holdout"
        status = json.loads(archive.read("metadata/release_status.json"))
        assert status["release_approved"] is False and status["dataset_license"] == ""
        metadata = json.loads(archive.read("dataset-metadata.json"))
        assert metadata["id"] is None and metadata["licenses"] == []
        for entry in archive.infolist():
            if not entry.filename.endswith(".png"):
                pkg.private_check(archive.read(entry))
        card = archive.read("DATASET_CARD.md").decode()
        assert "NOT an untouched test set" in card and "Physical lineage is inferred" in card
        inventory = json.loads(archive.read("metadata/archive_inventory.json"))
        assert set(inventory) == set(archive.namelist()) - {"metadata/archive_inventory.json"}
        for name, digest in inventory.items():
            assert pkg.sha256(archive.read(name)) == digest


@pytest.mark.parametrize(
    "path",
    [
        "../escape.png",
        "/escape.png",
        "C:/private.png",
        "a\\b.png",
        "a//b.png",
        "a/./b.png",
        "a/../b.png",
        "a /b.png",
        "a./b.png",
        "a:stream",
        "a\x00.png",
    ],
)
def test_path_containment(path, tmp_path):
    with pytest.raises(ValueError, match="path"):
        pkg.inside(tmp_path, path)


def test_symlink_escape(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not grant symlink creation")
    with pytest.raises(ValueError, match="escapes"):
        pkg.inside(root, "linked/image.png")


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda o, f: o[0].update(sha256="0" * 64), "hash mismatch"),
        (lambda o, f: o[0].update(original_split="holdout"), "historical split"),
        (lambda o, f: f[0].update(original_split="test"), "historical split"),
        (lambda o, f: f[0].update(source_group=o[0]["source_group"]), "source family"),
        (lambda o, f: o[4].update(source_group=o[0]["source_group"]), "source family"),
        (lambda o, f: o[0].update(image_path=o[1]["image_path"]), "duplicate image path"),
        (lambda o, f: f[0].update(sha256=o[0]["sha256"]), "duplicate image content"),
        (lambda o, f: o[0].update(pixel_sha256="0" * 64), "pixel hash"),
        (lambda o, f: o.pop(), "count mismatch"),
    ],
)
def test_population_and_integrity_rejections(corpus, change, match):
    change(corpus[1], corpus[2])
    with pytest.raises(ValueError, match=match):
        prepare(corpus)


def test_identical_pixels_with_different_png_bytes_rejected(corpus):
    root, old, *_ = corpus
    source, target = root / old[0]["image_path"], root / old[1]["image_path"]
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", "different encoding, identical image content")
    with Image.open(source) as im:
        im.save(target, pnginfo=info)
    old[1]["sha256"] = pkg.file_hash(target)
    assert old[1]["sha256"] != old[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate decoded"):
        prepare(corpus)


def test_archive_basename_conflicts(corpus):
    root, old, *_ = corpus
    original = root / old[0]["image_path"]
    new = original.parent / "different" / Path(old[1]["image_path"]).name
    new.parent.mkdir()
    new.write_bytes(original.read_bytes())
    old[0]["image_path"] = new.relative_to(root).as_posix()
    with pytest.raises(ValueError, match="archive image path conflict"):
        prepare(corpus)


@pytest.mark.parametrize("existing", ["zip", "sha256", "validation"])
def test_refuse_overwrite_any_output(corpus, existing):
    root = corpus[0]
    images, rows = prepare(corpus)
    output = root / "audit.zip"
    target = {
        "zip": output,
        "sha256": Path(str(output) + ".sha256"),
        "validation": output.with_suffix(".validation.json"),
    }[existing]
    target.write_bytes(b"PRESERVE")
    with pytest.raises(FileExistsError):
        pkg.write_archive(output, images, rows, {})
    assert target.read_bytes() == b"PRESERVE"


def test_source_change_during_packaging_leaves_no_outputs(corpus):
    images, rows = prepare(corpus)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(next(iter(images.values()))[0])
    output = corpus[0] / "audit.zip"
    with pytest.raises(ValueError, match="changed while packaging"):
        pkg.write_archive(output, images, rows, {})
    assert not output.exists()
    assert not Path(str(output) + ".sha256").exists()
    assert not output.with_suffix(".validation.json").exists()


def test_private_png_metadata_removed_without_changing_source_or_pixels(corpus):
    root, old, *_ = corpus
    source = root / old[0]["image_path"]
    info = PngImagePlugin.PngInfo()
    info.add_text("File", "C:\\Users\\private\\project\\scene.blend")
    info.add_text("Comment", "/home/private/project", zip=True)
    with Image.open(source) as im:
        im.save(source.with_suffix(".new.png"), pnginfo=info)
    source.write_bytes(source.with_suffix(".new.png").read_bytes())
    before = source.read_bytes()
    old[0]["sha256"] = pkg.sha256(before)
    images, rows = prepare(corpus)
    output = root / "private-free.zip"
    result = pkg.write_archive(output, images, rows, {})
    assert result["png_bytes_changed_by_metadata_removal"] == 1
    assert source.read_bytes() == before
    row = next(r for r in rows if r["lineage_image_path"] == old[0]["image_path"])
    assert row["source_sha256"] == old[0]["sha256"] != row["sha256"]
    with ZipFile(output) as archive:
        canonical = archive.read(row["image_path"])
    with Image.open(io.BytesIO(canonical)) as cleaned, Image.open(source) as original:
        assert "File" not in cleaned.info and "Comment" not in cleaned.info
        assert cleaned.tobytes() == original.tobytes()
    assert pkg.sha256(canonical) == row["sha256"]


def test_archive_crc_tampering_rejected(corpus):
    images, rows = prepare(corpus)
    output = corpus[0] / "tampered.zip"
    pkg.write_archive(output, images, rows, {})
    with ZipFile(output) as archive:
        expected = json.loads(archive.read("metadata/archive_inventory.json"))
        expected["metadata/archive_inventory.json"] = pkg.sha256(
            archive.read("metadata/archive_inventory.json")
        )
        manifest = archive.read("metadata/manifest.csv")
        info = archive.getinfo("DATASET_CARD.md")
        offset = info.header_offset + 30 + len(info.filename) + len(info.extra)
    with output.open("r+b") as stream:
        stream.seek(offset + 30)
        byte = stream.read(1)
        stream.seek(offset + 30)
        stream.write(bytes([byte[0] ^ 1]))
    from zipfile import BadZipFile

    with pytest.raises((ValueError, BadZipFile)):
        pkg.verify_archive(output, expected, manifest)


def test_text_hash_identity_and_privacy_normalization(tmp_path):
    raw = b"path,note\r\nC:/Users/private/project/materials/wood/a/,ok\r\n"
    (tmp_path / "original.csv").write_bytes(raw)
    book = pkg.Inputs(tmp_path)
    book.pin("original.csv", pkg.sha256(raw.replace(b"\r\n", b"\n")))
    entries = book.publication()
    canonical = entries["provenance/original.csv"]
    assert canonical == b"path,note\nmaterials/wood/a/,ok\n"
    record = json.loads(entries["metadata/input_hashes.json"])["text_inputs"][0]
    assert record["source_sha256"] == pkg.sha256(raw)
    assert record["canonical_sha256"] == pkg.sha256(canonical)
    assert record["source_sha256"] != record["canonical_sha256"]
    assert (tmp_path / "original.csv").read_bytes() == raw
    with pytest.raises(ValueError, match="hash mismatch"):
        book.pin("original.csv", "0" * 64)


def test_python_backslash_escaping_is_not_a_private_network_path():
    pkg.private_check(b'path.replace("\\\\", "/")\n')


@pytest.mark.parametrize(
    "private",
    [
        "C:/Users/private/file",
        "C:\\Users\\private\\file",
        "/home/private/file",
        "/Users/private/file",
        "\\\\server\\file",
    ],
)
def test_unknown_private_paths_block_archive(corpus, private):
    images, rows = prepare(corpus)
    with pytest.raises(ValueError, match="private absolute"):
        pkg.write_archive(
            corpus[0] / "audit.zip", images, rows, {"provenance/notes.md": private.encode()}
        )
    assert not (corpus[0] / "audit.zip").exists()


@pytest.mark.parametrize("entry", ["weights.pt", "textures/raw.exr", "training.log"])
def test_binary_or_log_provenance_rejected(corpus, entry):
    images, rows = prepare(corpus)
    with pytest.raises(ValueError, match="forbidden"):
        pkg.write_archive(corpus[0] / "audit.zip", images, rows, {entry: b"data"})


def handoff():
    registry = {
        "selection_sha256": "selection",
        "views_per_material": 1,
        "assets": [
            {
                "material_id": "fresh",
                "class": "wood",
                "provider": "polyhaven",
                "asset_id": "new_wood",
            }
        ],
    }
    receipt = {"selection_sha256": "selection", "files": []}
    run = {
        "selection_sha256": "selection",
        "source_sha256": pkg.fingerprint(receipt),
        "runtime": {"mode": "holdout", "engine": "BLENDER_EEVEE"},
    }
    run["run_sha256"] = pkg.fingerprint(run)
    row = {
        "image_path": "wood/fresh.png",
        "class": "wood",
        "material_id": "fresh",
        "source_group": "polyhaven:new_wood",
        "original_split": "holdout",
        "sha256": "a" * 64,
    }
    render = {
        **row,
        "source_asset_id": "new_wood",
        "provider": "polyhaven",
        "split": "holdout",
        "image_sha256": row["sha256"],
        **{k: v for k, v in run.items() if k != "runtime"},
    }
    qc = {
        "passed": True,
        "data_only": True,
        "failures": [],
        "images": 1,
        "groups": 1,
        "class_counts": {"wood": 1},
        "run_sha256": run["run_sha256"],
    }
    derived = {
        "run_sha256": run["run_sha256"],
        "selection_sha256": "selection",
        "model_predictions_used": False,
        "render_records_sha256": pkg.fingerprint([render]),
        "registry_sha256": pkg.fingerprint(registry),
        "qc_sha256": pkg.fingerprint(qc),
        "images": 1,
        "source_groups": 1,
    }
    return registry, receipt, qc, run, derived, [render], [row]


def test_small_synthetic_qc_handoff():
    assert set(pkg.validate_fresh_handoff(*handoff())) == {"fresh"}


@pytest.mark.parametrize(
    "index,field,value",
    [
        (1, "selection_sha256", "other"),
        (2, "passed", False),
        (2, "data_only", False),
        (2, "failures", ["bad"]),
        (2, "run_sha256", "other"),
        (3, "run_sha256", "other"),
        (4, "model_predictions_used", True),
        (4, "registry_sha256", "other"),
        (4, "render_records_sha256", "other"),
        (4, "images", 2),
    ],
)
def test_stale_receipt_qc_or_classifier_identity_rejected(index, field, value):
    args = handoff()
    args[index][field] = value
    with pytest.raises(ValueError):
        pkg.validate_fresh_handoff(*args)


def test_identity_audit_rejects_renamed_identical_source():
    registry = {
        "selection_sha256": "selection",
        "excluded_source_ids": ["old_wood"],
        "assets": [
            {
                "provider": "polyhaven",
                "provider_files_hash": "fresh",
                "maps": {"color": {"provider_md5": "reused"}},
            }
        ],
    }
    audit = {
        "selection_sha256": "selection",
        "collisions": [],
        "original_sources": [
            {
                "asset_id": "old_wood",
                "files_hash": "old",
                "map_md5_1k": {"color": {"jpg": "reused"}},
            }
        ],
    }
    with pytest.raises(ValueError, match="identity reused"):
        pkg.validate_identity_audit(registry, audit)


def test_receipt_requires_exact_maps_hashes_roles_and_selection(tmp_path):
    (tmp_path / "map.jpg").write_bytes(b"synthetic map")
    registry = {
        "selection_sha256": "selection",
        "assets": [{"maps": {"color": {"path": "map.jpg", "url": "https://polyhaven.com/source"}}}],
    }
    receipt = {
        "selection_sha256": "selection",
        "files": [
            {
                "path": "map.jpg",
                "role": "color",
                "url": "https://polyhaven.com/source",
                "size": 13,
                "sha256": pkg.file_hash(tmp_path / "map.jpg"),
            }
        ],
    }
    pkg.validate_receipt(registry, receipt, tmp_path)
    for field, value in (("path", "../map.jpg"), ("role", "normal"), ("sha256", "0" * 64)):
        changed = copy.deepcopy(receipt)
        changed["files"][0][field] = value
        with pytest.raises(ValueError):
            pkg.validate_receipt(registry, changed, tmp_path)


def test_default_population_cannot_accept_tiny_fixture(corpus):
    with pytest.raises(ValueError, match="count mismatch"):
        pkg.validate_population(corpus[1], corpus[2])


def registry_fixture(tmp_path):
    original = tmp_path / "material_manifest.csv"
    original.write_bytes(b"source_material_name\nold_source\n")
    assets = []
    for label in pkg.CLASSES:
        for suffix in "abc":
            aid = f"fresh_{label}_{suffix}"
            assets.append(
                {
                    "provider": "polyhaven",
                    "asset_id": aid,
                    "material_id": f"polyhaven_{aid}",
                    "class": label,
                    "split": "holdout",
                    "aliases": [],
                    "lineage_ids": [f"polyhaven:{aid}"],
                    "provider_relations": {},
                    "license": "CC0-1.0",
                    "authors": ["Synthetic fixture author"],
                    "label_rationale": "Synthetic fixture",
                    "independence_evidence": "Fixture only",
                    "canonical_url": f"https://polyhaven.com/a/{aid}",
                    "license_url": "https://polyhaven.com/license",
                    "normal_convention": "OpenGL",
                    "metallic_value": 0,
                    "render_assumptions": "Fixture",
                    "maps": {
                        role: {
                            "url": f"https://dl.polyhaven.org/{aid}/{role}.jpg",
                            "path": f"polyhaven/{aid}/{role}.jpg",
                            "resolution": 1024,
                            "colorspace": "sRGB" if role == "color" else "Non-Color",
                        }
                        for role in ("color", "roughness", "normal")
                    },
                }
            )
    registry = {
        "schema_version": 1,
        "split": "holdout",
        "groups_per_class": 3,
        "views_per_material": 30,
        "original_manifest_sha256": pkg.file_hash(original),
        "excluded_source_ids": ["old_source"],
        "excluded_lineage_ids": ["polyhaven:old_parent"],
        "assets": assets,
    }
    registry["selection_sha256"] = pkg.fingerprint(registry)
    return registry, original


def test_registry_requires_asset_exclusion_and_physical_lineage(tmp_path):
    registry, original = registry_fixture(tmp_path)
    pkg.validate_registry(registry, original)
    for field, value in (
        ("aliases", ["old_source"]),
        ("lineage_ids", ["polyhaven:old_parent"]),
        ("provider_relations", {"parents": ["old_source"]}),
        ("lineage_ids", registry["assets"][1]["lineage_ids"]),
    ):
        changed = copy.deepcopy(registry)
        changed["assets"][0][field] = value
        changed["selection_sha256"] = pkg.fingerprint(
            {k: v for k, v in changed.items() if k != "selection_sha256"}
        )
        with pytest.raises(ValueError, match="family|lineage|ancestry"):
            pkg.validate_registry(changed, original)
