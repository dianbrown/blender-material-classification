"""Real image fixtures exercise path containment, source dependence and release gates."""

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from material_classifier.v3.data import (
    audit_data,
    json_bytes,
    license_blockers,
    package_dataset,
    read_csv,
    relative_path,
    resolve_data_path,
    sha256_file,
    source_groups,
    validate_class_pure_groups,
)


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    materials, renders = [], []
    for index, (label, name, split) in enumerate(
        [
            ("wood", "Table", "train"),
            ("wood", "WornTable", "test"),
            ("stone", "UnrelatedRock", "val"),
        ]
    ):
        mid = f"{label}_{index:02d}"
        folder = root / "materials" / label / mid
        folder.mkdir(parents=True)
        # Shared roughness everywhere; base colors shared only by first two.
        Image.new("RGB", (8, 8), "red" if index < 2 else "blue").save(folder / "base.png")
        Image.new("L", (8, 8), 127).save(folder / "rough.png")
        materials.append(
            dict(
                material_id=mid,
                **{"class": label},
                source_material_name=name,
                source_folder="C:/Users/old/Desktop/private/irrelevant",
                base_color_file="base.png",
                roughness_file="rough.png",
                split=split,
            )
        )
        rel = f"dataset/{split}/{label}/{mid}.png"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (index, 50, 100)).save(path)
        renders.append(
            dict(
                image_path=rel,
                **{"class": label},
                material_id=mid,
                split=split,
                resolution_width="12",
                resolution_height="10",
            )
        )
    material_csv, render_csv = tmp_path / "materials.csv", tmp_path / "renders.csv"
    _csv(material_csv, materials)
    _csv(render_csv, renders)
    out = tmp_path / "audit"

    def run(**kwargs):
        return audit_data(
            root, render_csv, material_csv, out, expected_images=3, expected_materials=3, **kwargs
        )

    return root, out, material_csv, render_csv, run


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "dataset/../../secret",
        "/etc/passwd",
        "C:/Users/secret",
        "C:secret",
        "//server/share/secret",
        r"dataset\..\secret",
        "dataset/./x",
        "dataset//x",
        "dataset/x:secret",
        "dataset/x ",
        "dataset/x.",
        "dataset/NUL.png",
        "dataset/coM1.txt",
        "dataset/evil\n.png",
        "",
    ],
)
def test_rejects_malicious_paths(path):
    with pytest.raises(ValueError):
        relative_path(path)


def test_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("private")
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit unprivileged symlinks")
    with pytest.raises(ValueError, match="escapes"):
        resolve_data_path(root, "link/secret")


def test_audit_base_color_grouping_not_shared_roughness_and_determinism(corpus):
    root, out, _, _, run = corpus
    summary = run()
    rows = read_csv(out / "manifest.csv")
    groups = {r["material_id"]: r["source_group"] for r in rows}
    assert groups["wood_00"] == groups["wood_01"] != groups["stone_02"]
    assert summary["integrity_errors"] == []
    assert summary["source_group_count"] == 2
    assert len(summary["original_split_leakage"]) == 1
    assert summary["cross_class_conflicts"] == []
    assert len(summary["duplicate_source_files"]) == 2
    assert summary["readable_render_count"] == 3
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    run()
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    assert all("C:/Users" not in data.decode() for data in before.values())
    assert all(not Path(r["image_path"]).is_absolute() for r in rows)
    assert (root / rows[0]["image_path"]).is_file()


def test_alias_and_base_color_components_are_transitive():
    materials = [
        dict(material_id=mid, source_material_name=name)
        for mid, name in [
            ("a", "Oak Table"),
            ("b", "oak_table"),
            ("c", "Different"),
            ("d", "Unrelated"),
        ]
    ]
    maps = [
        dict(material_id=mid, role=role, sha256=digest)
        for mid, role, digest in [
            ("a", "base_color", "one"),
            ("b", "base_color", "two"),
            ("c", "base_color", "two"),
            ("a", "roughness", "constant"),
            ("d", "roughness", "constant"),
        ]
    ]
    groups, edges = source_groups(materials, maps)
    assert groups["a"] == groups["b"] == groups["c"] != groups["d"]
    assert {e["reason"] for e in edges} == {"source_name_alias", "base_color_sha256"}
    assert groups == source_groups(list(reversed(materials)), list(reversed(maps)))[0]


def test_family_override_merges_before_manifest_and_preserves_conflict(corpus):
    _, out, _, _, run = corpus
    summary = run(
        family_overrides=[
            dict(
                source_family="provider:generator",
                material_ids=["wood_01", "stone_02"],
                evidence_url="https://example.org/official/relationship",
                rationale="shared generator",
            )
        ]
    )
    assert summary["source_group_count"] == 1
    assert summary["cross_class_conflicts"][0]["classes"] == ["stone", "wood"]
    assert {r["class"] for r in read_csv(out / "manifest.csv")} == {"stone", "wood"}
    assert len(read_csv(out / "manifest.csv")) == 3
    assert summary["class_pure_source_groups"] is False
    assert summary["fold_ready"] is False
    with pytest.raises(ValueError, match="cross-class"):
        validate_class_pure_groups(read_csv(out / "manifest.csv"))


def test_invalid_override_is_rejected(corpus):
    *_, run = corpus
    with pytest.raises(ValueError, match="unknown"):
        run(
            family_overrides=[
                dict(
                    source_family="family",
                    material_ids=["wood_00", "missing"],
                    rationale="reason",
                    evidence_url="https://example.org",
                )
            ]
        )


def test_exact_duplicate_renders_detected(corpus):
    root, _, _, render_csv, run = corpus
    rows = read_csv(render_csv)
    (root / rows[1]["image_path"]).write_bytes((root / rows[0]["image_path"]).read_bytes())
    summary = run()
    assert len(summary["duplicate_renders"]) == 1
    assert len(summary["duplicate_render_pixels"]) == 1


def test_corrupt_wrong_dimensions_and_unmanifested_image(corpus):
    root, _, _, render_csv, run = corpus
    rows = read_csv(render_csv)
    (root / rows[0]["image_path"]).write_bytes(b"not an image")
    Image.new("RGB", (3, 4)).save(root / rows[1]["image_path"])
    Image.new("RGB", (1, 1)).save(root / "dataset" / "extra.png")
    summary = run()
    assert len(summary["integrity_errors"]) == 3
    assert summary["readable_render_count"] == 2
    assert summary["unmanifested_images"] == ["dataset/extra.png"]


def test_malicious_manifest_and_map_paths(corpus):
    _, _, material_csv, render_csv, run = corpus
    rows = read_csv(render_csv)
    rows[0]["image_path"] = "../private.png"
    _csv(render_csv, rows)
    with pytest.raises(ValueError, match="traversing"):
        run()
    materials = read_csv(material_csv)
    materials[0]["base_color_file"] = "../private.png"
    _csv(material_csv, materials)
    with pytest.raises(ValueError, match="traversing"):
        run()


def test_release_fails_closed_without_source_evidence(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    target = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="unresolved source license"):
        package_dataset(root, out, target)
    assert not target.exists()
    # Merely claiming CC0 in a guessed registry is insufficient.
    registry_path = out / "license_registry.json"
    registry = json.loads(registry_path.read_text())
    registry.update(dataset_license="CC0-1.0", release_approved=True)
    for entry in registry["sources"]:
        entry.update(
            verification_status="verified",
            license_spdx="CC0-1.0",
            redistribution_allowed=True,
            source_url=entry["candidate_source_url"],
        )
    registry_path.write_bytes(json_bytes(registry))
    with pytest.raises(ValueError, match="unresolved source license"):
        package_dataset(root, out, target)


def _approve_fixture_registry(out):
    # Artificial evidence is confined to this test, never the real source registry.
    evidence = out / "evidence.txt"
    evidence.write_text("Synthetic fixture permission to distribute under CC0-1.0.")
    registry_path = out / "license_registry.json"
    registry = json.loads(registry_path.read_text())
    registry.update(dataset_license="CC0-1.0", release_approved=True)
    for entry in registry["sources"]:
        entry.update(
            verification_status="verified",
            license_spdx="CC0-1.0",
            redistribution_allowed=True,
            source_url="https://example.org/fixture",
            evidence_url="https://example.org/fixture/license",
            evidence_path="evidence.txt",
            evidence_sha256=sha256_file(evidence),
            retrieved_at="2026-10-27",
        )
    registry_path.write_bytes(json_bytes(registry))
    return registry


def test_verified_registry_requires_matching_evidence_and_content(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    registry = _approve_fixture_registry(out)
    manifest, maps = read_csv(out / "manifest.csv"), read_csv(out / "source_maps.csv")
    assert license_blockers(registry, manifest, maps, out) == []
    target = tmp_path / "release.zip"
    status = package_dataset(root, out, target)
    assert status["publishable"] and not status["uploaded"]
    (out / "evidence.txt").write_text("changed")
    assert any("unresolved" in b for b in license_blockers(registry, manifest, maps, out))
    registry["sources"][0]["evidence_path"] = "../secret"
    with pytest.raises(ValueError):
        license_blockers(registry, manifest, maps, out)


def test_audit_only_archive_allowlist_hashes_determinism(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    (out / "private-notes.txt").write_text("C:/Users/private")
    (root / "weights.pt").write_bytes(b"model weights")
    one, two = tmp_path / "one.zip", tmp_path / "two.zip"
    status = package_dataset(root, out, one, audit_only=True)
    package_dataset(root, out, two, audit_only=True)
    assert one.read_bytes() == two.read_bytes()
    assert not status["publishable"] and status["audit_only"]
    with ZipFile(one) as archive:
        names = archive.namelist()
        assert len([name for name in names if name.startswith("dataset/")]) == 3
        assert not any(name.startswith("materials/") or name.endswith(".pt") for name in names)
        assert "metadata/private-notes.txt" not in names
        assert json.loads(archive.read("dataset-metadata.json"))["licenses"] == []
        assert b"NOT PUBLISHABLE" in archive.read("DATASET_CARD.md")
        for name in names:
            if not name.startswith("dataset/"):
                assert b"C:/Users" not in archive.read(name)
    with pytest.raises(FileExistsError):
        package_dataset(root, out, one, audit_only=True)


@pytest.mark.parametrize("kind", ["render", "map", "manifest"])
def test_stale_audit_rejected_and_no_partial_archive(corpus, tmp_path, kind):
    root, out, _, _, run = corpus
    run()
    if kind == "render":
        path = root / read_csv(out / "manifest.csv")[0]["image_path"]
    elif kind == "map":
        path = root / read_csv(out / "source_maps.csv")[0]["path"]
    else:
        path = out / "manifest.csv"
    path.write_bytes(path.read_bytes() + b"changed")
    target = tmp_path / "stale.zip"
    with pytest.raises(ValueError, match="changed"):
        package_dataset(root, out, target, audit_only=True)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_private_path_registry_cannot_leak_into_audit_archive(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    registry_path = out / "license_registry.json"
    registry = json.loads(registry_path.read_text())
    registry["private_note"] = "C:/Users/private/license.txt"
    registry_path.write_bytes(json_bytes(registry))
    with pytest.raises(ValueError, match="private path"):
        package_dataset(root, out, tmp_path / "private.zip", audit_only=True)


def test_audit_does_not_overwrite_existing_license_evidence(corpus):
    _, out, _, _, run = corpus
    run()
    _approve_fixture_registry(out)
    before = (out / "license_registry.json").read_bytes()
    run()
    assert (out / "license_registry.json").read_bytes() == before


def test_git_crlf_checkout_does_not_invalidate_staging(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    original = tmp_path / "lf.zip"
    package_dataset(root, out, original, audit_only=True)
    for name in ("manifest.csv", "source_maps.csv"):
        path = out / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    checked_out = tmp_path / "crlf.zip"
    package_dataset(root, out, checked_out, audit_only=True)
    assert original.read_bytes() == checked_out.read_bytes()


def test_verified_sources_do_not_choose_license_or_approve_publication(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    registry = _approve_fixture_registry(out)
    registry.update(dataset_license="", release_approved=False)
    for entry in registry["sources"]:
        entry.update(verified_by="agent", human_review_claimed=False)
    (out / "license_registry.json").write_bytes(json_bytes(registry))
    blockers = license_blockers(
        registry, read_csv(out / "manifest.csv"), read_csv(out / "source_maps.csv"), out
    )
    assert blockers == [
        "release has not been explicitly approved in the registry",
        "dataset license unresolved/unsupported",
    ]
    with pytest.raises(ValueError, match="release blocked"):
        package_dataset(root, out, tmp_path / "release.zip")
    status = package_dataset(root, out, tmp_path / "audit.zip", audit_only=True)
    assert status["blockers"] == blockers
    assert not status["publishable"] and not status["release_approved"]
    assert status["dataset_license"] == "" and not status["uploaded"]


@pytest.mark.parametrize("source_license,dataset_license,compatible", [
    ("CC0-1.0", "CC0-1.0", True),
    ("CC0-1.0", "CC-BY-4.0", True),
    ("CC-BY-4.0", "CC-BY-4.0", True),
    ("CC-BY-4.0", "CC0-1.0", False),
    ("unknown", "CC0-1.0", False),
    ("CC0-1.0", "unknown", False),
])
def test_supported_source_dataset_license_compatibility(
    corpus, tmp_path, source_license, dataset_license, compatible
):
    root, out, _, _, run = corpus
    run()
    registry = _approve_fixture_registry(out)
    registry["dataset_license"] = dataset_license
    for entry in registry["sources"]:
        entry.update(license_spdx=source_license, attribution="Synthetic fixture author",
                     license_url="https://example.org/fixture/license")
    (out / "license_registry.json").write_bytes(json_bytes(registry))
    target = tmp_path / "release.zip"
    if compatible:
        package_dataset(root, out, target)
        with ZipFile(target) as archive:
            assert json.loads(archive.read("dataset-metadata.json"))["licenses"] == [
                {"name": dataset_license}
            ]
            assert b"Synthetic fixture author" in archive.read("SOURCES.md")
    else:
        with pytest.raises(ValueError, match="release blocked"):
            package_dataset(root, out, target)
        assert not target.exists()


def test_cc_by_sources_require_carried_attribution(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    registry = _approve_fixture_registry(out)
    registry["dataset_license"] = "CC-BY-4.0"
    for entry in registry["sources"]:
        entry["license_spdx"] = "CC-BY-4.0"
    (out / "license_registry.json").write_bytes(json_bytes(registry))
    with pytest.raises(ValueError, match="source attribution missing"):
        package_dataset(root, out, tmp_path / "release.zip")


def test_evidence_is_self_contained_hash_bound_and_allowlisted(corpus, tmp_path):
    root, out, _, _, run = corpus
    run()
    registry = _approve_fixture_registry(out)
    evidence_dir = out / "license_evidence"
    evidence_dir.mkdir()
    evidence = evidence_dir / "render.json"
    evidence.write_bytes(json_bytes({"scope": "synthetic fixture"}))
    (evidence_dir / "private-unreferenced.txt").write_text("private")
    registry["supplemental_evidence"] = [dict(
        evidence_path="license_evidence/render.json", evidence_sha256=sha256_file(evidence)
    )]
    (out / "license_registry.json").write_bytes(json_bytes(registry))
    target = tmp_path / "evidence.zip"
    package_dataset(root, out, target, audit_only=True)
    with ZipFile(target) as archive:
        carried = json.loads(archive.read("metadata/license_registry.json"))
        for entry in carried["sources"] + carried["supplemental_evidence"]:
            payload = archive.read("metadata/" + entry["evidence_path"])
            assert hashlib.sha256(payload).hexdigest() == entry["evidence_sha256"]
        assert "metadata/license_evidence/private-unreferenced.txt" not in archive.namelist()
        assert archive.namelist().count("metadata/evidence.txt") == 1


@pytest.mark.parametrize("problem", ["changed", "missing", "traversal", "collision", "private"])
def test_audit_only_cannot_bypass_declared_evidence_integrity(corpus, tmp_path, problem):
    root, out, _, _, run = corpus
    run()
    registry = _approve_fixture_registry(out)
    evidence = out / "evidence.txt"
    if problem == "changed":
        evidence.write_text("changed")
    elif problem == "missing":
        evidence.unlink()
    elif problem == "traversal":
        registry["sources"][0]["evidence_path"] = "../outside.txt"
    elif problem == "collision":
        registry["sources"][0].update(
            evidence_path="manifest.csv", evidence_sha256=sha256_file(out / "manifest.csv")
        )
    else:
        evidence.write_text("C:/Users/private/evidence")
        for entry in registry["sources"]:
            entry["evidence_sha256"] = sha256_file(evidence)
    (out / "license_registry.json").write_bytes(json_bytes(registry))
    target = tmp_path / "bad.zip"
    with pytest.raises(ValueError):
        package_dataset(root, out, target, audit_only=True)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("approval", [False, "true", 1, None])
def test_publication_approval_requires_literal_true(corpus, tmp_path, approval):
    root, out, _, _, run = corpus
    run()
    registry = _approve_fixture_registry(out)
    registry["release_approved"] = approval
    (out / "license_registry.json").write_bytes(json_bytes(registry))
    with pytest.raises(ValueError, match="not been explicitly approved"):
        package_dataset(root, out, tmp_path / "release.zip")


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_completed_report_ingestion_preserves_fingerprints_and_release_defaults(tmp_path, newline):
    # Repository metadata only. Real-corpus byte validation is a separate staging run.
    from scripts.v3.record_provenance import record_provenance

    repo = Path(__file__).resolve().parents[1]
    audit = tmp_path / "audit"
    audit.mkdir()
    for name in ("license_registry.json", "source_maps.csv"):
        (audit / name).write_bytes((repo / "metadata/v3" / name).read_bytes())
    before = json.loads((audit / "license_registry.json").read_text())
    manifest = tmp_path / "material_manifest.csv"
    original = (repo / "metadata/material_manifest.csv").read_bytes()
    manifest.write_bytes(original.replace(b"\r\n", b"\n").replace(b"\n", newline))
    result = record_provenance(
        repo / "reports/results/v3/provenance_verification.json", audit,
        manifest,
    )
    after = json.loads((audit / "license_registry.json").read_text())
    assert result == dict(verified_sources=105, verified_maps=316,
                          dataset_license="", release_approved=False)
    assert [e["source_fingerprint"] for e in before["sources"]] == [
        e["source_fingerprint"] for e in after["sources"]
    ]
    assert not after["release_approved"] and after["dataset_license"] == ""
    for entry in after["sources"]:
        assert entry["verified_by"] == "agent" and not entry["human_review_claimed"]
        assert sha256_file(audit / entry["evidence_path"]) == entry["evidence_sha256"]

    # Line-ending compatibility must not accept changes to the manifest content.
    manifest.write_bytes(manifest.read_bytes().replace(b"wood", b"WOOD", 1))
    with pytest.raises(ValueError, match="original material manifest changed"):
        record_provenance(repo / "reports/results/v3/provenance_verification.json", audit, manifest)


def test_ingestion_rejects_changed_official_comparison_without_writing(tmp_path):
    from scripts.v3.record_provenance import record_provenance

    repo = Path(__file__).resolve().parents[1]
    audit = tmp_path / "audit"
    audit.mkdir()
    for name in ("license_registry.json", "source_maps.csv"):
        (audit / name).write_bytes((repo / "metadata/v3" / name).read_bytes())
    before = (audit / "license_registry.json").read_bytes()
    report = json.loads((repo / "reports/results/v3/provenance_verification.json").read_text())
    for match in report["materials"][0]["maps"][0]["official_matches"]:
        match["official_md5"] = "0" * 32
    changed = tmp_path / "report.json"
    changed.write_bytes(json_bytes(report))
    with pytest.raises(ValueError, match="no matching official bytes"):
        record_provenance(changed, audit, repo / "metadata/material_manifest.csv")
    assert (audit / "license_registry.json").read_bytes() == before
    assert not (audit / "license_evidence").exists()
