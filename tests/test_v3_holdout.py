"""Strict source isolation, protocol and resume tests (no Blender/GPU required)."""

import copy
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender/scripts/v3"))
sys.path.insert(0, str(ROOT / "scripts/v3"))
import holdout_common as common  # noqa: E402
import prepare_holdout as prepare  # noqa: E402
import render_holdout as renderer  # noqa: E402


@pytest.fixture
def registry():
    return common.read_json(ROOT / "metadata/v3_holdout/source_registry.json")


def reseal(registry):
    registry.pop("selection_sha256", None)
    registry["selection_sha256"] = common.fingerprint(registry)
    return registry


def validate(registry):
    common.validate_registry(registry, ROOT / "metadata/material_manifest.csv")


def test_frozen_sources_balanced_and_excluded(registry):
    validate(registry)
    assert len(registry["assets"]) == 21
    assert len(registry["excluded_source_ids"]) == 105
    assert len({a["material_id"] for a in registry["assets"]}) == 21
    leather = [a for a in registry["assets"] if a["class"] == "leather"]
    assert {a["asset_id"] for a in leather} == {"Leather037", "Leather038", "Leather039"}
    assert all(a["technique"] == "surface-photometric-stereo" for a in leather)


def test_identity_audit_uses_original_map_hashes(registry):
    audit = common.read_json(ROOT / "metadata/v3_holdout/source_identity_audit.json")
    common.validate_identity_audit(registry, audit)
    assert len(audit["original_sources"]) == 96
    registry["assets"][0]["maps"]["normal"]["provider_md5"] = next(
        iter(audit["original_sources"][0]["map_md5_1k"]["normal"].values())
    )
    with pytest.raises(ValueError, match="map identity"):
        common.validate_identity_audit(registry, audit)


@pytest.mark.parametrize(
    "source", ["wood_table", "wood_table_worn", "wood_table_02", "red_wood_table"]
)
def test_original_renamed_and_recolored_family_rejected(registry, source):
    registry["assets"][0]["asset_id"] = source
    with pytest.raises(ValueError, match="source family"):
        validate(reseal(registry))


def test_duplicate_new_family_rejected(registry):
    registry["assets"][1]["asset_id"] = "bamboo_veneer_02"
    with pytest.raises(ValueError, match="source family"):
        validate(reseal(registry))


def test_alias_identity_rejected(registry):
    registry["assets"][0]["aliases"] = ["wood_table_worn"]
    with pytest.raises(ValueError, match="source family"):
        validate(reseal(registry))


@pytest.mark.parametrize(
    "parent", ["LeatherSubstance001", "LeatherSubstance002", "LeatherSubstance006"]
)
def test_actual_provider_parent_rejected_even_if_claimed_lineage_differs(registry, parent):
    registry["assets"][-1]["provider_relations"]["parents"] = [parent]
    with pytest.raises(ValueError, match="lineage"):
        validate(reseal(registry))


def test_shared_new_recipe_rejected(registry):
    for asset in registry["assets"][-2:]:
        asset["provider_relations"]["parents"] = ["IndependentButSharedRecipe"]
    with pytest.raises(ValueError, match="lineage"):
        validate(reseal(registry))


@pytest.mark.parametrize("split", ["train", "val", "test", "development"])
def test_split_must_be_holdout(registry, split):
    registry["assets"][0]["split"] = split
    with pytest.raises(ValueError, match="CC0 holdout"):
        validate(reseal(registry))


def test_unbalanced_groups_rejected(registry):
    registry["assets"].pop()
    with pytest.raises(ValueError, match="balanced"):
        validate(reseal(registry))


def test_freeze_detects_unsealed_changes(registry):
    registry["assets"][0]["metallic_value"] = 0.7
    with pytest.raises(ValueError, match="fingerprint"):
        validate(registry)


@pytest.mark.parametrize(
    "value", ["../v1.png", "/absolute.png", "C:/v1.png", "a/../../v1", "a\\v1", "a/./file"]
)
def test_path_traversal_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        common.inside(tmp_path, value)


@pytest.mark.parametrize(
    "url",
    [
        "http://polyhaven.com/a/x",
        "https://polyhaven.com.evil.org/a/x",
        "https://evil.org/map",
        "https://user:secret@dl.polyhaven.org/map",
    ],
)
def test_only_official_https(url):
    with pytest.raises(ValueError):
        common.official_url(url, "polyhaven", download=True)


def base_cli(tmp_path, mode="holdout"):
    values = [
        "--mode",
        mode,
        "--config",
        str(tmp_path / "holdout.json"),
        "--out",
        str(tmp_path / "artifacts/v3/holdout_dataset"),
        "--manifest",
        str(tmp_path / "metadata/v3_holdout/render_manifest.csv"),
        "--original-manifest",
        str(tmp_path / "original.csv"),
    ]
    if mode == "holdout":
        for name in ("registry", "receipt", "sources-root", "identity-audit"):
            values.extend(
                [
                    "--" + name,
                    str(
                        tmp_path
                        / ("artifacts/v3/holdout_sources" if name == "sources-root" else name)
                    ),
                ]
            )
    else:
        values += [
            "--development-materials-root",
            str(tmp_path / "materials"),
            "--development-manifest",
            str(tmp_path / "split.csv"),
        ]
    return values


@pytest.mark.parametrize(
    "flag", ["--renders", "--materials", "--outp", "--engine", "--device", "--force", "--no-save"]
)
def test_unsupported_renderer_flags_rejected(tmp_path, flag):
    with pytest.raises(SystemExit):
        renderer.parse_args(base_cli(tmp_path) + [flag, "1"])


def test_relative_cli_paths_rejected(tmp_path):
    args = base_cli(tmp_path)
    args[args.index("--out") + 1] = "artifacts/v3/holdout_dataset"
    with pytest.raises(SystemExit):
        renderer.parse_args(args)


def test_no_v1_output_or_cross_project_manifest(tmp_path):
    args = renderer.parse_args(base_cli(tmp_path))
    renderer.validate_paths(args)
    args.out = tmp_path / "dataset"
    with pytest.raises(ValueError, match="isolated"):
        renderer.validate_paths(args)
    args.out = tmp_path / "other/artifacts/v3/holdout_dataset"
    with pytest.raises(ValueError, match="same explicit project"):
        renderer.validate_paths(args)


def test_development_and_fresh_inputs_cannot_mix(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        renderer.parse_args(
            base_cli(tmp_path) + ["--development-materials-root", str(tmp_path / "materials")]
        )


def test_smoke_cannot_write_fresh_manifest(tmp_path):
    args = renderer.parse_args(base_cli(tmp_path, "smoke"))
    args.out = tmp_path / "artifacts/v3/holdout_smoke"
    with pytest.raises(ValueError, match="isolated output"):
        renderer.validate_paths(args)


def test_prepare_requires_explicit_flags(tmp_path):
    args = []
    for name in ("registry", "receipt", "sources-root", "original-manifest", "identity-audit"):
        args.extend(["--" + name, str(tmp_path / name)])
    assert not prepare.parse_args(args).download
    with pytest.raises(SystemExit):
        prepare.parse_args(args + ["--resolution", "4K"])


def test_png_full_integrity_not_just_header(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (40, 50, 60)).save(path)
    assert common.validate_png(path, 8, 8) == common.digest(path)
    with pytest.raises(ValueError, match="dimensions"):
        common.validate_png(path, 16, 16)
    data = path.read_bytes()
    path.write_bytes(data[:24])
    with pytest.raises(ValueError):
        common.validate_png(path, 8, 8)
    corrupt = bytearray(data)
    corrupt[-20] ^= 1
    path.write_bytes(corrupt)
    with pytest.raises(ValueError):
        common.validate_png(path, 8, 8)


def test_resume_validates_pixels_fingerprints_and_orphans(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    image = root / "material.png"
    Image.new("RGB", (8, 8), (4, 5, 6)).save(image)
    manifest = tmp_path / "manifest.csv"
    with pytest.raises(ValueError, match="orphan"):
        renderer.existing_rows(manifest, root, {}, 8, 8)
    expected = {k: "test" for k in renderer.FIELDS if k != "image_sha256"}
    expected["image_path"] = "material.png"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=renderer.FIELDS)
        writer.writeheader()
        writer.writerow({**expected, "image_sha256": common.digest(image)})
    planned = {"material.png": expected}
    assert len(renderer.existing_rows(manifest, root, planned, 8, 8)) == 1
    for key in (
        "source_sha256",
        "scene_sha256",
        "renderer_sha256",
        "config_sha256",
        "parameters_sha256",
    ):
        changed = copy.deepcopy(planned)
        changed["material.png"][key] = "stale"
        with pytest.raises(ValueError, match="stale"):
            renderer.existing_rows(manifest, root, changed, 8, 8)
    Image.new("RGB", (8, 8), (7, 8, 9)).save(image)
    with pytest.raises(ValueError, match="stale/corrupt"):
        renderer.existing_rows(manifest, root, planned, 8, 8)


def test_config_frozen_and_seeded_envelopes(registry):
    config = common.read_json(ROOT / "configs/v3/holdout.json")
    renderer.validate_config(config)
    seen = set()
    for a in registry["assets"]:
        for index in range(1, 31):
            first = renderer.parameters(config, a["class"], a["material_id"], index)
            assert first == renderer.parameters(config, a["class"], a["material_id"], index)
            assert first["random_seed"] not in seen
            seen.add(first["random_seed"])
            for key in (
                "camera_azimuth_deg",
                "camera_elevation_deg",
                "camera_distance",
                "light_power",
            ):
                assert config[key][0] <= first[key] <= config[key][1]
    config["light_size"] = 10
    with pytest.raises(ValueError, match="fingerprint"):
        renderer.validate_config(config)


def test_nominal_1k_non_square_provider_map(tmp_path):
    path = tmp_path / "map.jpg"
    Image.new("RGB", (1024, 1026)).save(path)
    assert prepare.inspect_image(path, {"resolution": 1024}, "color")["height"] == 1026


def test_receipt_has_63_unique_hashed_maps(registry):
    receipt = common.read_json(ROOT / "metadata/v3_holdout/download_receipt.json")
    assert receipt["selection_sha256"] == registry["selection_sha256"]
    assert len(receipt["files"]) == 63
    assert len({f["path"] for f in receipt["files"]}) == 63
    assert all(len(f["sha256"]) == 64 for f in receipt["files"])
    assert all(f["role"] in common.MAPS for f in receipt["files"])


def test_receipt_rejects_changed_source_and_wrong_selection(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"verified official bytes")
    spec = {"path": path.name, "url": "https://polyhaven.com/example"}
    registry = {"selection_sha256": "frozen", "assets": [{"maps": {"color": spec}}]}
    receipt = {
        "selection_sha256": "frozen",
        "files": [
            {**spec, "role": "color", "sha256": common.digest(path), "size": path.stat().st_size}
        ],
    }
    common.validate_receipt(registry, receipt, tmp_path)
    receipt["selection_sha256"] = "different"
    with pytest.raises(ValueError, match="another selection"):
        common.validate_receipt(registry, receipt, tmp_path)
    receipt["selection_sha256"] = "frozen"
    path.write_bytes(b"replacement source")
    with pytest.raises(ValueError, match="source hash"):
        common.validate_receipt(registry, receipt, tmp_path)


def test_development_requires_train_sources(tmp_path):
    args = renderer.parse_args(base_cli(tmp_path, "development"))
    # With no train material, fail before opening source textures or any Blender API.
    with args.development_manifest.open("w", encoding="utf-8", newline="") as stream:
        stream.write("class,material_id,split\nwood,wood_01,test\n")
    args.original_manifest.write_text("class,material_id\nwood,wood_01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no old development source"):
        renderer.development_assets(args)


def test_features_manifest_contract():
    assert "original_split" in renderer.FIELDS
    assert "image_path" in renderer.FIELDS
    assert "material_id" in renderer.FIELDS


def test_scene_fingerprint_ignores_uv_storage_order_but_not_mapping():
    ns = SimpleNamespace
    vertices = [ns(co=(0.0, 0.0, 0.0)), ns(co=(1.0, 0.0, 0.0)), ns(co=(0.0, 1.0, 0.0))]
    world = ns(
        inputs={"Color": ns(default_value=(0.05, 0.05, 0.05, 1)), "Strength": ns(default_value=2)}
    )
    scene = ns(
        world=ns(node_tree=ns(nodes={"Background": world})),
        render=ns(engine="BLENDER_EEVEE", resolution_x=512, resolution_y=512),
        camera=ns(data=ns(lens=50, sensor_width=36)),
        objects={
            "HoldoutGround": ns(dimensions=(1000, 1000, 0)),
            "HoldoutArea": ns(data=ns(shape="SQUARE", size=5)),
        },
    )

    def sphere(order, uv):
        return ns(
            data=ns(
                vertices=vertices,
                loops=[ns(vertex_index=i) for i in order],
                polygons=[ns(loop_indices=range(3))],
                uv_layers=[ns(data=[ns(uv=uv[i]) for i in order])],
            )
        )

    uv = [(0, 0), (1, 0), (0, 1)]
    original = renderer.scene_fingerprint(scene, sphere([0, 1, 2], uv), {})
    rotated = renderer.scene_fingerprint(scene, sphere([1, 2, 0], uv), {})
    changed = renderer.scene_fingerprint(scene, sphere([1, 2, 0], [(0, 0), (0, 1), (1, 0)]), {})
    assert original == rotated
    assert changed != original
