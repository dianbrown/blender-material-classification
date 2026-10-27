"""Synthetic schema and integrity tests for the renderer-to-classifier handoff."""

import csv
import json

import pytest
from scripts.v3.holdout_manifest import derive_manifest

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3.features import file_hash, read_manifest


@pytest.fixture
def inputs(tmp_path):
    assets, rows = [], []
    for name in CLASS_NAMES:
        image = tmp_path / f"{name}.png"
        image.write_bytes(f"SYNTHETIC bytes for {name}".encode())
        assets.append(
            dict(material_id=name, asset_id=name, provider="synthetic", **{"class": name})
        )
        rows.append(
            dict(
                image_path=image.name,
                material_id=name,
                source_asset_id=name,
                provider="synthetic",
                original_split="holdout",
                selection_sha256="selection",
                run_sha256="run",
                image_sha256=file_hash(image),
                **{"class": name},
            )
        )
    manifest = tmp_path / "render.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            dict(
                assets=assets,
                views_per_material=1,
                groups_per_class=1,
                selection_sha256="selection",
            )
        )
    )
    qc = tmp_path / "qc.json"
    qc.write_text(
        json.dumps(
            dict(passed=True, data_only=True, failures=[], images=7, groups=7, run_sha256="run")
        )
    )
    return manifest, registry, qc, tmp_path, tmp_path / "classifier.csv"


def test_classifier_view_preserves_sealed_renderer_inputs(inputs):
    before = [file_hash(path) for path in inputs[:3]]
    result = derive_manifest(*inputs)
    frame = read_manifest(inputs[-1])
    assert len(frame) == 7 and frame.source_group.nunique() == 7
    assert set(frame.original_split) == {"holdout"}
    assert not result["model_predictions_used"]
    assert [file_hash(path) for path in inputs[:3]] == before
    with pytest.raises(FileExistsError):
        derive_manifest(*inputs)


def test_classifier_view_rejects_failed_qc(inputs):
    qc = json.loads(inputs[2].read_text())
    qc["passed"] = False
    inputs[2].write_text(json.dumps(qc))
    with pytest.raises(ValueError, match="QC"):
        derive_manifest(*inputs)
    assert not inputs[-1].exists()


def test_classifier_view_rejects_changed_image(inputs):
    (inputs[3] / "wood.png").write_bytes(b"changed after QC")
    with pytest.raises(ValueError, match="changed after QC"):
        derive_manifest(*inputs)
    assert not inputs[-1].exists()
