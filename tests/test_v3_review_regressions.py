"""Synthetic reproductions of independent review findings; no measured artifacts."""

import os
from types import SimpleNamespace

import pandas as pd
import pytest
from blender.scripts.v3.render_holdout import gpu_identity
from scripts.v3.build_report import ReportError, validate_predictions

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3.features import read_manifest, source_identity


def manifest_rows():
    return [
        dict(
            image_path=f"wood_{i}.png",
            material_id=f"wood_{i}",
            source_group=f"g{i}",
            original_split="train",
            **{"class": "wood"},
        )
        for i in range(2)
    ]


@pytest.mark.parametrize(
    "alias", ["./wood_0.png", "WOOD_0.PNG", ".\\wood_0.png", "wood_0.png.", "wood_0.png "]
)
def test_manifest_rejects_aliases_across_source_groups(tmp_path, alias):
    rows = manifest_rows()
    rows[1]["image_path"] = alias
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="aliases"):
        read_manifest(path)


def test_manifest_rejects_hardlink_identity_before_extraction(tmp_path):
    original = tmp_path / "wood_0.png"
    original.write_bytes(b"same file through distinct names")
    os.link(original, tmp_path / "wood_1.png")
    path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows()).to_csv(path, index=False)
    with pytest.raises(ValueError, match="resolved file identities"):
        source_identity(path, read_manifest(path), tmp_path)


def test_manifest_rejects_material_in_multiple_families(tmp_path):
    rows = manifest_rows()
    rows[1]["material_id"] = rows[0]["material_id"]
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="material_id"):
        read_manifest(path)


def test_report_rejects_mixed_class_source_family(tmp_path):
    rows = []
    for label in (0, 1):
        row = dict(
            image_path=f"image_{label}.png",
            material_id=f"material_{label}",
            source_group="shared_family",
            outer_fold=0,
            y_true=label,
            y_pred=label,
            **{"class": CLASS_NAMES[label]},
        )
        row.update({f"p_{name}": float(index == label) for index, name in enumerate(CLASS_NAMES)})
        rows.append(row)
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ReportError, match="multiple classes"):
        validate_predictions(path, {}, list(CLASS_NAMES), "nested_oof")


def test_background_gpu_metadata_records_unavailability():
    def unavailable():
        raise SystemError("GPU functions for drawing are not available in background mode")

    result = gpu_identity(SimpleNamespace(vendor_get=unavailable), lambda: "test GPU inventory")
    assert result["query_status"] == "drawing_context_unavailable_in_background"
    assert result["nvidia_device_inventory"] == "test GPU inventory"
    assert "renderer" not in result
