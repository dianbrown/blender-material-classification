"""Focused tests for the canonical dataset contract."""

from pathlib import Path

import numpy as np
from scripts.run_eda import _pooled_summary, representative_material_entries
from scripts.verify_dataset import verify

from material_classifier.data import (
    CLASS_ORDER,
    MaterialRenderDataset,
    class_to_idx,
    load_material_split_manifest,
    load_render_manifest,
    records_for_split,
)

ROOT = Path(__file__).resolve().parents[1]


def test_class_mapping_is_fixed_and_complete() -> None:
    assert class_to_idx() == {name: index for index, name in enumerate(CLASS_ORDER)}


def test_canonical_manifest_length_and_split_counts() -> None:
    rows = load_render_manifest(ROOT / "metadata/render_manifest.csv")
    assert len(rows) == 3150
    counts = {
        split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")
    }
    assert counts == {
        "train": 2310,
        "val": 420,
        "test": 420,
    }


def test_material_level_split_manifest_has_no_leakage() -> None:
    rows = load_render_manifest(ROOT / "metadata/render_manifest.csv")
    split_rows = load_material_split_manifest(ROOT / "metadata/material_manifest_split.csv")
    split_lookup = {row["material_id"]: row["split"] for row in split_rows}
    assert len(split_lookup) == 105
    assert all(row["split"] == split_lookup[row["material_id"]] for row in rows)
    assert all(
        len({item["split"] for item in rows if item["material_id"] == material_id}) == 1
        for material_id in split_lookup
    )


def test_records_preserve_material_identity_and_never_repartition() -> None:
    rows = load_render_manifest(ROOT / "metadata/render_manifest.csv")
    train = records_for_split(rows, "train")
    assert len(train) == 2310
    assert len({record.material_id for record in train}) == 77
    assert all(record.split == "train" for record in train)
    assert all(record.image_path.startswith("dataset/train/") for record in train)


def test_dataset_length_and_metadata() -> None:
    rows = load_render_manifest(ROOT / "metadata/render_manifest.csv")
    dataset = MaterialRenderDataset(
        records_for_split(rows, "val")[:2], ROOT / "dataset", return_metadata=True
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert item["material_id"] == dataset.records[0].material_id
    assert item["split"] == "val"
    assert item["label"] in range(7)


def test_full_dataset_verification_passes() -> None:
    result = verify(ROOT)
    assert result["passed"], result
    assert all(check["passed"] for check in result["checks"].values())


def test_eda_uses_pooled_pixel_standard_deviation() -> None:
    histogram = np.zeros(256, dtype=np.int64)
    histogram[0] = 1
    histogram[2] = 1
    items = [
        {
            "pixel_count": 2,
            "r_sum": 2.0,
            "r_sq_sum": 2.0,
            "g_sum": 2.0,
            "g_sq_sum": 2.0,
            "b_sum": 2.0,
            "b_sq_sum": 2.0,
            "luminance_sum": 2.0,
            "luminance_sq_sum": 2.0,
            "luminance_hist": histogram,
        },
        {
            "pixel_count": 2,
            "r_sum": 6.0,
            "r_sq_sum": 18.0,
            "g_sum": 6.0,
            "g_sq_sum": 18.0,
            "b_sum": 6.0,
            "b_sq_sum": 18.0,
            "luminance_sum": 2.0,
            "luminance_sq_sum": 2.0,
            "luminance_hist": histogram,
        },
    ]
    summary = _pooled_summary(items)
    assert summary["mean_r"] == 2.0
    assert summary["std_r"] == 1.0


def test_class_contact_sheet_represents_all_base_materials() -> None:
    rows = load_render_manifest(ROOT / "metadata/render_manifest.csv")
    entries = representative_material_entries(rows, "wood")
    assert len(entries) == 15
    assert {label for label, _ in entries} == {f"wood_{index:02d}" for index in range(1, 16)}
