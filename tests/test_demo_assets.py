"""Static resources must preserve measured results and contain no private metadata."""

import json
from collections import Counter
from pathlib import Path

from PIL import Image

from material_classifier.demo import CLASS_NAMES, MODEL_NAMES
from material_classifier.demo.assets import compact_ci, counts, read_csv, read_json

ROOT = Path(__file__).resolve().parents[1]


def test_static_study_uses_saved_raw_results():
    study = read_json(ROOT / "app/assets/study.json")
    assert study["class_names"] == list(CLASS_NAMES)
    assert study["primary_model"] == "frozen_selected"
    assert {m["id"] for m in study["models"]} == set(MODEL_NAMES)
    for model in study["models"]:
        raw = read_json(ROOT / f"reports/results/v3/fresh/{model['id']}/evaluation.json")
        result = raw["image_level"]
        assert model["accuracy"] == result["metrics"]["accuracy"]
        assert model["macro_f1"] == result["metrics"]["macro_f1"]
        assert model["macro_f1_ci"] == compact_ci(result["bootstrap"], "macro_f1")
        assert model["accuracy_ci"] == compact_ci(result["bootstrap"], "accuracy")
        assert model["confusion_matrix"] == result["confusion_matrix"]
        assert model["per_class"] == result["per_class"]
        benchmark = read_json(ROOT / model["sources"][1])
        warm = next(
            r for r in benchmark["results"] if r["scope"] == "warm" and r["batch_size"] == 1
        )
        assert model["latency"]["median_ms"] == warm["median_seconds"] * 1000
        assert (
            model["size"]["total_deployment_bytes"]
            == benchmark["deployment"]["total_deployment_bytes"]
        )
    assert study["dataset"]["development"] == counts(read_csv(ROOT / "metadata/v3/manifest.csv"))
    assert study["dataset"]["holdout"] == counts(
        read_csv(ROOT / "metadata/v3_holdout/classification_manifest.csv")
    )
    serialized = json.dumps(study).lower()
    assert "c:\\" not in serialized and "/users/" not in serialized


def test_gallery_complete_source_distinct_metadata_free():
    assets = ROOT / "app/assets"
    samples = read_json(assets / "study.json")["samples"]
    assert Counter(s["class_name"] for s in samples) == dict.fromkeys(CLASS_NAMES, 3)
    assert len({s["source_group"] for s in samples}) == 21
    for sample in samples:
        path = (assets / sample["path"]).resolve()
        assert path.is_relative_to(assets.resolve())
        assert sample["split"] == "development"
        assert "stored_predictions" not in sample
        assert sample["source_url"].startswith("https://")
        with Image.open(path) as image:
            assert image.size == (512, 512)
            assert image.mode == "RGB"
            assert image.info == {}
            image.load()
