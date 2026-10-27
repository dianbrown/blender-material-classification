"""Small CPU-only tests of the statistical boundaries and deployment recipe."""

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3 import learning as learning_module
from material_classifier.v3.features import (
    assemble_features,
    extract_features,
    load_feature_cache,
    load_protocol,
    read_json,
)
from material_classifier.v3.inference import load_artifact, predict_image, predict_manifest
from material_classifier.v3.learning import (
    _best_with_tolerance,
    class_probabilities,
    grouped_folds,
    labels_for,
    make_fold_plan,
    nested_oof,
    select_candidates,
    train_run,
)
from test_v3_features import dummy_factory

pytest_plugins = ["test_v3_features"]


@pytest.fixture
def tiny_learning_data():
    records = []
    for label, name in enumerate(CLASS_NAMES):
        for group in range(6):
            for view in range(2):
                records.append(
                    {
                        "image_path": f"{name}_{group}_{view}.png",
                        "class": name,
                        "material_id": f"{name}_{group}",
                        "source_group": f"{name}_{group}",
                        "original_split": "old_test",
                    }
                )
    frame = pd.DataFrame(records)
    protocol = load_protocol(Path(__file__).resolve().parents[1] / "configs/v3/protocol.json")
    protocol["C"] = [0.1, 1.0]
    protocol["candidates"] = [protocol["candidates"][0], protocol["candidates"][3]]
    y = labels_for(frame)
    rng = np.random.default_rng(11)
    X = np.eye(7, dtype=np.float32)[y] + rng.normal(0, 0.05, (len(y), 7)).astype(np.float32)
    features = {candidate["name"]: X.copy() for candidate in protocol["candidates"]}
    return frame, features, protocol


def test_group_plan_isolation_class_support_and_determinism(tiny_learning_data):
    frame, _, protocol = tiny_learning_data
    plan = make_fold_plan(frame, protocol)
    assert plan["records"] == make_fold_plan(frame, protocol)["records"]
    y = labels_for(frame)
    groups = frame["source_group"].to_numpy()
    covered = []
    for index, (train, test) in enumerate(plan["outer"]):
        assert not set(groups[train]) & set(groups[test])
        assert set(y[train]) == set(y[test]) == set(range(7))
        covered.extend(test)
        for fit, valid in plan["inner"][index]:
            assert not set(groups[train][fit]) & set(groups[train][valid])
            assert not set(groups[test]) & set(groups[train][valid])
            assert set(y[train][fit]) == set(y[train][valid]) == set(range(7))
    assert sorted(covered) == list(range(len(frame)))
    broken = groups.copy()
    broken[y == 1] = groups[0]
    with pytest.raises(ValueError, match="unique class"):
        grouped_folds(y, broken, 5, 42)
    too_few = frame.groupby("class", sort=False).head(4)
    with pytest.raises(ValueError, match="independent source groups"):
        make_fold_plan(too_few, protocol)


def test_inner_scalers_fit_only_inner_training_rows(tiny_learning_data, monkeypatch):
    frame, features, protocol = tiny_learning_data
    y = labels_for(frame)
    groups = frame["source_group"].to_numpy()
    folds = grouped_folds(y, groups, 3, 91)
    original = learning_module.fit_head
    calls = []

    def checked_fit(X, labels, C, configuration):
        model = original(X, labels, C, configuration)
        fit_indices = folds[len(calls) % 3][0]
        np.testing.assert_allclose(
            model.named_steps["scaler"].mean_,
            features["dino_cls224"][fit_indices].mean(axis=0),
            atol=1e-7,
        )
        calls.append(len(X))
        return model

    monkeypatch.setattr(learning_module, "fit_head", checked_fit)
    select_candidates(features, y, groups, protocol, seed=91, folds=folds)
    assert len(calls) == 12
    # A deployment transform on extreme held-out data cannot update fitted moments.
    model = original(features["dino_cls224"][:70], y[:70], 1, protocol)
    mean = model.named_steps["scaler"].mean_.copy()
    model.named_steps["scaler"].transform(np.full((2, 7), 1e6))
    np.testing.assert_array_equal(model.named_steps["scaler"].mean_, mean)


def test_nested_selection_receives_only_outer_train(tiny_learning_data, monkeypatch):
    frame, features, protocol = tiny_learning_data
    plan = make_fold_plan(frame, protocol)
    original = learning_module.select_candidates
    y, groups = labels_for(frame), frame["source_group"].to_numpy()
    visited = []

    def check_boundary(train_features, train_y, train_groups, specification, **kwargs):
        train, test = plan["outer"][len(visited)]
        np.testing.assert_array_equal(train_y, y[train])
        np.testing.assert_array_equal(train_groups, groups[train])
        assert not set(train_groups) & set(groups[test])
        for name in features:
            np.testing.assert_array_equal(train_features[name], features[name][train])
        visited.append(len(train))
        return original(train_features, train_y, train_groups, specification, **kwargs)

    monkeypatch.setattr(learning_module, "select_candidates", check_boundary)
    result = nested_oof(features, frame, protocol, cpu_threads=1, fold_plan=plan)
    assert len(visited) == 5
    assert result["primary"].shape == (len(frame), 7)
    assert np.isfinite(result["primary"]).all()
    assert set(result["outer_fold"]) == set(range(5))
    # Identical candidates force declared-order tie-breaking, never OOF ranking.
    assert all(choice["selected"]["candidate"] == "dino_cls224" for choice in result["choices"])
    for fold in range(5):
        rows = result["outer_fold"] == fold
        name = result["choices"][fold]["selected"]["candidate"]
        np.testing.assert_array_equal(result["primary"][rows], result["candidates"][name][rows])


def test_numerical_ties_use_predeclared_order(tiny_learning_data):
    _, _, protocol = tiny_learning_data
    scores = [
        {"candidate_index": 0, "C": 0.01, "inner_mean_macro_f1": 0.8},
        {"candidate_index": 1, "C": 10, "inner_mean_macro_f1": 0.8 + 1e-13},
    ]
    assert _best_with_tolerance(scores, protocol, across_candidates=True) is scores[0]
    scores[1]["inner_mean_macro_f1"] += 1e-5
    assert _best_with_tolerance(scores, protocol, across_candidates=True) is scores[1]


def test_nonconvergence_fails_explicitly(tiny_learning_data, monkeypatch):
    frame, features, protocol = tiny_learning_data

    def nonconverged(*args, **kwargs):
        warnings.warn("solver failed", ConvergenceWarning)

    monkeypatch.setattr(Pipeline, "fit", nonconverged)
    with pytest.raises(RuntimeError, match="did not converge"):
        learning_module.fit_head(features["dino_cls224"], labels_for(frame), 1, protocol)


def test_train_artifacts_and_new_image_inference_parity(tmp_path, small_protocol, monkeypatch):
    protocol = copy.deepcopy(small_protocol)
    protocol["C"] = [0.1]
    protocol["candidates"] = [protocol["candidates"][0], protocol["candidates"][3]]
    records = []
    for label, name in enumerate(CLASS_NAMES):
        for group in range(5):
            path = f"{name}_{group}.png"
            Image.new("RGB", (13, 17), (20 + label * 30, 20 + group * 30, 80)).save(tmp_path / path)
            records.append(
                {
                    "image_path": path,
                    "class": name,
                    "material_id": path,
                    "source_group": f"{name}_{group}",
                    "original_split": "development",
                }
            )
    manifest = tmp_path / "manifest.csv"
    frame = pd.DataFrame(records)
    frame.to_csv(manifest, index=False)
    cache_path = tmp_path / "features"
    extract_features(
        manifest,
        tmp_path,
        cache_path,
        protocol,
        device="cpu",
        batch_size=16,
        model_factory=dummy_factory,
    )
    output = tmp_path / "training"
    original = learning_module.fit_head

    def verify_frozen_before_fit(*args, **kwargs):
        assert read_json(output / "protocol.json") == protocol
        plan = read_json(output / "folds.json")
        assert len(plan["outer"]) == 5 and len(plan["inner"]) == 5
        assert len(plan["final_all_development"]) == 3
        assert len(plan["outer"][0]["validation_counts"]) == 7
        return original(*args, **kwargs)

    monkeypatch.setattr(learning_module, "fit_head", verify_frozen_before_fit)
    record = train_run(manifest, tmp_path, cache_path, output, protocol, cpu_threads=1)
    assert record["status"] == "complete"
    artifact = output / "model.joblib"
    bundle = load_artifact(artifact)
    control = load_artifact(output / "control_model.joblib")
    assert control["metadata"]["recipe"]["name"] == protocol["control"]
    assert control["metadata"]["final_selection"]["C"] == 0.1
    oof = pd.read_csv(output / "selected_oof.csv")
    assert oof["image_path"].tolist() == frame["image_path"].tolist()
    assert oof["y_true"].tolist() == labels_for(frame).tolist()
    assert all("p_" + name in oof for name in CLASS_NAMES)
    with pytest.raises(FileExistsError):
        train_run(manifest, tmp_path, cache_path, output, protocol, cpu_threads=1)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Inference must never fit")

    monkeypatch.setattr(Pipeline, "fit", forbid_fit)
    base, _ = load_feature_cache(cache_path, manifest=manifest, data_root=tmp_path)
    expected = class_probabilities(
        bundle["pipeline"], assemble_features(base, bundle["metadata"]["recipe"]["parts"])
    )
    direct = predict_image(
        artifact, tmp_path / frame.iloc[0]["image_path"], model_factory=dummy_factory, cpu_threads=1
    )
    np.testing.assert_allclose(list(direct["probabilities"].values()), expected[0], atol=1e-7)
    predicted = predict_manifest(
        artifact,
        manifest,
        tmp_path,
        tmp_path / "prediction",
        feature_cache=cache_path,
        cpu_threads=1,
    )
    np.testing.assert_allclose(predicted[["p_" + name for name in CLASS_NAMES]], expected)
    with pytest.raises(ValueError, match="overlap development"):
        predict_manifest(artifact, manifest, tmp_path, tmp_path / "invalid-holdout", evaluate=True)
    fresh_path = tmp_path / "fresh.png"
    Image.new("RGB", (17, 19), (25, 31, 87)).save(fresh_path)
    fresh = pd.DataFrame(
        [
            {
                **records[0],
                "image_path": fresh_path.name,
                "source_group": "new-independent-family",
                "material_id": "fresh",
            }
        ]
    )
    fresh_manifest = tmp_path / "fresh.csv"
    fresh.to_csv(fresh_manifest, index=False)
    fresh_output = tmp_path / "fresh-prediction"
    predict_manifest(
        artifact,
        fresh_manifest,
        tmp_path,
        fresh_output,
        evaluate=True,
        cpu_threads=1,
        model_factory=dummy_factory,
    )
    assert (fresh_output / "metrics.json").is_file()
    shared_cache = fresh_output / "features"
    predict_manifest(
        output / "control_model.joblib",
        fresh_manifest,
        tmp_path,
        tmp_path / "fresh-control",
        feature_cache=shared_cache,
        evaluate=True,
        cpu_threads=1,
    )
