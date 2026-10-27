"""CPU-only evaluator tests using generated unit fixtures, never benchmark data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from material_classifier.v3.evaluation import (
    cluster_bootstrap,
    evaluate_predictions,
    paired_cluster_bootstrap,
    validate_predictions,
    write_evaluation,
)

CLASSES = ["wood", "metal", "glass"]


def fixture_frame(truth=None, probabilities=None, groups=None, folds=None):
    truth = np.asarray([0, 0, 1, 1, 2, 2] if truth is None else truth)
    probabilities = np.eye(3)[truth] if probabilities is None else np.asarray(probabilities)
    n = len(truth)
    groups = [f"g{i // 2}" for i in range(n)] if groups is None else groups
    return pd.DataFrame(
        {
            "image_path": [f"image_{i}.png" for i in range(n)],
            "class": np.asarray(CLASSES)[truth],
            "material_id": [f"m_{group}" for group in groups],
            "source_group": groups,
            "outer_fold": [f"fold_{group}" for group in groups] if folds is None else folds,
            "y_true": truth,
            "y_pred": probabilities.argmax(axis=1),
            **{f"p_{c}": probabilities[:, i] for i, c in enumerate(CLASSES)},
        }
    )


def test_perfect_predictions_and_conventions():
    result = evaluate_predictions(fixture_frame(), CLASSES, n_bootstrap=30)
    score = result["image_level"]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "cohen_kappa"):
        assert score["metrics"][key] == pytest.approx(1)
    for key in ("brier_score", "log_loss", "ece"):
        assert score["metrics"][key] == pytest.approx(0)
    assert score["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert score["ranking"]["macro_auroc"] == pytest.approx(1)
    assert score["ranking"]["macro_average_precision"] == pytest.approx(1)
    assert score["reliability"][-1]["count"] == 6
    assert result["group_aggregated"]["evaluation"]["metrics"]["n_groups"] == 3
    assert "development" in result["estimate_type"]
    json.dumps(result, allow_nan=False)


def test_completely_incorrect_predictions():
    frame = fixture_frame(probabilities=np.roll(np.eye(3)[[0, 0, 1, 1, 2, 2]], 1, axis=1))
    score = evaluate_predictions(frame, CLASSES, n_bootstrap=0)["image_level"]
    assert score["metrics"]["accuracy"] == 0
    assert score["metrics"]["macro_f1"] == 0
    assert score["metrics"]["brier_score"] == pytest.approx(2)
    assert score["metrics"]["ece"] == 1
    assert np.isfinite(score["metrics"]["log_loss"])
    assert score["metrics"]["mcc"] == pytest.approx(-0.5)
    assert score["metrics"]["cohen_kappa"] == pytest.approx(-0.5)


def test_metrics_match_sklearn_on_nontrivial_predictions():
    rng = np.random.default_rng(17)
    truth = np.tile(np.arange(3), 10)
    probabilities = rng.dirichlet(np.ones(3), len(truth))
    frame = fixture_frame(
        truth=truth, probabilities=probabilities, groups=[f"g{i}" for i in range(len(truth))]
    )
    result = evaluate_predictions(frame, CLASSES, n_bootstrap=0)
    metrics = result["image_level"]["metrics"]
    predicted = probabilities.argmax(axis=1)
    assert metrics["accuracy"] == pytest.approx(accuracy_score(truth, predicted))
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, predicted))
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(truth, predicted))
    assert metrics["cohen_kappa"] == pytest.approx(cohen_kappa_score(truth, predicted))
    assert metrics["log_loss"] == pytest.approx(log_loss(truth, probabilities, labels=[0, 1, 2]))
    for average in ("macro", "weighted"):
        expected = precision_recall_fscore_support(
            truth, predicted, average=average, zero_division=0
        )
        for metric, value in zip(("precision", "recall", "f1"), expected[:3]):
            assert metrics[f"{average}_{metric}"] == pytest.approx(value)


def test_missing_classes_and_undefined_scores():
    score = evaluate_predictions(fixture_frame(truth=[0, 0, 0, 0]), CLASSES, n_bootstrap=10)[
        "image_level"
    ]
    assert score["metrics"]["balanced_accuracy"] == 1
    assert score["metrics"]["macro_f1"] == pytest.approx(1 / 3)
    assert score["metrics"]["mcc"] is None
    assert score["metrics"]["cohen_kappa"] is None
    assert score["ranking"]["n_defined_classes"] == 0
    assert score["ranking"]["macro_auroc"] is None
    assert score["per_class"][1]["support"] == 0
    assert score["confusion_normalized_true"][1] == [0, 0, 0]
    assert score["bootstrap"]["intervals"]["mcc"]["n_undefined"] == 10
    partial = evaluate_predictions(fixture_frame(truth=[0, 0, 1, 1]), CLASSES, n_bootstrap=0)[
        "image_level"
    ]["ranking"]
    assert partial["n_defined_classes"] == 2
    assert partial["macro_auroc"] is None
    assert partial["macro_auroc_defined_classes"] == 1


def test_cluster_bootstrap_determinism_and_cluster_resampling():
    # One nine-image correct source and one single-image incorrect source. Whole
    # cluster draws can yield 0 or 1; independent image draws cannot produce this
    # 95% interval reliably. Unequal cluster size also checks image weighting.
    truth = [0] * 9 + [1]
    frame = fixture_frame(
        truth=truth, probabilities=[[1, 0, 0]] * 10, groups=["large"] * 9 + ["small"]
    )
    a = cluster_bootstrap(frame, CLASSES, n_bootstrap=300, seed=5, stratify_by_class=False)
    b = cluster_bootstrap(
        frame.sample(frac=1, random_state=4),
        CLASSES,
        n_bootstrap=300,
        seed=5,
        stratify_by_class=False,
    )
    assert a == b
    assert a["resampling_unit"] == "source_group"
    assert a["intervals"]["accuracy"]["estimate"] == pytest.approx(0.9)
    assert a["intervals"]["accuracy"]["lower"] == 0
    assert a["intervals"]["accuracy"]["upper"] == 1


def test_one_cluster_has_no_claimed_confidence_interval():
    result = cluster_bootstrap(fixture_frame(truth=[0, 0]), CLASSES, n_bootstrap=20)
    assert result["status"] == "insufficient_source_groups"
    assert result["n_performed"] == 0
    assert result["intervals"]["accuracy"]["lower"] is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda f: f.drop(columns="source_group"), "Missing"),
        (lambda f: f.iloc[:0], "empty"),
        (lambda f: pd.concat([f, f.iloc[:1]]), "Duplicate"),
        (lambda f: f.assign(p_wood=np.nan), "finite"),
        (lambda f: f.assign(p_wood=np.inf), "finite"),
        (lambda f: f.assign(p_wood=-0.1), "finite"),
        (lambda f: f.assign(p_wood=0.2), "sum to one"),
        (lambda f: f.assign(y_pred=2), "argmax"),
        (lambda f: f.assign(y_true=2), "agree with y_true"),
        (lambda f: f.assign(y_true=5), "valid zero-based"),
        (lambda f: f.assign(y_pred=0.1), "valid zero-based"),
        (lambda f: f.assign(source_group=""), "empty"),
        (lambda f: f.assign(p_unknown=0), "outside ordered"),
        (lambda f: f.assign(**{"class": "unknown"}), "outside ordered"),
        (lambda f: f.assign(material_id="same_material"), "material_id"),
    ],
)
def test_invalid_input(mutation, match):
    with pytest.raises(ValueError, match=match):
        evaluate_predictions(mutation(fixture_frame()), CLASSES, n_bootstrap=0)


def test_class_order_and_name_labels():
    frame = fixture_frame()
    names = validate_predictions(frame, CLASSES)
    assert names.y_true.tolist() == ["wood", "wood", "metal", "metal", "glass", "glass"]
    assert (
        evaluate_predictions(names, CLASSES[::-1], n_bootstrap=0)["image_level"]["metrics"][
            "accuracy"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="unique"):
        validate_predictions(frame, ["wood", "wood"])
    with pytest.raises(ValueError, match="ordered sequence"):
        validate_predictions(frame, "wood")
    tie = fixture_frame(truth=[0, 0], probabilities=[[0.5, 0.5, 0]] * 2)
    validate_predictions(tie, CLASSES)
    with pytest.raises(ValueError, match="argmax"):
        validate_predictions(tie.assign(y_pred=1), CLASSES)


def test_source_fold_isolation_and_summaries():
    frame = fixture_frame()
    frame.loc[0, "outer_fold"] = "leaked"
    with pytest.raises(ValueError, match="multiple outer folds"):
        validate_predictions(frame, CLASSES)
    result = evaluate_predictions(fixture_frame(), CLASSES, n_bootstrap=0)
    assert len(result["per_fold"]) == 3
    assert [r["metrics"]["n_images"] for r in result["per_fold"]] == [2, 2, 2]
    assert len(result["per_source_group"]) == 3


def test_group_aggregation_and_mixed_truth_rejection():
    frame = fixture_frame(
        truth=[0, 0, 1, 2],
        probabilities=[[0.6, 0.4, 0], [0.2, 0.8, 0], [0, 1, 0], [0, 0, 1]],
        groups=["g0", "g0", "g1", "g2"],
    )
    result = evaluate_predictions(frame, CLASSES, n_bootstrap=0)
    groups = result["group_aggregated"]
    assert groups["n_eligible_groups"] == 3
    assert groups["n_excluded_groups"] == 0
    assert groups["predictions"][0]["p_metal"] == pytest.approx(0.6)
    assert groups["evaluation"]["metrics"]["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="class-pure"):
        evaluate_predictions(fixture_frame(truth=[0, 1, 1, 2]), CLASSES, n_bootstrap=0)


def test_selective_ties_and_ece():
    frame = fixture_frame(
        truth=[0, 1, 2, 2],
        probabilities=[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]],
        groups=["g0", "g1", "g2", "g2"],
    )
    score = evaluate_predictions(frame, CLASSES, n_bootstrap=0, n_bins=5)["image_level"]
    assert [r["coverage"] for r in score["selective_risk_coverage"]] == [0.5, 1]
    assert [r["risk"] for r in score["selective_risk_coverage"]] == [0.5, 0.25]
    assert score["metrics"]["ece"] == pytest.approx(0.35)


def test_paired_bootstrap_alignment_and_difference():
    a = fixture_frame()
    b = fixture_frame(probabilities=np.roll(np.eye(3)[[0, 0, 1, 1, 2, 2]], 1, axis=1))
    result = paired_cluster_bootstrap(a, b.sample(frac=1, random_state=1), CLASSES, n_bootstrap=30)
    assert result["intervals"]["accuracy"]["estimate"] == 1
    assert result["intervals"]["accuracy"]["lower"] == 1
    assert result["intervals"]["brier_score"]["estimate"] == -2
    identical = paired_cluster_bootstrap(a, a, CLASSES, n_bootstrap=20)
    assert identical["intervals"]["macro_f1"]["upper"] == 0
    with pytest.raises(ValueError, match="identical image_path"):
        paired_cluster_bootstrap(a, b.iloc[:-1], CLASSES)
    with pytest.raises(ValueError, match="differ in outer_fold"):
        paired_cluster_bootstrap(a, b.assign(outer_fold="different"), CLASSES)


@pytest.mark.parametrize("kwargs", [{"n_bins": 0}, {"n_bootstrap": -1}, {"n_bootstrap": 1.5}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        evaluate_predictions(fixture_frame(), CLASSES, **kwargs)


def test_csv_round_trip_preserves_ids_and_writes_strict_outputs(tmp_path):
    frame = fixture_frame()
    frame["source_group"] = ["001", "001", "NA", "NA", "null", "null"]
    path = tmp_path / "unit_fixture.csv"
    frame.to_csv(path, index=False)
    canonical = validate_predictions(path, CLASSES)
    assert canonical.source_group.tolist() == frame.source_group.tolist()
    result = evaluate_predictions(path, CLASSES, n_bootstrap=5)
    paths = write_evaluation(result, tmp_path / "output", figures=False)
    assert json.loads(Path(paths["evaluation"]).read_text())["classes"] == CLASSES
    table = pd.read_csv(paths["per_fold"])
    assert len(table) == 3
    assert table.estimate_type.str.contains("development").all()


def test_duplicate_csv_headers_and_aliased_image_paths(tmp_path):
    path = tmp_path / "duplicate_headers.csv"
    frame = fixture_frame()
    frame.to_csv(path, index=False)
    text = path.read_text()
    lines = text.splitlines()
    lines[0] += ",p_wood"
    lines[1:] = [line + ",0" for line in lines[1:]]
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="columns must be unique"):
        validate_predictions(path, CLASSES)
    frame.loc[0, "image_path"] = "folder/../image_1.png"
    with pytest.raises(ValueError, match="Duplicate image_path"):
        validate_predictions(frame, CLASSES)


@pytest.mark.parametrize("truth", [None, [0, 0, 0, 0]])
def test_static_figures_written_without_display(tmp_path, truth):
    import matplotlib.pyplot as plt

    from material_classifier.v3.plots import plot_evaluation

    result = evaluate_predictions(fixture_frame(truth=truth), CLASSES, n_bootstrap=10)
    paths = plot_evaluation(result, tmp_path)
    assert len(paths) == 16
    assert all(Path(path).stat().st_size > 1000 for path in paths.values())
    assert not plt.get_fignums()


def test_cli_metadata_and_conflicting_classes(tmp_path):
    path = tmp_path / "unit_fixture.csv"
    fixture_frame().to_csv(path, index=False)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"classes": CLASSES}))
    script = Path(__file__).resolve().parents[1] / "scripts" / "v3" / "evaluate.py"
    command = [
        sys.executable,
        str(script),
        "--predictions",
        str(path),
        "--metadata",
        str(metadata),
        "--output-dir",
        str(tmp_path / "output"),
        "--n-bootstrap",
        "0",
        "--no-figures",
    ]
    env = dict(os.environ)
    completed = subprocess.run(command, text=True, capture_output=True, env=env, check=True)
    assert "development estimates" in completed.stdout
    assert (tmp_path / "output" / "evaluation.json").is_file()
    rejected = subprocess.run(
        command + ["--classes", "glass", "metal", "wood"], text=True, capture_output=True, env=env
    )
    assert rejected.returncode != 0
    assert "disagree" in rejected.stderr


def test_class_stratification_secondary_estimands_and_thresholds():
    frame = fixture_frame(
        truth=[0, 0, 0, 0, 0, 1, 2],
        probabilities=[[1, 0, 0]] * 4 + [[0, 1, 0]] * 2 + [[0, 0, 1]],
        groups=["large"] * 4 + ["small", "metal", "glass"],
    )
    result = evaluate_predictions(frame, CLASSES, n_bootstrap=300, seed=2)
    metrics = result["image_level"]["metrics"]
    assert metrics["accuracy"] == pytest.approx(6 / 7)
    assert metrics["source_balanced_accuracy"] == pytest.approx(3 / 4)
    assert metrics["class_balanced_source_accuracy"] == pytest.approx(5 / 6)
    boot = result["image_level"]["bootstrap"]
    assert boot["stratify_by_class"] is True
    assert boot["singleton_strata"] == ["metal", "glass"]
    shuffled = cluster_bootstrap(
        frame.sample(frac=1, random_state=2), CLASSES, n_bootstrap=300, seed=2
    )
    assert boot == shuffled
    assert result["accuracy_threshold_assessment"]["point_estimate_at_least_threshold"] is True
    assert result["accuracy_threshold_assessment"]["ci_lower_bound_at_least_threshold"] is False
    perfect = cluster_bootstrap(fixture_frame(), CLASSES, n_bootstrap=30)
    assert perfect["intervals"]["macro_f1"]["lower"] == 1
    assert perfect["intervals"]["class_balanced_source_accuracy"]["lower"] == 1


@pytest.mark.parametrize("fold", ["holdout", -1])
def test_explicit_fresh_holdout_provenance(tmp_path, fold):
    frame = fixture_frame().assign(outer_fold=fold)
    result = evaluate_predictions(frame, CLASSES, n_bootstrap=5, estimate_type="fresh_holdout")
    assert result["estimate_kind"] == "fresh_holdout"
    assert "OOF" not in result["estimate_type"]
    assert "Fresh independent holdout" in result["estimate_type"]
    paths = write_evaluation(result, tmp_path, figures=False)
    assert pd.read_csv(paths["image_metrics"]).estimate_type.str.contains("holdout").all()
    with pytest.raises(ValueError, match="estimate_type"):
        evaluate_predictions(frame, CLASSES, n_bootstrap=0, estimate_type="test")
