"""CPU-only evidence assembly; all test measurements are explicitly synthetic."""

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.v3 import assemble_report as assembly

from material_classifier.v3.evaluation import evaluate_predictions


@pytest.fixture
def inputs(tmp_path):
    classes = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]
    rows = []
    for fold in range(5):
        for i, label in enumerate(classes):
            for view in range(2):
                pred = i if view == 0 else (i + 1) % 7
                rows.append(
                    dict(
                        image_path=f"SYNTHETIC/{fold}/{label}/{view}.png",
                        material_id=f"m_{fold}_{label}",
                        source_group=f"g_{fold}_{label}",
                        outer_fold=str(fold),
                        y_true=i,
                        y_pred=pred,
                        **{"class": label},
                        **{f"p_{c}": 0.7 if j == pred else 0.05 for j, c in enumerate(classes)},
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp_path / "predictions.csv", index=False)
    evaluation = evaluate_predictions(frame, classes, n_bootstrap=20)
    assembly.write_json(tmp_path / "evaluation.json", evaluation)
    protocol = json.loads(Path("configs/v3/protocol.json").read_text())
    assembly.write_json(tmp_path / "protocol.json", protocol)
    assembly.write_json(tmp_path / "environment.json", {"gpu": "SYNTHETIC CPU fixture"})
    assembly.write_json(
        tmp_path / "audit.json",
        dict(
            image_count=70,
            material_count=35,
            source_group_count=35,
            classes={c: 10 for c in classes},
        ),
    )
    feature = dict(identity={"implementation_sha256": "a" * 64}, complete=True)
    feature["identity_sha256"] = assembly.json_hash(feature["identity"])
    assembly.write_json(tmp_path / "cache.json", feature)
    histories = [
        dict(
            candidate=c["name"],
            C=value,
            inner_fold=fold,
            macro_f1=0.5,
            n_fit=40,
            n_valid=20,
            stage="outer",
            outer_fold=0,
        )
        for c in protocol["candidates"]
        for value in protocol["C"]
        for fold in range(3)
    ]
    pd.DataFrame(histories).to_csv(tmp_path / "search.csv", index=False)
    (tmp_path / "weights.bin").write_bytes(b"SYNTHETIC - NOT A MODEL")
    run = dict(
        status="complete",
        seconds=1.5,
        run_identity_sha256="b" * 64,
        run_identity=dict(
            protocol_sha256=assembly.json_hash(protocol),
            feature_cache_identity_sha256=feature["identity_sha256"],
        ),
        outputs={
            "search_history.csv": assembly.report.sha256(tmp_path / "search.csv"),
            "selected_oof.csv": assembly.report.sha256(tmp_path / "predictions.csv"),
        },
    )
    run["run_identity_sha256"] = assembly.json_hash(run["run_identity"])
    assembly.write_json(tmp_path / "run.json", run)
    sidecar = dict(
        class_order=classes,
        protocol_sha256=assembly.json_hash(protocol),
        run_identity_sha256=run["run_identity_sha256"],
        feature_implementation_sha256="a" * 64,
        model_sha256=assembly.report.sha256(tmp_path / "weights.bin"),
        interpretation="SYNTHETIC all-development fit; no test claim.",
    )
    assembly.write_json(tmp_path / "model.json", sidecar)
    spec = dict(
        schema_version=1,
        evidence_kind="synthetic_fixture",
        title="Synthetic evidence assembly test",
        run_id="synthetic",
        code_commit="c" * 40,
        protocol="protocol.json",
        environment="environment.json",
        audit="audit.json",
        run="run.json",
        model_sidecar="model.json",
        weights="weights.bin",
        feature_record="cache.json",
        search_history="search.csv",
        evaluations=[
            dict(
                id="selected",
                label="SYNTHETIC selected",
                scope="nested_oof",
                endpoint="nested_oof",
                evaluation="evaluation.json",
                predictions="predictions.csv",
                label_format="indices",
                description="Synthetic fixture, no real-world results.",
            )
        ],
        unavailable_endpoints={
            "fresh_holdout": "Not supplied; unopened.",
            "historical_test": "Historical evidence intentionally omitted.",
        },
    )
    assembly.write_json(tmp_path / "spec.json", spec)
    return tmp_path, spec, evaluation


def run_bundle(inputs, monkeypatch=None):
    root, spec, _ = inputs
    assembly.write_json(root / "spec.json", spec)
    if monkeypatch:
        monkeypatch.setattr(assembly.Bundle, "plot", lambda *args, **kwargs: None)
    return assembly.assemble_report(root / "spec.json", root / "output", allow_synthetic=True)


def test_normalization_preserves_values_and_intervals(inputs):
    _, _, raw = inputs
    summary = assembly.normalize_evaluation(raw, raw["classes"])
    assert summary["metrics"]["accuracy"]["value"] == raw["image_level"]["metrics"]["accuracy"]
    ci = summary["metrics"]["accuracy"]["ci"]
    assert ci["low"] == raw["image_level"]["bootstrap"]["intervals"]["accuracy"]["lower"]
    assert ci["n_resamples"] == 20
    assert sum(b["count"] for b in summary["calibration_bins"]) == 70


def test_bundle_is_builder_compatible_with_real_charts(inputs):
    out = run_bundle(inputs)
    manifest, endpoints, hashes = assembly.report.load_report(out, allow_synthetic=True)
    assert manifest["primary_endpoint"] == "nested_oof"
    assert endpoints[0]["data"]["n_images"] == 70
    figures = [s for s in manifest["supplements"] if s["kind"] == "figure"]
    assert len(figures) == 2  # recorded C search and outer folds, never fabricated loss
    assert all((out / s["path"]).stat().st_size > 1000 for s in figures)
    inventory = json.loads((out / "derived/source_inventory.json").read_text())
    for record in inventory:
        assert hashes[record["path"]] == record["sha256"]
        assert record["source_sha256"] == assembly.report.sha256(Path(record["source"]))
        content = (out / record["path"]).read_bytes()
        if record["normalization"] != "none":
            assert b"\r" not in content and not content.startswith(b"\xef\xbb\xbf")
        else:
            assert content == Path(record["source"]).read_bytes()


def test_appendix_routing_and_preview_label_preserve_measurements(inputs):
    _, spec, raw = inputs
    spec["appendix_details"] = True
    spec["status_label"] = "INTERNAL PRELIMINARY PREVIEW | v3"
    out = run_bundle(inputs)
    manifest, endpoints, _ = assembly.report.load_report(out, allow_synthetic=True)
    assert manifest["status_label"] == spec["status_label"]
    for supplement in manifest["supplements"]:
        if (
            supplement["path"].startswith("derived/search_")
            or supplement["path"] == "derived/compute.csv"
        ):
            assert supplement["section"] == "reproduction"
    assert (
        endpoints[0]["data"]["metrics"]["accuracy"]["value"]
        == raw["image_level"]["metrics"]["accuracy"]
    )


def test_appendix_routing_requires_boolean(inputs):
    _, spec, _ = inputs
    spec["appendix_details"] = "true"
    with pytest.raises(assembly.ReportError, match="appendix_details must be boolean"):
        run_bundle(inputs)


def test_requires_synthetic_opt_in(inputs):
    root, _, _ = inputs
    with pytest.raises(assembly.ReportError, match="Synthetic"):
        assembly.assemble_report(root / "spec.json", root / "output")
    assert not (root / "output").exists()


def test_preserves_existing_destination_and_old_reports(inputs, monkeypatch):
    root, _, _ = inputs
    out = run_bundle(inputs, monkeypatch)
    sentinel = out / "keep.txt"
    sentinel.write_text("existing results")
    (root / "report.pdf").write_bytes(b"old PDF")
    with pytest.raises(assembly.ReportError, match="already exists"):
        run_bundle(inputs, monkeypatch)
    assert sentinel.read_text() == "existing results"
    assert (root / "report.pdf").read_bytes() == b"old PDF"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("metric", "disagrees"),
        ("classes", "class order"),
        ("weights", "weights hash"),
        ("audit", "population"),
        ("run", "complete"),
        ("cache", "implementation"),
        ("search", "Search history"),
        ("fold", "Fold accuracy"),
        ("missing", "Missing explicit"),
        ("bootstrap", "bootstrap"),
        ("endpoint", "endpoint provenance"),
    ],
)
def test_rejects_inconsistent_evidence_atomically(inputs, monkeypatch, mutation, match):
    root, spec, raw = inputs
    if mutation in ("metric", "classes", "bootstrap", "fold", "endpoint"):
        if mutation == "metric":
            raw["image_level"]["metrics"]["accuracy"] = 0.91
        elif mutation == "classes":
            raw["classes"].reverse()
        elif mutation == "bootstrap":
            raw["image_level"]["bootstrap"]["status"] = "disabled"
        elif mutation == "fold":
            raw["per_fold"][0]["metrics"]["accuracy"] = 0.99
        else:
            raw["estimate_kind"] = "fresh_holdout"
        assembly.write_json(root / "evaluation.json", raw)
    elif mutation == "weights":
        (root / "weights.bin").write_bytes(b"changed")
    elif mutation == "audit":
        audit = json.loads((root / "audit.json").read_text())
        audit["source_group_count"] = 99
        assembly.write_json(root / "audit.json", audit)
    elif mutation == "run":
        run = json.loads((root / "run.json").read_text())
        run["status"] = "running"
        assembly.write_json(root / "run.json", run)
    elif mutation == "cache":
        sidecar = json.loads((root / "model.json").read_text())
        sidecar["feature_implementation_sha256"] = "bad"
        assembly.write_json(root / "model.json", sidecar)
    elif mutation == "search":
        with (root / "search.csv").open("a") as stream:
            stream.write("incomplete")
    else:
        spec["environment"] = "no-such-file.json"
    with pytest.raises(assembly.ReportError, match=match):
        run_bundle(inputs, monkeypatch)
    assert not (root / "output").exists()
    assert not list(root.glob(".assembly-*"))


def test_named_labels_and_matched_population_guard(inputs, monkeypatch):
    root, spec, raw = inputs
    frame = pd.read_csv(root / "predictions.csv", dtype={"outer_fold": str})
    for col in ("y_true", "y_pred"):
        frame[col] = frame[col].map(lambda i: raw["classes"][i])
    frame.to_csv(root / "named.csv", index=False)
    entry = {
        **spec["evaluations"][0],
        "id": "control",
        "predictions": "named.csv",
        "label_format": "names",
    }
    entry.pop("endpoint")
    spec["evaluations"].append(entry)
    out = run_bundle(inputs, monkeypatch)
    assert (out / "derived/control_summary.json").exists()
    # A numerically identical evaluation from different images is not matched.
    spec["evaluations"][1]["predictions"] = "different.csv"
    frame["image_path"] = "different/" + frame["image_path"]
    frame.to_csv(root / "different.csv", index=False)
    assembly.write_json(root / "spec2.json", spec)
    with pytest.raises(assembly.ReportError, match="Matched population"):
        assembly.assemble_report(root / "spec2.json", root / "different", allow_synthetic=True)


def add_fresh(inputs, *, overlap=False):
    root, spec, raw = inputs
    frame = pd.read_csv(root / "predictions.csv")
    frame = frame[frame.outer_fold == 0].copy()
    frame["outer_fold"] = "holdout"
    if not overlap:
        for col in ("image_path", "source_group", "material_id"):
            frame[col] = "fresh_" + frame[col]
    frame.to_csv(root / "fresh.csv", index=False)
    evaluation = evaluate_predictions(
        frame, raw["classes"], n_bootstrap=20, estimate_type="fresh_holdout"
    )
    assembly.write_json(root / "fresh.json", evaluation)
    assembly.write_json(
        root / "lock.json",
        {
            "model_sha256": assembly.report.sha256(root / "weights.bin"),
            "selection_frozen_at": "2026-10-27T10:00:00+00:00",
        },
    )
    assembly.write_json(root / "independence.json", {"status": "SYNTHETIC independently audited"})
    spec["evaluations"].append(
        dict(
            id="fresh",
            label="SYNTHETIC fresh",
            scope="fresh_holdout",
            endpoint="fresh_holdout",
            evaluation="fresh.json",
            predictions="fresh.csv",
            label_format="indices",
            description="SYNTHETIC independent endpoint",
            selection_record="lock.json",
            prediction_started_at="2026-10-27T11:00:00+00:00",
            independence_audit="independence.json",
        )
    )


def test_fresh_becomes_primary_only_with_lock_and_disjoint_population(inputs, monkeypatch):
    add_fresh(inputs)
    out = run_bundle(inputs, monkeypatch)
    manifest, endpoints, _ = assembly.report.load_report(out, allow_synthetic=True)
    assert manifest["primary_endpoint"] == "fresh_holdout"
    assert len(endpoints) == 2


@pytest.mark.parametrize("bad", ["overlap", "late_lock", "wrong_model"])
def test_rejects_invalid_fresh_evidence(inputs, monkeypatch, bad):
    root, spec, _ = inputs
    add_fresh(inputs, overlap=bad == "overlap")
    if bad == "late_lock":
        spec["evaluations"][-1]["prediction_started_at"] = "2026-10-27T09:00:00+00:00"
    if bad == "wrong_model":
        lock = json.loads((root / "lock.json").read_text())
        lock["model_sha256"] = "bad"
        assembly.write_json(root / "lock.json", lock)
    with pytest.raises(assembly.ReportError):
        run_bundle(inputs, monkeypatch)
    assert not (root / "output").exists()


def test_recorded_training_curves_only(inputs):
    root, spec, _ = inputs
    pd.DataFrame({"epoch": [1, 2], "train_loss": [1.2, 0.8]}).to_csv(
        root / "history.csv", index=False
    )
    spec["training_histories"] = [
        dict(
            path="history.csv",
            x="epoch",
            caption="SYNTHETIC training loss only; no validation measurements.",
            plots=[dict(ys=["train_loss"], ylabel="Cross entropy")],
        )
    ]
    out = run_bundle(inputs)
    assert (out / "derived/training_0_0.png").is_file()


def test_three_fold_exploration_cannot_be_promoted(inputs, monkeypatch):
    root, spec, raw = inputs
    frame = pd.read_csv(root / "predictions.csv")
    frame = frame[frame.outer_fold < 3]
    frame.to_csv(root / "exploratory.csv", index=False)
    evaluation = evaluate_predictions(frame, raw["classes"], n_bootstrap=20)
    assembly.write_json(root / "exploratory.json", evaluation)
    entry = dict(
        id="exploratory",
        label="Post-result exploration",
        scope="exploratory_development",
        evaluation="exploratory.json",
        predictions="exploratory.csv",
        label_format="indices",
        description="Post-result search, not a new unbiased nested estimate.",
    )
    spec["evaluations"].append(entry)
    out = run_bundle(inputs, monkeypatch)
    assert (out / "derived/comparison_exploratory_development.csv").is_file()
    entry["endpoint"] = "nested_oof"
    assembly.write_json(root / "promoted.json", spec)
    with pytest.raises(assembly.ReportError, match="Only nested/fresh"):
        assembly.assemble_report(root / "promoted.json", root / "promoted", allow_synthetic=True)


def test_duplicate_json_keys_rejected(inputs):
    root, _, _ = inputs
    (root / "spec.json").write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(assembly.ReportError, match="Duplicate JSON key"):
        assembly.assemble_report(root / "spec.json", root / "output")


def test_crlf_inputs_use_original_hashes_but_publish_lf(inputs, monkeypatch):
    root, spec, _ = inputs
    # Model the Windows training writers and Git eol=lf publication contract.
    for name in ("predictions.csv", "search.csv", "evaluation.json", "environment.json"):
        path = root / name
        value = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        path.write_bytes(value.replace("\n", "\r\n").encode("utf-8"))
    run = json.loads((root / "run.json").read_text())
    run["outputs"]["selected_oof.csv"] = assembly.report.sha256(root / "predictions.csv")
    run["outputs"]["search_history.csv"] = assembly.report.sha256(root / "search.csv")
    assembly.write_json(root / "run.json", run)
    spec["copy_weights"] = True
    out = run_bundle(inputs, monkeypatch)
    inventory = json.loads((out / "derived/source_inventory.json").read_text())
    assert any(r["source_sha256"] != r["sha256"] for r in inventory)
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in assembly.Bundle.TEXT_SUFFIXES:
            value = path.read_bytes()
            assert b"\r" not in value
            # Simulate checkout-normalization and prove manifest hashes survive.
            path.write_bytes(value.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8"))
    assembly.report.load_report(out, allow_synthetic=True)
    weights = next(r for r in inventory if r["source"].endswith("weights.bin"))
    assert weights["source_sha256"] == weights["sha256"]
    assert b"\r\n" in (root / "predictions.csv").read_bytes()


def test_json_finetune_histories_and_selection_tables(inputs):
    root, spec, _ = inputs
    assembly.write_json(
        root / "progress.json",
        {
            "history": [
                {
                    "epoch": 1,
                    "train": {"loss": 1.2},
                    "validation_development_only": {"macro_f1": 0.4},
                },
                {
                    "epoch": 2,
                    "train": {"loss": 0.8},
                    "validation_development_only": {"macro_f1": 0.5},
                },
            ]
        },
    )
    assembly.write_json(
        root / "selection.json",
        {
            "fits": [
                {"lr": 0.000003, "fold": 0, "best_epoch": 2, "best_score": 0.5},
            ]
        },
    )
    spec["training_histories"] = [
        dict(
            path="progress.json",
            rows_key="history",
            x="epoch",
            caption="SYNTHETIC post-result exploratory development, selected epochs.",
            plots=[
                dict(ys=["train.loss"], ylabel="Train loss"),
                dict(ys=["validation_development_only.macro_f1"], ylabel="Development macro F1"),
            ],
        )
    ]
    spec["json_tables"] = [
        dict(
            path="selection.json",
            rows_key="fits",
            columns={
                "LR": "lr",
                "fold": "fold",
                "best_epoch": "best_epoch",
                "dev_F1": "best_score",
            },
            section="comparison_ablations",
            caption="SYNTHETIC exploratory selection; not unbiased CV.",
        )
    ]
    out = run_bundle(inputs)
    assert (out / "derived/training_0_1.png").is_file()
    _, rows = assembly.read_csv(out / "derived/json_table_0.csv")
    assert rows[0]["dev_F1"] == "0.5"


def test_explicit_later_deployment_does_not_replace_nested_run(inputs, monkeypatch):
    root, spec, raw = inputs
    (root / "later.bin").write_bytes(b"SYNTHETIC LATER MODEL")
    assembly.write_json(
        root / "later.json",
        {"class_order": raw["classes"], "model_sha256": assembly.report.sha256(root / "later.bin")},
    )
    spec["deployment_model"] = dict(
        sidecar="later.json",
        weights="later.bin",
        description="Explicit parent choice, not made by assembly.",
    )
    out = run_bundle(inputs, monkeypatch)
    manifest, _, _ = assembly.report.load_report(out, allow_synthetic=True)
    assert manifest["deployment_fit"]["artifact"] == "derived/selected_weights_reference.json"
    assert manifest["primary_endpoint"] == "nested_oof"


@pytest.mark.parametrize("mutation", ["missing_inner_fold", "missing_C", "nonfinite_score"])
def test_search_curves_reject_missing_or_nonfinite_measurements(inputs, monkeypatch, mutation):
    root, _, _ = inputs
    history = pd.read_csv(root / "search.csv")
    if mutation == "missing_inner_fold":
        history = history.iloc[1:]
    elif mutation == "missing_C":
        history = history[history.C != 0.01]
    else:
        history.loc[0, "macro_f1"] = float("inf")
    history.to_csv(root / "search.csv", index=False)
    run = json.loads((root / "run.json").read_text())
    run["outputs"]["search_history.csv"] = assembly.report.sha256(root / "search.csv")
    assembly.write_json(root / "run.json", run)
    with pytest.raises(assembly.ReportError):
        run_bundle(inputs, monkeypatch)
    assert not (root / "output").exists()


def test_missing_history_values_are_not_fabricated(inputs, monkeypatch):
    root, spec, _ = inputs
    pd.DataFrame({"epoch": [1, 2], "train_loss": [1.2, 0.8]}).to_csv(
        root / "history.csv", index=False
    )
    spec["training_histories"] = [
        dict(
            path="history.csv",
            x="epoch",
            caption="Synthetic recorded training only.",
            plots=[dict(ys=["val_loss"], ylabel="Validation cross entropy")],
        )
    ]
    with pytest.raises(assembly.ReportError, match="val_loss"):
        run_bundle(inputs, monkeypatch)


def test_publication_assets_are_normalized_and_in_complete_inventory(inputs, monkeypatch):
    root, spec, _ = inputs
    (root / "publication.csv").write_bytes(b"Label,Accuracy\r\nSYNTHETIC,0.5\r\n")
    spec["publication_pages"] = [
        dict(
            title="Synthetic publication",
            blocks=[dict(kind="table", path="publication.csv", caption="Synthetic evidence")],
        )
    ]
    out = run_bundle(inputs, monkeypatch)
    manifest, _, _ = assembly.report.load_report(out, allow_synthetic=True)
    path = manifest["publication_pages"][0]["blocks"][0]["path"]
    assert b"\r" not in (out / path).read_bytes()
    inventory = json.loads((out / "derived/source_inventory.json").read_text())
    item = next(row for row in inventory if row["path"] == path)
    assert item["source_sha256"] == assembly.report.sha256(root / "publication.csv")
    assert item["sha256"] == assembly.report.sha256(out / path)
    assert item["sha256"] != item["source_sha256"]
