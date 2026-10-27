"""Build and optionally execute the evidence-only V3 results notebook.

No model imports, training, inference, network calls, or global kernel installs.
Use --root to select the evidence checkout and --output for the deliverable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import textwrap
from pathlib import Path

import nbformat as nbf

NAMES = ("frozen_selected", "frozen_control", "resnet_fixed", "dinov2_finetuned")


def evidence_paths(root: Path) -> list[str]:
    base = "reports/results/v3/"
    paths = ["metadata/v3/manifest.csv", "metadata/v3_holdout/classification_manifest.csv"]
    paths += [
        base + name
        for name in (
            "final_model_lock.json",
            "model_choice.json",
            "final_execution_summary.json",
            "neural_execution_summary.json",
            "exploratory_robustness_amendment.json",
            "extraction_summary.json",
            "frozen_initial/protocol.json",
            "frozen_initial/run.json",
            "frozen_initial/fold_assignments.csv",
            "frozen_initial/fold_choices.json",
            "frozen_initial/selected_oof.csv",
            "frozen_initial/control_oof.csv",
            "resnet_baseline/resnet_oof.csv",
            "finetune_development/selection.json",
            "resnet_baseline/final_history.csv",
        )
    ]
    for name in ("nested_selected", "nested_control", "nested_resnet"):
        paths.append(base + name + "/evaluation.json")
    for name in NAMES:
        paths += [
            base + f"fresh/{name}/{file}"
            for file in (
                "evaluation.json",
                "predictions.csv",
                "prediction_run.json",
                "figures/image_confusion.png",
                "figures/calibration.png",
            )
        ]
        paths += [base + f"benchmarks/{name}/benchmark.json"]
    paths += [
        str(p.relative_to(root)).replace("\\", "/")
        for p in sorted((root / base / "fresh").glob("paired_*.json"))
    ]
    for name in ("primary", "dinov2_exploratory"):
        paths.append(base + f"robustness/{name}/comparison.csv")
    paths += [base + f"resnet_baseline/fold_{i}/history.csv" for i in range(5)]
    paths += [
        str(p.relative_to(root)).replace("\\", "/")
        for p in sorted((root / base / "finetune_development").glob("*/progress.json"))
    ]
    return sorted(paths)


def build(root: Path):
    inventory = {}
    for name in evidence_paths(root):
        data = (root / name).read_bytes()
        if Path(name).suffix in {".json", ".csv"}:
            data = data.replace(b"\r\n", b"\n")
        inventory[name] = hashlib.sha256(data).hexdigest()
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    md("""
    # V3 material classification: completed evidence review

    **Locked primary:** frozen DINOv2 multiscale + ConvNeXt features with a saved
    scaler/logistic head. Its fresh accuracy is **80.1587%**, macro-F1 **0.7952**,
    with a source-group bootstrap accuracy interval of **66.03–93.65%**.
    Fine-tuned DINOv2 reaches **86.8254%**, macro-F1 **0.8631**, with an interval of
    **76.98–95.87%**, as an **exploratory comparison**. It does not replace the primary.

    This executable notebook reads committed evidence only. It never loads model
    weights, renders images, trains, scores images, or fits calibration. Recomputed
    metrics are checks on saved predictions. Saved bootstrap intervals are read,
    not regenerated. All paths resolve from a repository root, without machine paths.
    Historical progress notes in `reports/V3_STATE.md` are superseded by the completed
    execution summaries for final results. The last cell records successful checks.
    """)
    code(r"""
    from pathlib import Path
    from datetime import datetime
    import hashlib, json, io, platform
    from importlib.metadata import version
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    from IPython.display import display, Image, Markdown

    ROOT = next(
        (
            p
            for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "metadata/v3/manifest.csv").is_file()
        ),
        None,
    )
    assert ROOT is not None, "Run from the repository root or notebooks directory"
    BASE = "reports/results/v3/"
    names = ["frozen_selected", "frozen_control", "resnet_fixed", "dinov2_finetuned"]
    labels = dict(
        zip(
            names,
            [
                "Frozen selected (PRIMARY)",
                "Frozen ConvNeXt control",
                "ResNet fixed baseline",
                "DINOv2 fine-tuned (exploratory)",
            ],
        )
    )
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 150)
    plt.rcParams.update({"figure.dpi": 100, "font.size": 10})


    def read_json(name):
        return json.loads((ROOT / BASE / name).read_text(encoding="utf-8"))


    def read_csv(name):
        return pd.read_csv(ROOT / BASE / name)


    def table(rows, digits=4):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        display(frame.round(digits))


    def saved_plot(name, width=900):
        # Embed a resized copy of the saved figure; the source file is unchanged.
        with PILImage.open(ROOT / BASE / name) as im:
            im = im.convert("RGB")
            im.thumbnail((width, 950))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
        display(Image(data=buf.getvalue()))


    lock = read_json("final_model_lock.json")
    choice = read_json("model_choice.json")
    final = read_json("final_execution_summary.json")
    neural = read_json("neural_execution_summary.json")
    classes = lock["class_order"]
    print("Evidence-only CPU analysis; Python", platform.python_version())
    print(
        "Notebook packages:", {n: version(n) for n in ["nbformat", "nbclient", "ipykernel"]}
    )
    """)
    md("""
    ## Evidence identity and integrity

    The following SHA-256 inventory pins every file consumed by this notebook.
    Text digests normalize **CRLF to LF only** to survive Git checkout settings;
    PNG digests are exact bytes. Original execution hashes are checked separately
    against exact or newline-equivalent text bytes. Hashes establish consistency
    with saved records, not independent proof of provenance. Raw images and model
    weights are deliberately unnecessary for this analysis.
    """)
    code(
        "EXPECTED_SHA256 = "
        + json.dumps(inventory, indent=4)
        + "\n"
        + textwrap.dedent(r"""
    for relative, expected in EXPECTED_SHA256.items():
        data = (ROOT / relative).read_bytes()
        if Path(relative).suffix in {'.json', '.csv'}:
            data = data.replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == expected, relative
    def recorded_hash(relative, expected):
        raw = (ROOT / relative).read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        variants = (raw, lf, lf.replace(b'\n', b'\r\n'))
        assert expected in {hashlib.sha256(b).hexdigest() for b in variants}, relative
    recorded_hash(BASE + 'model_choice.json', lock['choice_sha256'])
    recorded_hash(BASE + 'final_model_lock.json', final['lock_sha256'])
    recorded_hash('metadata/v3_holdout/classification_manifest.csv',
                  lock['holdout']['manifest_sha256'])
    recorded_hash(BASE + 'finetune_development/selection.json',
                  choice['fine_tuning_selection_sha256'])
    assert final['status'] == neural['status'] == 'complete'
    assert choice['fresh_holdout_used_for_choice'] is False
    assert (lock['primary_model'] == choice['primary_model']
            == final['primary_model'] == 'frozen_selected')
    print(f'{len(EXPECTED_SHA256)} evidence file digests verified; '
          'choice, lock and selection hashes verified.')
    """)
    )
    md("""
    ## Dataset and source-group design

    All **3,150 historical images** (including the old test split) form development
    data in V3: **105 material IDs, 101 source groups**. Shared texture families
    can span material IDs, so splitting by image or material alone is insufficient.
    Every source group stays together in a fold. The fresh set adds **630 images,
    21 materials and 21 new groups**, three groups and 90 images per class.
    Combined: **3,780 images, 126 materials, 122 source groups**.

    Group labels and image hashes are checked below. Distinct group strings alone
    cannot prove distinct physical sources; the source audit and sealed holdout
    registry provide the separate provenance evidence. Views of one source are
    correlated, so 630 images do not provide 630 independent source observations.
    """)
    code(r"""
    dev = pd.read_csv(ROOT / "metadata/v3/manifest.csv")
    fresh = pd.read_csv(ROOT / "metadata/v3_holdout/classification_manifest.csv")
    for frame, n, groups in [(dev, 3150, 101), (fresh, 630, 21)]:
        assert len(frame) == frame.image_path.nunique() == n
        assert frame.source_group.nunique() == groups
        assert frame.groupby("source_group")["class"].nunique().eq(1).all()
        assert set(frame["class"]) == set(classes)
    assert not set(dev.source_group) & set(fresh.source_group)
    assert not set(dev.material_id) & set(fresh.material_id)
    assert not set(dev.sha256) & set(fresh.sha256)
    assert fresh.groupby("class").size().eq(90).all()
    assert fresh.groupby("class").source_group.nunique().eq(3).all()
    combined = pd.concat([dev, fresh], ignore_index=True)
    assert (
        len(combined),
        combined.material_id.nunique(),
        combined.source_group.nunique(),
    ) == (3780, 126, 122)
    table(
        [
            {
                "partition": label,
                "images": len(frame),
                "materials": frame.material_id.nunique(),
                "source groups": frame.source_group.nunique(),
            }
            for label, frame in [
                ("Development", dev),
                ("Fresh", fresh),
                ("Combined", combined),
            ]
        ]
    )
    table(
        pd.concat(
            {
                label: frame.groupby("class")
                .agg(
                    images=("image_path", "size"),
                    materials=("material_id", "nunique"),
                    source_groups=("source_group", "nunique"),
                )
                .reindex(classes)
                for label, frame in [("Development", dev), ("Fresh", fresh)]
            },
            axis=1,
        )
    )
    """)
    md("""
    ## Development: nested estimates and selection scores

    Frozen features use five outer source-group folds, with three inner grouped
    folds selecting among five feature recipes and four regularization values.
    Scaling and the logistic head are fitted within each training partition.
    The ConvNeXt control selects its regularization independently. ResNet is a
    matched, fixed ten-epoch grouped OOF baseline, not nested architecture selection.

    Nested frozen accuracy is **79.3968%**; fixed ResNet OOF accuracy is **71.5556%**.
    The all-development frozen selector score **0.80352 macro-F1** and DINO's
    selected best-development mean **0.77248** are selection scores, not outer-CV
    estimates. DINO was an exploratory follow-up after the initial frozen result:
    two learning rates × three grouped folds, best-epoch selection on those folds,
    then a prescribed five-epoch all-development fit. No DINO outer-CV result exists.
    """)
    code(r"""
    protocol = read_json("frozen_initial/protocol.json")
    frozen_run = read_json("frozen_initial/run.json")
    folds = read_csv("frozen_initial/fold_assignments.csv")
    assert protocol["outer_folds"] == 5 and protocol["inner_folds"] == 3
    assert folds.groupby("source_group").outer_fold.nunique().eq(1).all()
    assert set(folds.image_path) == set(dev.image_path)
    dev_rows = []
    for directory, prediction_file, label in [
        ("nested_selected", "frozen_initial/selected_oof.csv", "Frozen nested selection"),
        ("nested_control", "frozen_initial/control_oof.csv", "Frozen nested control"),
        ("nested_resnet", "resnet_baseline/resnet_oof.csv", "ResNet fixed OOF"),
    ]:
        ev = read_json(directory + "/evaluation.json")
        pred = read_csv(prediction_file).set_index("image_path").sort_index()
        expected = folds.set_index("image_path").sort_index()
        assert len(pred) == pred.index.nunique() == 3150
        pd.testing.assert_series_equal(
            pred.outer_fold, expected.outer_fold, check_names=False
        )
        assert pred.groupby("source_group").outer_fold.nunique().eq(1).all()
        if directory != "nested_resnet":
            recorded_hash(
                BASE + prediction_file, frozen_run["outputs"][Path(prediction_file).name]
            )
        m = ev["image_level"]["metrics"]
        assert np.isclose((pred.y_true == pred.y_pred).mean(), m["accuracy"])
        ci = ev["image_level"]["bootstrap"]["intervals"]["accuracy"]
        dev_rows.append(
            {
                "estimate": label,
                "accuracy %": 100 * m["accuracy"],
                "macro F1": m["macro_f1"],
                "accuracy CI %": f"{100 * ci['lower']:.2f}–{100 * ci['upper']:.2f}",
            }
        )
    table(dev_rows)
    table(
        [
            {"outer fold": f["outer_fold"], **f["selected"]}
            for f in read_json("frozen_initial/fold_choices.json")
        ]
    )
    selection = read_json("finetune_development/selection.json")
    table(selection["scores"])
    print("Final frozen selection:", frozen_run["final_selection"])
    print(
        "DINO selected LR:",
        selection["backbone_lr"],
        "; final epochs:",
        selection["final_epochs"],
    )
    """)
    md("""
    ## Preselection lock and fresh-scoring chronology

    The development rule chose frozen selected before opening fresh scores.
    The endpoint lock records all four checkpoint identities and roles, the
    holdout manifest hash, and the rule that primary status cannot change with
    fresh ranking. The DINO final fit finished before the lock; the choice record
    predates its completion because the development choice was already available.
    The additional DINO corruption comparison was declared **after** clean scores
    and primary robustness were observed, and remains post hoc.
    """)
    code(r"""
    clean_commands = [c for c in final["commands"] if c["stage"] == "clean"]
    assert len(clean_commands) == 4 and {c["model"] for c in clean_commands} == set(names)
    parse = datetime.fromisoformat
    lock_time = parse(lock["frozen_at_utc"])
    assert parse(choice["decided_at_utc"]) < lock_time < parse(final["started_at_utc"])
    assert all(parse(s["end_utc"]) < lock_time for s in neural["stages"])
    assert all(
        lock_time < parse(c["start_utc"]) and c["exit_code"] == 0 for c in clean_commands
    )
    assert all(c["lock_sha256"] == final["lock_sha256"] for c in clean_commands)
    amendment = read_json("exploratory_robustness_amendment.json")
    assert all(
        parse(c["end_utc"]) < parse(amendment["declared_at_utc"]) for c in clean_commands
    )
    assert amendment["primary_model_unchanged"] == "frozen_selected"
    table(
        [
            {"event": "Development choice", "UTC": choice["decided_at_utc"]},
            {"event": "Endpoint checkpoint lock", "UTC": lock["frozen_at_utc"]},
            *[
                {"event": "Clean scoring: " + c["model"], "UTC": c["start_utc"]}
                for c in clean_commands
            ],
            {
                "event": "Post hoc DINO robustness amendment",
                "UTC": amendment["declared_at_utc"],
            },
        ]
    )
    table(
        [
            {
                "model": name,
                "role": lock["models"][name]["role"],
                "checkpoint SHA256": lock["models"][name]["artifact_sha256"],
            }
            for name in names
        ]
    )
    """)
    md("""
    ## Four fresh endpoints and uncertainty

    These are image-weighted metrics on exactly the same 630 images. The saved
    95% percentile intervals use 10,000 class-stratified source-group bootstrap
    draws, keeping every view with its source. There are only **21 groups**, three
    per class. These intervals do not include refitting or selection uncertainty.
    Mean-probability group voting is a separate secondary estimand, not the image
    accuracy presented here. Passing an 80% point target does not establish an
    80% lower confidence bound or a real-world guarantee.
    """)
    code(r"""
    evaluations, predictions, fresh_rows = {}, {}, []
    identity = fresh.set_index("image_path").sort_index()
    for name in names:
        directory = f"fresh/{name}/"
        ev, run = (
            read_json(directory + "evaluation.json"),
            read_json(directory + "prediction_run.json"),
        )
        recorded_hash(BASE + directory + "predictions.csv", run["predictions_sha256"])
        recorded_hash(
            BASE + directory + "evaluation.json",
            final["evaluations"][name]["evaluation_sha256"],
        )
        assert run["model_sha256"] == lock["models"][name]["artifact_sha256"]
        source = run if "manifest_sha256" in run else run["source"]
        assert source["manifest_sha256"] == lock["holdout"]["manifest_sha256"]
        assert run["predictions_sha256"] == final["clean_results"][name]["predictions_sha256"]
        assert ev["classes"] == classes and ev["estimate_kind"] == "fresh_holdout"
        pred = read_csv(directory + "predictions.csv").set_index("image_path").sort_index()
        assert len(pred) == pred.index.nunique() == 630
        pd.testing.assert_frame_equal(
            pred[["class", "material_id", "source_group"]],
            identity[["class", "material_id", "source_group"]],
        )
        assert pred.outer_fold.eq("holdout").all()
        assert np.array_equal(
            pred.y_true, pred["class"].map({c: i for i, c in enumerate(classes)})
        )
        probs = pred[["p_" + c for c in classes]].to_numpy()
        assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
        assert np.array_equal(probs.argmax(axis=1), pred.y_pred)
        cm = np.zeros((7, 7), dtype=int)
        np.add.at(cm, (pred.y_true, pred.y_pred), 1)
        assert np.array_equal(cm, ev["image_level"]["confusion_matrix"])
        f1 = np.divide(
            2 * np.diag(cm),
            cm.sum(0) + cm.sum(1),
            out=np.zeros(7),
            where=(cm.sum(0) + cm.sum(1)) != 0,
        ).mean()
        m = ev["image_level"]["metrics"]
        assert np.isclose(m["accuracy"], np.trace(cm) / cm.sum()) and np.isclose(
            m["macro_f1"], f1
        )
        bs = ev["image_level"]["bootstrap"]
        assert bs["n_performed"] == 10000 and bs["n_source_groups"] == 21
        assert bs["resampling_unit"] == "source_group" and bs["stratify_by_class"]
        ci, fi = bs["intervals"]["accuracy"], bs["intervals"]["macro_f1"]
        fresh_rows.append(
            {
                "model": labels[name],
                "accuracy %": 100 * m["accuracy"],
                "accuracy 95% CI %": f"{100 * ci['lower']:.2f}–{100 * ci['upper']:.2f}",
                "macro F1": m["macro_f1"],
                "F1 95% CI": f"{fi['lower']:.4f}–{fi['upper']:.4f}",
            }
        )
        evaluations[name], predictions[name] = ev, pred
    table(fresh_rows)
    print(
        "Verified all four prediction identities, probabilities, confusion matrices, "
        "accuracies and macro-F1 values."
    )
    """)
    code(r"""
    paired_rows = []
    for path in sorted((ROOT / BASE / "fresh").glob("paired_*.json")):
        paired = read_json("fresh/" + path.name)
        assert paired["n_source_groups"] == 21 and paired["difference"] == "A minus B"
        for metric in ["accuracy", "macro_f1"]:
            ci = paired["intervals"][metric]
            assert ci["lower"] <= 0 <= ci["upper"]
            paired_rows.append(
                {
                    "comparison (A minus B)": path.stem.removeprefix("paired_"),
                    "metric": metric,
                    "difference": ci["estimate"],
                    "95% lower": ci["lower"],
                    "95% upper": ci["upper"],
                }
            )
    table(paired_rows)
    """)
    md("""
    All displayed paired accuracy and F1 intervals include zero. DINO's observed
    gain over the primary is 6.67 percentage points, with a paired accuracy interval
    of approximately −2.06 to +16.19 points. The fresh evidence does **not establish
    superiority**. It also does not establish equivalence. No retrospective primary
    relabeling or test-driven hyperparameter choice is made.
    """)
    md("""
    ## Per-class errors and calibration

    The next table shows precision, recall and F1 for all classes and models.
    The saved confusion and calibration panels follow for each endpoint. Confusion
    rows are true classes. Calibration statistics describe uncalibrated saved
    probabilities; no holdout calibration was fitted. ECE uses 15 equal-width
    top-label bins; multiclass Brier is the sum of squared probability errors.
    """)
    code(r"""
    perclass = pd.concat(
        {
            labels[name]: pd.DataFrame(
                evaluations[name]["image_level"]["per_class"]
            ).set_index("class")[["precision", "recall", "f1", "support"]]
            for name in names
        },
        names=["model", "class"],
    )
    table(perclass)
    table(
        [
            {
                "model": labels[name],
                **{
                    k: evaluations[name]["image_level"]["metrics"][k]
                    for k in ["log_loss", "brier_score", "ece"]
                },
            }
            for name in names
        ]
    )
    errors = []
    for name in names:
        cm = np.array(evaluations[name]["image_level"]["confusion_matrix"])
        pairs = sorted(
            [
                (int(cm[i, j]), classes[i], classes[j])
                for i in range(7)
                for j in range(7)
                if i != j and cm[i, j]
            ],
            reverse=True,
        )[:3]
        errors.extend(
            {"model": labels[name], "true": a, "predicted": b, "images": n}
            for n, a, b in pairs
        )
    table(errors)
    """)
    for name in NAMES:
        code(
            f"display(Markdown('### ' + labels[{name!r}]))\n"
            f"saved_plot('fresh/{name}/figures/image_confusion.png', 760)\n"
            f"saved_plot('fresh/{name}/figures/calibration.png', 760)"
        )
    md("""
    The primary recalls all fresh wood, metal and fabric views, but only 57.78% of
    ceramic views; ceramic → stone accounts for 30 errors. With only three sources
    per class, these class results can be dominated by individual texture families.
    DINO's higher accuracy does not imply better calibration: inspect its ECE
    alongside loss and Brier rather than treating accuracy as probability quality.
    """)
    md("""
    ## Actual recorded training histories

    ResNet has five ten-epoch fold fits plus a ten-epoch all-development fit.
    Its histories record training loss/accuracy; outer assessments occur only
    after training, so there is no fabricated per-epoch outer-validation curve.
    DINO plots include all six development fits and the final five-epoch fit.
    Dashed DINO curves are development validation reused for checkpoint selection.
    The gap between near-perfect training and lower validation is consistent with
    overfitting to the available synthetic families. Final fits have no validation.
    """)
    code(r"""
    resnet_histories = {
        f"fold_{i}": read_csv(f"resnet_baseline/fold_{i}/history.csv") for i in range(5)
    }
    resnet_histories["all_development"] = read_csv("resnet_baseline/final_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for fit, history in resnet_histories.items():
        assert list(history.epoch) == list(range(1, 11))
        assert history.amp_skipped_updates.sum() == 0
        axes[0].plot(history.epoch, history.train_loss, label=fit)
        axes[1].plot(history.epoch, history.train_accuracy, label=fit)
    for ax, title in zip(axes, ["ResNet training loss", "ResNet training accuracy"]):
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    plt.show()
    plt.close(fig)

    dino_histories = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, path in zip(
        axes.flat, sorted((ROOT / BASE / "finetune_development").glob("lr_*/progress.json"))
    ):
        progress = read_json(f"finetune_development/{path.parent.name}/progress.json")
        assert progress["status"] == "complete"
        h = pd.json_normalize(progress["history"])
        dino_histories[path.parent.name] = h
        assert h["train.amp_skipped_steps"].sum() == 0
        ax.plot(h.epoch, h["train.macro_f1"], label="Train F1")
        ax.plot(
            h.epoch, h["validation_development_only.macro_f1"], "--", label="Development F1"
        )
        ax.axvline(progress["best_epoch"], color="grey", alpha=0.5, label="Selected epoch")
        ax.set(title=path.parent.name, xlabel="Epoch", ylim=(0.4, 1.03))
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    plt.show()
    plt.close(fig)
    dino_final = pd.json_normalize(
        read_json("finetune_development/all_development/progress.json")["history"]
    )
    assert list(dino_final.epoch) == list(range(1, 6))
    assert len(dino_histories) == 6
    table(
        dino_final[
            [
                "epoch",
                "head_only",
                "train.loss",
                "train.accuracy",
                "train.macro_f1",
                "train.optimizer_steps",
                "train.amp_skipped_steps",
                "seconds",
            ]
        ]
    )
    table(
        [
            {"model": "ResNet", "fit": k, "epochs": len(h), "epoch seconds": h.seconds.sum()}
            for k, h in resnet_histories.items()
        ]
        + [
            {"model": "DINO", "fit": k, "epochs": len(h), "epoch seconds": h.seconds.sum()}
            for k, h in {**dino_histories, "all_development": dino_final}.items()
        ]
    )
    """)
    md("""
    ## Fixed corruptions: primary and post hoc DINO comparison

    Both use the same five previously staged corruptions and fixed clean checkpoints.
    Primary corruption evaluation was specified by the endpoint lock. DINO's
    extension was added after observing fresh and primary corruption results.
    These are paired descriptive sensitivity results, not new independent holdouts
    or an established robustness-superiority test. Negative drops mean improvement.
    JPEG quality 30 lowers primary accuracy to **59.05%**, versus **75.40%** for DINO.
    """)
    code(r"""
    corruptions = []
    for directory, name, summary_key in [
        ("primary", "frozen_selected", "robustness"),
        ("dinov2_exploratory", "dinov2_finetuned", "exploratory_robustness"),
    ]:
        frame = read_csv(f"robustness/{directory}/comparison.csv")
        saved = (
            pd.DataFrame(final[summary_key]["comparison"]).set_index("condition").sort_index()
        )
        actual = frame.set_index("condition").sort_index()
        assert len(frame) == 5 and frame.n_pairs.eq(630).all()
        assert np.allclose(actual[["accuracy", "macro_f1"]], saved[["accuracy", "macro_f1"]])
        assert np.allclose(
            frame.clean_accuracy, evaluations[name]["image_level"]["metrics"]["accuracy"]
        )
        assert np.allclose(
            frame.clean_accuracy - frame.accuracy, frame.accuracy_drop_from_clean
        )
        frame["model"] = labels[name]
        corruptions.append(frame)
    corruption_table = pd.concat(corruptions, ignore_index=True)
    table(
        corruption_table[
            [
                "model",
                "condition",
                "accuracy",
                "macro_f1",
                "accuracy_drop_from_clean",
                "correct_to_incorrect",
                "incorrect_to_correct",
            ]
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    corruption_table.pivot(index="condition", columns="model", values="accuracy").mul(
        100
    ).plot.bar(ax=ax)
    ax.set(
        ylabel="Accuracy (%)",
        xlabel="",
        ylim=(0, 100),
        title="Fixed corruptions (DINO comparison is post hoc)",
    )
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
    plt.show()
    plt.close(fig)
    """)
    md("""
    ## Measured efficiency and deployment footprint

    Saved GPU benchmarks use an RTX 4070 Laptop GPU, four requested CPU threads,
    five warmups and 30 repetitions, at batch sizes 1 and 16. Warm timings include
    preprocessing, transfers, networks and probability output, but exclude file
    reading/decoding and model loading. The separate three-trial reload measurement
    is **process-warm**, not OS-cold startup. CUDA allocator peaks are not total
    board usage. These measurements are hardware- and scope-specific.

    Frozen selected's small scaler/head is not a standalone classifier: it needs
    two unique pretrained files, with DINO instantiated at two resolutions.
    Count that shared stored file once, while retaining separate runtime instances.
    Total deployment bytes include saved sidecar metadata. DINO's fine-tuned loader
    does not need the original pretrained cache. No FLOPs or CPU-RSS result is claimed.
    """)
    code(r"""
    benchmark_rows, footprint_rows = [], []
    for name in names:
        b = read_json(f"benchmarks/{name}/benchmark.json")
        recorded_hash(
            BASE + f"benchmarks/{name}/benchmark.json",
            final["benchmarks"][name]["benchmark_json_sha256"],
        )
        assert b["class_order"] == classes and b["status"] == "complete"
        assert (
            b["deployment"]["artifact"]["sha256"] == lock["models"][name]["artifact_sha256"]
        )
        assert b["settings"]["repetitions"] == 30 and b["settings"]["warmup"] == 5
        for r in b["results"]:
            benchmark_rows.append(
                {
                    "model": labels[name],
                    "scope": r["scope"],
                    "batch": r["batch_size"],
                    "median ms/batch": 1000 * r["median_seconds"],
                    "p95 ms/batch": 1000 * r["p95_seconds"],
                    "images/s": r["images_per_second"],
                    "allocated MiB": r["peak_allocated_bytes"] / 2**20,
                    "reserved MiB": r["peak_reserved_bytes"] / 2**20,
                }
            )
        d = b["deployment"]
        footprint_rows.append(
            {
                "model": labels[name],
                "artifact MB": d["artifact"]["bytes"] / 1e6,
                "unique pretrained MB": d["unique_pretrained_weight_bytes"] / 1e6,
                "total deployment MB": d["total_deployment_bytes"] / 1e6,
                "dependency": d["dependency_note"],
            }
        )
    table(benchmark_rows, 2)
    table(footprint_rows, 3)
    table(
        [
            {
                "stage": s["label"],
                "wall seconds": s["duration_seconds"],
                "status": s["status"],
            }
            for s in neural["stages"]
        ],
        2,
    )
    extraction = read_json("extraction_summary.json")
    print(
        "Initial feature extraction session seconds:",
        round(sum(j["session_seconds"] for j in extraction["jobs"].values()), 2),
    )
    print("Initial frozen nested/head run seconds:", round(frozen_run["seconds"], 2))
    print(
        "Initial timings exclude later guard revalidation; "
        "neural wall times include stage overhead."
    )
    """)
    md("""
    ## Limitations and interpretation

    - Synthetic-only evidence, a restricted seven-class fresh taxonomy, and just
      three fresh source groups per class limit generalization and CI resolution.
      More camera views cannot substitute for more independent materials/sources.
    - Fresh rendering uses a different renderer version; softer shadows and other
      renderer/domain differences are possible. Source novelty and renderer shift
      are not disentangled by this study.
    - No backbone-pretraining overlap audit was performed. Novelty relative to
      this project's development corpus does not establish novelty relative to
      pretrained backbone corpora. There is no arbitrary real-world guarantee.
    - Historical data, including the old test split, were reused for development.
      Nested OOF evaluates the frozen selection procedure; DINO's chosen best
      development mean is optimistic and must not be presented as outer CV.
    - DINO is exploratory; its extra corruption evaluation is post hoc. Fresh
      paired intervals include zero, and no superiority claim or primary switch
      follows from the point ranking. Fixed-prediction intervals omit refitting
      and model-selection uncertainty and depend on the source independence assumption.
    - This notebook checks committed evidence and cross-record identities. It does
      not re-audit raw images, rerun training/inference/bootstrap, or remeasure speed.
      Integrity hashes are not a substitute for independent experimental replication.
    """)
    md("""
    ## Runnable reproduction (saved evidence only)

    From the repository root, use Python 3.13 and the existing V3 environment.
    Optional notebook packages are isolated in `requirements-notebook.txt`; the ML
    lock is unchanged. On Windows replace `.venv/bin/python` with
    `.venv/Scripts/python.exe`:

    ```sh
    .venv/bin/python -m pip install -c requirements-v3-lock.txt -r requirements-notebook.txt
    .venv/bin/python scripts/v3/build_results_notebook.py --execute
    ```

    The builder uses **nbclient** and a temporary task-local kernelspec pointing
    at that interpreter; it installs no global kernel. Without `--execute` it
    builds an unexecuted notebook. To keep this committed snapshot intact, choose
    a separate `--output` path. The notebook can also be run top-to-bottom in an
    existing Python kernel from the root or `notebooks/` directory. No raw dataset,
    model checkpoint, GPU, or network is needed after dependencies are installed.

    The embedded inventory pins the saved snapshot. Rebuilding intentionally
    refreshes those pins from `--root`; executing the committed notebook directly
    verifies the existing pins. Review changed evidence before accepting new pins.
    Full training/scoring command records remain in `neural_execution_summary.json`
    and `final_execution_summary.json`; they are not executed here.

    To independently regenerate metrics and plots from the **saved** primary
    predictions (optional, not run by this notebook), use a new output directory:

    ```sh
    .venv/bin/python scripts/v3/evaluate.py \\
      --predictions reports/results/v3/fresh/frozen_selected/predictions.csv \\
      --metadata reports/results/v3/final_model_lock.json \\
      --output-dir artifacts/v3/notebook_recheck_primary \\
      --estimate-type fresh_holdout --label-format indices \\
      --n-bootstrap 10000 --seed 62642 --n-bins 15
    ```

    This evaluator reads prediction CSVs, not images or checkpoints. The saved
    evaluator seed is 62642 for this endpoint. Do not install the historical
    `.[dev]` extras into the validated V3 environment: their older scikit-learn and
    ipykernel pins differ from the V3 lock and these optional notebook pins.
    The multiline example uses shell backslash continuations; in PowerShell,
    join those lines into one command and use `.venv/Scripts/python.exe`.
    """)
    code(r"""
    assert len(evaluations) == 4 and len(resnet_histories) == 6 and len(dino_histories) == 6
    print("PASS: pinned evidence hashes; lock/choice chronology; source-group counts;")
    print("four saved prediction identities and metrics; paired intervals; actual histories;")
    print(
        "corruption consistency; benchmark/checkpoint identity. No training or scoring performed."
    )
    print("Evidence files verified:", len(EXPECTED_SHA256))
    """)
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform_version()},
        "v3_evidence": {
            "file_count": len(inventory),
            "hash_policy": "SHA256; CRLF-to-LF for text only",
        },
    }
    for i, cell in enumerate(notebook.cells):
        cell.id = f"v3-results-{i:02d}"
    return notebook


def platform_version():
    import platform

    return platform.python_version()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("notebooks/05_v3_results.ipynb"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    notebook = build(root)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{i}", "exec")
    if args.execute:
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        from nbclient import NotebookClient

        # Scoped manager searches only this temporary directory; no global changes.
        with tempfile.TemporaryDirectory(prefix="v3-notebook-kernel-") as temp:
            kernel_dir = Path(temp) / "v3-results"
            kernel_dir.mkdir()
            (kernel_dir / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "V3 evidence (temporary)",
                        "language": "python",
                        "env": {"MPLBACKEND": "module://matplotlib_inline.backend_inline"},
                    }
                ),
                encoding="utf-8",
            )
            manager = KernelManager(
                kernel_name="v3-results", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp])
            )
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=180,
                resources={"metadata": {"path": str(root)}},
                record_timing=True,
            )
            try:
                client.execute()
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
        assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
        assert not any(
            o.output_type == "error"
            for c in notebook.cells
            if c.cell_type == "code"
            for o in c.outputs
        )
        notebook.metadata["v3_evidence"]["execution"] = (
            "nbclient; all code cells passed; saved evidence only"
        )
    nbf.validate(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(
        json.dumps(
            {
                "output": args.output.name,
                "executed": args.execute,
                "cells": len(notebook.cells),
                "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
