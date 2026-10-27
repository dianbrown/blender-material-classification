"""CPU-only, training-partition-only scaling and nested source-group selection."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import CLASS_NAMES
from .features import (
    assemble_features,
    environment,
    file_hash,
    json_hash,
    load_feature_cache,
    read_manifest,
    write_json,
)


def labels_for(frame):
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if not frame["class"].isin(mapping).all():
        raise ValueError("Unknown class in training metadata")
    return frame["class"].map(mapping).to_numpy(dtype=np.int64)


def grouped_folds(y, groups, n_splits, seed):
    """Stratify independent groups, then expand to renders; no group is divided."""
    y, groups = np.asarray(y), np.asarray(groups, dtype=str)
    if len(y) != len(groups) or not len(y):
        raise ValueError("Labels and groups must be nonempty and row-aligned")
    table = pd.DataFrame({"group": groups, "y": y})
    if table.groupby("group")["y"].nunique().gt(1).any():
        raise ValueError("Each source_group must have a unique class")
    table = table.drop_duplicates("group").sort_values("group").reset_index(drop=True)
    if set(table["y"]) != set(range(len(CLASS_NAMES))):
        raise ValueError("Every training partition must contain all seven classes")
    counts = table.groupby("y").size()
    if counts.min() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} independent source groups per class; "
            f"found {counts.to_dict()}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    seen = np.zeros(len(y), dtype=np.int32)
    result = []
    for train_groups, test_groups in splitter.split(table["group"], table["y"]):
        train = np.flatnonzero(np.isin(groups, table["group"].iloc[train_groups]))
        test = np.flatnonzero(np.isin(groups, table["group"].iloc[test_groups]))
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Source group leakage")
        if set(y[train]) != set(range(len(CLASS_NAMES))) or set(y[test]) != set(
            range(len(CLASS_NAMES))
        ):
            raise ValueError("A training or validation fold is missing a class")
        seen[test] += 1
        result.append((train, test))
    if not np.all(seen == 1):
        raise AssertionError("Fold assignment must cover each row once")
    return result


def make_head(C, protocol):
    spec = protocol["head"]
    if spec["penalty"] != "l2" or spec["solver"] != "lbfgs":
        raise ValueError("This frozen protocol supports only L2 lbfgs logistic regression")
    # sklearn >=1.8 specifies L2 via l1_ratio=0; penalty='l2' is deprecated.
    return Pipeline(
        [
            ("scaler", StandardScaler(**protocol["scaler"])),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=spec["max_iter"],
                    tol=spec["tol"],
                    class_weight=spec["class_weight"],
                    random_state=protocol["seed"],
                ),
            ),
        ]
    )


def fit_head(X, y, C, protocol):
    head = make_head(C, protocol)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            head.fit(X, y)
        except ConvergenceWarning as exc:
            raise RuntimeError(
                "Logistic regression did not converge under the frozen solver settings; "
                "no result was accepted. Change the protocol and start a new run."
            ) from exc
    return head


def class_probabilities(head, X):
    classes = head.named_steps["classifier"].classes_
    if not np.array_equal(classes, np.arange(len(CLASS_NAMES))):
        raise ValueError("Classifier probability order does not match the frozen class order")
    probabilities = np.asarray(head.predict_proba(X), dtype=np.float64)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("Invalid classifier probabilities")
    # sklearn may preserve float32 internally. Normalize in float64 before OOF
    # storage so later log_loss validation does not mistake rounding for bad inputs.
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _best_with_tolerance(scores, protocol, *, across_candidates=False):
    best_score = max(item["inner_mean_macro_f1"] for item in scores)
    tied = [
        item
        for item in scores
        if best_score - item["inner_mean_macro_f1"] <= protocol["tie_tolerance"]
    ]
    return min(
        tied, key=lambda item: (item["candidate_index"] if across_candidates else 0, item["C"])
    )


def select_candidates(features_train, y_train, groups_train, protocol, *, seed, folds=None):
    """Only training rows enter this API. Each inner scaler sees only its fit split.

    Returns independently C-selected candidates as well as the primary winning
    representation. Ties within the frozen tolerance use declaration order then C.
    """
    if folds is None:
        folds = grouped_folds(y_train, groups_train, protocol["inner_folds"], seed)
    history, choices = [], []
    for candidate_index, candidate in enumerate(protocol["candidates"]):
        name = candidate["name"]
        X = features_train[name]
        if X.shape[0] != len(y_train):
            raise ValueError("Candidate features and training labels are not aligned")
        candidate_scores = []
        for C in sorted(protocol["C"]):
            scores = []
            for fold, (fit, valid) in enumerate(folds):
                model = fit_head(X[fit], y_train[fit], C, protocol)
                predictions = class_probabilities(model, X[valid]).argmax(axis=1)
                score = f1_score(
                    y_train[valid],
                    predictions,
                    average="macro",
                    labels=np.arange(len(CLASS_NAMES)),
                    zero_division=0,
                )
                scores.append(float(score))
                history.append(
                    {
                        "candidate": name,
                        "C": C,
                        "inner_fold": fold,
                        "macro_f1": float(score),
                        "n_fit": len(fit),
                        "n_valid": len(valid),
                    }
                )
            candidate_scores.append(
                {
                    "candidate": name,
                    "candidate_index": candidate_index,
                    "C": C,
                    "inner_mean_macro_f1": float(np.mean(scores)),
                }
            )
        choices.append(_best_with_tolerance(candidate_scores, protocol))
    winner = _best_with_tolerance(choices, protocol, across_candidates=True)
    return winner, choices, history, folds


def fold_record(folds, indices, groups, y):
    def counts(rows):
        return {
            name: {
                "images": int(np.sum(y[rows] == label)),
                "source_groups": len(set(groups[rows][y[rows] == label])),
            }
            for label, name in enumerate(CLASS_NAMES)
        }

    return [
        {
            "fold": fold,
            "train_rows": indices[train].tolist(),
            "validation_rows": indices[valid].tolist(),
            "train_groups": sorted(set(groups[train])),
            "validation_groups": sorted(set(groups[valid])),
            "train_counts": counts(train),
            "validation_counts": counts(valid),
        }
        for fold, (train, valid) in enumerate(folds)
    ]


def make_fold_plan(frame, protocol):
    """Validate every planned fit partition before any model is fitted or scored."""
    y = labels_for(frame)
    groups = frame["source_group"].to_numpy(dtype=str)
    outer = grouped_folds(y, groups, protocol["outer_folds"], protocol["seed"])
    inner = [
        grouped_folds(
            y[train], groups[train], protocol["inner_folds"], protocol["seed"] + index + 1
        )
        for index, (train, _) in enumerate(outer)
    ]
    final = grouped_folds(y, groups, protocol["inner_folds"], protocol["seed"] + 1000)
    records = {
        "outer": fold_record(outer, np.arange(len(y)), groups, y),
        "inner": [
            {
                "outer_fold": index,
                "folds": fold_record(inner[index], train, groups[train], y[train]),
            }
            for index, (train, _) in enumerate(outer)
        ],
        "final_all_development": fold_record(final, np.arange(len(y)), groups, y),
    }
    return {"outer": outer, "inner": inner, "final": final, "records": records}


def nested_oof(features, frame, protocol, *, cpu_threads=4, fold_plan=None):
    """Generate primary OOF and descriptive per-candidate OOF without post-hoc selection."""
    y = labels_for(frame)
    groups = frame["source_group"].to_numpy(dtype=str)
    if set(features) != {candidate["name"] for candidate in protocol["candidates"]}:
        raise ValueError("Features must exactly match predeclared candidates")
    for name, X in features.items():
        if X.ndim != 2 or len(X) != len(frame) or not np.isfinite(X).all():
            raise ValueError(f"Invalid or unaligned candidate features: {name}")
    plan = fold_plan if fold_plan is not None else make_fold_plan(frame, protocol)
    outer = plan["outer"]
    primary = np.full((len(y), len(CLASS_NAMES)), np.nan)
    candidate_oof = {name: np.full_like(primary, np.nan) for name in features}
    assignments = np.full(len(y), -1, dtype=np.int64)
    history, choices = [], []
    threads = max(1, min(int(cpu_threads), os.cpu_count() or 1, 8))
    with threadpool_limits(limits=threads):
        for outer_fold, (train, test) in enumerate(outer):
            # This boundary is intentional: outer labels are never passed to selection.
            train_features = {name: X[train] for name, X in features.items()}
            winner, candidates, search, inner = select_candidates(
                train_features,
                y[train],
                groups[train],
                protocol,
                seed=protocol["seed"] + outer_fold + 1,
                folds=plan["inner"][outer_fold],
            )
            history.extend({**row, "stage": "outer", "outer_fold": outer_fold} for row in search)
            choices.append(
                {"outer_fold": outer_fold, "selected": winner, "per_candidate": candidates}
            )
            for choice in candidates:
                name = choice["candidate"]
                model = fit_head(train_features[name], y[train], choice["C"], protocol)
                probabilities = class_probabilities(model, features[name][test])
                candidate_oof[name][test] = probabilities
                if name == winner["candidate"]:
                    primary[test] = probabilities
            assignments[test] = outer_fold
    if not np.isfinite(primary).all() or (assignments < 0).any():
        raise AssertionError("OOF predictions do not cover each row")
    return {
        "primary": primary,
        "candidates": candidate_oof,
        "outer_fold": assignments,
        "choices": choices,
        "search_history": history,
        "folds": plan["records"],
    }


def prediction_frame(frame, probabilities, outer_fold=None):
    result = frame[["image_path", "class", "material_id", "source_group"]].copy()
    if outer_fold is not None:
        result["outer_fold"] = outer_fold
    result["y_true"] = labels_for(frame)
    result["y_pred"] = probabilities.argmax(axis=1)
    for index, name in enumerate(CLASS_NAMES):
        result["p_" + name] = probabilities[:, index]
    return result


def probability_metrics(y, probabilities):
    pred = probabilities.argmax(axis=1)
    labels = np.arange(len(CLASS_NAMES))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=labels)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "class_order": list(CLASS_NAMES),
    }


def train_run(manifest, data_root, feature_cache, output, protocol, *, cpu_threads=4):
    """Freeze a unique output, compute nested OOF, then select/refit all development data."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    frame = read_manifest(manifest)
    base, cache = load_feature_cache(
        feature_cache, manifest=manifest, data_root=data_root, protocol=protocol
    )
    features = {
        candidate["name"]: assemble_features(base, candidate["parts"])
        for candidate in protocol["candidates"]
    }
    threads = min(cpu_threads, os.cpu_count() or 1, 8)
    identity = {
        "protocol_sha256": json_hash(protocol),
        "feature_cache_identity_sha256": cache["identity_sha256"],
        "feature_arrays": cache["arrays"],
        "environment": environment(),
        "learning_implementation_sha256": file_hash(__file__),
        "inference_implementation_sha256": file_hash(Path(__file__).with_name("inference.py")),
        "cpu_threads": threads,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    # Persist the complete selection protocol and immutable input identity BEFORE predictions.
    write_json(output / "protocol.json", protocol)
    record = {
        "run_identity": identity,
        "run_identity_sha256": json_hash(identity),
        "status": "running",
        "interpretation": protocol["interpretation"],
    }
    write_json(output / "run.json", record)
    try:
        plan = make_fold_plan(frame, protocol)
        write_json(output / "folds.json", plan["records"])
        record["folds_sha256"] = file_hash(output / "folds.json")
        write_json(output / "run.json", record)
        result = nested_oof(features, frame, protocol, cpu_threads=threads, fold_plan=plan)
        prediction_frame(frame, result["primary"], result["outer_fold"]).to_csv(
            output / "selected_oof.csv", index=False
        )
        control = protocol["control"]
        prediction_frame(frame, result["candidates"][control], result["outer_fold"]).to_csv(
            output / "control_oof.csv", index=False
        )
        ablations = output / "candidate_oof"
        ablations.mkdir()
        for name, probabilities in result["candidates"].items():
            prediction_frame(frame, probabilities, result["outer_fold"]).to_csv(
                ablations / (name + ".csv"), index=False
            )
        fold_frame = frame[list(("image_path", "class", "material_id", "source_group"))].copy()
        fold_frame["outer_fold"] = result["outer_fold"]
        fold_frame.to_csv(output / "fold_assignments.csv", index=False)
        write_json(output / "fold_choices.json", result["choices"])
        y = labels_for(frame)
        groups = frame["source_group"].to_numpy(dtype=str)
        metrics = {
            "primary_selection_procedure": probability_metrics(y, result["primary"]),
            "fixed_control": probability_metrics(y, result["candidates"][control]),
            "descriptive_candidates": {
                name: probability_metrics(y, proba) for name, proba in result["candidates"].items()
            },
            "interpretation": protocol["interpretation"],
        }
        write_json(output / "metrics.json", metrics)
        with threadpool_limits(limits=threads):
            winner, choices, search, final_folds = select_candidates(
                features, y, groups, protocol, seed=protocol["seed"] + 1000, folds=plan["final"]
            )
            final_head = fit_head(features[winner["candidate"]], y, winner["C"], protocol)
            control_choice = next(choice for choice in choices if choice["candidate"] == control)
            control_head = (
                final_head
                if winner["candidate"] == control
                else fit_head(features[control], y, control_choice["C"], protocol)
            )
        final_history = [
            {**row, "stage": "final_all_development", "outer_fold": -1} for row in search
        ]
        pd.DataFrame(result["search_history"] + final_history).to_csv(
            output / "search_history.csv", index=False
        )
        write_json(
            output / "final_folds.json", fold_record(final_folds, np.arange(len(y)), groups, y)
        )
        recipe = next(
            candidate
            for candidate in protocol["candidates"]
            if candidate["name"] == winner["candidate"]
        )
        metadata = {
            "schema_version": 1,
            "class_order": list(CLASS_NAMES),
            "protocol": protocol,
            "protocol_sha256": json_hash(protocol),
            "recipe": recipe,
            "final_selection": winner,
            "per_candidate": choices,
            "input_features": int(features[winner["candidate"]].shape[1]),
            "feature_implementation_sha256": cache["identity"]["implementation_sha256"],
            "feature_environment": cache["identity"]["environment"],
            "learning_environment": identity["environment"],
            "training_source": {
                **cache["identity"]["source"],
                "source_groups": sorted(set(groups)),
            },
            "run_identity_sha256": record["run_identity_sha256"],
            "interpretation": protocol["interpretation"]["final"],
        }
        artifact = output / "model.joblib"
        joblib.dump({"pipeline": final_head, "metadata": metadata}, artifact, compress=3)
        write_json(
            output / "model_metadata.json", {**metadata, "model_sha256": file_hash(artifact)}
        )
        control_recipe = next(
            candidate for candidate in protocol["candidates"] if candidate["name"] == control
        )
        control_metadata = {
            **metadata,
            "recipe": control_recipe,
            "final_selection": control_choice,
            "input_features": int(features[control].shape[1]),
            "interpretation": "Predeclared ConvNeXt control; C selected by grouped "
            "inner CV on all development data, then refitted on all development.",
        }
        control_artifact = output / "control_model.joblib"
        joblib.dump(
            {"pipeline": control_head, "metadata": control_metadata}, control_artifact, compress=3
        )
        write_json(
            output / "control_model_metadata.json",
            {**control_metadata, "model_sha256": file_hash(control_artifact)},
        )
        record.update(
            {
                "status": "complete",
                "seconds": time.perf_counter() - start,
                "folds_sha256": file_hash(output / "folds.json"),
                "final_selection": winner,
                "outputs": {
                    path.relative_to(output).as_posix(): file_hash(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file() and path.name != "run.json"
                },
            }
        )
        write_json(output / "run.json", record)
        return record
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.perf_counter() - start,
            }
        )
        write_json(output / "run.json", record)
        raise
