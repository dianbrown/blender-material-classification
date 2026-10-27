"""Assemble explicit measured evidence for build_report.py; never train or discover results.

CLI: python scripts/v3/assemble_report.py --spec assembly.json --output-dir NEW_DIR
See README.md for the version-1 assembly specification. Paths in the spec are
absolute or relative to the spec file, never relative to the process directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as report  # noqa: E402

ReportError = report.ReportError
require = report.require


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    require(fields and rows, f"Empty CSV: {path}")
    require(len(set(fields)) == len(fields), f"Duplicate CSV columns: {path}")
    require(all(None not in r and None not in r.values() for r in rows), "Ragged CSV")
    return fields, rows


def write_csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8"))


def normalize_evaluation(raw, classes):
    """Translate evaluate.py JSON without estimating or substituting any metrics."""
    require(raw.get("classes") == classes, "Evaluation class order differs from protocol")
    level = raw["image_level"]
    bootstrap = level["bootstrap"]
    require(bootstrap.get("status") == "ok", "Completed source bootstrap required")
    require(bootstrap.get("resampling_unit") == "source_group", "Wrong bootstrap unit")
    metrics = {}
    for name, value in level["metrics"].items():
        if name in ("n_images", "n_source_groups"):
            continue
        metrics[name] = {"value": value}
        if value is None:
            metrics[name]["reason"] = (
                "Undefined in supplied evaluator JSON; see metric conventions."
            )
        interval = bootstrap["intervals"].get(name)
        if interval and interval.get("lower") is not None:
            require(value is not None, "Interval for undefined metric")
            require(
                math.isclose(value, interval["estimate"], abs_tol=1e-12),
                f"Bootstrap point estimate disagrees: {name}",
            )
            metrics[name]["ci"] = {
                "low": interval["lower"],
                "high": interval["upper"],
                "level": bootstrap["confidence_level"],
                "unit": "source_group",
                "method": bootstrap["method"] + "; " + bootstrap["estimand"],
                "n_resamples": bootstrap["n_performed"],
                "seed": bootstrap["seed"],
                "n_valid": interval["n_valid"],
            }
    for source, target in (
        ("macro_auroc", "auroc_macro_ovr"),
        ("macro_average_precision", "average_precision_macro"),
    ):
        if source in level.get("ranking", {}):
            value = level["ranking"][source]
            metrics[target] = {"value": value}
            if value is None:
                metrics[target]["reason"] = "Undefined in evaluator ranking output."
    summary = {
        "class_order": classes,
        "n_images": level["metrics"]["n_images"],
        "n_source_groups": level["metrics"]["n_source_groups"],
        "metrics": metrics,
        "confusion_matrix": level["confusion_matrix"],
        "per_class": level["per_class"],
        "calibration_bins": [
            {"confidence": b["mean_confidence"], "accuracy": b["accuracy"], "count": b["count"]}
            for b in level.get("reliability", [])
            if b["count"] > 0
        ],
        "selective_risk": level.get("selective_risk_coverage", []),
        "metric_conventions": raw.get("metric_conventions", {}),
        "bootstrap_caveat": bootstrap.get("caveat", "Not supplied"),
    }
    report.validate_summary(summary, classes)
    return summary


def normalize_predictions(source, destination, classes, label_format):
    require(
        label_format in ("indices", "names"), "Explicit label_format: indices or names required"
    )
    fields, rows = read_csv(source)
    for row in rows:
        for key in ("y_true", "y_pred"):
            if label_format == "names":
                require(row[key] in classes, f"Unknown {key}: {row[key]}")
                row[key] = str(classes.index(row[key]))
            else:
                require(row[key] in [str(i) for i in range(len(classes))], "Invalid label index")
    write_csv(destination, fields, rows)
    return rows


def population(rows):
    return {
        r["image_path"]: tuple(
            r[k] for k in ("class", "material_id", "source_group", "outer_fold", "y_true")
        )
        for r in rows
    }


def validate_folds(raw, rows, classes):
    """Validate displayed fold summaries, not just the pooled confusion matrix."""
    folds = raw.get("per_fold", [])
    if not folds:
        return
    require(
        {str(f["outer_fold"]) for f in folds} == {r["outer_fold"] for r in rows}
        and len(folds) == len({r["outer_fold"] for r in rows}),
        "Fold summary coverage mismatch",
    )
    for fold in folds:
        selected = [r for r in rows if r["outer_fold"] == str(fold["outer_fold"])]
        cm = [[0] * len(classes) for _ in classes]
        for row in selected:
            cm[int(row["y_true"])][int(row["y_pred"])] += 1
        require(cm == fold["confusion_matrix"], "Fold confusion disagrees with predictions")
        accuracy = sum(cm[i][i] for i in range(len(classes))) / len(selected)
        f1 = sum(
            2 * cm[i][i] / (sum(cm[i]) + sum(r[i] for r in cm))
            if sum(cm[i]) + sum(r[i] for r in cm)
            else 0
            for i in range(len(classes))
        ) / len(classes)
        for key, value in (("accuracy", accuracy), ("macro_f1", f1)):
            require(
                math.isclose(fold["metrics"][key], value, abs_tol=1e-6),
                f"Fold {key} disagrees with predictions",
            )


class Bundle:
    TEXT_SUFFIXES = {
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".txt",
        ".md",
        ".yaml",
        ".yml",
        ".toml",
        ".lock",
        ".log",
        ".py",
        ".ps1",
    }

    def __init__(self, root, base, synthetic):
        self.root, self.base, self.synthetic = root, base, synthetic
        self.evidence, self.supplements, self.sources = [], [], []
        (root / "evidence").mkdir()
        (root / "derived").mkdir()

    def register(self, path, role):
        relative = path.relative_to(self.root).as_posix()
        self.evidence.append({"path": relative, "sha256": report.sha256(path), "role": role})
        return relative

    def copy(self, value, role):
        report.nonempty(value, role)
        source = (self.base / value).resolve()
        require(source.is_file(), f"Missing explicit {role}: {source}")
        digest = report.sha256(source)
        # Prefix makes identically named files from different experiments unambiguous.
        dest = self.root / "evidence" / f"{len(self.sources):03d}_{source.name}"
        shutil.copyfile(source, dest)
        require(report.sha256(dest) == digest, f"Source changed during copy: {source}")
        is_text = source.suffix.lower() in self.TEXT_SUFFIXES
        if is_text:
            # Git enforces LF for published evidence. Raw training hashes continue
            # to describe source bytes, not this portable representation.
            value = dest.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            dest.write_bytes(value.encode("utf-8"))
        relative = self.register(dest, role)
        self.sources.append(
            {
                "source": str(source),
                "path": relative,
                "source_sha256": digest,
                "sha256": report.sha256(dest),
                "normalization": "UTF-8 without BOM; LF" if is_text else "none",
            }
        )
        return relative

    def original_hash(self, relative):
        return next(item["source_sha256"] for item in self.sources if item["path"] == relative)

    def json(self, name, value, role):
        path = self.root / "derived" / f"{name}.json"
        write_json(path, value)
        return self.register(path, role)

    def supplement(self, path, section, kind, caption):
        self.supplements.append(dict(path=path, section=section, kind=kind, caption=caption))

    def table(self, name, fields, rows, section, caption):
        require(bool(rows), f"No rows for {name}")
        path = self.root / "derived" / f"{name}.csv"
        write_csv(path, fields, rows)
        relative = self.register(path, caption)
        self.supplement(relative, section, "table", caption)

    def plot(
        self, name, series, xlabel, ylabel, caption, *, logx=False, section="comparison_ablations"
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
        try:
            for label, xs, ys in series:
                require(xs and len(xs) == len(ys), "Empty or mismatched plot series")
                require(all(math.isfinite(float(v)) for v in xs + ys), "Nonfinite plot data")
                ax.plot(xs, ys, marker="o", markersize=3, label=label)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            if xlabel.startswith("Fold index"):
                ax.set_xticks(series[0][1])
            if logx:
                ax.set_xscale("log")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
            if self.synthetic:
                fig.suptitle("SYNTHETIC FIXTURE - NOT EXPERIMENTAL EVIDENCE", fontsize=10)
            path = self.root / "derived" / f"{name}.png"
            fig.savefig(path, dpi=180)
        finally:
            plt.close(fig)
        relative = self.register(path, caption)
        self.supplement(relative, section, "figure", caption)


def search_curves(bundle, path, protocol, *, section="comparison_ablations"):
    _, rows = read_csv(bundle.root / path)
    groups = defaultdict(list)
    for row in rows:
        require(row["stage"] in ("outer", "final_all_development"), "Unknown search stage")
        require(
            row["candidate"] in {c["name"] for c in protocol["candidates"]},
            "Search candidate not in protocol",
        )
        c, score = float(row["C"]), float(row["macro_f1"])
        require(c in protocol["C"], "Search C not in protocol")
        report.number(score, "search macro_f1", 0, 1)
        groups[(row["stage"], row["outer_fold"], row["candidate"], c)].append(row)
    means = defaultdict(list)
    for (stage, fold, candidate, c), values in groups.items():
        require(
            len(values) == protocol["inner_folds"]
            and len({v["inner_fold"] for v in values}) == protocol["inner_folds"],
            "Search curve requires every distinct inner fold",
        )
        means[(stage, fold, candidate)].append(
            (c, sum(float(v["macro_f1"]) for v in values) / len(values))
        )
    panels = defaultdict(list)
    for (stage, fold, candidate), values in means.items():
        require({c for c, _ in values} == set(protocol["C"]), "Incomplete C curve")
        values.sort()
        panels[(stage, fold)].append((candidate, [c for c, _ in values], [s for _, s in values]))
    for index, ((stage, fold), series) in enumerate(sorted(panels.items())):
        require(len(series) == len(protocol["candidates"]), "Incomplete candidate search panel")
        bundle.plot(
            f"search_{index}",
            series,
            "Inverse regularization C (log scale)",
            "Mean inner-fold macro F1 (fraction)",
            f"Actual C search: stage={stage}, outer_fold={fold}. Arithmetic mean over "
            f"{protocol['inner_folds']} recorded inner folds; selection evidence, not "
            f"an independent test or epoch learning curve. Source: {path}.",
            logx=True,
            section=section,
        )


def training_curves(bundle, entries):
    for index, entry in enumerate(entries):
        path = bundle.copy(entry["path"], "recorded training history")
        if Path(path).suffix.lower() == ".json":
            rows = field_value(report.read_json(bundle.root / path), entry["rows_key"])
            require(isinstance(rows, list) and rows, "JSON history must contain recorded rows")
        else:
            _, rows = read_csv(bundle.root / path)
        report.nonempty(entry["caption"], "Training history caption")
        # Column mappings make this compatible with ResNet and future fine-tune logs,
        # without guessing that a training or outer-test score is a validation score.
        for j, plot in enumerate(entry["plots"]):
            xs = [float(field_value(r, entry["x"])) for r in rows]
            require(len(xs) == len(set(xs)) and xs == sorted(xs), "History x must increase")
            series = [
                (column, xs, [float(field_value(r, column)) for r in rows]) for column in plot["ys"]
            ]
            bundle.plot(
                f"training_{index}_{j}",
                series,
                entry["x"],
                plot["ylabel"],
                entry["caption"] + f" Source: {path}. No interpolation or missing epochs.",
            )


def field_value(record, key):
    """Explicit dotted JSON field selection; literal CSV names take precedence."""
    if key in record:
        return record[key]
    for part in key.split("."):
        record = record[part]
    return record


def json_tables(bundle, entries):
    for index, entry in enumerate(entries):
        path = bundle.copy(entry["path"], entry["caption"])
        rows = field_value(report.read_json(bundle.root / path), entry["rows_key"])
        require(isinstance(rows, list) and rows, "JSON table rows must be a nonempty array")
        columns = entry["columns"]
        require(
            isinstance(columns, dict) and 1 <= len(columns) <= 7, "1-7 JSON table columns required"
        )
        selected = [{name: field_value(row, key) for name, key in columns.items()} for row in rows]
        require(
            all(
                value is None or type(value) in (str, int, float, bool)
                for row in selected
                for value in row.values()
            ),
            "JSON table cells must be scalar",
        )
        bundle.table(
            f"json_table_{index}",
            list(columns),
            selected,
            entry["section"],
            entry["caption"] + f" Source: {path}#/{entry['rows_key']}.",
        )


def narrative(protocol, audit, refs, model, primary, supplements):
    """Evidence-specific prose; optional editorial overrides remain explicit spec inputs."""
    models = "; ".join(m["name"] for m in protocol["models"].values())
    return {
        "motivation": "This study asks whether visual representations can distinguish seven "
        "material labels on previously unseen source families in a controlled Blender-render "
        "corpus. The engineering target is 80% image accuracy. Repeated views test camera and "
        "lighting variation but do not create new independent materials. This report separates "
        "the selection procedure, deployment fitting, and any independently collected endpoint.",
        "taxonomy_provenance": "Labels describe the archived rendering taxonomy, not verified "
        "physical composition. Ceramic includes mosaic/terrazzo-like appearances and concrete "
        "includes plaster/painted-wall appearances; a restricted fresh taxonomy changes the "
        "sampling domain. Provider maps are third-party inputs; project authorship concerns "
        "renders and experimental code. Source verification, family links, map hashes and "
        "release permissions must be read with their scope in the supplied audit/evidence. "
        f"Source: {refs['audit']}. A provenance match is not proof of independent physical origin.",
        "data_audit": f"The supplied audit records {audit['image_count']} images, "
        f"{audit['material_count']} local material IDs and {audit['source_group_count']} "
        f"source-family groups. Dataset count tables are computed from the supplied nested "
        f"prediction population and checked against this audit ({refs['audit']}). Grouping "
        "conservatively joins known dependencies. Historical directory splits are descriptive; "
        "they are not the v3 evaluation partition. Audit limitations remain part of the evidence.",
        "protocol": f"The frozen protocol specifies {protocol['outer_folds']} outer and "
        f"{protocol['inner_folds']} inner source-group folds, seed {protocol['seed']}, selection "
        f"metric {protocol['selection_metric']}, and C={protocol['C']}. Tie rule: "
        f"{protocol['tie_break']}. Scalers and heads are fitted within training partitions. "
        "Outer predictions estimate the selection procedure on the existing corpus, including "
        "the previously consulted historical test. They are not an untouched new test. "
        "Source-group bootstrap intervals resample fixed predictions, not model refits; they "
        "do not include selection or correlated-training-set uncertainty. Efron (1979), "
        "Bootstrap Methods: Another Look at the Jackknife, https://doi.org/10.1214/aos/1176344552, "
        "provides the resampling foundation, not a guarantee for this design. "
        f"Source: {refs['protocol']}.",
        "representations_heads": f"Recorded backbones: {models}. The frozen feature grid "
        "compares CLS, CLS with patch mean, patch mean/std at multiple resolutions, pooled "
        "convolutional descriptors and concatenation. Register tokens are excluded from patch "
        "statistics. These are local design hypotheses, not established improvements from "
        "the cited DINOv2 or ConvNeXt papers. Exact preprocessing, revisions and weight hashes "
        f"are in {refs['protocol']}; solver settings: {json.dumps(protocol['head'])}. "
        f"Frozen all-data recipe and training identity are in {refs['model_sidecar']}. "
        "Any later explicitly chosen deployment artifact is identified in deployment_fit. "
        "Frozen extraction and logistic fitting have no neural epoch loss curve.",
        "reproducibility": f"Environment: {refs['environment']}. Run: {refs['run']}. "
        "The inventory records original and staged hashes; text copies use UTF-8/LF and "
        "binary copies retain original bytes, alongside normalized numeric summaries. "
        "SHA-256 detects changes; it does not establish experimental independence. Supplied "
        "compute records distinguish run scope and units. Missing latency, energy or peak "
        "memory measurements remain unmeasured. The model reference records the exact verified "
        "weight file without deserializing it. Validation-rerun equivalence must be checked "
        "against its own predictions; initial scores are not substituted for a later run.",
        "comparison_ablations": "Matched comparisons require identical image identities, labels, "
        "source groups and outer-fold assignments. Candidate OOF variants are descriptive "
        "ablations, not a second model-selection opportunity. Recorded C-search curves show "
        "inner selection scores. A three-fold fine-tuning search conducted after initial nested "
        "results is exploratory development work; its score must not be called a new unbiased "
        "nested estimate. A fixed ResNet baseline only counts as the third assessed architecture "
        "once matching-fold predictions exist. Missing experiments receive no estimated scores.",
        "errors": "Confusion and per-class tables describe supplied predictions. Correlated "
        "views can concentrate errors within a small number of sources; inspect source-level "
        "tables and captioned diagnostics before attributing a confusion to composition. "
        "Diagnostic galleries, if supplied, must state the selection policy and image/source "
        "identities. Confidence-ranked examples are illustrative, not representative prevalence. "
        f"Supplied error supplements: {sum(s['section'] == 'errors' for s in supplements)}.",
        "robustness_compute": "Compute values below are copied from explicitly supplied records "
        "with their original field names and scope. Timing one extraction or head-fitting stage "
        "does not measure end-to-end deployment latency. No robustness result is inferred from "
        "clean accuracy. Corruption and pilot results, when supplied, require separate captions "
        "and paired-population evidence. An earlier EEVEE pilot is development evidence and "
        "does not replace the independent fresh endpoint.",
        "limitations_ethics": "This is synthetic-domain evidence with a finite source-family "
        "population. Unknown provider derivations, pretrained-data overlap, shader conventions "
        "and scene shortcuts limit inference. New renders under Blender 5.1.1 and a restricted "
        "ceramic/concrete taxonomy introduce domain differences. No real-photo, physical "
        "composition or safety-critical validation is claimed. Source-map licensing and the "
        "creator's dataset release license are distinct; public publication requires the "
        "release decision and complete scoped provenance evidence.",
        "conclusion": f"The primary endpoint is explicitly {primary}; its estimate and "
        "interval are printed from the validated evaluator JSON. A point estimate above 80% "
        "would not imply that the confidence lower bound exceeds 80%. The all-data deployment "
        "fit creates an inference artifact, not an independent performance estimate. Any fresh "
        "endpoint must follow a locked choice made before its predictions. Remaining "
        "experiments and document review must finish before describing a final submission.",
        "reproduction": "Use the archived assembly_spec.json to recover explicit source inputs "
        "and the source inventory to verify their SHA-256 values. Run python "
        "scripts/v3/assemble_report.py --spec PATH --output-dir NEW_EVIDENCE_DIR, then python "
        "scripts/v3/build_report.py --results-dir NEW_EVIDENCE_DIR --output-dir NEW_REPORT_DIR. "
        "Neither command trains, chooses a model, downloads data, or overwrites the historical "
        "report. Upstream audit, feature extraction, nested training and evaluation commands "
        "are documented in README.md; supplied run-specific commands take precedence.",
    }


def _assemble(spec, base, root, spec_path):
    require(
        type(spec.get("schema_version")) is int and spec["schema_version"] == 1,
        "Assembly schema_version must be 1",
    )
    kind = spec.get("evidence_kind", "measured")
    require(kind in ("measured", "synthetic_fixture"), "Invalid evidence_kind")
    require(type(spec.get("appendix_details", False)) is bool, "appendix_details must be boolean")
    bundle = Bundle(root, base, kind == "synthetic_fixture")
    bundle.copy(str(spec_path), "Assembly specification as supplied (UTF-8/LF staging)")
    refs = {
        key: bundle.copy(spec[key], key)
        for key in (
            "protocol",
            "environment",
            "audit",
            "run",
            "model_sidecar",
            "search_history",
            "feature_record",
        )
    }
    data = {key: report.read_json(root / refs[key]) for key in refs if key != "search_history"}
    protocol, audit, run, model = (data[k] for k in ("protocol", "audit", "run", "model_sidecar"))
    classes = protocol["class_order"]
    require(model["class_order"] == classes, "Model class order mismatch")
    require(run.get("status") == "complete", "Training run must be complete")
    require(
        json_hash(run["run_identity"]) == run["run_identity_sha256"], "Run identity hash mismatch"
    )
    require(
        model["run_identity_sha256"] == run["run_identity_sha256"], "Model/run identity mismatch"
    )
    require(model["protocol_sha256"] == json_hash(protocol), "Model/protocol hash mismatch")
    require(run["run_identity"]["protocol_sha256"] == json_hash(protocol), "Run/protocol mismatch")
    feature = data["feature_record"]
    require(feature.get("complete") is True, "Feature record must be complete")
    require(
        feature["identity_sha256"] == json_hash(feature["identity"]),
        "Feature identity hash mismatch",
    )
    require(
        feature["identity_sha256"] == run["run_identity"]["feature_cache_identity_sha256"],
        "Run/feature identity mismatch",
    )
    require(
        feature["identity"]["implementation_sha256"] == model["feature_implementation_sha256"],
        "Model/feature implementation mismatch",
    )
    require(
        bundle.original_hash(refs["search_history"]) == run["outputs"]["search_history.csv"],
        "Search history differs from frozen run output",
    )
    weights = (base / spec["weights"]).resolve()
    require(weights.is_file(), f"Missing weights reference: {weights}")
    digest = report.sha256(weights)
    require(digest == model["model_sha256"], "Model weights hash mismatch")
    weight_ref = bundle.json(
        "weights_reference",
        {
            "source": str(weights),
            "sha256": digest,
            "size_bytes": weights.stat().st_size,
            "sidecar": refs["model_sidecar"],
            "copied": bool(spec.get("copy_weights", False)),
        },
        "Verified deployment weights reference (not serialized weights)",
    )
    if spec.get("copy_weights", False):
        weight_ref = bundle.copy(spec["weights"], "deployment weights")
    weight_sources = [(weights, digest)]
    deployment_description = model["interpretation"]
    # The parent may explicitly lock a later model. It does not replace the
    # frozen nested run or its feature provenance and is never selected here.
    if spec.get("deployment_model"):
        deployment = spec["deployment_model"]
        sidecar = bundle.copy(deployment["sidecar"], "explicitly selected deployment sidecar")
        metadata = report.read_json(root / sidecar)
        require(metadata["class_order"] == classes, "Deployment class order mismatch")
        weights = (base / deployment["weights"]).resolve()
        require(weights.is_file(), "Missing deployment weights")
        digest = report.sha256(weights)
        require(digest == metadata["model_sha256"], "Deployment weights hash mismatch")
        weight_sources.append((weights, digest))
        weight_ref = bundle.json(
            "selected_weights_reference",
            {
                "source": str(weights),
                "sha256": digest,
                "size_bytes": weights.stat().st_size,
                "sidecar": sidecar,
                "copied": bool(spec.get("copy_weights", False)),
            },
            "Explicit deployment model reference; no model selection performed",
        )
        if spec.get("copy_weights", False):
            weight_ref = bundle.copy(deployment["weights"], "explicit deployment weights")
        deployment_description = report.nonempty(
            deployment["description"], "deployment description"
        )
    endpoints, evaluations, comparisons = {}, {}, []
    for entry in spec["evaluations"]:
        identifier = entry["id"]
        require(re.fullmatch(r"[a-z][a-z0-9_]*", identifier) is not None, "Unsafe evaluation id")
        require(identifier not in evaluations, "Duplicate evaluation id")
        scope = entry["scope"]
        require(
            scope in ("nested_oof", "fresh_holdout", "exploratory_development"), "Invalid scope"
        )
        report.nonempty(entry["label"], "Evaluation label")
        report.nonempty(entry["description"], "Evaluation description")
        raw_path = bundle.copy(entry["evaluation"], f"{identifier}: evaluator JSON ({scope})")
        predictions = bundle.copy(entry["predictions"], f"{identifier}: raw predictions")
        raw = report.read_json(root / raw_path)
        expected = "nested_oof" if scope == "exploratory_development" else scope
        require(raw.get("estimate_kind") == expected, "Evaluation endpoint provenance mismatch")
        summary = normalize_evaluation(raw, classes)
        normalized = root / "derived" / f"{identifier}_predictions.csv"
        rows = normalize_predictions(root / predictions, normalized, classes, entry["label_format"])
        # Exploratory development predictions can have three folds; never promote to endpoint.
        report.validate_predictions(
            normalized,
            summary,
            classes,
            "nested_oof" if scope == "nested_oof" else "historical_test",
        )
        validate_folds(raw, rows, classes)
        normalized_path = bundle.register(
            normalized, f"{identifier}: normalized prediction indices"
        )
        summary_path = bundle.json(
            f"{identifier}_summary", summary, f"{identifier}: normalized metrics"
        )
        evaluations[identifier] = (entry, raw, summary, rows)
        if entry.get("endpoint"):
            endpoint = entry["endpoint"]
            require(
                endpoint == scope and endpoint in ("nested_oof", "fresh_holdout"),
                "Only nested/fresh evaluations can be report endpoints",
            )
            require(endpoint not in endpoints, "Duplicate endpoint")
            endpoints[endpoint] = dict(
                id=endpoint, status="measured", summary=summary_path, predictions=normalized_path
            )
            if endpoint == "nested_oof":
                require(
                    bundle.original_hash(predictions) == run["outputs"]["selected_oof.csv"],
                    "Nested endpoint predictions differ from run output",
                )
            if endpoint == "fresh_holdout":
                lock_path = bundle.copy(entry["selection_record"], "pre-prediction selection lock")
                lock = report.read_json(root / lock_path)
                require(lock["model_sha256"] == digest, "Fresh selection lock model mismatch")
                frozen = datetime.fromisoformat(lock["selection_frozen_at"])
                started = datetime.fromisoformat(entry["prediction_started_at"])
                require(
                    frozen.tzinfo is not None and started.tzinfo is not None and frozen < started,
                    "Selection must be frozen before fresh predictions (timezone required)",
                )
                endpoints[endpoint].update(
                    selection_frozen_at=lock["selection_frozen_at"],
                    independence_audit=bundle.copy(
                        entry["independence_audit"], "fresh independence audit"
                    ),
                )
        comparisons.append(
            {
                "model": entry["label"],
                "scope": scope,
                "images": summary["n_images"],
                "sources": summary["n_source_groups"],
                "accuracy": f"{summary['metrics']['accuracy']['value']:.6f}",
                "macro_f1": f"{summary['metrics']['macro_f1']['value']:.6f}",
                "evidence": summary_path,
            }
        )
        bundle.table(
            f"{identifier}_interpretation",
            ["label", "scope", "interpretation"],
            [dict(label=entry["label"], scope=scope, interpretation=entry["description"])],
            "comparison_ablations",
            f"Experiment status and interpretation. Source: {raw_path}.",
        )
        if raw.get("per_fold"):
            folds = raw["per_fold"]
            bundle.plot(
                f"{identifier}_folds",
                [
                    (metric, list(range(len(folds))), [f["metrics"][metric] for f in folds])
                    for metric in ("accuracy", "macro_f1")
                ],
                "Fold index (recorded order)",
                "Score (fraction)",
                f"{entry['label']}: {scope}; fold IDs {[f['outer_fold'] for f in folds]}. "
                f"Fold variation is not independent replication. Source: {raw_path}#/per_fold.",
            )
    require("nested_oof" in endpoints, "Nested evidence required even with fresh endpoint")
    nested = next(v for v in evaluations.values() if v[0].get("endpoint") == "nested_oof")
    for entry, raw, summary, rows in evaluations.values():
        if entry["scope"] == "nested_oof":
            require(population(rows) == population(nested[3]), "Matched population/folds mismatch")
        elif entry["scope"] == "fresh_holdout":
            fresh = next(
                (v for v in evaluations.values() if v[0].get("endpoint") == "fresh_holdout"), None
            )
            require(fresh is not None, "Fresh supplements require a locked primary fresh endpoint")
            require(
                population(rows) == population(fresh[3]), "Fresh comparison population mismatch"
            )
    rows = nested[3]
    require(
        len(rows) == audit["image_count"]
        and len({r["source_group"] for r in rows}) == audit["source_group_count"]
        and len({r["material_id"] for r in rows}) == audit["material_count"],
        "Audit population mismatch",
    )
    counts = []
    for label in classes:
        selected = [r for r in rows if r["class"] == label]
        require(len(selected) == audit["classes"][label], "Audit class counts mismatch")
        counts.append(
            dict(
                class_name=label,
                images=len(selected),
                materials=len({r["material_id"] for r in selected}),
                source_groups=len({r["source_group"] for r in selected}),
            )
        )
    bundle.table(
        "dataset_counts",
        ["class_name", "images", "materials", "source_groups"],
        counts,
        "data_audit",
        f"Existing-corpus counts from validated nested predictions; audit {refs['audit']}.",
    )
    # Separate tables keep exploratory and fresh results out of nested comparisons.
    for scope in ("nested_oof", "fresh_holdout", "exploratory_development"):
        selected = [r for r in comparisons if r["scope"] == scope]
        if selected:
            bundle.table(
                f"comparison_{scope}",
                list(selected[0]),
                selected,
                "comparison_ablations",
                f"{scope}: image-weighted accuracy and macro F1 as fractions. "
                "Rounded to six decimals; full precision in summary JSON. "
                "Declaration order, not ranking. Sources in evidence column.",
            )
    search_curves(
        bundle,
        refs["search_history"],
        protocol,
        section="reproduction" if spec.get("appendix_details") else "comparison_ablations",
    )
    training_curves(bundle, spec.get("training_histories", []))
    json_tables(bundle, spec.get("json_tables", []))
    for entry in spec.get("evidence", []):
        bundle.copy(entry["path"], entry["role"])
    for entry in spec.get("supplements", []):
        path = bundle.copy(entry["path"], entry["caption"])
        bundle.supplement(path, entry["section"], entry["kind"], entry["caption"])
    compute = []
    for key in ("environment", "run"):
        for field in ("seconds", "gpu", "vram_bytes", "python", "platform", "cuda"):
            if field in data[key]:
                compute.append(
                    dict(record=key, field=field, value=str(data[key][field]), source=refs[key])
                )
    compute.append(
        dict(
            record="deployment",
            field="weights_size_bytes",
            value=weights.stat().st_size,
            source=weight_ref,
        )
    )
    for job, record in feature.get("jobs", {}).items():
        for field, value in record.get("metrics", {}).items():
            compute.append(
                dict(
                    record=f"extraction/{job}",
                    field=field,
                    value=value,
                    source=refs["feature_record"],
                )
            )
    bundle.table(
        "compute",
        ["record", "field", "value", "source"],
        compute,
        "reproduction" if spec.get("appendix_details") else "robustness_compute",
        "Recorded compute and artifact size; bytes/seconds where named. "
        "Run time does not include separate feature extraction unless its record says so.",
    )
    unavailable = spec["unavailable_endpoints"]
    for endpoint in report.ENDPOINTS:
        if endpoint not in endpoints:
            reason = report.nonempty(unavailable.get(endpoint), f"Unavailable {endpoint} reason")
            endpoints[endpoint] = dict(id=endpoint, status="not_available", reason=reason)
    primary = (
        "fresh_holdout"
        if "fresh_holdout" in [k for k, v in endpoints.items() if v["status"] == "measured"]
        else "nested_oof"
    )
    sections = narrative(protocol, audit, refs, model, primary, bundle.supplements)
    overrides = spec.get("sections", {})
    require(set(overrides) <= set(report.SECTIONS), "Unknown narrative section")
    sections.update(overrides)
    bundle.json(
        "assembly_spec",
        spec,
        "Exact parsed assembly specification; paths relative to original spec",
    )
    bundle.json("source_inventory", bundle.sources, "Original paths and copied evidence hashes")
    manifest = {
        "schema_version": 1,
        "evidence_kind": kind,
        "title": spec["title"],
        "run_id": spec["run_id"],
        "code_commit": spec["code_commit"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "class_order": classes,
        "authors": spec.get("authors", []),
        "primary_endpoint": primary,
        "sections": sections,
        "endpoints": list(endpoints.values()),
        "deployment_fit": {
            "status": "fitted",
            "artifact": weight_ref,
            "description": deployment_description + " Weights reference: " + weight_ref,
        },
        "evidence_files": bundle.evidence,
        "supplements": bundle.supplements,
    }
    if "status_label" in spec:
        manifest["status_label"] = spec["status_label"]
    write_json(root / "report_manifest.json", manifest)
    report.load_report(root, allow_synthetic=kind == "synthetic_fixture")
    # Reject concurrent producer changes instead of combining files from different instants.
    for source in bundle.sources:
        require(
            report.sha256(Path(source["source"])) == source["source_sha256"],
            f"Input changed during assembly: {source['source']}",
        )
    for weights, digest in weight_sources:
        require(report.sha256(weights) == digest, "Weights changed during assembly")


def assemble_report(spec_path, output_dir, *, allow_synthetic=False):
    """Publish a validated evidence bundle atomically; failure leaves destination absent."""
    spec_path, output = Path(spec_path).resolve(), Path(output_dir).resolve()
    require(not output.exists(), f"Output directory already exists: {output}")
    spec = report.read_json(spec_path)
    require(
        allow_synthetic or spec.get("evidence_kind", "measured") != "synthetic_fixture",
        "Synthetic fixtures require --allow-synthetic",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".assembly-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        try:
            _assemble(spec, spec_path.parent, stage, spec_path)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ReportError(f"Cannot assemble evidence: {exc}") from exc
        require(not output.exists(), f"Output appeared during assembly: {output}")
        stage.rename(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = assemble_report(args.spec, args.output_dir, allow_synthetic=args.allow_synthetic)
    except (ReportError, OSError) as exc:
        parser.error(str(exc))
    print(output / "report_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
