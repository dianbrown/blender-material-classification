"""Build an evidence-bound v3 Markdown/PDF report from report_manifest.json.

The normalized contract is documented in reports/V3_REPORT_METHODS.md. No result
discovery, training, model selection, or historical metric fallback occurs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

CLASSES = {"wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"}
ENDPOINTS = {
    "historical_test": "Historical test (previously consulted)",
    "nested_oof": "Nested existing-corpus OOF (selection procedure)",
    "fresh_holdout": "Fresh independent-source holdout",
}
SECTIONS = {
    "motivation": "Motivation and scope",
    "taxonomy_provenance": "Taxonomy and provenance",
    "data_audit": "Data audit and exclusions",
    "protocol": "Experimental protocol",
    "representations_heads": "Representations and classification heads",
    "reproducibility": "Reproducibility and compute environment",
    "comparison_ablations": "Model comparison and ablations",
    "errors": "Error analysis",
    "robustness_compute": "Robustness and compute",
    "limitations_ethics": "Limitations and ethics",
    "conclusion": "Conclusion",
    "reproduction": "Appendix: reproduction",
}
REFERENCES = [
    (
        "1",
        "Oquab et al. (2024). DINOv2: Learning Robust Visual Features without Supervision.",
        "https://arxiv.org/abs/2304.07193",
    ),
    (
        "2",
        "Darcet et al. (2024). Vision Transformers Need Registers.",
        "https://arxiv.org/abs/2309.16588",
    ),
    (
        "3",
        "Cimpoi et al. (2015). Deep filter banks for texture recognition, "
        "description, and segmentation.",
        "https://arxiv.org/abs/1507.02620",
    ),
    ("4", "Liu et al. (2022). A ConvNet for the 2020s.", "https://arxiv.org/abs/2201.03545"),
    (
        "5",
        "Cawley and Talbot (2010). On Over-fitting in Model Selection and "
        "Subsequent Selection Bias in Performance Evaluation. JMLR 11:2079-2107.",
        "https://jmlr.org/papers/v11/cawley10a.html",
    ),
    ("6", "Poly Haven. Asset license (CC0).", "https://polyhaven.com/license"),
    ("7", "ambientCG. License (CC0).", "https://docs.ambientcg.com/license/"),
]
RELATED = (
    "DINOv2 investigates transferable self-supervised visual representations [1]. "
    "Registers provide extra transformer tokens for internal computation [2]; they "
    "must not be treated as image patches. Deep Filter Banks motivates pooling local "
    "descriptors for texture recognition [3]. Patch mean/std pooling here is an "
    "experimental extension, not a reproduced Deep Filter Banks method. ConvNeXt "
    "provides a modern convolutional architecture [4]. Nested evaluation separates "
    "selection from performance estimation to address selection bias [5]. None of "
    "these sources establishes performance on this rendered corpus."
)


class ReportError(ValueError):
    """Incomplete, inconsistent, or unsafe report inputs."""


def require(condition, message):
    if not condition:
        raise ReportError(message)


def nonempty(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty text required")
    return value


def number(value, name, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value), f"{name}: finite number required")
    require(low is None or value >= low, f"{name}: below {low}")
    require(high is None or value <= high, f"{name}: above {high}")
    return value


def positive_int(value, name):
    require(type(value) is int and value > 0, f"{name}: positive integer required")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Cannot read JSON {path}: {exc}") from exc


def asset(root, relative):
    nonempty(relative, "artifact path")
    require(not Path(relative).is_absolute(), "Artifact paths must be relative to results-dir")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root), f"Artifact escapes results-dir: {relative}")
    require(resolved.is_file(), f"Required artifact missing: {relative}")
    return resolved


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_predictions(path, summary, classes, endpoint):
    """Verify the declared population and scalar metrics against supplied predictions."""
    matrix = [[0] * len(classes) for _ in classes]
    groups, materials, images, folds = {}, {}, set(), set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = {
            "image_path",
            "class",
            "material_id",
            "source_group",
            "outer_fold",
            "y_true",
            "y_pred",
        } | {f"p_{c}" for c in classes}
        require(columns <= set(reader.fieldnames or []), "Prediction CSV columns incomplete")
        for row in reader:
            try:
                truth, pred = int(row["y_true"]), int(row["y_pred"])
                require(
                    0 <= truth < len(classes) and 0 <= pred < len(classes),
                    "Prediction indices outside class_order",
                )
                require(row["class"] == classes[truth], "Prediction class/y_true mismatch")
                image = nonempty(row["image_path"], "image_path")
                require(image not in images, "Duplicate image in endpoint predictions")
                images.add(image)
                group = nonempty(row["source_group"], "source_group")
                material = nonempty(row["material_id"], "material_id")
                fold = row["outer_fold"]
                if endpoint == "nested_oof":
                    nonempty(fold, "outer_fold")
                    require(
                        group not in groups or groups[group] == fold,
                        "Source group crosses outer folds",
                    )
                require(
                    material not in materials or materials[material] == (group, truth),
                    "Material changes source group or class",
                )
                materials[material] = (group, truth)
                groups[group] = fold
                folds.add(fold)
                probabilities = [float(row[f"p_{c}"]) for c in classes]
                for p in probabilities:
                    number(p, "probability", 0, 1)
                require(
                    math.isclose(sum(probabilities), 1, abs_tol=1e-6),
                    "Prediction probabilities must sum to one",
                )
                require(
                    probabilities[pred] >= max(probabilities) - 1e-8,
                    "y_pred disagrees with probability argmax",
                )
                matrix[truth][pred] += 1
            except (TypeError, KeyError, ValueError) as exc:
                raise ReportError(
                    f"Invalid predictions at CSV line {reader.line_num}: {exc}"
                ) from exc
    require(len(images) == summary["n_images"], "n_images disagrees with predictions")
    require(len(groups) == summary["n_source_groups"], "n_source_groups disagrees with predictions")
    require(matrix == summary["confusion_matrix"], "Confusion matrix disagrees with predictions")
    if endpoint == "nested_oof":
        require(len(folds) == 5, "Nested OOF requires all five outer folds")
    return set(groups), images, set(materials)


def validate_summary(summary, classes):
    require(isinstance(summary, dict), "Summary must be an object")
    require(summary.get("class_order") == classes, "Summary class_order mismatch")
    for key in ("n_images", "n_source_groups"):
        positive_int(summary.get(key), key)
    require(summary["n_source_groups"] <= summary["n_images"], "Too many source groups")
    cm = summary.get("confusion_matrix")
    k = len(classes)
    require(isinstance(cm, list) and len(cm) == k, "Confusion matrix must be K by K")
    require(all(isinstance(row, list) and len(row) == k for row in cm), "Invalid confusion shape")
    require(
        all(type(x) is int and x >= 0 for row in cm for x in row),
        "Confusion counts must be nonnegative integers",
    )
    require(sum(map(sum, cm)) == summary["n_images"], "Confusion matrix count mismatch")
    per_class = summary.get("per_class")
    require(isinstance(per_class, list) and len(per_class) == k, "per_class required")
    for i, item in enumerate(per_class):
        require(
            isinstance(item, dict) and item.get("class") == classes[i], "per_class order mismatch"
        )
        support, predicted, correct = sum(cm[i]), sum(row[i] for row in cm), cm[i][i]
        require(item.get("support") == support, "Per-class support mismatch")
        require(support > 0, "Each evaluation class must have support")
        precision = correct / predicted if predicted else 0.0
        recall = correct / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for key, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
            number(item.get(key), key, 0, 1)
            require(math.isclose(item[key], expected, abs_tol=1e-6), f"Per-class {key} mismatch")
    metrics = summary.get("metrics")
    require(isinstance(metrics, dict), "metrics object required")
    expected = {
        "accuracy": sum(cm[i][i] for i in range(k)) / summary["n_images"],
        "balanced_accuracy": sum(p["recall"] for p in per_class) / k,
        "macro_f1": sum(p["f1"] for p in per_class) / k,
        "weighted_f1": sum(p["f1"] * p["support"] for p in per_class) / summary["n_images"],
    }
    require(expected.keys() <= metrics.keys(), "Required scalar metrics missing")
    for key, metric in metrics.items():
        require(isinstance(metric, dict), f"metrics.{key}: object required")
        if metric.get("value") is None:
            require(key not in expected, f"Required metric {key} cannot be undefined")
            nonempty(metric.get("reason"), f"metrics.{key}.reason")
            continue
        number(metric["value"], key)
        if key in expected:
            require(
                math.isclose(metric["value"], expected[key], abs_tol=1e-6),
                f"Metric {key} disagrees with confusion matrix",
            )
        if "ci" in metric:
            ci = metric["ci"]
            require(isinstance(ci, dict), f"{key}.ci must be an object")
            number(ci.get("low"), "ci.low")
            number(ci.get("high"), "ci.high")
            require(ci["low"] <= ci["high"], "CI bounds reversed")
            number(ci.get("level"), "ci.level", 0.01, 0.9999)
            require(ci.get("unit") == "source_group", "CI must resample source groups")
            nonempty(ci.get("method"), "ci.method")
            positive_int(ci.get("n_resamples"), "ci.n_resamples")
            require(type(ci.get("seed")) is int, "ci.seed required")
    require("ci" in metrics["accuracy"], "Source-group accuracy CI is required")
    for key in ("calibration_bins", "selective_risk"):
        require(isinstance(summary.get(key, []), list), f"{key} must be an array")
        require(
            all(isinstance(row, dict) for row in summary.get(key, [])),
            f"{key} entries must be objects",
        )
    for b in summary.get("calibration_bins", []):
        for key in ("confidence", "accuracy"):
            number(b.get(key), f"calibration {key}", 0, 1)
        positive_int(b.get("count"), "calibration bin count")
    if summary.get("calibration_bins"):
        require(
            sum(b["count"] for b in summary["calibration_bins"]) == summary["n_images"],
            "Calibration bin counts mismatch",
        )
    for row in summary.get("selective_risk", []):
        number(row.get("coverage"), "coverage", 0, 1)
        number(row.get("risk"), "risk", 0, 1)


def load_report(results_dir, allow_synthetic=False):
    root = Path(results_dir).resolve()
    manifest = read_json(root / "report_manifest.json")
    require(
        isinstance(manifest, dict)
        and type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 1,
        "report_manifest.json schema_version must be 1",
    )
    require(
        manifest.get("evidence_kind") in ("measured", "synthetic_fixture"),
        "evidence_kind must be measured or synthetic_fixture",
    )
    require(
        allow_synthetic or manifest["evidence_kind"] == "measured",
        "Synthetic fixtures require --allow-synthetic",
    )
    for key in ("title", "run_id", "created_utc", "code_commit"):
        nonempty(manifest.get(key), key)
    classes = manifest.get("class_order")
    require(
        isinstance(classes, list)
        and len(classes) == 7
        and all(isinstance(c, str) for c in classes)
        and set(classes) == CLASSES,
        "class_order must contain the seven unique project classes",
    )
    require(
        isinstance(manifest.get("authors", []), list)
        and all(isinstance(x, str) and x.strip() for x in manifest.get("authors", [])),
        "authors must be a list of supplied names",
    )
    sections = manifest.get("sections", {})
    require(isinstance(sections, dict), "sections object required")
    for key in SECTIONS:
        nonempty(sections.get(key), f"sections.{key}")
    evidence = manifest.get("evidence_files")
    require(isinstance(evidence, list) and evidence, "Hashed evidence_files required")
    hashes = {}
    for item in evidence:
        require(isinstance(item, dict), "Evidence entry must be an object")
        path = asset(root, item.get("path"))
        digest = sha256(path)
        require(item.get("sha256") == digest, f"Evidence hash mismatch: {item.get('path')}")
        nonempty(item.get("role"), "evidence role")
        hashes[item["path"]] = digest
    endpoints = manifest.get("endpoints")
    require(isinstance(endpoints, list) and endpoints, "endpoints required")
    seen, loaded = set(), []
    for endpoint in endpoints:
        require(isinstance(endpoint, dict), "Endpoint must be an object")
        key = endpoint.get("id")
        nonempty(key, "endpoint.id")
        require(key in ENDPOINTS and key not in seen, "Invalid or duplicate endpoint id")
        seen.add(key)
        status = endpoint.get("status")
        require(status in ("measured", "not_available"), "Invalid endpoint status")
        if status == "not_available":
            nonempty(endpoint.get("reason"), "Unavailable endpoint reason")
            require(
                "summary" not in endpoint and "predictions" not in endpoint,
                "Unavailable endpoint cannot carry results",
            )
        else:
            for key_path in ("summary", "predictions"):
                nonempty(endpoint.get(key_path), key_path)
                require(endpoint.get(key_path) in hashes, f"{key_path} must be hashed evidence")
            summary = read_json(asset(root, endpoint["summary"]))
            validate_summary(summary, classes)
            groups, images, materials = validate_predictions(
                asset(root, endpoint["predictions"]), summary, classes, key
            )
            loaded.append(
                {
                    **endpoint,
                    "data": summary,
                    "groups": groups,
                    "images": images,
                    "materials": materials,
                }
            )
    require(seen == set(ENDPOINTS), "Declare all three endpoint statuses explicitly")
    primary = manifest.get("primary_endpoint")
    require(primary in ("nested_oof", "fresh_holdout"), "Historical test cannot be primary")
    require(any(e["id"] == primary for e in loaded), "Primary endpoint must have measured results")
    nested = next((e for e in loaded if e["id"] == "nested_oof"), None)
    fresh = next((e for e in loaded if e["id"] == "fresh_holdout"), None)
    if fresh:
        require(nested is not None, "Fresh holdout requires existing-corpus OOF for overlap checks")
        require(not fresh["groups"] & nested["groups"], "Fresh holdout source-group overlap")
        require(not fresh["images"] & nested["images"], "Fresh holdout image overlap")
        require(not fresh["materials"] & nested["materials"], "Fresh holdout material overlap")
        nonempty(fresh.get("selection_frozen_at"), "fresh_holdout.selection_frozen_at")
        nonempty(fresh.get("independence_audit"), "fresh_holdout.independence_audit")
        require(fresh["independence_audit"] in hashes, "Independence audit must be hashed evidence")
    deployment = manifest.get("deployment_fit")
    require(
        isinstance(deployment, dict) and deployment.get("status") in ("not_available", "fitted"),
        "deployment_fit status required",
    )
    nonempty(deployment.get("description"), "deployment_fit.description")
    if deployment["status"] == "fitted":
        nonempty(deployment.get("artifact"), "Deployment artifact")
        require(deployment.get("artifact") in hashes, "Deployment artifact must be hashed evidence")
    supplements = manifest.get("supplements", [])
    require(isinstance(supplements, list), "supplements must be a list")
    for item in supplements:
        require(
            isinstance(item, dict) and item.get("section") in SECTIONS,
            "Supplement must name a narrative section",
        )
        require(item.get("path") in hashes, "Supplement must be hashed evidence")
        nonempty(item.get("caption"), "supplement.caption")
        require(item.get("kind") in ("figure", "table"), "Supplement kind must be figure or table")
        path = asset(root, item["path"])
        if item["kind"] == "figure":
            require(path.suffix.lower() in (".png", ".jpg", ".jpeg"), "Supplement must be PNG/JPEG")
        else:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            require(
                len(rows) > 1
                and 1 <= len(rows[0]) <= 7
                and all(len(row) == len(rows[0]) for row in rows),
                "Supplement table needs header, data and 1-7 consistent columns",
            )
    return manifest, loaded, hashes


def metric_text(metric):
    if metric.get("value") is None:
        return f"Not defined: {metric['reason']}"
    result = f"{metric['value']:.4f}"
    if "ci" in metric:
        ci = metric["ci"]
        result += f" ({ci['level']:.0%} CI {ci['low']:.4f}-{ci['high']:.4f})"
    return result


def metric_line(endpoint):
    data = endpoint["data"]
    accuracy = data["metrics"]["accuracy"]
    ci = accuracy["ci"]
    return (
        f"{ENDPOINTS[endpoint['id']]}: accuracy {accuracy['value']:.2%} "
        f"({ci['level']:.0%} source-group CI {ci['low']:.2%}-{ci['high']:.2%}); "
        f"macro-F1 {data['metrics']['macro_f1']['value']:.4f}; "
        f"{data['n_images']:,} images from {data['n_source_groups']:,} source groups. "
        f"Source: {endpoint['summary']}#/metrics. "
        "The 80% target is a point-accuracy target, not a lower confidence bound."
    )


def make_plots(endpoint, classes, directory, synthetic):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data, prefix = endpoint["data"], endpoint["id"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figures = []

    def save(fig, name, caption):
        if synthetic:
            fig.suptitle(
                "SYNTHETIC FIXTURE - NOT EXPERIMENTAL EVIDENCE", fontsize=10, color="#ad2727"
            )
        filename = f"{prefix}_{name}.png"
        fig.savefig(directory / filename, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append((filename, f"{caption} Source: {endpoint['summary']}."))

    cm = np.array(data["confusion_matrix"])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.4), layout="constrained")
    heatmap = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{normalized[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.55 else "#172b40",
                fontsize=9,
            )
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted class",
        ylabel="True class",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(heatmap, ax=ax, label="Fraction of true class", shrink=0.8)
    save(fig, "confusion", "Confusion matrix: counts and row-normalized percentages.")
    fig, ax = plt.subplots(figsize=(8.4, 4.7), layout="constrained")
    x = np.arange(len(classes))
    for offset, (key, color) in enumerate(
        (("precision", "#245b83"), ("recall", "#178679"), ("f1", "#c18026"))
    ):
        ax.bar(
            x + (offset - 1) * 0.24,
            [p[key] for p in data["per_class"]],
            width=0.24,
            label=key.capitalize(),
            color=color,
        )
    ax.set(xticks=x, xticklabels=classes, ylim=(0, 1.08), ylabel="Score")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    save(fig, "per_class", "Per-class precision, recall and F1; class support is in the table.")
    bins = data.get("calibration_bins", [])
    if bins:
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
        ax.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
        ax.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            "o-",
            color="#245b83",
            label="Observed bins",
        )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy")
        ax.legend(frameon=False)
        save(fig, "calibration", "Reliability diagram using evaluator-supplied nonempty bins.")
    risks = sorted(data.get("selective_risk", []), key=lambda row: row["coverage"])
    if risks:
        fig, ax = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
        ax.plot([r["coverage"] for r in risks], [r["risk"] for r in risks], color="#178679")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Coverage (fraction retained)",
            ylabel="Risk (error rate among retained)",
        )
        save(fig, "selective_risk", "Selective risk from supplied confidence-ranked predictions.")
    return figures


def report_blocks(manifest, endpoints, hashes, plots, supplements):
    """One block stream drives both formats so prose and metric strings cannot drift."""
    primary = next(e for e in endpoints if e["id"] == manifest["primary_endpoint"])
    synthetic = manifest["evidence_kind"] == "synthetic_fixture"
    blocks = [("title", manifest["title"])]
    if synthetic:
        blocks.append(
            (
                "p",
                "SYNTHETIC FIXTURE - NOT EXPERIMENTAL EVIDENCE. "
                "All scores and figures in this test document are fabricated test inputs.",
            )
        )
    if manifest.get("authors"):
        blocks.append(("p", "Authors: " + ", ".join(manifest["authors"])))
    blocks += [
        (
            "p",
            f"Run: {manifest['run_id']} | Created: {manifest['created_utc']} | "
            f"Code commit: {manifest['code_commit']}",
        ),
        ("h", "Abstract"),
        (
            "p",
            "This report evaluates seven-class material "
            "classification within synthetic Blender renders. " + metric_line(primary),
        ),
        ("h", "Evidence and endpoint status"),
    ]
    for e in manifest["endpoints"]:
        blocks.append(
            (
                "p",
                f"{ENDPOINTS[e['id']]}: {e['status']}. "
                + (
                    e["reason"]
                    if e["status"] == "not_available"
                    else f"Metrics source: {e['summary']}."
                ),
            )
        )
    blocks.append(
        (
            "p",
            "All-data deployment fit: "
            + manifest["deployment_fit"]["status"]
            + ". "
            + manifest["deployment_fit"]["description"]
            + " A deployment fit is not an independent performance estimate.",
        )
    )

    def section(key):
        blocks.extend([("h", SECTIONS[key]), ("p", manifest["sections"][key])])
        blocks.extend(supplements.get(key, []))

    for key in list(SECTIONS)[:6]:
        section(key)
        if key == "motivation":
            blocks += [("h", "Related work"), ("p", RELATED)]
    section("comparison_ablations")
    for e in endpoints:
        blocks += [("page", ""), ("h", ENDPOINTS[e["id"]]), ("p", metric_line(e))]
        data = e["data"]
        blocks.append(
            (
                "table",
                (
                    ["Metric (units defined in methods)", "Value / interval"],
                    [[k.replace("_", " "), metric_text(v)] for k, v in data["metrics"].items()],
                ),
            )
        )
        for key, metric in data["metrics"].items():
            if "ci" in metric:
                ci = metric["ci"]
                blocks.append(
                    (
                        "p",
                        f"{key} uncertainty: {ci['method']}; "
                        f"{ci['n_resamples']} source-group resamples; seed {ci['seed']}. "
                        "Intervals describe this sampled source population; they do not "
                        "establish real-world transfer or fully quantify training variability.",
                    )
                )
        blocks.append(
            (
                "table",
                (
                    ["Class", "Support", "Precision", "Recall", "F1"],
                    [
                        [
                            p["class"],
                            str(p["support"]),
                            f"{p['precision']:.3f}",
                            f"{p['recall']:.3f}",
                            f"{p['f1']:.3f}",
                        ]
                        for p in data["per_class"]
                    ],
                ),
            )
        )
        for figure in plots[e["id"]]:
            blocks.append(("figure", figure))
        for key, label in (
            ("calibration_bins", "Calibration diagram"),
            ("selective_risk", "Selective risk curve"),
        ):
            if not data.get(key):
                blocks.append(("p", f"{label}: not supplied; no plot generated."))
    for key in ("errors", "robustness_compute", "limitations_ethics", "conclusion"):
        section(key)
    blocks += [("h", "References")]
    for index, title, url in REFERENCES:
        blocks.append(("p", f"[{index}] {title} {url}"))
    section("reproduction")
    blocks += [
        ("p", "Report input: report_manifest.json (SHA-256 " + manifest["_manifest_sha256"] + ")."),
        ("h", "Appendix: evidence inventory"),
    ]
    for path, digest in hashes.items():
        blocks.append(("evidence", f"{path}\nSHA-256: {digest}"))
    return blocks


def write_markdown(blocks, path):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = []
    for kind, value in blocks:
        if kind == "title":
            lines.append(f"# {value}")
        elif kind == "h":
            lines.append(f"## {value}")
        elif kind in ("p", "evidence"):
            lines.append(value)
        elif kind == "table":
            header, rows = value
            lines.append(
                "\n".join(
                    [
                        "| " + " | ".join(map(cell, header)) + " |",
                        "| " + " | ".join(["---"] * len(header)) + " |",
                    ]
                    + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
                )
            )
        elif kind == "figure":
            filename, caption = value
            lines.append(f"![{caption}](figures/{filename})\n\n{caption}")
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def write_pdf(blocks, path, figures_dir, synthetic):
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        CondPageBreak,
        Image,
        KeepTogether,
        LongTable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        TableStyle,
    )

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, filename in (
        ("ReportSans", "DejaVuSans.ttf"),
        ("ReportSansBold", "DejaVuSans-Bold.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    styles = getSampleStyleSheet()
    for name in ("Normal", "Title", "Heading1"):
        styles[name].fontName = "ReportSans" if name == "Normal" else "ReportSansBold"
        styles[name].textColor = colors.HexColor("#172b40")
    styles["Normal"].fontSize, styles["Normal"].leading = 9.5, 14
    styles["Normal"].spaceAfter = 9
    styles["Normal"].splitLongWords = True
    styles["Title"].fontSize, styles["Title"].leading = 23, 29
    styles["Title"].alignment = TA_LEFT
    styles["Heading1"].fontSize, styles["Heading1"].leading = 14, 19
    styles["Heading1"].keepWithNext = True
    styles.add(ParagraphStyle("CaptionV3", parent=styles["Normal"], fontSize=8, leading=11))
    styles.add(
        ParagraphStyle("EvidenceV3", parent=styles["Normal"], fontSize=8, leading=11, spaceAfter=6)
    )
    styles.add(
        ParagraphStyle("CellV3", parent=styles["Normal"], fontSize=8, leading=11, spaceAfter=0)
    )
    width = A4[0] - 96
    story, figure_index = [], 0

    def paragraph(text, style="Normal"):
        # Input narrative is plain text, never interpreted as ReportLab markup.
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    for kind, value in blocks:
        if kind in ("title", "h", "p"):
            story.append(paragraph(value, {"title": "Title", "h": "Heading1", "p": "Normal"}[kind]))
        elif kind == "evidence":
            story.append(paragraph(value, "EvidenceV3"))
        elif kind == "page":
            story.append(CondPageBreak(150))
        elif kind == "table":
            header, rows = value
            widths = (
                [width * 0.43, width * 0.57]
                if len(header) == 2
                else [width / len(header)] * len(header)
            )
            table = LongTable(
                [[paragraph(c, "CellV3") for c in row] for row in [header] + rows],
                colWidths=widths,
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce7ef")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f3f6f8")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.extend([table, Spacer(1, 12)])
        elif kind == "figure":
            figure_index += 1
            filename, caption = value
            picture = Image(str(figures_dir / filename))
            scale = min(width / picture.imageWidth, 350 / picture.imageHeight)
            picture.drawWidth, picture.drawHeight = (
                picture.imageWidth * scale,
                picture.imageHeight * scale,
            )
            story.append(
                KeepTogether(
                    [
                        picture,
                        paragraph(f"Figure {figure_index}. {caption}", "CaptionV3"),
                        Spacer(1, 10),
                    ]
                )
            )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c4d0da"))
        canvas.line(48, 43, A4[0] - 48, 43)
        canvas.setFont("ReportSans", 8)
        canvas.setFillColor(colors.HexColor("#536577"))
        label = (
            "SYNTHETIC FIXTURE - NOT EXPERIMENTAL EVIDENCE"
            if synthetic
            else "Material classification | v3"
        )
        canvas.drawString(48, 29, label)
        canvas.drawRightString(A4[0] - 48, 29, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=46,
        bottomMargin=58,
        title=blocks[0][1],
        author="",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def build_report(results_dir, output_dir, *, allow_synthetic=False):
    """Validate first; publish a new output directory only after both formats succeed."""
    manifest, endpoints, hashes = load_report(results_dir, allow_synthetic)
    manifest["_manifest_sha256"] = sha256(Path(results_dir) / "report_manifest.json")
    output = Path(output_dir).resolve()
    require(not output.exists(), "Output directory already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    synthetic = manifest["evidence_kind"] == "synthetic_fixture"
    staging = Path(tempfile.mkdtemp(prefix=".v3-report-", dir=output.parent))
    try:
        figures = staging / "figures"
        figures.mkdir()
        plots = {
            e["id"]: make_plots(e, manifest["class_order"], figures, synthetic) for e in endpoints
        }
        supplements = {}
        for index, item in enumerate(manifest.get("supplements", [])):
            path = asset(Path(results_dir).resolve(), item["path"])
            entries = supplements.setdefault(item["section"], [])
            caption = f"{item['caption']} Source: {item['path']}."
            if synthetic:
                caption = "SYNTHETIC FIXTURE. " + caption
            if item["kind"] == "figure":
                name = f"supplement_{index}{path.suffix.lower()}"
                shutil.copyfile(path, figures / name)
                entries.append(("figure", (name, caption)))
            else:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.reader(stream))
                entries.extend([("p", caption), ("table", (rows[0], rows[1:]))])
        blocks = report_blocks(manifest, endpoints, hashes, plots, supplements)
        write_markdown(blocks, staging / "report.md")
        write_pdf(blocks, staging / "report.pdf", figures, synthetic)
        # Preserve exact manifest bytes, including its original formatting, for provenance.
        shutil.copyfile(
            Path(results_dir) / "report_manifest.json", staging / "report_manifest.json"
        )
        require(
            sha256(staging / "report_manifest.json") == manifest["_manifest_sha256"],
            "Manifest changed during report generation",
        )
        for relative, digest in hashes.items():
            require(
                sha256(asset(Path(results_dir).resolve(), relative)) == digest,
                f"Evidence changed during report generation: {relative}",
            )
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/v3"))
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="TEST ONLY: visibly label synthetic fixture outputs",
    )
    args = parser.parse_args(argv)
    try:
        output = build_report(
            args.results_dir, args.output_dir, allow_synthetic=args.allow_synthetic
        )
    except (ReportError, OSError) as exc:
        parser.exit(2, f"Report not generated: {exc}\n")
    print(f"Created {output / 'report.pdf'} and {output / 'report.md'}")


if __name__ == "__main__":
    main()
