"""Create reproducible descriptive report assets from explicit saved predictions.

Selection is deterministic, not a statistical representative sample: the overview
chooses one image per source first; galleries show lowest-confidence successes and
highest-confidence failures within each family. PCA is descriptive, not evaluation.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from material_classifier.v3 import features as ft  # noqa: E402
from material_classifier.v3.evaluation import validate_predictions  # noqa: E402
from material_classifier.v3.robustness import relative_path  # noqa: E402


def _font(size):
    # Pillow's bundled font is portable; record Pillow in the provenance runtime.
    return ImageFont.load_default(size=size)


def _sheet(rows, data_root, output, title, subtitle, per_page=12):
    paths = []
    pages = max(1, math.ceil(len(rows) / per_page))
    for page in range(pages):
        selected = rows.iloc[page * per_page : (page + 1) * per_page]
        columns, tile_width = 3, 360
        path_lines = max((len(textwrap.wrap(p, width=40)) for p in selected.image_path), default=1)
        tile_height = max(350, 260 + (path_lines + 2) * 20)
        canvas = Image.new(
            "RGB",
            (
                columns * tile_width + 48,
                120 + max(1, math.ceil(len(selected) / columns)) * tile_height,
            ),
            "#F5F7FA",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), title, font=_font(26), fill="#172A3A")
        draw.text((24, 58), subtitle, font=_font(15), fill="#475569")
        draw.text(
            (24, 84),
            f"Page {page + 1}/{pages} | Selected images: {len(rows)}",
            font=_font(14),
            fill="#475569",
        )
        if selected.empty:
            draw.text((24, 150), "No images meet this criterion.", font=_font(20), fill="#475569")
        for index, row in enumerate(selected.itertuples(index=False)):
            x, y = 24 + index % columns * tile_width, 120 + index // columns * tile_height
            with Image.open(ft.image_file(data_root, row.image_path)) as source:
                thumb = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (332, 224))
                canvas.paste(thumb, (x + (332 - thumb.width) // 2, y))
            caption = f"True: {row.y_true} | Pred: {row.y_pred}\nConfidence: {row.confidence:.3f}"
            caption += "\n" + "\n".join(textwrap.wrap(row.image_path, width=40))
            draw.multiline_text((x, y + 232), caption, font=_font(14), fill="#172A3A", spacing=4)
        path = output.with_name(output.stem + f"_{page + 1:02d}.png")
        canvas.save(path)
        paths.append(path)
    return paths


def _pca(frame, classes, data_root, output, feature_cache, feature_manifest, parts):
    import matplotlib
    from sklearn.decomposition import PCA

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = ft.read_manifest(feature_manifest)
    paths = manifest.image_path.map(relative_path)
    if paths.duplicated().any() or set(paths) != set(frame.image_path):
        raise ValueError("PCA feature manifest must match prediction image identities exactly")
    indexed = frame.set_index("image_path")
    for column in ("class", "material_id", "source_group"):
        if manifest[column].tolist() != indexed.loc[paths, column].tolist():
            raise ValueError(f"PCA feature manifest differs in {column}")
    arrays, record = ft.load_feature_cache(
        feature_cache, manifest=feature_manifest, data_root=data_root
    )
    matrix = ft.assemble_features(arrays, parts)
    if min(matrix.shape) < 2:
        raise ValueError("Descriptive PCA needs at least two images and two features")
    # Labels are deliberately absent from the PCA fitting call.
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(matrix)
    if not np.isfinite(pca.explained_variance_ratio_).all():
        raise ValueError("Descriptive PCA is undefined for zero-variance features")
    table = manifest.copy()
    table["PC1"], table["PC2"] = coordinates[:, 0], coordinates[:, 1]
    table.to_csv(output / "pca_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for label in classes:
        selected = manifest["class"].eq(label).to_numpy()
        ax.scatter(*coordinates[selected].T, label=label, alpha=0.7, s=22)
    ax.set(
        title="Frozen features: descriptive PCA (not evaluation)",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Plot labels only", fontsize=9)
    fig.savefig(output / "pca_descriptive.png", dpi=180)
    plt.close(fig)
    return {
        "interpretation": "Descriptive projection only; not evaluation or class separability "
        "evidence. Labels used only for plot colors, never PCA fitting.",
        "feature_cache_identity_sha256": record["identity_sha256"],
        "parts": parts,
        "feature_manifest_sha256": ft.file_hash(feature_manifest),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def create_diagnostics(
    predictions,
    data_root,
    classes,
    output_dir,
    *,
    per_class=3,
    feature_cache=None,
    feature_manifest=None,
    parts=None,
):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    if any((feature_cache, feature_manifest, parts)) and not all(
        (feature_cache, feature_manifest, parts)
    ):
        raise ValueError("PCA requires feature-cache, feature-manifest and explicit parts together")
    frame = validate_predictions(predictions, classes)
    frame["image_path"] = frame.image_path.map(relative_path)
    resolved = [ft.image_file(data_root, path) for path in frame.image_path]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Prediction image paths alias the same source file")
    frame["image_sha256"] = [ft.file_hash(path) for path in resolved]
    frame["confidence"] = frame[[f"p_{c}" for c in classes]].max(axis=1)
    frame["correct"] = frame.y_true.eq(frame.y_pred)
    selections, summaries = [], []
    for label in classes:
        family = frame[frame["class"] == label].sort_values(["source_group", "image_path"])
        representatives = (
            pd.concat([family.drop_duplicates("source_group"), family])
            .drop_duplicates("image_path")
            .head(per_class)
        )
        correct = family[family.correct].sort_values(["confidence", "image_path"]).head(per_class)
        incorrect = (
            family[~family.correct]
            .sort_values(["confidence", "image_path"], ascending=[False, True])
            .head(per_class)
        )
        for kind, selected in (
            ("representative", representatives),
            ("correct", correct),
            ("incorrect", incorrect),
        ):
            selections.append(selected.assign(gallery=kind))
        errors = family[~family.correct]
        destinations = errors.y_pred.value_counts().sort_index()
        summaries.append(
            {
                "class": label,
                "n_images": len(family),
                "n_sources": family.source_group.nunique(),
                "n_errors": len(errors),
                "recall": float(family.correct.mean()) if len(family) else None,
                "mean_error_confidence": float(errors.confidence.mean()) if len(errors) else None,
                "error_destinations": "; ".join(f"{k}: {v}" for k, v in destinations.items()),
            }
        )
    chosen = pd.concat(selections, ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "predictions": str(Path(predictions).resolve()),
        "predictions_sha256": ft.file_hash(predictions),
        "data_root": str(Path(data_root).resolve()),
        "class_order": classes,
        "per_class": per_class,
        "selection_policy": __doc__,
        "implementation_sha256": ft.file_hash(__file__),
        "environment": ft.environment(),
        "source_images": frame[["image_path", "image_sha256"]].to_dict(orient="records"),
        "interpretation": "Descriptive diagnostics of supplied predictions; no new "
        "evaluation, training, or generalization claim.",
    }
    ft.write_json(output / "diagnostics.json", record)
    try:
        chosen.to_csv(output / "chosen_images.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "family_failures.csv", index=False)
        frame.to_csv(output / "source_predictions.csv", index=False)
        for kind, title, subtitle in (
            (
                "representative",
                "Material family overview",
                "Deterministic examples; diverse sources first",
            ),
            ("correct", "Correct predictions", "Lowest-confidence successes within each family"),
            (
                "incorrect",
                "Incorrect predictions",
                "Highest-confidence failures within each family",
            ),
        ):
            _sheet(
                chosen[chosen.gallery == kind], data_root, output / f"{kind}.png", title, subtitle
            )
        if feature_cache:
            record["pca"] = _pca(
                frame, classes, data_root, output, feature_cache, feature_manifest, parts
            )
        if ft.file_hash(predictions) != record["predictions_sha256"] or any(
            ft.file_hash(ft.image_file(data_root, row["image_path"])) != row["image_sha256"]
            for row in record["source_images"]
        ):
            raise ValueError("Diagnostic inputs changed during rendering")
        record.update(
            status="complete",
            files={
                path.name: ft.file_hash(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "diagnostics.json"
            },
        )
        ft.write_json(output / "diagnostics.json", record)
        return record
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        ft.write_json(output / "diagnostics.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--classes", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--parts", nargs="+")
    args = parser.parse_args()
    record = create_diagnostics(
        args.predictions,
        args.data_root,
        args.classes,
        args.output_dir,
        per_class=args.per_class,
        feature_cache=args.feature_cache,
        feature_manifest=args.feature_manifest,
        parts=args.parts,
    )
    print(f"Descriptive assets: {args.output_dir} ({record['status']})")


if __name__ == "__main__":
    main()
