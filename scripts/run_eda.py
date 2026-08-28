"""Generate deterministic dataset EDA figures from the canonical manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from verify_dataset import CLASSES, SPLITS, verify
except ImportError:  # Allows ``import scripts.run_eda`` from the repository root.
    from scripts.verify_dataset import CLASSES, SPLITS, verify


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports/figures/dataset"
RESULTS_DIR = ROOT / "reports/results"
COLORS = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(CLASSES)}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def _open_thumbnail(root: Path, path: str, size: int = 180) -> Image.Image:
    with Image.open(root / path) as image:
        return image.convert("RGB").resize((size, size))


def representative_material_entries(
    rows: list[dict[str, str]], class_name: str
) -> list[tuple[str, str]]:
    """Select render 1 from every base material in one class, deterministically."""

    materials: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["class"] == class_name:
            materials[row["material_id"]].append(row)
    entries = []
    for material_id in sorted(materials):
        item = sorted(materials[material_id], key=lambda row: int(row["render_index"]))[0]
        entries.append((material_id, item["image_path"]))
    return entries


def _image_grid(
    root: Path, entries: list[tuple[str, str]], path: Path, *, columns: int, title: str
) -> None:
    rows = max(1, (len(entries) + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.4, rows * 2.7), squeeze=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for index, (label, image_path) in enumerate(entries):
        ax = axes[index // columns][index % columns]
        ax.imshow(_open_thumbnail(root, image_path))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    for index in range(len(entries), rows * columns):
        axes[index // columns][index % columns].axis("off")
    _save(fig, path)


def _stats(
    root: Path, rows: list[dict[str, str]]
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Collect streaming-friendly full-resolution pixel sufficient statistics.

    Only sums, sums of squares, and a 256-bin luminance histogram are retained
    for each image; the full image array is released before the next image is
    read.  This gives pooled dataset/channel moments without retaining the
    corpus in memory.
    """

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        with Image.open(root / row["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
            luminance = np.asarray(image.convert("L"), dtype=np.uint8)
        channels = (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])
        pixel_count = int(rgb.shape[0] * rgb.shape[1])
        grouped[(row["class"], row["split"])].append(
            {
                "pixel_count": pixel_count,
                "r_sum": float(channels[0].sum()),
                "r_sq_sum": float(np.square(channels[0]).sum()),
                "g_sum": float(channels[1].sum()),
                "g_sq_sum": float(np.square(channels[1]).sum()),
                "b_sum": float(channels[2].sum()),
                "b_sq_sum": float(np.square(channels[2]).sum()),
                "luminance_sum": float(luminance.sum()),
                "luminance_sq_sum": float(np.square(luminance.astype(np.float64)).sum()),
                "luminance_hist": np.bincount(luminance.ravel(), minlength=256),
            }
        )
    return grouped


def _pooled_summary(items: list[dict[str, object]]) -> dict[str, float]:
    """Return pooled pixel moments, rather than an average of image moments."""

    count = sum(int(item["pixel_count"]) for item in items)
    if not count:
        return {
            key: 0.0
            for key in (
                "mean_r",
                "std_r",
                "mean_g",
                "std_g",
                "mean_b",
                "std_b",
                "luminance_mean",
                "luminance_std",
                "luminance_p05",
                "luminance_p50",
                "luminance_p95",
            )
        }

    result: dict[str, float] = {}
    for short, label in (("r", "r"), ("g", "g"), ("b", "b")):
        total = sum(float(item[f"{short}_sum"]) for item in items)
        square_total = sum(float(item[f"{short}_sq_sum"]) for item in items)
        mean = total / count
        result[f"mean_{label}"] = mean
        result[f"std_{label}"] = float(np.sqrt(max(0.0, square_total / count - mean**2)))

    luminance_total = sum(float(item["luminance_sum"]) for item in items)
    luminance_square_total = sum(float(item["luminance_sq_sum"]) for item in items)
    luminance_mean = luminance_total / count
    result["luminance_mean"] = luminance_mean
    result["luminance_std"] = float(
        np.sqrt(max(0.0, luminance_square_total / count - luminance_mean**2))
    )
    histogram = np.sum(
        np.stack([np.asarray(item["luminance_hist"], dtype=np.int64) for item in items]), axis=0
    )
    cumulative = np.cumsum(histogram)
    for quantile, suffix in ((0.05, "p05"), (0.50, "p50"), (0.95, "p95")):
        position = int(quantile * (count - 1)) + 1
        result[f"luminance_{suffix}"] = float(np.searchsorted(cumulative, position))
    return result


SUMMARY_STAT_KEYS = (
    "mean_r",
    "std_r",
    "mean_g",
    "std_g",
    "mean_b",
    "std_b",
    "luminance_mean",
    "luminance_std",
    "luminance_p05",
    "luminance_p50",
    "luminance_p95",
)


def write_summary(root: Path, rows: list[dict[str, str]], output: Path) -> None:
    grouped = _stats(root, rows)
    material_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        material_groups[(row["class"], row["split"])].add(row["material_id"])
    fields = [
        "scope",
        "class",
        "split",
        "n_images",
        "n_materials",
        "renders_per_material",
        "width",
        "height",
        "mean_r",
        "std_r",
        "mean_g",
        "std_g",
        "mean_b",
        "std_b",
        "luminance_mean",
        "luminance_std",
        "luminance_p05",
        "luminance_p50",
        "luminance_p95",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cls in CLASSES:
            for split in SPLITS:
                items = grouped[(cls, split)]
                values = _pooled_summary(items)
                writer.writerow(
                    {
                        "scope": "class_split",
                        "class": cls,
                        "split": split,
                        "n_images": len(items),
                        "n_materials": len(material_groups[(cls, split)]),
                        "renders_per_material": 30,
                        "width": 512,
                        "height": 512,
                        **{key: round(values[key], 4) for key in SUMMARY_STAT_KEYS},
                    }
                )
        all_items = [item for values in grouped.values() for item in values]
        values = _pooled_summary(all_items)
        writer.writerow(
            {
                "scope": "all",
                "class": "ALL",
                "split": "ALL",
                "n_images": len(rows),
                "n_materials": len({row["material_id"] for row in rows}),
                "renders_per_material": 30,
                "width": 512,
                "height": 512,
                **{key: round(values[key], 4) for key in SUMMARY_STAT_KEYS},
            }
        )


def generate_figures(root: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = Counter(row["class"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    class_split = Counter((row["class"], row["split"]) for row in rows)
    materials = defaultdict(list)
    for row in rows:
        materials[row["material_id"]].append(row)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(CLASSES, [class_counts[c] for c in CLASSES], color=[COLORS[c] for c in CLASSES])
    _style(ax, "Class distribution")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir / "class_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(SPLITS, [split_counts[s] for s in SPLITS], color=["#4472c4", "#70ad47", "#ed7d31"])
    _style(ax, "Split distribution")
    ax.set_ylabel("Images")
    _save(fig, output_dir / "split_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(CLASSES))
    for split, color in zip(SPLITS, ("#4472c4", "#70ad47", "#ed7d31")):
        values = np.array([class_split[(c, split)] for c in CLASSES])
        ax.bar(CLASSES, values, bottom=bottom, label=split, color=color)
        bottom += values
    _style(ax, "Class by split distribution")
    ax.set_ylabel("Images")
    ax.legend(frameon=False, ncol=3)
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir / "class_split_stacked.png")

    by_class = {
        c: sorted(
            (r for r in rows if r["class"] == c),
            key=lambda r: (r["material_id"], r["render_index"]),
        )[0]
        for c in CLASSES
    }
    _image_grid(
        root,
        [(c, by_class[c]["image_path"]) for c in CLASSES],
        output_dir / "representative_images_all_classes.png",
        columns=4,
        title="Representative renders by class",
    )
    entries = []
    for split in SPLITS:
        for cls in CLASSES:
            candidate = sorted(
                (r for r in rows if r["split"] == split and r["class"] == cls),
                key=lambda r: (r["material_id"], r["render_index"]),
            )[0]
            entries.append((f"{split} / {cls}", candidate["image_path"]))
    _image_grid(
        root,
        entries,
        output_dir / "representative_images_by_split.png",
        columns=7,
        title="Representative renders across locked splits",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = [len(materials[mid]) for mid in sorted(materials)]
    ax.hist(counts, bins=np.arange(0.5, max(counts) + 1.5), color="#5b9bd5", rwidth=0.8)
    _style(ax, "Renders per base material")
    ax.set_xlabel("Renders per material")
    ax.set_ylabel("Materials")
    _save(fig, output_dir / "renders_per_material.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts_by_class = [len({r["material_id"] for r in rows if r["class"] == c}) for c in CLASSES]
    ax.bar(CLASSES, counts_by_class, color=[COLORS[c] for c in CLASSES])
    _style(ax, "Base-material count per class")
    ax.set_ylabel("Base materials")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir / "base_material_count_per_class.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["width", "height"], [512, 512], color="#7f8c8d")
    _style(ax, "Image dimensions")
    ax.set_ylabel("Pixels")
    ax.set_ylim(0, 560)
    _save(fig, output_dir / "image_dimensions.png")

    stats = _stats(root, rows)
    all_stats = [item for values in stats.values() for item in values]
    pooled = _pooled_summary(all_stats)
    means = [pooled[key] for key in ("mean_r", "mean_g", "mean_b")]
    stds = [pooled[key] for key in ("std_r", "std_g", "std_b")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    width = 0.36
    ax.bar(x - width / 2, means, width, label="mean", color="#4472c4")
    ax.bar(x + width / 2, stds, width, label="pooled pixel std", color="#ed7d31")
    ax.set_xticks(x, ["R", "G", "B"])
    _style(ax, "RGB channel statistics")
    ax.set_ylabel("8-bit intensity")
    ax.legend(frameon=False)
    _save(fig, output_dir / "rgb_channel_mean_std.png")

    luminance_hist = np.sum(
        np.stack([np.asarray(item["luminance_hist"], dtype=np.int64) for item in all_stats]), axis=0
    )
    histogram = luminance_hist.reshape(32, 8).sum(axis=1)
    centers = np.arange(32) * 8 + 3.5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, histogram, width=8, color="#8064a2", alpha=0.9, align="center")
    _style(ax, "Luminance distribution (64 x 64 thumbnails)")
    ax.set_xlabel("Luminance")
    ax.set_ylabel("Pixels")
    _save(fig, output_dir / "luminance_distribution.png")

    diversity = []
    for cls in CLASSES:
        mids = sorted(mid for mid in materials if mid.startswith(f"{cls}_"))[:3]
        for mid in mids:
            first = sorted(materials[mid], key=lambda r: int(r["render_index"]))[0]
            diversity.append((f"{cls} / {mid}", first["image_path"]))
    _image_grid(
        root,
        diversity,
        output_dir / "material_diversity_contact_sheet.png",
        columns=3,
        title="Material diversity: three base materials per class",
    )

    for cls in CLASSES:
        entries = representative_material_entries(rows, cls)
        _image_grid(
            root,
            entries,
            output_dir / f"class_{cls}.png",
            columns=5,
            title=f"{cls.title()} material contact sheet",
        )


def run(project_root: str | Path = ROOT) -> Path:
    root = Path(project_root).resolve()
    validation = verify(root)
    if not validation.get("passed"):
        raise RuntimeError("dataset verification failed; EDA was not generated")
    rows = _read_rows(root / "metadata/render_manifest.csv")
    summary = root / "reports/results/dataset_summary.csv"
    write_summary(root, rows, summary)
    generate_figures(root, rows, root / "reports/figures/dataset")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run(args.project_root)
    print(f"summary: {summary}")
    print(f"figures: {args.project_root / 'reports/figures/dataset'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
