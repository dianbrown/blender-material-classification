"""Publication-sized static figures for grouped OOF development estimates.

``plot_evaluation(result, output_dir)`` consumes evaluate_predictions() output.
All figures are saved as 200 dpi PNG and vector PDF without requiring a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_evaluation(result: dict, output_dir: str | Path) -> dict[str, str]:
    """Save confusion, class, calibration, ROC/PR, risk and fold figures."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    classes = result["classes"]
    score = result["image_level"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
            "axes.labelcolor": "#273444",
            "text.color": "#273444",
        }
    ):

        def save(fig: plt.Figure, name: str, title: str) -> None:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
            fig.text(0.5, 0.012, result["estimate_type"], ha="center", fontsize=8, color="#526171")
            fig.tight_layout(rect=(0, 0.045, 1, 0.94))
            try:
                for extension in ("png", "pdf"):
                    path = directory / f"{name}.{extension}"
                    fig.savefig(path, dpi=200, bbox_inches="tight")
                    paths[f"{name}_{extension}"] = str(path)
            finally:
                plt.close(fig)

        def confusion(evaluation: dict, name: str, title: str) -> None:
            size = max(5, 0.75 * len(classes))
            fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1, size + 1))
            for ax, key, label in zip(
                axes,
                ("confusion_matrix", "confusion_normalized_true"),
                ("Observation counts", "Fraction within true class"),
            ):
                matrix = np.asarray(evaluation[key])
                vmax = max(1, float(matrix.max())) if key == "confusion_matrix" else 1
                im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
                ax.set(
                    xticks=range(len(classes)),
                    yticks=range(len(classes)),
                    xticklabels=classes,
                    yticklabels=classes,
                    xlabel="Predicted class",
                    ylabel="True class",
                    title=label,
                )
                plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        value = matrix[i, j]
                        text = str(int(value)) if key == "confusion_matrix" else f"{value:.2f}"
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="white" if value > 0.55 * vmax else "#19344d",
                        )
                fig.colorbar(im, ax=ax, shrink=0.75, pad=0.025)
            save(fig, name, title)

        confusion(score, "image_confusion", "Image-level classification")
        if result["group_aggregated"]["evaluation"] is not None:
            grouped = result["group_aggregated"]
            confusion(
                grouped["evaluation"],
                "group_confusion",
                f"Source-group classification · {grouped['n_eligible_groups']} eligible, "
                f"{grouped['n_excluded_groups']} excluded",
            )

        fig, ax = plt.subplots(figsize=(max(8, len(classes)), 5))
        positions = np.arange(len(classes))
        for offset, metric, color in (
            (-0.24, "precision", "#3b7ea1"),
            (0, "recall", "#d58b32"),
            (0.24, "f1", "#54987c"),
        ):
            ax.bar(
                positions + offset,
                [r[metric] for r in score["per_class"]],
                width=0.24,
                color=color,
                label=metric.capitalize(),
            )
        ax.set(
            xticks=positions,
            xticklabels=[f"{c}\n(n={r['support']})" for c, r in zip(classes, score["per_class"])],
            ylim=(0, 1.08),
            ylabel="Score",
        )
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(ncol=3, frameon=False, loc="upper center")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        save(fig, "per_class", "Image-level precision, recall and F1")

        reliability = score["reliability"]
        populated = [r for r in reliability if r["count"]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        axes[0].plot([0, 1], [0, 1], "--", color="#8b97a3", label="Perfect calibration")
        axes[0].plot(
            [r["mean_confidence"] for r in populated],
            [r["accuracy"] for r in populated],
            "o-",
            color="#3b7ea1",
            label=f"ECE = {score['metrics']['ece']:.3f}",
        )
        axes[0].set(
            xlim=(0, 1),
            ylim=(0, 1.03),
            xlabel="Mean confidence",
            ylabel="Observed accuracy",
            title="Reliability diagram",
        )
        axes[0].legend(frameon=False, loc="upper left")
        axes[1].bar(
            [(r["lower"] + r["upper"]) / 2 for r in reliability],
            [r["count"] for r in reliability],
            width=0.9 / result["n_bins"],
            color="#3b7ea1",
        )
        axes[1].set(
            xlim=(0, 1),
            xlabel="Top-label confidence",
            ylabel="Images",
            title="Confidence distribution",
        )
        save(fig, "calibration", "Image-level calibration")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
        colors = plt.get_cmap("tab10")
        for i, row in enumerate(score["ranking"]["per_class"]):
            if not row["defined"]:
                continue
            curve = score["ranking"]["curves"][row["class"]]
            axes[0].plot(
                curve["fpr"],
                curve["tpr"],
                color=colors(i % 10),
                label=f"{row['class']} ({row['auroc']:.3f})",
            )
            axes[1].step(
                curve["recall"],
                curve["precision"],
                where="post",
                color=colors(i % 10),
                label=f"{row['class']} ({row['average_precision']:.3f})",
            )
        axes[0].plot([0, 1], [0, 1], "--", color="#8b97a3", linewidth=1)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC · AUROC")
        axes[1].set(
            xlabel="Recall", ylabel="Precision", title="Precision–recall · Average precision"
        )
        for ax in axes:
            ax.set(xlim=(0, 1), ylim=(0, 1.03))
            if score["ranking"]["n_defined_classes"]:
                ax.legend(frameon=False, fontsize=8, loc="lower left")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Undefined: positive and negative\nobservations required",
                    ha="center",
                    transform=ax.transAxes,
                )
        save(fig, "roc_pr", "One-vs-rest ranking · Undefined classes omitted")

        selective = score["selective_risk_coverage"]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(
            [r["coverage"] for r in selective],
            [r["risk"] for r in selective],
            "o-",
            markersize=3,
            color="#3b7ea1",
        )
        ax.axhline(
            1 - score["metrics"]["accuracy"],
            color="#8b97a3",
            linestyle="--",
            label="All-image error rate",
        )
        ax.set(
            xlim=(0, 1.02),
            ylim=(0, 1.02),
            xlabel="Coverage · Fraction retained",
            ylabel="Selective risk · Error rate among retained images",
        )
        ax.legend(frameon=False)
        save(fig, "selective_risk", "Confidence-based selection · Ties retained together")

        folds = result["per_fold"]
        fig, ax = plt.subplots(figsize=(max(7, len(folds) * 0.8), 4.6))
        for metric, label, color in (
            ("accuracy", "Accuracy", "#3b7ea1"),
            ("macro_f1", "Macro F1", "#d58b32"),
        ):
            ax.plot(
                range(len(folds)),
                [r["metrics"][metric] for r in folds],
                "o-",
                label=label,
                color=color,
            )
        ax.set(
            xticks=range(len(folds)),
            xticklabels=[str(r["outer_fold"]) for r in folds],
            xlabel="Outer fold",
            ylabel="Score",
            ylim=(0, 1.05),
        )
        ax.legend(frameon=False)
        save(
            fig,
            "per_fold",
            "Outer-fold development performance"
            if result.get("estimate_kind", "nested_oof") == "nested_oof"
            else "Fresh holdout partition performance",
        )

        fig, ax = plt.subplots(figsize=(9, 4.8))
        selected_metrics = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        levels = [("Image level", score, "#3b7ea1", -0.12)]
        if result["group_aggregated"]["evaluation"] is not None:
            levels.append(
                ("Source-group level", result["group_aggregated"]["evaluation"], "#d58b32", 0.12)
            )
        any_intervals = False
        for label, level, color, offset in levels:
            for i, metric in enumerate(selected_metrics):
                interval = level["bootstrap"]["intervals"][metric]
                y = i + offset
                if interval["lower"] is not None:
                    # Percentile CIs need not contain the original point estimate.
                    ax.hlines(y, interval["lower"], interval["upper"], color=color, linewidth=2)
                    ax.vlines(
                        [interval["lower"], interval["upper"]],
                        y - 0.045,
                        y + 0.045,
                        color=color,
                        linewidth=1.5,
                    )
                    any_intervals = True
                ax.scatter(
                    interval["estimate"],
                    y,
                    color=color,
                    s=40,
                    label=label if i == 0 else None,
                    zorder=3,
                )
        ax.set(
            yticks=range(len(selected_metrics)),
            yticklabels=[name.replace("_", " ").capitalize() for name in selected_metrics],
            xlabel="Score · Points are estimates; bars are percentile 95% intervals",
            xlim=(-0.02, 1.02),
            ylim=(-0.5, len(selected_metrics) - 0.5),
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.legend(frameon=False, loc="lower left")
        if not any_intervals:
            ax.text(
                0.98,
                0.04,
                "Intervals disabled or fewer than two source groups",
                ha="right",
                transform=ax.transAxes,
                fontsize=8,
            )
        save(fig, "confidence_intervals", "Uncertainty · Resample whole source groups")
    return paths
