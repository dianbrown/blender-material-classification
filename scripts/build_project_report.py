# ruff: noqa: E501
"""Build the image-rich ITRI626 project report from locked local artifacts.

The script is deliberately read-only with respect to the dataset, checkpoints and
TEST results.  It consumes cached CSV/JSON/PNG artifacts and writes only
``report.docx`` (or an explicitly supplied output path).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "reports" / "figures"
RESULT_DIR = ROOT / "reports" / "results"
CLASSES = ["wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather"]
MODEL_LABELS = {
    "custom_cnn": "CustomCNN",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b16": "ViT-B/16",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text: str, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.size = Pt(8.5)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[Any]],
    caption: str | None = None,
    widths: list[float] | None = None,
):
    if caption:
        p = doc.add_paragraph(style="Caption")
        p.add_run(caption)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, color="FFFFFF")
        set_cell_shading(table.rows[0].cells[i], "1F4E79")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], str(value))
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr)
    run._r.append(fld_char2)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.add_run("ITRI626 • Blender Material Classification  |  ")
    add_field(paragraph, "PAGE")


def set_alt_text(inline_shape, title: str, description: str) -> None:
    # python-docx exposes the drawing XML but not the accessibility properties.
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("title", title)
    doc_pr.set("descr", description)


def add_figure(
    doc: Document, rel_path: str, caption: str, width: float = 6.15, alt: str | None = None
) -> bool:
    path = ASSET_DIR / rel_path
    if not path.exists():
        p = doc.add_paragraph(f"[Figure unavailable: reports/figures/{rel_path}]")
        p.style = "Intense Quote"
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    shape = p.add_run().add_picture(str(path), width=Inches(width))
    set_alt_text(shape, caption, alt or caption)
    cp = doc.add_paragraph(style="Caption")
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.add_run(caption)
    return True


def add_figure_pair(
    doc: Document,
    left: tuple[str, str],
    right: tuple[str, str],
    alt_left: str = "",
    alt_right: str = "",
) -> int:
    """Place two related figures on one readable row to keep the report concise."""
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    # Hide table borders while retaining a stable two-column layout.
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tbl_pr.append(borders)
    count = 0
    for cell, (rel_path, caption), alt in zip(
        table.rows[0].cells, (left, right), (alt_left, alt_right)
    ):
        cell.text = ""
        path = ASSET_DIR / rel_path
        if not path.exists():
            cell.paragraphs[0].add_run(f"[Unavailable: {rel_path}]")
            continue
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        shape = p.add_run().add_picture(str(path), width=Inches(2.92))
        set_alt_text(shape, caption, alt or caption)
        cp = cell.add_paragraph(style="Caption")
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.add_run(caption)
        count += 1
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return count


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Number")


def fmt(x: Any, digits: int = 3) -> str:
    try:
        value = float(x)
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x: Any, digits: int = 1) -> str:
    try:
        return f"{100 * float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return str(x)


def build(output: Path) -> dict[str, Any]:
    comparison = load_csv(RESULT_DIR / "model_comparison.csv")
    env = load_json(RESULT_DIR / "environment.json")
    resnet_test = load_json(RESULT_DIR / "resnet50" / "test" / "metrics.json")
    resnet_val = load_json(RESULT_DIR / "resnet50" / "val" / "metrics.json")

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.7)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Aptos")
    styles["Normal"].font.size = Pt(10)
    styles["Normal"].paragraph_format.space_after = Pt(5)
    styles["Normal"].paragraph_format.line_spacing = 1.08
    for name, size, color in [
        ("Title", 29, "123B5D"),
        ("Heading 1", 17, "1F4E79"),
        ("Heading 2", 12.5, "2B6F8A"),
        ("Heading 3", 10.5, "2B6F8A"),
    ]:
        st = styles[name]
        st.font.name = "Aptos Display" if name != "Heading 3" else "Aptos"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), st.font.name)
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string(color)
    styles["Caption"].font.name = "Aptos"
    styles["Caption"].font.size = Pt(8.5)
    styles["Caption"].font.italic = True
    styles["Caption"].font.color.rgb = RGBColor.from_string("4D5C66")
    header = sec.header.paragraphs[0]
    header.text = "ITRI626  |  Deep Learning for Image Classification  |  Project Report"
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor.from_string("607D8B")
    footer = sec.footer.paragraphs[0]
    add_page_number(footer)
    footer.runs[0].font.size = Pt(8)

    # Cover
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(45)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("ITRI626\nPROJECT REPORT")
    r.bold = True
    r.font.size = Pt(29)
    r.font.color.rgb = RGBColor.from_string("123B5D")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Blender Material Classification")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor.from_string("2B6F8A")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("A leakage-safe benchmark of synthetic PBR material recognition")
    r.italic = True
    r.font.size = Pt(12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(16)
    p.add_run("Group members: ________________________________\n").bold = True
    p.add_run("Student numbers: ______________________________\n")
    p.add_run(
        "Lecturer: Prof. Absalom Ezugwu\nModule: ITRI626\nSubmission date: ______________________________"
    )
    doc.add_page_break()

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "This report evaluates whether visual cues learned from procedural Blender materials transfer to unseen base materials. "
        "A balanced dataset of 3,150 RGB EEVEE renders was generated from 105 independent PBR base materials across wood, metal, stone, concrete, ceramic, fabric and leather. "
        "The decisive split is grouped by material identity, so all 30 views of a base material remain in one split. A from-scratch CustomCNN is compared with ImageNet-pretrained ResNet-50, ConvNeXt-Tiny and ViT-B/16 using the same locked train/validation/test protocol. "
        "On the sealed TEST set, ResNet-50 is the recommended model (accuracy 69.8%, Macro-F1 68.3%); grouped five-fold development CV gives Macro-F1 71.7% ± 10.9%. "
        "The validation-to-TEST drop and weak metal/ceramic recall show a material-domain shift, so the model is a reproducible research benchmark rather than a validated real-world detector."
    )
    doc.add_paragraph(
        "Keywords: image classification; Blender; PBR; transfer learning; grouped cross-validation; domain shift; reproducibility."
    )
    doc.add_heading("Contents", level=1)
    # A static, visible contents list keeps the exported PDF useful in viewers
    # that do not calculate Word fields. Re-run this generator after edits to
    # refresh the listed pagination.
    contents = [
        "1  Introduction, problem and aim ................................................ 3",
        "2  Dataset provenance, creation and quality assurance .................... 4",
        "3  Methodology and model families ................................................. 6",
        "4  Experimental design and evaluation protocol ............................... 7",
        "5  Results and model comparison ..................................................... 8",
        "6  Discrimination, calibration, efficiency and interpretation ............ 10",
        "7  Discussion, limitations and Version-2 proposal .......................... 12",
        "8  Kaggle publication: conditional future work .................................. 12",
        "9  Conclusion .................................................................................... 13",
        "References ........................................................................................ 13",
        "Appendices ......................................................................................... 14",
    ]
    for line in contents:
        q = doc.add_paragraph(line)
        q.paragraph_format.space_after = Pt(1)
        q.paragraph_format.left_indent = Inches(0.2)
    doc.add_page_break()

    # 1
    doc.add_heading("1. Introduction, problem and aim", level=1)
    doc.add_heading("1.1 Problem definition and motivation", level=2)
    doc.add_paragraph(
        "Material recognition is useful in asset search, synthetic-scene inspection and graphics workflows, but appearance depends on texture, reflectance, illumination, viewpoint and scale. "
        "This project asks a deliberately bounded question: can a classifier identify the material class of a rendered sphere when it has not seen that base material during training? The scope is seven-way single-label classification of Blender EEVEE images, not arbitrary photographs or safety-critical inspection."
    )
    doc.add_heading("1.2 Aim, value and target classes", level=2)
    doc.add_paragraph(
        "The aim is to create a documented, leakage-safe benchmark; compare distinct neural model families fairly; quantify both predictive quality and deployment cost; and explain where errors arise. The classes are defined by the dominant PBR surface family:"
    )
    add_table(
        doc,
        ["Class", "Operational definition", "Why included"],
        [
            [
                "wood",
                "Wood-grain or timber-like PBR surface",
                "Common structural/decorative material",
            ],
            [
                "metal",
                "Conductive/metallic reflective surface",
                "Tests reflectance and highlight cues",
            ],
            ["stone", "Natural stone/mineral surface", "Tests mottled, coarse texture"],
            ["concrete", "Cementitious, granular grey surface", "Near-neighbour to stone"],
            ["ceramic", "Glazed or fired ceramic surface", "Tests smooth/glossy ambiguity"],
            ["fabric", "Woven/textile surface", "Distinct fine-scale texture"],
            ["leather", "Leather-like hide surface", "Near-neighbour to fabric/wood in appearance"],
        ],
        "Table 1. Target classes and practical motivation.",
    )
    add_figure(
        doc,
        "dataset/representative_images_all_classes.png",
        "Figure 1. Representative renders from every target class.",
        6.1,
        "Seven panels of representative renders, one panel per material class.",
    )

    # 2
    doc.add_heading("2. Dataset provenance, creation and quality assurance", level=1)
    doc.add_heading("2.1 Provenance and generation", level=2)
    doc.add_paragraph(
        "The dataset is an original synthetic benchmark generated locally with Blender EEVEE and the retained RenderSynth pipeline. The generation report records seed 62642, 512×512 RGB PNG output, 105 independent PBR base materials (15 per class), and 30 randomized renders per material. The run completed in one pass: 3,150 rendered, zero skipped and zero failed, with 41 min 48 s wall-clock time."
    )
    add_figure_pair(
        doc,
        ("dataset/split_distribution.png", "Figure 2. Locked split distribution."),
        (
            "dataset/material_diversity_contact_sheet.png",
            "Figure 3. Base-material diversity across randomized views.",
        ),
        "Split counts by material and image.",
        "Contact sheet of varied synthetic material views.",
    )
    doc.add_heading("2.2 Cleaning, label QA and class balance", level=2)
    doc.add_paragraph(
        "Class and material IDs were inherited from the curated source-material folder structure and the authoritative material manifest. Automated checks then verified class/split inheritance, material IDs, texture-node resolution and render-record consistency; this report does not claim an unsupported manual relabelling review. The final dataset report records pass results for file count, manifest alignment, image dimensions, PNG headers, filename collisions, render indices, material counts, unique seeds, blank-image checks and sphere framing. Pixel sanity was sampled on 420 stratified images (four per base material), while structural and geometric framing checks were applied to all 3,150 manifest rows. The base-material audit found 105 clean and zero flagged materials; all texture nodes resolved to files on disk."
    )
    add_table(
        doc,
        [
            "Class",
            "Target/rendered before QA",
            "Final accepted",
            "Removed/corrupt/failed/uncertain",
        ],
        [[c, "450", "450", "0"] for c in CLASSES] + [["TOTAL", "3,150", "3,150", "0"]],
        "Table 2. Before-versus-after cleaning audit (source: final_dataset_report.md).",
    )
    doc.add_paragraph(
        "Each class has exactly 450 accepted images (330/60/60 train/validation/test), so no class-weight correction was required."
    )
    add_figure_pair(
        doc,
        ("dataset/class_distribution.png", "Figure 4. Balanced class counts after cleaning."),
        ("dataset/class_split_stacked.png", "Figure 5. Counts by class and split."),
        "Balanced class distribution.",
        "Per-class train, validation and TEST counts.",
    )
    doc.add_heading("2.3 Leakage control, ethics and licensing", level=2)
    doc.add_paragraph(
        "The authoritative manifests are metadata/material_manifest_split.csv and metadata/render_manifest.csv. Splitting is by material_id, never by individual render: all 30 views of one base material stay together. This directly prevents near-duplicate views from crossing train, validation and TEST. The data contain no people, faces, health records or personal information, so human-subject privacy risk is not introduced by the images."
    )
    doc.add_paragraph(
        "A material-source licensing registry is not yet normalized into a machine-readable record. This is an important unresolved limitation: generated renders may inherit restrictions from source texture assets. The dataset is therefore not presented as publicly redistributable in this report. Before a Kaggle release, every source asset needs a source URL, licence/usage condition and attribution entry, and the redistribution terms must be checked.",
        style="Intense Quote",
    )
    doc.add_paragraph(
        "Repository: https://github.com/dianbrown/blender-material-classification. Dataset access URL: unavailable/not public at this stage because licensing clearance is incomplete."
    )
    add_table(
        doc,
        ["Split", "Base materials", "Images", "Images/class", "Leakage status"],
        [
            ["Train", "77", "2,310", "330", "Development"],
            ["Validation", "14", "420", "60", "Model selection"],
            ["TEST", "14", "420", "60", "Locked; evaluated once"],
        ],
        "Table 3. Dataset split and leakage policy.",
    )

    # 3
    doc.add_heading("3. Methodology and model families", level=1)
    doc.add_paragraph(
        "All models receive the same 224×224 RGB input. Train-only augmentation is RandomResizedCrop(224, scale=0.88–1.00, ratio=0.95–1.05), horizontal flip (p=0.5), and ColorJitter (brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02). These transformations represent modest viewpoint, framing, illumination and colour changes while avoiding physically implausible deformations of a material sphere. Validation and TEST use resize 256, center crop 224 and ImageNet mean/std normalization. Training uses AdamW, label smoothing 0.10, warmup/cosine scheduling, AMP where available, a 40-epoch ceiling and early stopping (patience 8), selecting the best validation Macro-F1 checkpoint."
    )
    add_table(
        doc,
        ["Model", "Effective batch", "Learning rate", "Weight decay", "Other fixed settings"],
        [
            [
                "CustomCNN",
                "64",
                "3e−4",
                "1e−4",
                "smoothing .10; warmup 3; max 40; patience 8; clip 1; AMP",
            ],
            [
                "ResNet-50",
                "32",
                "head 3e−4 / backbone 3e−5",
                "1e−4",
                "freeze 3; smoothing .10; warmup 3; max 40; patience 8; clip 1; AMP",
            ],
            [
                "ConvNeXt-Tiny",
                "16",
                "head 3e−4 / backbone 3e−5",
                "1e−4",
                "freeze 3; smoothing .10; warmup 3; max 40; patience 8; clip 1; AMP",
            ],
            [
                "ViT-B/16",
                "8",
                "head 3e−4 / backbone 1e−5",
                "1e−4",
                "freeze 3; smoothing .10; warmup 3; max 40; patience 8; clip 1; AMP",
            ],
        ],
        "Table 4. Exact per-model hyperparameters from the resolved experiment configurations.",
    )
    add_table(
        doc,
        ["Family", "Model", "Training mode", "Role in comparison"],
        [
            ["Custom convolutional", "CustomCNN", "From scratch", "Capacity floor/baseline"],
            [
                "Residual CNN",
                "ResNet-50",
                "ImageNet head stage then fine-tune",
                "Recommended candidate",
            ],
            [
                "Modern CNN",
                "ConvNeXt-Tiny",
                "ImageNet head stage then fine-tune",
                "Contemporary convolutional design",
            ],
            [
                "Vision transformer",
                "ViT-B/16",
                "ImageNet head stage then fine-tune",
                "Patch-token architecture",
            ],
        ],
        "Table 5. Four distinct model families represented in the study.",
    )
    doc.add_heading("3.1 Transfer learning workflow", level=2)
    doc.add_paragraph(
        "For pretrained models, the classification head is trained while the backbone is frozen, followed by whole-model fine-tuning under the fixed training policy. This gives pretrained representations a stable adaptation stage before allowing texture- and reflectance-sensitive features to change. CustomCNN provides a genuine from-scratch baseline rather than a fourth pretrained variant."
    )
    doc.add_heading("3.2 Reproducibility and fairness", level=2)
    add_table(
        doc,
        ["Item", "Recorded setting"],
        [
            ["Random seed", str(env["seed"])],
            [
                "Python / PyTorch / torchvision",
                f"{env['python_version']} / {env['pytorch_version']} / {env['torchvision_version']}",
            ],
            [
                "CUDA / GPU",
                f"CUDA {env['cuda_version']} / {env['gpu_name']} (≈{env['gpu_vram_bytes'] / 1024**3:.1f} GB)",
            ],
            ["Input", "224×224 RGB; resize 256 then center crop 224"],
            ["Optimizer", "AdamW; label smoothing; warmup/cosine schedule"],
            ["Selection", "Best validation Macro-F1; same locked TEST split"],
            ["Grouping", "StratifiedGroupKFold, 5 folds, development materials only"],
        ],
        "Table 6. Exact recorded environment and experimental controls.",
    )
    doc.add_paragraph(
        "The comparison is fair in data, preprocessing, split, selection metric and evaluation protocol. Architecture-specific effective batch sizes reflect memory constraints and are recorded in efficiency_metrics.json; this is reported transparently rather than hidden."
    )

    # 4
    doc.add_heading("4. Experimental design and evaluation protocol", level=1)
    doc.add_paragraph(
        "The train/validation/TEST split is fixed before final evaluation. Five-fold StratifiedGroupKFold is used only on development materials for transfer models. The test gate is represented by methodology_lock.json and hashed test_evaluated markers. No post-TEST tuning is claimed in this report. Macro-F1 is emphasized because it weights all seven classes equally; accuracy, precision, recall, balanced accuracy, ROC-AUC, log loss and calibration are supporting measures."
    )
    add_figure_pair(
        doc,
        ("training/combined_val_f1.png", "Figure 6. Validation Macro-F1 trajectories."),
        ("training/combined_val_loss.png", "Figure 7. Validation loss trajectories."),
        "Development Macro-F1 curves.",
        "Development validation-loss curves.",
    )
    add_figure_pair(
        doc,
        ("cv/macro_f1_mean_std.png", "Figure 8. Grouped five-fold Macro-F1 mean ± SD."),
        ("cv/fold_heatmap.png", "Figure 9. Fold-level grouped-CV heatmap."),
        "Grouped CV mean and sample standard deviation.",
        "Fold-level CV variability.",
    )

    # 5
    doc.add_heading("5. Results and model comparison", level=1)
    doc.add_paragraph(
        "The following results are copied from completed cached artifacts. TEST figures are treated as descriptive held-out evidence; grouped CV remains the development stability estimate. The current recommendation is ResNet-50 because its composite selection rule combines TEST Macro-F1/accuracy, grouped-CV stability, per-class consistency, checkpoint size and batch-1 latency."
    )
    rows = []
    for row in comparison:
        rows.append(
            [
                MODEL_LABELS.get(row["model"], row["model"]),
                row["family"],
                pct(row["test_accuracy"]),
                pct(row["test_macro_f1"]),
                pct(row["cv_macro_f1_mean"])
                if row["cv_macro_f1_mean"] not in ("", "—", "â€”")
                else "N/A",
                row["best_epoch"],
                fmt(row["inference_ms_per_image"], 2),
                fmt(row["model_size_mb"], 1),
            ]
        )
    add_table(
        doc,
        [
            "Model",
            "Family",
            "TEST acc.",
            "TEST Macro-F1",
            "CV Macro-F1",
            "Best epoch",
            "Latency ms/img",
            "Size MB",
        ],
        rows,
        "Table 7. Locked comparison across all reported models.",
    )
    add_figure_pair(
        doc,
        ("comparison/model_comparison_dashboard.png", "Figure 10. Model comparison dashboard."),
        ("comparison/model_rankings.png", "Figure 11. Composite model ranking."),
        "Held-out metrics, CV and efficiency.",
        "Composite ranking supporting ResNet-50.",
    )
    doc.add_heading("5.1 ResNet-50 held-out performance", level=2)
    doc.add_paragraph(
        f"ResNet-50 reaches {pct(resnet_test['accuracy'])} TEST accuracy, {pct(resnet_test['precision_macro'])} macro precision, {pct(resnet_test['recall_macro'])} macro recall and {pct(resnet_test['f1_macro'])} Macro-F1. Its grouped-CV Macro-F1 is 71.7% ± 10.9%. TEST ROC-AUC is {fmt(resnet_test['roc_auc_macro'], 3)}, log loss is {fmt(resnet_test['log_loss'], 3)}, and expected calibration error is {pct(resnet_test['ece'])}. Validation accuracy at the selected epoch is {pct(resnet_val['accuracy'])}, so the validation-to-TEST accuracy drop is {pct(resnet_val['accuracy'] - resnet_test['accuracy'])} points; this is consistent with a material-level domain shift and motivates cautious claims."
    )
    add_figure_pair(
        doc,
        (
            "confusion_matrices/resnet50_normalized.png",
            "Figure 12. ResNet-50 TEST confusion matrix (normalized).",
        ),
        (
            "confusion_matrices/resnet50_raw.png",
            "Figure 13. ResNet-50 TEST confusion matrix (counts).",
        ),
        "Rows are true classes; columns are predictions.",
        "Integer counts, 60 examples per class.",
    )
    doc.add_heading("5.2 Per-class analysis and important errors", level=2)
    per_rows = [
        [m["class"], pct(m["precision"]), pct(m["recall"]), pct(m["f1"]), str(m["support"])]
        for m in resnet_test["per_class"]
    ]
    add_table(
        doc,
        ["Class", "Precision", "Recall", "F1", "Support"],
        per_rows,
        "Table 8. ResNet-50 per-class TEST metrics.",
    )
    doc.add_paragraph(
        "Concrete is strongest (F1 97.4%), with perfect precision and 95.0% recall. Wood is also strong (F1 84.6%), while fabric and leather reach 78.9% F1. Metal is the clearest weakness: only 23.3% recall, with many examples predicted as leather, ceramic or stone. Ceramic is also difficult (42.0% F1) and is frequently confused with fabric. These patterns suggest that smooth highlights, texture scale and shared shading cues are more informative than class names alone. They also identify targeted next experiments without changing the sealed TEST result."
    )
    add_figure(
        doc,
        "per_class/grouped_f1.png",
        "Figure 14. Per-class F1 comparison across the four models; the metal and ceramic weaknesses are visible.",
        6.0,
    )
    add_figure_pair(
        doc,
        ("errors/resnet50_incorrect.png", "Figure 15. Curated ResNet-50 TEST errors."),
        (
            "errors/resnet50_pair_metal_vs_leather.png",
            "Figure 16. Metal-versus-leather confusion pair.",
        ),
        "Examples of incorrect held-out predictions.",
        "Representative material confusion pair.",
    )
    add_figure(
        doc,
        "materials/resnet50_accuracy.png",
        "Figure 17. ResNet-50 accuracy by base material; variation within a class motivates more diverse materials.",
        5.95,
    )

    # 6
    doc.add_heading("6. Discrimination, calibration, efficiency and interpretation", level=1)
    doc.add_paragraph(
        "The supplementary diagnostics separate ranking ability from thresholded accuracy. The ResNet-50 macro ROC-AUC of 0.916 indicates useful class ranking even though the selected class decision is imperfect. Calibration is not ideal (ECE ≈9.7%), so confidence scores should not be interpreted as probabilities without calibration on development data."
    )
    add_figure_pair(
        doc,
        ("roc_pr/macro_comparison.png", "Figure 18. Macro ROC/precision-recall comparison."),
        (
            "calibration/reliability_and_confidence.png",
            "Figure 19. Reliability and confidence diagnostics.",
        ),
        "Macro discrimination curves.",
        "Confidence reliability and calibration.",
    )
    add_figure(
        doc,
        "comparison/efficiency_tradeoffs.png",
        "Figure 20. Accuracy/latency/model-size trade-offs on the RTX 4060 benchmark.",
        5.95,
    )
    doc.add_paragraph(
        "Efficiency matters for a practical recognizer. CustomCNN is smallest (38.9 MB) and fastest at batch 1 (1.38 ms/image), but its TEST Macro-F1 is only 49.0%. ResNet-50 provides the strongest held-out score at 269.6 MB and 6.44 ms/image. ViT-B/16 is by far the largest (982.1 MB) and slowest (7.30 ms/image) while scoring below ResNet-50, so it is not the preferred deployment trade-off for this dataset."
    )
    doc.add_heading("6.1 Embeddings and explainability", level=2)
    doc.add_paragraph(
        "PCA/UMAP views provide qualitative evidence about feature organization, while the curated saliency gallery tests whether attention overlaps the rendered object rather than the neutral background. These are interpretation aids, not causal proof and not additional performance metrics."
    )
    add_figure_pair(
        doc,
        ("embeddings/resnet50_pca.png", "Figure 21. ResNet-50 TEST embedding PCA."),
        ("embeddings/resnet50_umap.png", "Figure 22. ResNet-50 TEST embedding UMAP."),
        "PCA projection of learned embeddings.",
        "UMAP projection of learned embeddings.",
    )
    add_figure(
        doc,
        "explainability/resnet50_curated.png",
        "Figure 23. Curated ResNet-50 saliency/Grad-CAM gallery; qualitative evidence only.",
        5.8,
    )

    # 7
    doc.add_heading("7. Discussion, limitations and Version-2 proposal", level=1)
    doc.add_heading("7.1 What the results support", level=2)
    add_bullets(
        doc,
        [
            "The project satisfies the core experimental design: a defined seven-way task, balanced and audited data, four distinct model families, transfer learning, grouped leakage prevention and a sealed common TEST set.",
            "ResNet-50 is the defensible current recommendation for this benchmark because it is best on held-out accuracy and Macro-F1 and has a much smaller footprint than ViT-B/16.",
            "Grouped CV is valuable: the standard deviations are large (about 0.10–0.11 Macro-F1), so a single split score should not be treated as a precise universal estimate.",
            "The large validation-to-TEST gap is a scientifically useful finding, not something to hide: unseen base materials expose texture-source and material-domain variation.",
        ],
    )
    doc.add_heading("7.2 Limitations", level=2)
    add_bullets(
        doc,
        [
            "Only 105 base materials and one Blender/EEVEE rendering setup are represented; real photographs, different renderers and different camera/object geometries are untested.",
            "The source-texture licensing/attribution registry remains unresolved, so public redistribution is conditional rather than available now.",
            "The TEST set contains 14 base materials (60 renders per class), which is adequate for a course benchmark but not a production validation sample.",
            "The baseline has no grouped-CV score because it was not run in the cached CV protocol; this is explicitly reported as N/A rather than fabricated.",
            "Saliency and embedding plots are descriptive. They do not establish that a model uses physically meaningful material properties.",
        ],
    )
    doc.add_heading("7.3 Version-2 improvement proposal (no new results claimed)", level=2)
    doc.add_paragraph(
        "A sound Version-2 study would preserve the current TEST set and use only development materials for selection. First, build the licensing registry and add more independent base materials, especially diverse metals and ceramics. Second, run a small Optuna search over learning rate, weight decay, label smoothing, augmentation strength, freeze-stage duration and classifier dropout using grouped CV or a fixed development validation split. Third, evaluate targeted material-aware augmentations (lighting, roughness/specular perturbation, blur/noise and modest crop/scale changes) only when physically realistic. Fourth, compare temperature scaling or other calibration fitted on validation predictions. Finally, unlock a new untouched TEST set and report whether improvements survive domain shift. These are proposals; no higher accuracy is claimed here."
    )

    # 8
    doc.add_heading("8. Kaggle publication: conditional future work", level=1)
    doc.add_paragraph(
        "Kaggle publication status: NO-GO at the current project state. The dataset would be a useful teaching benchmark because it is balanced, reproducible and includes material-level group metadata, but publication must wait for a completed per-asset source-texture licence/attribution audit and explicit permission to redistribute every generated render. A step-by-step release guide is provided separately in kaggle.md as future work; it is not evidence that the dataset has been published.",
        style="Intense Quote",
    )
    add_numbered(
        doc,
        [
            "Audit every source PBR asset and record source, licence, attribution and redistribution permission.",
            "Prepare a Kaggle-ready package with images, manifests, DATASET.md, licence/attribution file, checksums and a minimal notebook.",
            "Run the local verification and inspect a clean extraction before upload.",
            "Create the dataset on Kaggle with accurate title, description, labels, metadata fields and licence; do not imply real-world material recognition.",
            "Publish only after confirming that every included file is permitted, then link the stable Kaggle URL back into README and this report.",
        ],
    )

    # 9
    doc.add_heading("9. Conclusion", level=1)
    doc.add_paragraph(
        "This study treats the benchmark as an experiment rather than only a leaderboard: the dataset is generated and audited, the split is leakage-safe, architectures are meaningfully different, metrics are broad, errors and efficiency are analyzed, and held-out limitations are explicit. The evidence supports ResNet-50 as the current model for this synthetic benchmark (69.8% TEST accuracy; 68.3% Macro-F1), while showing that synthetic material-domain shift is the central unresolved challenge. Future work should expand independent materials, complete licence clearance, evaluate synthetic-to-real transfer and test whether development-only tuning survives a new untouched TEST set."
    )

    # References
    doc.add_heading("References", level=1)
    refs = [
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of CVPR. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
        "Liu, Z., et al. (2022). A ConvNet for the 2020s. Proceedings of CVPR. https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html",
        "Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR. https://openreview.net/forum?id=YicbFdNTTy",
        "Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision (label smoothing). Proceedings of CVPR. https://openaccess.thecvf.com/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html",
        "Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR. https://openreview.net/forum?id=Bkg6RiCqY7",
        "Guo, C., et al. (2017). On Calibration of Modern Neural Networks. Proceedings of ICML. https://proceedings.mlr.press/v70/guo17a.html",
        "Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV. https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html",
        "McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. Journal of Open Source Software. https://joss.theoj.org/papers/10.21105/joss.00861",
        "Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS. https://papers.nips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
        "PyTorch authors. (2024). PyTorch documentation. https://pytorch.org/docs/stable/",
        "PyTorch authors. Reproducibility notes. https://docs.pytorch.org/docs/stable/notes/randomness.html",
        "Torchvision authors. (2024). Torchvision documentation and model implementations. https://pytorch.org/vision/stable/",
        "scikit-learn developers. StratifiedGroupKFold documentation. https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold",
        "Blender Foundation. Blender Manual: EEVEE and rendering documentation. https://docs.blender.org/manual/en/latest/",
        "Local project sources. DATASET.md, MODEL_CARD.md, reports/REPORT_ASSETS.md and retained Blender/RenderSynth generation manifests (accessed in this repository).",
    ]
    for index, ref in enumerate(refs, start=1):
        paragraph = doc.add_paragraph(f"{index}. {ref}")
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        for run in paragraph.runs:
            run.font.size = Pt(8.5)

    # Appendices
    doc.add_page_break()
    doc.add_heading("Appendix A. Reproduction commands", level=1)
    doc.add_paragraph(
        "Run from the repository root in the pinned environment. The TEST evaluation command is guarded by the methodology lock and cached hashes; do not tune after it."
    )
    p = doc.add_paragraph(style="Intense Quote")
    p.add_run(
        "uv sync --extra dev --extra explainability --extra embeddings --extra reports\npython scripts/verify_dataset.py\npython scripts/train_all.py\npython scripts/run_group_cv.py --folds 5\npython scripts/lock_methodology.py\npython scripts/evaluate_all.py --split test\npython scripts/generate_report.py\npython scripts/generate_visuals.py --explainability\npython scripts/update_docs.py\npython scripts/build_project_report.py"
    )
    doc.add_heading("Appendix B. Figure inventory", level=1)
    doc.add_paragraph(
        "The source index reports/REPORT_ASSETS.md remains the authoritative inventory and explains which figures are development, CV or held-out TEST evidence. This DOCX embeds 23 selected figures to avoid repeating all 111 available PNGs."
    )
    inventory = [
        "Dataset: material_diversity_contact_sheet; representative_images_all_classes; split_distribution; class_distribution; class_split_stacked.",
        "Training/CV: combined_val_f1; combined_val_loss; macro_f1_mean_std; fold_heatmap.",
        "Comparison/efficiency: model_comparison_dashboard; model_rankings; efficiency_tradeoffs.",
        "TEST evaluation: resnet50_normalized/raw; grouped_f1; resnet50_incorrect; resnet50_pair_metal_vs_leather; resnet50_accuracy.",
        "Diagnostics: macro_comparison; reliability_and_confidence; resnet50_pca; resnet50_umap; resnet50_curated.",
    ]
    add_bullets(doc, inventory)
    doc.add_heading("Appendix C. Submission checklist against the rubric", level=1)
    add_table(
        doc,
        ["Criterion", "Evidence in report", "Completion check"],
        [
            [
                "Problem / motivation (2)",
                "Sections 1 and 9",
                "Insert group names and student numbers",
            ],
            [
                "Dataset / preparation (4)",
                "Section 2, Tables 1–3, Figures 1–5",
                "Complete texture-source licensing registry",
            ],
            [
                "Models / methodology (4)",
                "Section 3, Tables 4–5",
                "Prepare oral explanation of transfer stages",
            ],
            [
                "Setup / reproducibility (3)",
                "Section 4, Table 6, Appendix A",
                "Confirm pinned environment on a clean machine",
            ],
            [
                "Results / evaluation (4)",
                "Sections 5–6, Tables 7–8, Figures 6–23",
                "Refresh fields/TOC and export PDF",
            ],
            [
                "Discussion / conclusion (2)",
                "Sections 6–9",
                "Keep limitations and TEST protocol explicit",
            ],
            [
                "Quality / references (1)",
                "References and figure captions",
                "Check citation style required by lecturer",
            ],
        ],
        "Table 9. Rubric-oriented final checklist.",
    )

    # Keep update fields and core properties useful.
    doc.core_properties.title = "ITRI626 Project Report — Blender Material Classification"
    doc.core_properties.subject = "Leakage-safe synthetic PBR material image classification"
    doc.core_properties.author = "ITRI626 project group"
    doc.core_properties.comments = (
        "Generated from cached local artifacts by scripts/build_project_report.py"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return {
        "figures": 23,
        "tables": len(doc.tables),
        "paragraphs": len(doc.paragraphs),
        "sections": len(doc.sections),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report.docx")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
