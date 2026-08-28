# Report asset index

This index lists the publication-ready figures generated from the locked
experiment artifacts. Links are relative to this file, so they also work from
a GitHub checkout. Use TEST figures for the final results section; training and
CV figures describe the development protocol and should not be presented as
held-out TEST performance.

## Primary comparison figures

- [Model comparison dashboard](figures/comparison/model_comparison_dashboard.png) — compact overview of TEST metrics, grouped CV, parameter count, and latency.
- [Model rankings](figures/comparison/model_rankings.png) — ranking view supporting the recommendation.
- [Efficiency trade-offs](figures/comparison/efficiency_tradeoffs.png) — accuracy/latency/size trade-off summary.
- [ResNet-50 normalized confusion matrix](figures/confusion_matrices/resnet50_normalized.png) — recommended primary confusion matrix for the report because ResNet-50 is the selected model; rows are true classes and values are row-normalized recall.

## Confusion matrices

Use raw matrices when reporting counts and normalized matrices when comparing
class recall. The class order is wood, metal, stone, concrete, ceramic, fabric,
leather.

| Model | Raw counts | Row-normalized |
| --- | --- | --- |
| CustomCNN | [raw](figures/confusion_matrices/custom_cnn_raw.png) | [normalized](figures/confusion_matrices/custom_cnn_normalized.png) |
| ResNet-50 | [raw](figures/confusion_matrices/resnet50_raw.png) | [normalized](figures/confusion_matrices/resnet50_normalized.png) |
| ConvNeXt-Tiny | [raw](figures/confusion_matrices/convnext_tiny_raw.png) | [normalized](figures/confusion_matrices/convnext_tiny_normalized.png) |
| ViT-B/16 | [raw](figures/confusion_matrices/vit_b16_raw.png) | [normalized](figures/confusion_matrices/vit_b16_normalized.png) |

## Recommended supporting figures

- Training: [combined validation accuracy](figures/training/combined_val_accuracy.png), [combined validation Macro-F1](figures/training/combined_val_f1.png), and [combined validation loss](figures/training/combined_val_loss.png). Use these to show optimization and early-stopping behavior.
- Cross-validation: [Macro-F1 mean ± SD](figures/cv/macro_f1_mean_std.png), [fold distributions](figures/cv/f1_macro_distribution.png), [accuracy distributions](figures/cv/accuracy_distribution.png), and [fold heatmap](figures/cv/fold_heatmap.png).
- Discrimination and calibration: [ROC/PR macro comparison](figures/roc_pr/macro_comparison.png), [ResNet-50 per-class ROC/PR](figures/roc_pr/resnet50_per_class.png), and [reliability/calibration](figures/calibration/reliability_and_confidence.png).
- Embeddings: [ResNet-50 PCA](figures/embeddings/resnet50_pca.png) and [ResNet-50 UMAP](figures/embeddings/resnet50_umap.png). These are the recommended embedding views for the selected model.
- Errors: [ResNet-50 correct examples](figures/errors/resnet50_correct.png), [incorrect examples](figures/errors/resnet50_incorrect.png), [lowest-confidence examples](figures/errors/resnet50_lowest_confidence.png), and [material-pair error view](figures/errors/resnet50_pair_metal_vs_leather.png).
- Per-material best/worst: [ResNet-50 material accuracy](figures/materials/resnet50_accuracy.png) (the companion CSV is [here](figures/materials/resnet50_accuracy.csv)).
- Explainability: [ResNet-50 curated gallery](figures/explainability/resnet50_curated.png), [ConvNeXt-Tiny curated gallery](figures/explainability/convnext_tiny_curated.png), and [ViT-B/16 curated gallery](figures/explainability/vit_b16_curated.png). These are qualitative saliency/Grad-CAM evidence, not additional metrics.

## Dataset context

- [Representative renders by class](figures/dataset/representative_images_all_classes.png)
- [Class distribution](figures/dataset/class_distribution.png)
- [Split distribution](figures/dataset/split_distribution.png)
- [Material diversity contact sheet](figures/dataset/material_diversity_contact_sheet.png)

All report figures are derived from cached outputs. Do not alter the sealed
TEST split or tune a model using these figures after TEST evaluation.
