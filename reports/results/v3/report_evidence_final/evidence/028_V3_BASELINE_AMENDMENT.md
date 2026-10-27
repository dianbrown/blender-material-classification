# Supplemental matched ResNet baseline

Declared 2026-10-27 before the orchestrator inspected any V3 scored results.
The primary nested DINOv2/ConvNeXt selection run has started, and its candidate
set and selection rules remain unchanged.

The report audit identified the assignment's requirement for three distinct
architectures evaluated on the same split. Pooling variants of two backbones
do not meet that requirement, and historical ResNet scores use a different
protocol. Add a fixed, separately reported ResNet-50 transfer baseline on the
same five outer source-group folds. Its design is motivated by this requirement,
not by observed V3 model scores.

The supplemental model has ten fixed training epochs, with two head-only
epochs and eight fine-tuning epochs, explicit ImageNet-1K V2 pretrained weights,
and a fixed optimizer/augmentation recipe. No outer-fold early stopping,
checkpoint selection or hyperparameter search is allowed. Persist the precise
configuration before executing it. Report epoch training histories, final
outer predictions, compute cost and an independently fitted all-development
artifact. The fresh holdout remains unopened until all final models are locked.

The ResNet baseline is not eligible to replace the primary selection-procedure
estimate or final selected descriptor on the strength of outer-fold results.
Its score is a matched supplementary comparison, with different adaptation and
pretraining budgets explicitly reported.
