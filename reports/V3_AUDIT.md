# V3 audit findings and design rationale

Audited 27 October 2026 by separate data and modelling agents. This records
verified repository findings and primary-source research before v3 experiments.

## Data and independence

The supplied archives contain 3,150 PNG images and 316 PBR maps. Image paths
match the manifest exactly, with 30 renders for each of 105 local material IDs.
The old split contains 77 training, 14 validation and 14 test IDs. Thirty views
of a texture are correlated observations, not thirty independent materials.

The original 105-independent-material claim is too strong. The official
[ambientCG relationship API](https://ambientcg.com/api/v2/full_json?q=leather&type=Material&include=relationshipData&limit=100)
identifies Leather011 and Leather009 as children of LeatherSubstance002,
Leather030 and Leather032 as children of LeatherSubstance005, and Leather034B
and Leather036B as children of LeatherSubstance006. Leather011 was validation
and Leather009 training. This establishes generator-family overlap, not exact
image duplication. V3 conservatively keeps each source family together.

Ninety-six source identifiers match the
[Poly Haven catalog](https://api.polyhaven.com/assets?t=textures); nine leather
identifiers match ambientCG. Both providers publish CC0 terms
([Poly Haven](https://polyhaven.com/license),
[ambientCG](https://docs.ambientcg.com/license/)). Identifier matches alone do
not verify downloaded file identity. The release registry must distinguish
provider evidence, local map hashes and unresolved provenance. This project
authors the rendered dataset, not the third-party source texture assets.

Semantic ambiguities include marble mosaic/terrazzo under ceramic and several
plaster or painted-wall sources under concrete. All old metal shaders use
Metallic=1 even for coated or rusty surfaces. Historical quality checks detect
gross defects, not correct material semantics. Preserve existing labels for
the archival benchmark, disclose ambiguity, and use a clearer taxonomy for
fresh data. Do not remove difficult samples based on model predictions.

## Historical modelling limitations

V1 best published test accuracy is 69.76% (ResNet-50). ConvNeXt development CV
accuracy was already 81.24%, but test accuracy was 61.19%. Fold standard
deviations of macro-F1 were around 0.10--0.11. These are not interchangeable
endpoints or evidence that the 80% unseen-material target has been achieved.

The old CV validation folds also selected checkpoints through early stopping.
The report recommendation used test metrics. Cached CV results were keyed by
model/fold without full data/config identity. Configured weight versions and
augmentation options were not always passed into execution. Resume omitted RNG
states. OOM recovery changed effective batch size. Reported model size included
optimizer state. The v3 implementation avoids these paths and preserves V1.

All thirteen inspected locked source files and the data config matched their
archived hashes after CRLF-to-LF normalization, but failed raw-byte hashes in
this Windows checkout. This is a line-ending reproducibility issue; it is not
evidence of substantive tampering. Do not rewrite the archived lock to hide it.

## Chosen initial experiment

Use public [DINOv2 small with registers](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m)
and [ConvNeXt Tiny with ImageNet-22K pretraining](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k).
Anonymous weight accessibility was verified by the auditor. DINOv3 weights
required access acceptance and are not an initial dependency.

DINOv2 features compare CLS, CLS plus mean patch tokens, and multiscale CLS,
patch mean and standard deviation at 224/336 pixels. Register tokens must be
excluded. CLS plus patch averaging is supported by the
[official linear evaluation implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py).
The mean/std multiscale extension is an experimental hypothesis informed by
[Deep Filter Banks](https://www.robots.ox.ac.uk/~vgg/publications/2015/Cimpoi15/),
not an established result on this dataset.

Five outer source-group folds and three inner folds select representations and
regularization entirely inside training partitions, following
[nested CV principles](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Fit feature scaling on each training partition only. Frozen feature extraction
can be cached because it learns nothing from this corpus. Cache provenance
includes weights, preprocessing and image/manifest identity.

Report the primary nested selection-procedure out-of-fold estimate separately
from individual model ablations and the final all-data deployment fit. A fresh
independent material holdout is being prepared without access to model results.
Evaluate it only after model selection is frozen. Neither endpoint establishes
arbitrary real-world performance. Bootstrap independent source groups, not
individual correlated renders, and report both point estimates and intervals.

## Local compute

Current GPU is RTX 4070 Laptop with 8 GB VRAM; historical timings came from a
different RTX 4060. Use one GPU process, small extraction batches and sequential
resolutions. Record actual runtime and peak VRAM. Library/platform changes can
alter numerical reproducibility; see
[PyTorch guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
