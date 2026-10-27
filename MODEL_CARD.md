# Model card: Material Studio

## Model identity and use

The preselected **primary** is frozen multiscale DINOv2 small with registers plus
ConvNeXt-Tiny features and a scaled balanced logistic classifier. Use for
recognition experiments and inspection of comparable synthetic renders. It is not
validated for arbitrary photographs, physical composition or safety-critical
material decisions.

Material Studio defaults to the exploratory fine-tuned DINOv2 model. Its
observed accuracy and lower batch-one latency than fusion support a provisional
application default for comparable renders. This does not change the
preselected primary or establish superiority. Evaluation scores describe the
saved models on the recorded dataset; they do not validate arbitrary app inputs.

Input is RGB with recorded model-specific resize, crop and normalization.
Output is a probability vector over `wood`, `metal`, `stone`, `concrete`,
`ceramic`, `fabric`, `leather`, in that order. Probabilities are not calibrated
guarantees. The recipe `dino_multiscale_convnext`, C=1, combines CLS, patch mean
and population standard deviation at 224/336 pixels with ConvNeXt pooled features
at 224: 3,072 features. Register tokens are excluded from patch statistics.
Backbones remain frozen. Scaling and balanced L2 logistic fitting occur within
each training partition. Exact revisions and preprocessing are in metadata.

## Artifacts and selection

All four trained models are available in the
[bundled Windows installer](https://github.com/dianbrown/blender-material-classification/releases/tag/v0.5),
including the backbone files and runtime needed for offline inference. The paths
below identify the development artifacts, which are not stored in Git.

| Role | Local checkpoint (adjacent metadata required) |
| --- | --- |
| Primary | `artifacts/v3/run_final/model.joblib` |
| ConvNeXt control | `artifacts/v3/run_final/control_model.joblib` |
| Fixed ResNet-50 | `artifacts/v3/resnet-final/final/model.pt` |
| Exploratory DINOv2 | `artifacts/v3/finetune/candidates/finetuned/model.pt` |

[Model choice](reports/results/v3/model_choice.json) used development Macro F1
0.8035245993060354 versus 0.7724822182541141 for DINOv2. These are selection
scores. The [checkpoint lock](reports/results/v3/final_model_lock.json) at
**09:51:05 UTC** preceded first fresh prediction at **09:54:06 UTC**, 27 October
2026. Primary choice did not change after the fresh ranking.

Primary checkpoint SHA-256:
`8722128562fcef9837089b39c97d0f631c8480c71378f05158fe90a30c106b8e`.
Use the lock for all artifact and sidecar hashes.
Load only trusted serialized files. Canonical `features_final` and `run_final`
exactly reproduce seven earlier feature arrays and seven OOF prediction files;
initial/intermediate runs remain preserved.

## Completed evaluation

The same **630 images / 21 groups**, 90 images and three groups per class:

| Pipeline | Fresh accuracy | Macro F1 |
| --- | ---: | ---: |
| Preselected primary | **80.16%** | **0.7952** |
| ConvNeXt control | 75.56% | 0.7566 |
| Fixed ResNet-50 | 77.62% | 0.7665 |
| Exploratory DINOv2 | **86.83%** | **0.8631** |

Primary accuracy 95% CI: **66.03-93.65%**; DINOv2: **76.98-95.87%**.
DINOv2 is the highest observed comparison, not a substituted primary. Its paired
advantage is **+6.67 percentage points (95% CI -2.06 to 16.19)**. All three
saved fresh paired accuracy intervals include zero; superiority is not confirmed.
The **80% point target is met by the primary**, but not by its lower CI bound.

Intervals use 10,000 class-stratified source-group bootstrap draws, seed 62642,
retaining all views per sampled group. They condition on fixed predictions,
excluding refitting/selection uncertainty. Sources:
[fresh outputs](reports/results/v3/fresh) and
[paired comparison](reports/results/v3/fresh/paired_dinov2_finetuned_vs_frozen_selected.json).
Primary ceramic recall is 57.78%. Perfect sampled recall for wood, metal and
fabric does not imply perfect performance on new families.

The initial five-fold existing-corpus endpoint uses 3,150 images / 101 families.
Frozen nested selection achieved 79.3968%, Macro F1 0.795485, accuracy CI
72.65-85.57%; control 68.8571%; fixed ResNet 71.5556%, Macro F1 0.713501.
The initial target was missed. Paired primary-control difference is +10.54 pp
(CI 4.83-16.45); primary-ResNet +7.84 pp (CI 1.46-14.14). These initial endpoint
comparisons do not establish corresponding fresh-population effects.

## Adaptation and robustness

ResNet was declared before V3 scores and uses ten fixed epochs without
outer-validation checkpoint selection. DINOv2 adaptation followed the initial
79.4% outcome: six development fits, two learning rates, selected 1e-5, best
epochs [6, 5, 3], five final epochs. It remains an exploratory follow-up.

| Corruption | Primary accuracy | DINOv2 accuracy |
| --- | ---: | ---: |
| Low brightness | 78.25% | 85.24% |
| High brightness | 83.02% | 89.05% |
| Gaussian blur | 77.14% | 80.00% |
| JPEG quality 30 | **59.05%** | **75.40%** |
| Resolution 128 | 75.08% | 83.17% |

Primary tests were predeclared. DINOv2 tests were added **post hoc**, after clean
and primary-corruption scores, under the
[amendment](reports/results/v3/exploratory_robustness_amendment.json). No model
changed. JPEG is a major limitation; the target is not met in all conditions.
[Outputs](reports/results/v3/robustness) retain all class recall changes.

Primary log loss is 0.6745, Brier score 0.3018, 15-bin ECE 0.0948. DINOv2 has
better observed log loss/Brier but higher ECE, 0.1411. Calibration and risk/coverage
curves are descriptive. No calibration fit or abstention threshold was selected
on the holdout; operating thresholds need separate development validation.

## Measured deployment costs

| Pipeline | Warm batch-1 median ms | Batch-16 images/s | Complete deployment bytes |
| --- | ---: | ---: | ---: |
| Primary | 42.876 | 46.361 | 203,629,625 |
| Control | 14.205 | 174.416 | 115,271,775 |
| ResNet | 9.463 | 294.688 | 95,315,284 |
| DINOv2 | 13.862 | 101.438 | 88,027,439 |

[Benchmarks](reports/results/v3/benchmarks) use the same inputs, four CPU threads
and CUDA on an RTX 4070 Laptop GPU. Scope: predecoded RGB through preprocessing,
model computation and probabilities; disk I/O/decode excluded. Reload is
process-warm, not OS-cold. Memory is process allocator peak, not total board use.
These model deployment sizes exclude the application's bundled Python and CUDA
runtime. The installed application occupies approximately 4.8 GiB, including
that runtime.
The approximately 311 KB primary head is **not** total model size: pretrained
files and sidecars are required. DINOv2 exports contain standalone weights with
no pretrained-cache loading dependency. See [README](README.md) for setup and
the [results notebook](notebooks/05_v3_results.ipynb) for saved evaluation evidence.

## Limitations and release

Three fresh families per class, incomplete physical ancestry, restricted class
semantics and Blender 5.1.1 softer shadows limit transfer. Public texture overlap
with backbone pretraining has not been audited. Different pretraining,
resolution, adaptation and search budgets confound architecture-only comparisons.
No real-world validation exists. Historical V1 is a previously consulted,
different endpoint, not direct improvement evidence.

All 316 original and 63 fresh maps have official CC0 provenance. The project
authors renders/code and attributes third-party maps. The full dataset is not
published; a Kaggle release is planned, and no dataset upload has been made.
Its rendered-image license remains unassigned.
Pretrained-weight terms, map permissions and dataset licensing are
distinct. See [DATASET.md](DATASET.md).
