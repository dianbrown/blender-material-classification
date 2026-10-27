# Model card: V3 material classification (draft)

## Status and identity

The completed initial model is a **frozen DINOv2/ConvNeXt feature selector with
a logistic head**. Final deployment choice, exploratory fine-tuning, matched
ResNet training, independent fresh evaluation and the final report remain in
progress. This card does not claim a measured final fresh accuracy.

The initial all-development fit selected `dino_multiscale_convnext` with C=1:
DINOv2 small register CLS, patch mean and patch population standard deviation at
224 and 336 pixels, concatenated with ConvNeXt-Tiny pooled features at 224 pixels
(3,072 input features). Scaling and a class-balanced L2 logistic head use the
settings in [the frozen protocol](configs/v3/protocol.json). Initial artifacts
are `artifacts/v3/run/model.joblib` and `model_metadata.json`; the ConvNeXt
control is retained alongside them. Exact checkpoint revisions, weight hashes,
preprocessing, class order and implementation/environment hashes are recorded
in protocol, cache and model metadata. Do not reconstruct them from this prose.

The completed input-validation rerun uses canonical `artifacts/v3/features_final`
and `artifacts/v3/run_final`. It follows fixes for feature provenance and path
identity (including Windows dot/space aliases and resolved file identities).
No actual dataset changes were reported. Initial and intermediate runs are
preserved. All seven feature arrays and seven OOF prediction files reproduced
exactly (maximum probability difference zero). A report must still bind the
sidecar to its own feature-cache record and actual weight bytes.

## Evaluation evidence

| Completed initial endpoint | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Five-by-three nested frozen selection procedure | 0.793968253968254 | 0.7954847513984545 |
| Matched nested ConvNeXt control | 0.6885714285714286 | 0.6861276769966136 |

Sources: `reports/results/v3/nested_selected/evaluation.json` and
`reports/results/v3/nested_control/evaluation.json`, with raw predictions in
`artifacts/v3/run/selected_oof.csv` and `control_oof.csv`. Both endpoints use
3,150 images and 101 frozen source-family groups with identical outer folds.
The selector's initial 95% accuracy CI is [0.7265355511, 0.8556604537], based on
10,000 class-stratified source-group bootstrap resamples, seed 62642. Each
resampled source retains its views; point metrics are image-weighted. The CI
is conditional on fixed predictions and does not capture refitting, selection
uncertainty or correlated outer training sets. The 80% target is not achieved
by the initial point estimate and is not supported by its lower bound.

The all-development inner selection mean macro F1 of 0.8035245993060354 is an
optimization/selection score, not a new test estimate. The nested endpoint
estimates the complete frozen selection procedure, not necessarily the one
recipe fitted to all data. Broad historical data reuse also limits any claim
that the entire research process was untouched by earlier results.

The pending DINOv2 follow-up unfreezes the final two blocks, searches two learning
rates over three development folds, and uses development best-epoch selection.
It was initiated after the initial nested result and is exploratory. Its
comparison to the frozen all-data selector (0.8035245993 macro F1) cannot be
relabelled unbiased CV. Planned outputs are under `artifacts/v3/finetune`.
The fixed ten-epoch ResNet-50 comparison was declared before v3 scores; it
satisfies the third-architecture assessment only after matched-fold evaluation.
No unavailable scores or plausible training curves are supplied here.

The independently collected fresh endpoint has 21 registered new CC0 sources,
three per class, and 630 rendered images that passed model-free QC. It has not
been scored. Model choice must be explicitly frozen before fresh predictions.
Its measured result will become the primary endpoint;
it must not overwrite or retroactively justify the initial nested experiment.

## Inputs, outputs and use

Input is an RGB render with the recorded model-specific resize, center crop and
normalization. Output is a probability vector and one of the ordered labels:
`wood`, `metal`, `stone`, `concrete`, `ceramic`, `fabric`, `leather`. Use the exact
saved scaler, recipe and head; probability values are not calibrated guarantees
of correctness. Preserve model/metadata checksums and environment records.

Use for reproducible synthetic-material research, teaching and inspection of
renders from a comparable scene. The artifact is not validated for physical
composition identification, arbitrary photographs, industrial safety or
high-stakes material decisions. Real-image and out-of-distribution testing
have not been established. See `scripts/v3/predict.py --help` for explicit
artifact-based inference; fresh batch inference belongs only to the locked
endpoint protocol, not routine experimentation.

## Limitations and interpretation

Repeated renders share source texture content, so image count overstates the
number of independent examples. Known provider/generator links are grouped;
unknown derivatives and pretrained-data overlap may remain. Labels describe
appearance-based archive categories. Ceramic includes mosaic/terrazzo-like
surfaces and concrete includes plaster/painted-wall-like surfaces. Restricted
fresh definitions alter the evaluated population. The historic metallic shader
convention and shared geometry/lighting can provide shortcuts.

Blender 5.1.1 is a rendering-domain caveat. A seven-image old-TRAIN EEVEE pilot
before bulk rendering visually matched framing and texture placement but had
softer ground shadows. This observation is development evidence, not a causal
explanation of performance or an independent validation. New source variation
and renderer/taxonomy changes cannot be separated by a single holdout score.

Error galleries must state their selection policy; high-confidence errors are
illustrative rather than prevalence estimates. Inspect per-source concentration,
class confusions and the actual bootstrap intervals before making general
claims. Robustness, latency and peak memory are reported only where measured;
head-fit time is not end-to-end inference latency.

## Provenance, release and historical model

All 316 original source maps are verified; source-level CC0 proof and local
release preparation are complete. Dataset-license and publication choices are
still the owner's decisions. Pretrained model terms, source-map permissions,
render authorship and the project's dataset license are distinct. See
[DATASET.md](DATASET.md) and [release proof](reports/V3_KAGGLE_RELEASE.md).

The V1 ResNet-50 recommendation and its approximately 69.76% historical test
accuracy remain archived in the original reports and result files. Those
previously consulted test results, different epoch-selection procedure and
material-ID split are not comparable to v3 nested or fresh endpoints. The V1
recommendation is not a current final deployment choice.

This is a manually maintained v3 draft. The old `scripts/update_docs.py` emits
V1 content and must not overwrite it. Final measured updates must cite the new
assembled manifest and the completed run's exact artifacts.
