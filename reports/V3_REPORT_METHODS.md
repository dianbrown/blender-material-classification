# V3 report methods and normalized input contract

This is a reusable narrative outline and generator contract, not a results report.
The initial frozen nested experiment is now complete (27 October 2026); final
validation-rerun, matched ResNet, exploratory fine-tuning and fresh evidence are
still being integrated. This document remains a method/contract reference.
The generator never searches old result folders or fills missing measurements.
Final interpretation and visual verification must use the actual experiment
artifacts in a later round. Do not copy the test fixture into a publication.

## Evidence established before experiments

The current parent `reports/V3_AUDIT.md`, `V3_PLAN.md`, and `V3_STATE.md` are the
local evidence record. The supplied archives contain 3,150 rendered PNGs, 105
local material IDs, and 30 views per ID. Three confirmed ambientCG leather
generator-family merges and the frozen conservative wood merge give **101**
source-family groups (14 wood, 12 leather, 15 in each other class). Thirty
views are correlated observations, not thirty independent materials. The old
77/14/14 material-ID train/validation/test partition does not by itself prevent
source-family leakage. Use the completed audit and frozen family registry for
these counts; 101 groups do not prove 101 unrelated physical sources.

The seven retained labels are wood, metal, stone, concrete, ceramic, fabric and
leather. Discuss mosaic/terrazzo labels under ceramic, plaster/painted-wall
labels under concrete, and the old metallic shader convention explicitly.
Retain archival labels; any fresh-data taxonomy changes need a frozen rule and
an exclusion/change ledger independent of predictions.

Historical reported values (audit evidence, not new measurements) include
ResNet-50 test accuracy 69.76%, ConvNeXt development CV accuracy 81.24%, and
ConvNeXt test accuracy 61.19%. The old folds selected epochs and the old final
recommendation consulted test metrics. Those endpoints cannot become untouched
v3 tests. The generator will not automatically reproduce these numbers.

The project authors its Blender renders and experimental code, **not** the
third-party texture maps. Completed verification covers 96 Poly Haven sources
and nine ambientCG sources: all 315 referenced maps plus one additional local map
are verified (316 total). Poly Haven uses official MD5/size comparisons and
ambientCG uses official archive-member SHA-256/size comparisons; local SHA-256
pins every file. Both providers publish CC0 terms [6, 7]. Original-source
permissions and local release proof are prepared; only the owner dataset-license
and later publication choices remain. See V3_PROVENANCE.md and
V3_KAGGLE_RELEASE.md; earlier unverified-license audit text is historical.

## Narrative outline and evidence checklist

Use the following section keys in `report_manifest.json`. Values are nonempty
plain text, with paragraph breaks allowed. The PDF escapes markup; Markdown is
published as supplied. These are editorial inputs, not machine-verified factual
assertions. Cite supporting hashed evidence paths in each section. Human review
must check that prose does not overstate the numeric artifacts.

| Key / report section | Required narrative and evidence for finalization |
| --- | --- |
| `motivation` | Seven-class task; independent-material generalization; why rendered data is useful; the predeclared 80% point-accuracy target; synthetic-domain scope. The abstract is generated from the primary endpoint's JSON, not manually typed scores. |
| `taxonomy_provenance` | Define each class and ambiguities; asset providers, URLs, licenses, local hashes, render authorship and source-group rules. Include representative class examples through hashed supplementary figures. |
| `data_audit` | Counts by class before/after cleaning, materials and independent groups; duplicates, corruption, exclusions and label review; unknown relationships; split distributions and audit evidence. Never remove difficult samples because of model errors. |
| `protocol` | Five outer/three inner source-family grouped folds, seeds, split manifest hash, selection objective, tie rules and finite candidate grid; all preprocessing fitting, selection and calibration inside training partitions. Describe fresh holdout collection and freeze times. |
| `representations_heads` | Exact timm model/weight IDs and hashes, frozen parameters, input sizes and normalization, CLS and patch pooling, register exclusion, feature dimensions, logistic solver and regularization. Report class weighting and convergence, not invented neural-network epochs. |
| `reproducibility` | Code commit, dirty-tree status, environment lock, commands/config hashes, seeds, feature cache identity, batch sizes, platform/GPU, measured runtime/memory; enumerate settings that are inapplicable to frozen extraction. |
| `comparison_ablations` | Fair same-population comparison, predeclared pooling/resolution/backbone ablations, inner selections and fold variation. Outer-fold ablations are descriptive and must not select the final procedure. Separate historical endpoints. Explain missing experiments explicitly. |
| `errors` | Correct and incorrect examples with prediction/source IDs; recurring class confusions; source-group concentration; ambiguity hypotheses distinguished from established causes. Never infer physical composition from appearance alone. |
| `robustness_compute` | Declared corruption definitions/severities and paired results, or explicit not measured; extraction versus head-fitting/inference timings, hardware, peak VRAM and model artifact size; no invented timings or curves. |
| `limitations_ethics` | Finite independent sources, remaining related-family uncertainty, label ambiguity, rendering shortcuts, provider/pretraining bias, license/attribution, missing real-world evaluation and failure modes. |
| `conclusion` | Answer the predeclared question using the primary point estimate and CI; distinguish exceeding 80% as a point estimate from confidence-bound evidence. Recommend fresh independent sources and real-image testing; no universal-performance claims. |
| `reproduction` | Exact data audit, split, feature extraction, nested selection, evaluation and report commands; explicit paths, seeds, hashes and dependencies. Say which artifacts cannot be redistributed. |

The generator supplies related work, references, endpoint status, metric tables,
per-class tables, figures, and an evidence inventory. Do not add fabricated
student names; `authors` may be omitted until names are supplied.

## Method rationale and primary sources

DINOv2 develops transferable self-supervised visual features [1]; the register
variant adds dedicated internal tokens [2]. The planned small register model is
`vit_small_patch14_reg4_dinov2.lvd142m`. Exclude register tokens from patch
statistics. The official DINOv2 linear evaluation code demonstrates CLS with
patch averaging [8]. The CLS/patch-mean/patch-standard-deviation extension and
224/336 multiscale concatenation are hypotheses to evaluate, not established
texture recognition improvements. Deep Filter Banks motivates pooling local
descriptors but uses a different representation [3]. ConvNeXt provides a
convolutional comparison [4], initially
`convnext_tiny.fb_in22k_ft_in1k` (ImageNet-22K pretraining, 1K fine-tuning).

The frozen initial grid uses logistic heads with C in {0.01, 0.1, 1, 10}. Five
outer folds estimate the complete selection procedure; three inner folds select
representation and regularization. Fit scaling on each relevant training
partition only. Any later calibration must also respect training partitions.
Selection on evaluation estimates can bias performance [5]. A deterministic,
frozen feature cache may be shared across folds only when extraction learns
nothing from the evaluated corpus and records image, weight and preprocessing
identity. The report does not itself prove the training pipeline respected this
rule: retain split/selection logs and review them.

Treat these as separate endpoints:

1. **Historical test:** previously consulted V1 evidence; contextual only.
2. **Nested existing-corpus OOF:** estimate of the frozen selection procedure;
   correlated views remain clustered by source family.
3. **Fresh holdout:** the registry now specifies 21 new CC0 sources, three per
   class, and 630 images. Rendering/integrity completion and scoring are pending.
   Audit source-family independence, freeze selection, then evaluate once.
   Restricted ceramic/concrete definitions and Blender 5.1.1 are domain caveats;
   a seven-image old-TRAIN EEVEE pilot matched framing/texture placement but had
   softer ground shadows. The pilot was development evidence before bulk rendering.
4. **All-data deployment fit:** an inference artifact after selection; no
   independent score can be attributed to the act of fitting all data. Record
   exactly which data it includes and whether a holdout remains untouched.

Report image-weighted point metrics with source-group bootstrap intervals;
record method, seed, confidence level and resample count. Unequal group sizes
can change the estimand: state whether each bootstrap replicate pools sampled
images or averages source scores. These intervals do not capture all training
variability or domain shift. Report uncertainty even if the point target fails.

## Assignment alignment and outstanding gaps

`ITRI626_Project_New.md` recommends roughly 10-15 pages excluding references and
appendices; substance determines actual length. It requests three distinct model
families on the same split. The completed frozen comparison has DINOv2 and
ConvNeXt. A fixed ten-epoch ResNet-50 baseline on the same outer folds was declared before v3 scores and is
pending execution; it resolves the third-family requirement only once evaluated.
Pooling/resolution variants and historical ResNet scores do not count as a matched
third architecture. The later two-LR/three-development-fold DINOv2 partial
fine-tune is exploratory after the initial result, not a new unbiased nested CV.

The assignment also requests training/validation loss and validation score
curves. Frozen feature extraction and a converged logistic solver are not an
epoch-trained deep classifier. Include genuine optimization histories where
recorded, explain inapplicable curves, and preserve historical curves only with
their historical labels. Never create plausible-looking curves. Dataset examples,
class counts, split distributions, learning histories, example errors, robustness
and compute evidence must be supplied before calling the final submission complete.
The generator is a report tool, not an automatic rubric-compliance certificate.

## Strict normalized input API (schema version 1)

Run from the repository root with the existing environment; no installation is
performed by the script:

```powershell
& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe `
  scripts/v3/build_report.py --results-dir reports/results/v3/ACTUAL_RUN `
  --output-dir reports/v3
```

`--results-dir` is mandatory. The only entry point is its
`report_manifest.json`; there is no directory globbing, V1 fallback or automatic
"best" model discovery. The destination must not already exist. Outputs are
`report.md`, `report.pdf`, `report_manifest.json` (exact input bytes) and
`figures/*.png` (220 dpi). Validation occurs before output creation; rendering
uses a staging directory and publishes only after both formats succeed. Root
`report.pdf` and `report.docx` are never overwritten. DOCX is deliberately not
implemented in this first round; adding it later requires the documents skill
and render-and-verify workflow.

Python API: `load_report(results_dir, allow_synthetic=False)` returns the
manifest, loaded measured endpoints and evidence hashes;
`build_report(results_dir, output_dir, *, allow_synthetic=False)` returns the
output `Path`. Contract violations raise `ReportError` (`ValueError` subclass).
CLI validation failures exit 2. Runtime dependencies: Python, matplotlib, NumPy,
ReportLab. Embedded DejaVu Sans fonts come from matplotlib. Tests also use pytest,
pypdf and PyMuPDF for document checks. No GPU is required.

### Manifest fields

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1`. |
| `evidence_kind` | `measured` or `synthetic_fixture`. Synthetic inputs require `--allow-synthetic`; every PDF page and generated plot is visibly labelled, and Markdown begins with the warning. This declaration is not a cryptographic proof of real measurement. |
| `title`, `run_id`, `created_utc`, `code_commit` | Nonempty strings identifying the actual run; use an ISO UTC timestamp and full Git SHA. |
| `authors` | Optional list of supplied nonempty names; no inferred names. |
| `class_order` | Exactly the seven unique lowercase labels in the evaluator's order; the builder never silently sorts them. |
| `primary_endpoint` | `nested_oof` or `fresh_holdout`; must be measured. Historical test cannot be primary. |
| `sections` | All twelve plain-text keys in the outline above, nonempty even when explicitly documenting missing evidence. |
| `evidence_files` | Nonempty list of `{path, sha256, role}`. Paths are relative to results-dir and cannot escape through `..` or symlinks. Hashes must match actual bytes. Include normalized summaries, prediction CSVs, audit, protocol, environment and any displayed supplemental figures. |
| `endpoints` | Exactly one entry for each `historical_test`, `nested_oof`, `fresh_holdout`. Each has `id` and `status`: `measured` or `not_available`. An unavailable endpoint requires a nonempty `reason` and cannot contain summary/prediction paths. Measured endpoints require `summary` and `predictions`, both registered in `evidence_files`. |
| `deployment_fit` | `{status: "not_available" or "fitted", description: "..."}`. A fitted artifact additionally requires `artifact`, a path in evidence_files. Never count its fit score as an independent evaluation endpoint. |
| `supplements` | Optional list of `{section, kind, path, caption}`. `section` is one of the twelve narrative keys. `kind` is `figure` (PNG/JPEG) or `table` (CSV with a header and 1-7 consistent columns). The path must be hashed evidence. Use for same-endpoint model comparisons, ablations, fold distributions, data examples, learning histories, errors, robustness and compute. Captions must state units, endpoint and provenance; no automatic cross-endpoint ranking occurs. |

A measured fresh holdout additionally requires `selection_frozen_at` and
`independence_audit` (hashed evidence path). Existing-corpus OOF predictions must
also be supplied for overlap checks. The builder rejects overlapping source-group
IDs, material IDs and image paths; human audit must still detect aliases, related assets or
test-driven selection that no report schema can infer.

### Summary JSON (one file per measured endpoint)

Required fields:

- `class_order`: identical to manifest order.
- `n_images`, `n_source_groups`: positive integers matching the CSV population.
- `confusion_matrix`: K by K nonnegative integer counts; rows true, columns
  predicted, in class order. Every class must have nonzero true support.
- `per_class`: ordered objects `{class, support, precision, recall, f1}`.
  Undefined precision for a never-predicted class uses zero, as does its F1.
- `metrics`: object mapping names to `{value: number}`. Required names:
  `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`. Optional names may
  include `macro_precision`, `macro_recall`, `ece`, `brier`, `log_loss`, `mcc`,
  `kappa`, `auroc_macro_ovr`, `average_precision_macro`. Use fractions, not
  percentages. An undefined optional value is `{value: null, reason: "..."}`;
  never NaN, Infinity or a silently substituted zero. Include metric conventions
  (e.g. Brier normalization, ECE bins and AUC missing-class policy) in methods.
- Each defined metric may have `ci: {low, high, level, unit, method,
  n_resamples, seed}`. `accuracy.ci` is required. `unit` must be `source_group`,
  `level` is a fraction, `method` describes the actual bootstrap/estimand,
  `n_resamples` is a positive integer, `seed` is an integer. Bounds must be finite
  and ordered; the point need not be inside a percentile interval.

Optional arrays:

- `calibration_bins`: nonempty-bin objects `{confidence, accuracy, count}`,
  fractional scores and positive counts summing to n_images. No empty-bin
  interpolation. If absent, the report explicitly says the diagram was not supplied.
- `selective_risk`: `{coverage, risk}` fractions. Sorted by coverage for display;
  risk refers to errors among retained observations, not all observations.

The normalized scalar and per-class values are validated against the confusion
matrix to 1e-6 tolerance and that matrix is checked against the prediction CSV.
Printed metrics use the JSON values themselves; the builder does not perform
evaluation, confidence interval estimation, calibration or model selection.
Optional metric values, bins and intervals remain evaluator responsibilities.
The literal source JSON path appears in each endpoint's metric line and captions.
Every input manifest and registered evidence file has its SHA-256 in the report.

### Prediction CSV

Required columns are `image_path,class,material_id,source_group,outer_fold,y_true,
y_pred,p_<class>` for each class in manifest order. `y_true` and `y_pred` are
zero-based integer indices into `class_order`; `class` must equal the true label.
Probabilities are finite in [0,1], sum to one within 1e-6, and agree with y_pred
argmax (ties allowed). Image paths are unique within each endpoint. A material
cannot change class or source group. For nested OOF, each source group belongs to
one nonempty outer_fold and exactly five outer folds must appear. This is not a
substitute for checking training membership and all three inner folds in logs.
The builder does not open image_path entries, which may be archived identifiers.

## Test and finalization workflow

```powershell
& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe `
  -m pytest tests/test_v3_report.py -q
```

Tests create clearly marked fixtures under pytest temporary directories. No
fixture results, PDF or plots belong in `reports/v3`. Coverage includes missing
results, synthetic opt-in, hashes, class order, grouping, probabilities, metric
alignment, endpoint semantics, output preservation and PDF/Markdown generation.

When measured outputs exist: normalize evaluator artifacts without changing
their values; freeze the manifest and hashes; check prose against evidence;
generate into a new directory; render **every** PDF page with Poppler and inspect
PNGs for clipping, overlap, missing glyphs, table breaks and caption placement.
Where Poppler is unavailable, PyMuPDF can support local developer layout checks,
but final delivery must disclose that substitution and retain human visual QA.
Check extracted PDF text against the Markdown metric lines. Re-render after
meaningful layout changes. Do not label the report final before this review.

First-round developer validation (27 October 2026): 36 focused tests passed;
Ruff checks passed. The six-page synthetic fixture was rendered with portable
Poppler 26.09.0 and all six pages were visually inspected after layout fixes.
PyMuPDF also rasterized every page and checked text bounds. These checks validate
the report machinery, not model performance or a completed v3 submission.

## References

1. Oquab et al. (2024). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
2. Darcet et al. (2024). [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).
3. Cimpoi, Maji, Kokkinos and Vedaldi (2015). [Deep filter banks for texture recognition, description, and segmentation](https://arxiv.org/abs/1507.02620).
4. Liu et al. (2022). [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545).
5. Cawley and Talbot (2010). [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html). JMLR 11:2079-2107.
6. Poly Haven. [Asset license](https://polyhaven.com/license). Verify against asset registry at release.
7. ambientCG. [License](https://docs.ambientcg.com/license/). Verify against asset registry at release.
8. Meta DINOv2 authors. [Official linear evaluation implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py). Pin the actual used commit in the experiment record.
