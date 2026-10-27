# Blender Material Classification

Seven-class classification of synthetic Blender PBR renders, with source-family
separation and an auditable path from raw predictions to the report. **V3 work is
ongoing; the final fresh-source evaluation and final report are pending.** The
completed initial frozen experiment is documented below, not presented as the
unknown future holdout result.

## Verified status (27 October 2026)

- The existing corpus contains **3,150 images, 105 local materials and 101 frozen
  source-family groups**. Thirty views of a material are correlated observations.
- All **316 local source maps** have verified provenance: 315 manifest-referenced
  maps plus one additional verified displacement map. The original 105 sources
  no longer have unknown source permissions. The local release ZIP and source
  proof are prepared; the owner's dataset-license choice and later publication
  decision remain pending. Nothing has been uploaded. See
  [release preparation](reports/V3_KAGGLE_RELEASE.md) and [DATASET.md](DATASET.md).
- Five-by-three nested source-group CV for the frozen DINOv2/ConvNeXt feature
  selector and matched ConvNeXt control is complete. Input-validation fixes are
  being rerun into `artifacts/v3/features_final` and `artifacts/v3/run_final`.
  The initial `run` and intermediate `run_validated` are preserved. Equality
  against the initial predictions must be verified; it is not assumed.
- A fixed ten-epoch ResNet-50 baseline on the same outer folds was declared
  before v3 scores. Its GPU training is pending. It supplies the third assessed
  architecture only when its matched predictions exist.
- Partial DINOv2 fine-tuning (last two blocks; two learning rates, three development
  folds) is a bounded **exploratory follow-up after the initial result**. It is
  pending and does not create a new unbiased nested-CV estimate.
- The fresh registry specifies **21 new CC0 sources, three per class, and 630
  images**. The fresh endpoint has not been scored. Complete the render/identity
  checks and lock the model choice before predictions; the fresh endpoint becomes
  primary when its independently audited results are available.

## Completed initial frozen experiment

The following values are read from the initial evaluator JSONs, not copied from
historical V1 reports or inferred from the all-data model fit.

| Initial existing-corpus endpoint | Accuracy | Macro F1 | Evidence |
| --- | ---: | ---: | --- |
| Nested frozen selection procedure | 0.793968253968254 | 0.7954847513984545 | `reports/results/v3/nested_selected/evaluation.json` |
| Matched ConvNeXt control | 0.6885714285714286 | 0.6861276769966136 | `reports/results/v3/nested_control/evaluation.json` |

Both cover the same 3,150 images and 101 source groups with the same outer folds.
The initial selector's 95% accuracy interval is **[0.7265355511, 0.8556604537]**,
from 10,000 class-stratified source-group bootstrap draws (seed 62642). These
intervals resample fixed OOF predictions; they do not include refitting,
selection uncertainty, or dependence between CV training sets. Neither the
point estimate nor its lower bound establishes the 80% target.

The all-data grouped selector chose `dino_multiscale_convnext`, C=1, with mean
inner macro F1 0.8035245993060354. That is a **development selection score**, not
an independent test score. The exploratory fine-tuning comparison uses that
three-fold development reference. It must not be compared to the nested score
as if both were independent tests.

## Method and interpretation

The frozen grid in [protocol.json](configs/v3/protocol.json) uses DINOv2 small
with registers at 224/336 pixels and ConvNeXt-Tiny at 224 pixels. DINO variants
use CLS, CLS plus patch mean, or CLS/patch mean/std at both resolutions; one
candidate concatenates DINO and convolutional features. Register tokens are
excluded from patch statistics. The backbone weights remain fixed. Standard
scaling and class-balanced logistic heads are fitted inside each relevant
training partition. Three inner folds select the representation and C from
{0.01, 0.1, 1, 10}; five outer folds estimate the full selection procedure.

DINOv2 supplies transferable self-supervised representations
([Oquab et al.](https://arxiv.org/abs/2304.07193)); ConvNeXt supplies a modern
convolutional comparison ([Liu et al.](https://arxiv.org/abs/2201.03545)). These
papers do not establish performance on this corpus. Separating selection from
assessment addresses the selection-bias problem described by
[Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html). The source-group
bootstrap adapts the resampling principle introduced by
[Efron](https://doi.org/10.1214/aos/1176344552) to correlated views; the reported
interval remains conditional on the saved predictions.

The old train/val/test directories are retained for lineage, but all existing
images are development data for v3. Known leather-generator and wood-family
relationships can cross those old directory splits. Source-family grouping is
therefore stricter than material-ID grouping. Ceramic and concrete retain broad
archival labels (including mosaic/terrazzo and plaster/painted-wall appearances).
The fresh collection uses restricted definitions. Blender 5.1.1 also introduces
a rendering-domain caveat: a seven-image old-TRAIN EEVEE pilot visually matched
framing and texture placement but showed softer ground shadows. That pilot was
development evidence before bulk rendering, not an independent endpoint.

## Reproduction

Use Python 3.13 and the project environment; retain the actual runtime package
versions, CUDA/hardware record, code commit, protocol hash and weight hashes.
The v3 feature runtime also requires `timm`, `safetensors`, `huggingface-hub`,
`joblib` and `threadpoolctl`. Reporting requires matplotlib; the separate PDF
builder also requires ReportLab. `environment.json` and cache/run metadata
record the executed environment; do not assume the older V1 dependency lock
exactly describes it. Run from the repository root with local images and source
maps placed as documented in [DATASET.md](DATASET.md).

The following is a reproduction template. Use **new** output names; existing
caches, runs and evidence are immutable. Do not launch it over an active GPU run.

```powershell
python scripts/v3/audit_data.py --data-root . --output-dir metadata/v3_reproduction --report reports/results/v3/audit_reproduction.json
python scripts/v3/extract_features.py --data-root . --manifest metadata/v3/manifest.csv --protocol configs/v3/protocol.json --output artifacts/v3/features_reproduction
python scripts/v3/train.py --data-root . --manifest metadata/v3/manifest.csv --feature-cache artifacts/v3/features_reproduction --protocol configs/v3/protocol.json --output artifacts/v3/run_reproduction
python scripts/v3/evaluate.py --predictions artifacts/v3/run_reproduction/selected_oof.csv --metadata configs/v3/protocol.json --output-dir reports/results/v3/nested_reproduction --n-bootstrap 10000 --seed 62642
python scripts/v3/evaluate.py --predictions artifacts/v3/run_reproduction/control_oof.csv --metadata configs/v3/protocol.json --output-dir reports/results/v3/control_reproduction --n-bootstrap 10000 --seed 62642
```

Evaluate each explicitly named `candidate_oof/*.csv` with the same evaluator if
its descriptive ablation is to appear in the comparison table. Never select a
new candidate from the outer-fold results. The fixed ResNet configuration is
`configs/v3/resnet_baseline.json`; its planned output directories are
`artifacts/v3/resnet-oof` and `artifacts/v3/resnet-final`. Exploratory fine-tuning
uses `configs/v3/finetune.json` and planned `artifacts/v3/finetune`. Consult the
respective CLI `--help` and frozen configurations before execution. No fresh
scoring command is part of this development reproduction sequence.

## Assemble report evidence

`assemble_report.py` consumes explicitly named artifacts, copies evidence
with SHA-256 hashes, normalizes evaluator JSON and prediction indices, validates
the bundle with the existing builder, and publishes to a **new directory**.
It performs no training, model choice, result discovery or holdout access unless
those exact input paths are supplied. It generates no PDF. Existing `report.pdf`,
`report.docx`, V1 files and previous v3 results remain unchanged.

```powershell
python scripts/v3/assemble_report.py --spec C:/path/to/assembly.json --output-dir reports/results/v3/report_evidence_final
python scripts/v3/build_report.py --results-dir reports/results/v3/report_evidence_final --output-dir reports/v3_final
python -m pytest tests/test_v3_report_assembly.py -q
```

Python API: `assemble_report(spec_path, output_dir, *, allow_synthetic=False)`
returns the published `Path`. Contract failures raise `ReportError` and leave
the destination absent; CLI errors exit 2. Synthetic fixtures require explicit
opt-in. No GPU is used. The builder contract and final PDF visual-QA requirements
are in [V3_REPORT_METHODS.md](reports/V3_REPORT_METHODS.md).

### Assembly specification, version 1

All source paths are absolute or relative to the **spec file's directory**.
There is no automatic newest-run or best-model selection. This minimal example
uses the completed initial evidence and deliberately keeps its status explicit;
replace paths with verified final-run artifacts only after their evaluation.
Save it in the repository root, and replace `code_commit` with the full actual
experiment commit before use.

```json
{
  "schema_version": 1,
  "evidence_kind": "measured",
  "title": "V3 material classification: initial frozen evidence",
  "run_id": "initial-frozen-v3",
  "code_commit": "REPLACE_WITH_ACTUAL_EXPERIMENT_COMMIT",
  "protocol": "configs/v3/protocol.json",
  "environment": "reports/results/v3/environment.json",
  "audit": "reports/results/v3/data_audit.json",
  "feature_record": "artifacts/v3/features/cache.json",
  "run": "artifacts/v3/run/run.json",
  "model_sidecar": "artifacts/v3/run/model_metadata.json",
  "weights": "artifacts/v3/run/model.joblib",
  "search_history": "artifacts/v3/run/search_history.csv",
  "evaluations": [
    {
      "id": "selected",
      "label": "Initial frozen selection procedure",
      "scope": "nested_oof",
      "endpoint": "nested_oof",
      "evaluation": "reports/results/v3/nested_selected/evaluation.json",
      "predictions": "artifacts/v3/run/selected_oof.csv",
      "label_format": "indices",
      "description": "Initial predeclared five-by-three source-group nested selection."
    },
    {
      "id": "control",
      "label": "Matched ConvNeXt control",
      "scope": "nested_oof",
      "evaluation": "reports/results/v3/nested_control/evaluation.json",
      "predictions": "artifacts/v3/run/control_oof.csv",
      "label_format": "indices",
      "description": "Independent inner C selection on identical outer folds."
    }
  ],
  "unavailable_endpoints": {
    "historical_test": "V1 was previously consulted; preserved separately.",
    "fresh_holdout": "Not scored; independent endpoint pending."
  },
  "evidence": [
    {"path": "reports/results/v3/provenance_verification.json", "role": "Original-source verification"},
    {"path": "reports/V3_KAGGLE_RELEASE.md", "role": "Prepared release and remaining owner decisions"}
  ]
}
```

Required keys are shown above except `evidence` is optional. `authors` is an
optional list of supplied names. `copy_weights: true` copies verified weights;
by default the bundle stores a verified path/hash/byte-size reference and copies
the sidecar, without deserializing weights. `feature_record` binds the sidecar's
feature-implementation hash and the training run to the actual cache identity.
It does not require that the old run used today's implementation. Final rerun
comparability must be established separately. Class order comes from the
protocol and must agree with evaluations and sidecar.

Each evaluation requires `id` (lowercase letters/digits/underscores), `label`,
`description`, `scope`, `evaluation`, `predictions`, and explicit `label_format`
(`indices` or `names`). JSON must be the `evaluate.py` schema with completed
source-group bootstrap CIs; missing/undefined numbers are never invented.
`scope` is `nested_oof`, `fresh_holdout`, or `exploratory_development`. Exactly
one nested evaluation has `endpoint: "nested_oof"`. Additional matched ResNet
or candidate evaluations omit `endpoint`. Nested comparison populations and
fold assignments must match exactly. Three-fold exploratory evaluation stays
in a separate table and cannot become a report endpoint. Fine-tune selection
summaries without evaluator JSON belong in explicit evidence/supplements.

For a future primary fresh result, add an evaluation with
`scope` and `endpoint` both `fresh_holdout`, plus `independence_audit` (file path),
`selection_record` (JSON path), and timezone-aware `prediction_started_at`.
The selection record must contain `model_sha256` matching the deployment weights
and `selection_frozen_at`, strictly earlier than prediction start. The builder
checks image/material/group overlap with the nested population; the audit must
also address source aliases and family relationships. A supplied measured fresh
endpoint automatically becomes primary. All missing endpoint reasons are
mandatory in `unavailable_endpoints`; historical V1 remains separate.

Optional inputs:

- `training_histories`: entries with `path` (CSV or JSON), `x` (for example `epoch`),
  `caption` stating experiment/endpoint, and `plots`, each with `ys` (actual
  column names) and `ylabel` including units. JSON histories additionally require
  `rows_key`, e.g. `history`; dotted fields such as `train.loss` and
  `validation_development_only.macro_f1` select actual nested values. No validation or epoch curves are
  inferred when absent. Example: `{"path":"history.csv","x":"epoch",
  "caption":"ResNet outer fold 0: training only; fixed epoch budget",
  "plots":[{"ys":["train_loss"],"ylabel":"Training cross entropy"}]}`.
- `supplements`: `{path, section, kind, caption}` for explicit diagnostic
  PNG/JPEG figures or CSV tables (header, 1-7 columns), as required by the builder.
  Include endpoint, units, selection policy and source IDs in gallery captions.
  Use `scripts/v3/diagnostics.py` for real-image galleries; assembly never opens
  paths embedded in predictions. Fine-tuning results, robustness and external
  compute summaries can be supplied this way without fabricating measurements.
- `json_tables`: explicit JSON-to-table mappings with `path`, `rows_key` (array),
  `columns` (1-7 output names mapped to dotted source fields), `section` and
  `caption`. For fine-tune selection use `rows_key: "fits"` and the recorded
  learning-rate/fold/best-epoch/development-score fields. These tables do not
  promote selection scores to independent endpoints.
- `deployment_model`: optional `{sidecar, weights, description}` for a later
  model explicitly chosen by the parent. The sidecar requires `class_order` and
  `model_sha256`. This changes the deployment reference and future lock check,
  not the frozen nested run or its provenance. The assembler makes no choice.
- `evidence`: `{path, role}` for extra fold assignments, candidate OOF files,
  experiment histories, configurations, release proof or compute records.
- `sections`: overrides for any of the twelve builder narrative keys. Review
  these against copied evidence before publication. Defaults provide substantial
  methods and limitations but do not certify the final submission as complete.

Actual C curves come from every recorded inner fold at each declared C, grouped
by stage and outer fold. Fold charts use checked evaluator fold summaries.
Dataset tables come from validated predictions and agree with the audit.
Environment, extraction-job compute, run time and model size retain their scopes
and units. Known text inputs are staged as UTF-8 without BOM and LF before hashing;
images and weights retain their bytes. Raw run-output checks use original
artifact hashes. The inventory records both `source_sha256` (original) and
`sha256` (staged), so Git LF normalization cannot invalidate the report hashes.
Every staged input and derived table/figure is hashed. The output
`derived/source_inventory.json` preserves original paths and hashes; `report_manifest.json`
is directly accepted by the existing builder. Do not publish a generated PDF
until every page and caption has passed visual review.

## Historical V1 and retained deliverables

Root [report.pdf](report.pdf) and [report.docx](report.docx), historical training
curves, [comparison CSV](reports/results/model_comparison.csv), and old
[recommendation](reports/results/recommended_model.json) are preserved V1
artifacts. The historical recommendation was ResNet-50 (test accuracy about
69.76%). Its test had been consulted, epoch selection and populations differ,
and some source families crossed the old material-ID splits. It is **not a
matched third v3 architecture, an untouched v3 endpoint, or the current final
model choice**. Old `scripts/update_docs.py` generates V1 documentation; do not
run it over these draft v3 cards.

See [MODEL_CARD.md](MODEL_CARD.md) for artifact-specific use and limitations and
[DATASET.md](DATASET.md) for provenance and release status. Real-photo validation,
synthetic-to-real transfer and production deployment remain unestablished.
