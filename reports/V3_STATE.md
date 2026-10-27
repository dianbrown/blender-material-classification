# V3 orchestration state

## Verified initial state

- Base commit: f6a52b8; new branch: v3/material-classification.
- Remote V2 adds planning documents only (84960c5).
- dataset.zip: 3,175 ZIP entries, 840,761,942 uncompressed bytes.
- materials.zip: 430 ZIP entries, 1,335,486,748 uncompressed bytes.
- Archive paths checked to remain inside repository before local extraction.
- Both archives extracted locally; raw archives, dataset and textures excluded
  from version control.
- GPU: RTX 4070 Laptop, 8,188 MiB; driver 591.44.
- Default Python 3.13.5; isolated .venv being prepared.
- Historical requirements resolution failed on tqdm==4.70.0; actual compatible
  versions will be captured in a dedicated v3 lock without rewriting V1 pins.
- Linear project P-PER-18; phases PER-368 through PER-373, assigned only to
  HectoLogic (50bc7339-0c0c-49f6-83a5-1c97ff8a3f2e).
- Linear get_agent_context is advertised but backend returned tool-not-found;
  use ordinary project documents and issues for durable memory.

## Active audit tasks

- v3-data-audit-20261027-r1: source/archive/protocol audit (read-only).
- v3-model-audit-20261027-r1: training/model/primary-source audit (read-only).

## Artifact contract (draft, freeze before experiments)

V3 manifest rows will include image_path, class, material_id and source_group.
Prediction rows will include image_path, class, material_id, source_group,
outer_fold, y_true, y_pred and p_<class> for each ordered class. JSON metadata
must record the class order. Evaluators require explicit prediction paths and
must never read V1 test predictions for v3 selection.

## Implementation handoff 07:12 UTC

- Parent branch commits: d6fff49 plan, c7cef2a audits, 63adb2b environment.
- Parent .venv ready, CUDA smoke and both pretrained backbone forwards passed.
- Baseline suite: 42 passed, one existing V1 scheduler-order warning (60.05s).
- Timm weights cached locally for DINOv2 small registers and ConvNeXt Tiny22K.
- Original corpus remains development in v3; no v3 performance results yet.
- Independent fresh-source holdout preparation started with no model-result access.
- Source provenance agent verifies original assets against provider metadata.

T3 task IDs share prefix
`node:delegated-task:command%3Amcp%3A432140c9-e9ff-453b-a0c6-4b2557810a8f%3Adelegate-task%3A`.
Append the following request IDs to query status (not child-thread continuation
for reviews):

| Request ID | Ownership | Worktree directory suffix |
| --- | --- | --- |
| v3-data-audit-20261027-r1 | completed read-only audit | shared |
| v3-model-audit-20261027-r1 | completed read-only audit | shared |
| v3-data-impl-20261027-r1 | data.py, audit, packaging, metadata/v3 | t3-0ec32f83 |
| v3-evaluation-impl-20261027-r1 | evaluation.py, plots.py, evaluation CLI | t3-0c43fbbb |
| v3-learning-impl-20261027-r1 | features, nested learning, inference | t3-aa1708c5 |
| v3-holdout-impl-20261027-r1 | fresh sources, strict Blender renderer | t3-c03fb685 |
| v3-report-builder-20261027-r1 | report generator and methods | discover with git worktree list |
| v3-protocol-review-20261027-r1 | independent pre-experiment review | shared read-only |
| v3-provenance-audit-20261027-r1 | provenance_verification.json and V3_PROVENANCE.md only | shared |

Worktree root: `C:/Users/dianb/.t3/worktrees/blender-material-classification/`.
Parent schedules GPU, integrates commits and owns Linear records. Linear memory
document ID: 742e8b58-e455-46f0-9dd2-8b1d7f651cec. Project P-PER-18, phases
PER-368 (Done), PER-369 through PER-373 (implementation pending).

## Latest execution state after review fixes

This section supersedes earlier progress notes. Main branch remains
`v3/material-classification`; draft PR is
https://github.com/dianbrown/blender-material-classification/pull/1 (linked to T3).

**GPU ownership:** only task `v3-gpu-experiments-20261027-r1` may execute GPU jobs.
It is a shared-checkout execution agent, using the same task-ID prefix above.
Parent must not launch concurrent training, rendering or extraction.

Completed implementation: data audit/packaging, frozen features, nested learning
and inference, metrics/plots, report generator, robustness/diagnostics, strict
holdout renderer, matched ResNet baseline, bounded DINO fine-tuning. All agents'
code integrated. Report assembly/docs agent is still active.

Initial measured frozen results (`artifacts/v3/run`, preserved):

- Primary nested selection procedure: accuracy 0.793968253968254; macro-F1
  0.7954847513984545. Class-stratified source-group bootstrap accuracy CI
  [0.7265355511136493, 0.8556604536850461], 10,000 draws.
- Matched frozen ConvNeXt control accuracy 0.6885714285714286.
- All-development selector chose DINO multiscale + ConvNeXt, C=1; development
  mean macro-F1 0.8035245993060354. This is not an untouched test score.
- Initial extraction 188 seconds; CPU nested fits 50 seconds. Raw predictions
  copied to `reports/results/v3/frozen_initial`; comprehensive evaluations in
  `reports/results/v3/nested_selected` and `nested_control`; diagnostic figures
  in `reports/figures/v3/nested_diagnostics`.

Review fixes: 7ed06ff resolves prediction `outer_fold` and mixed-class report
validation; 91eac7d rejects Windows dot/space aliases and resolved physical-file
duplicates. Actual canonical corpus was unaffected. Because feature code hashes
changed, preserve earlier caches/runs and use the new `features_final` /
`run_final` for final deployment artifacts. `run_validated` is an intermediate
completed rerun, not the canonical final run.

`features_final` completed with identity
`33a7e56080d09232e36983aee287fbd7f4583e4745d2bf62d92b028e40f94ac4`.
The GPU agent is executing: fresh holdout retry + QC, `run_final` nested heads,
`resnet-oof` five fixed ten-epoch folds, `resnet-final` ten-epoch all-data fit.

Holdout: 21 source assets, 3 per class, 630 planned images. All 63 source maps
verified. Seven old TRAIN pilot images rendered and visually inspected: framing
and texture placement match, ground shadows are softer; report renderer/domain
caveat. First bulk launch failed before images due CRLF/LF source-audit hash.
3b07616 accepts only newline-equivalent textual fingerprints, preserving frozen
config/registry values and strict binary hashes; 50 holdout tests passed.
Retry log: `artifacts/v3/holdout-render-retry.log`.
**No fresh holdout predictions/evaluation have run.**

Supplemental ResNet was declared before any V3 scores were read, to satisfy
three architecture comparison. DINO partial fine-tuning was declared after the
79.4% result as an explicitly exploratory development follow-up (two learning
rates, three grouped folds, max ten epochs). Code 294729d; 19 specific tests pass.
Do not call its best-validation CV unbiased. GPU owner has NOT yet been told to
run fine-tuning; wait for `v3-deep-training-review-20261027-r1`, fix findings,
then steer the GPU owner to execute select/final stages using `run_final`.
Parent must freeze final candidate choice before opening holdout scores.

Active non-GPU tasks:

- `v3-report-assembly-20261027-r1`: assembler + README/MODEL_CARD/DATASET refresh.
- `v3-deep-training-review-20261027-r1`: ResNet/fine-tune correctness review.

Source provenance complete: all 316 local maps verified against official CC0
provider files/hashes. Commit bf1e4df integrates source and render-lineage
evidence; 52 packaging tests pass, one Windows symlink skip. Staged ZIP:
`artifacts/v3/release-staging/original-corpus-verified-audit-20261027.zip`,
SHA256 `0b19b667d48150c9b272a28c26a685b6519c96fd2830a9922e5aaad882f39681`.
Only user dataset-license/publication decisions remain; do not upload yet.

Outstanding: complete GPU experiments, lock model choice, score fresh holdout
once for fixed candidates, run frozen corruptions, evaluate all endpoints and
paired comparisons, build and visually verify final report, finalize docs,
final independent review/tests, update PR and Linear. Never mark complete merely
because a point estimate or the first implementation exists.

## Latest neural execution handoff

Core code is now at 28cb026 (report assembly) with reviewed neural fixes
44aaaaf. The ResNet/DINO fixes cover AMP overflow recovery and update accounting,
fresh CSV fold markers, decoded-pixel overlap for DINO and material-ID overlap
for ResNet. Seventy-seven integrated neural/report-assembly tests passed.

Previous GPU execution task completed: holdout rendering 630/21 and QC passed;
`features_final` and `run_final` completed. All seven feature arrays and seven
OOF prediction CSVs exactly reproduce the initial run (max probability
difference 0). See `reports/results/v3/gpu_execution_summary.json` and
`artifacts/v3/guard_revalidation_comparison.json`.

**New sole GPU owner:** `v3-neural-gpu-execution-20261027-r2` (same full task-ID
prefix). It has started sequential ResNet OOF/final, then DINO select/final.
Do not launch another GPU job. It must finish before fresh scoring. Reviewed
code/configs must not change under these runs. GPU agent reports progress and
completion automatically.

The clean renderer manifest lacks `source_group` by design; use the derived
`metadata/v3_holdout/classification_manifest.csv` for every model/classifier
command. Its 630 rows are verified against the sealed registry, passed QC and
image hashes; sidecar records canonical source hashes. Use data-root
`artifacts/v3/holdout_dataset`. Both renderer and classifier manifests are
preserved separately. Three handoff integrity tests passed.

All five corruption datasets are staged, not scored, at
`artifacts/v3/robustness/fresh_holdout`. Staging identity:
`e1574a95f957683689f8002c0c6cf287bb2a897bee055f91be0dd287ef8698b3`.
Frozen-model corruption evaluator is implemented. If DINO fine-tune becomes
the final selected model, add an adapter to score the same staged corruptions
with its fixed checkpoint; do not introduce new conditions.

Report assembler integrated (`scripts/v3/assemble_report.py`) plus draft cards.
Read README input-spec example when preparing final evidence. It normalizes
published text to UTF-8/LF before evidence hashing, preserving original and
staged hashes. `scripts/v3/build_report.py` creates PDF/Markdown from assembled
evidence, but final actual report has not yet been generated. Old report files
remain unchanged. Dataset release package is ready locally, pending only the
user's eventual license/publication choices.

The full suite previously reported 292 pass / one Windows skip / one assertion
wording failure (aliased vs aliases); that wording was corrected and all27
robustness tests then passed. A full suite rerun is active after final neural
and assembler integration. Do not report a full-suite total until it completes.

Once neural execution completes: freeze primary candidate on the prescribed
development comparison; record timestamp and exact artifact hashes BEFORE
holdout predictions; score locked frozen selected/control, fixed ResNet and
fine-tuned DINO once on the fresh classifier manifest; retain all results.
Primary cannot be swapped based on the fresh ranking. Evaluate CIs/paired
differences, run fixed corruptions, finalize report/cards and visual QA,
independent final claim review, full checks and PR/Linear completion.

Full-suite rerun completed: **344 passed, 1 Windows symlink-permission skip,
1 existing V1 scheduler warning**, 104.47 seconds. Record:
`reports/results/v3/integrated_validation.json`. Real CPU image inference passed.
Neural GPU owner reports ResNet first two epochs ~34 seconds each; long runs are
active. Report-preview task `v3-report-preview-20261027-r1` builds/visually checks
an internal actual-data PDF while training proceeds; it may return report-only
fix commits. No final report is available yet.

Additional CPU-only task `v3-neural-robustness-20261027-r1` implements corruption
evaluation adapters for ResNet/DINO without changing any active training code.
It owns only new neural_robustness module/CLI/tests. Proposed final lock schema:
schema_version=1; frozen_at_utc; primary_model; class_order; selection_rule;
models mapping names to artifact_path/artifact_sha256/metadata_path/
metadata_sha256/role; holdout containing manifest_sha256, n_images=630,
source_group_count=21. Create the actual lock only after all development choices
are complete and BEFORE fresh inference. This adapter requires that lock and
matches each clean prediction run to the named model and holdout manifest.

GPU milestone: ResNet outer fold0 completed ten epochs plus final-only
assessment in438.37 seconds; fold1 running. Latest fold peak allocated/reserved
GPU memory443371008/526385152 bytes, no skipped updates. Keep the fixed recipe.

Neural robustness adapter integrated at 2b2c9c8; all 71 adapter CPU tests pass
in the parent checkout. CLI: `scripts/v3/neural_robustness.py`; use family
`resnet` or `dinov2`, named model in the pre-score endpoint lock, actual clean
prediction CSV/run, existing staging directory and a NEW output directory.
Relative artifact paths in the lock resolve against the lock file's directory.
The lock's holdout object must contain exactly manifest_sha256, n_images and
source_group_count. Keep the prescribed 630/21 values. This adapter never fits
models and rejects incomplete outputs rather than resuming them.

Latest GPU milestone: ResNet 2/5 outer folds complete, 20/50 fixed epochs;
fold1 took473.59 seconds and fold2 started. No failures/skipped updates and no
fresh scores. Do not adjust epochs or select a checkpoint from outer results.

Internal measured report preview complete, and report-only layout fixes merged
as46450f3. Parent reran report/assembly tests:75 pass. Preview:
`artifacts/v3/report_preview/report.pdf` (21pages; body through14), with
`qa/verification.json` recording all-page rendered inspection. Assembly spec:
`artifacts/v3/report_preview_inputs/assembly.json`. Agent inspected all21pages;
parent additionally inspected pages1,6,11. It remains preliminary and fresh
endpoint unavailable; never deliver it as the completed final report.

For final editorial pass: replace preliminary title/footer and pending prose
only when actual final evidence exists. Prefer human-readable metric labels
(Macro F1, Accuracy) and endpoint names over raw keys such as nested_oof or
not_available. Move repeated raw evidence paths out of abstract/main prose into
captions/appendix references where possible, retaining full hashed inventory.
Main narrative should interpret measured errors/uncertainty and architecture
tradeoffs, not read like a tool status log. Include final neural histories,
three-architecture comparison, fresh endpoint and fixed corruption results.

## Matched ResNet benchmark completed

GPU owner reports all5fixed10-epoch ResNet folds completed in2262.88seconds.
Accuracy0.7155555555555555; macro-F1 0.7135009154776019. Coverage3150 unique
images, exact primary fold map, class/group constraints and probabilities all
verified. Raw snapshots and five histories: reports/results/v3/resnet_baseline.
Comprehensive evaluation: reports/results/v3/nested_resnet (standard evaluator
scope includes grouped OOF; label this fixed-policy baseline, not nested model
selection). Paired primary-minus-ResNet accuracy difference0.0784126984, 95%
family-bootstrap interval[0.0146397786,0.1414450879]; saved paired JSON.
ResNet final all-development fit running, latest milestone6/10epochs; DINO
selection/final follows. Fresh holdout still unscored; no model choice changes.

CPU-only task v3-inference-benchmark-20261027-r1 implements NEW benchmark
module/CLI/tests for latency, throughput, VRAM and actual weight footprints.
It must not touch active training code. Actual GPU benchmarking waits for the
GPU owner to finish and final inference phase. In report, distinguish a frozen
classifier's ~311KB head artifact from its required ~202MB pretrained backbone
files; use measured file sizes/counts, not this approximate note. The same DINO
weight file at two resolutions counts once as stored weights, while runtime
instances may occupy separate memory. Neural loaders may also require cached
pretrained files; report measured dependency requirements honestly.

ResNet final all-development fit completed successfully in576.59seconds,
tenepochs on3150images. Export: artifacts/v3/resnet-final/final/model.pt and
model.json; run identity
cf91792f1b0d36d85cf7ba0b160e3a7ee61aed800c2f80d6e3d2dd6bd42c3990.
Training source/config hashes unchanged. Sole GPU owner has now started DINO
development selection: exactly2learning rates x3folds, physical batch8,
accumulation4, four CPU threads. No fresh holdout access. Wait for complete
selection/final-fit results; do not choose from intermediate validation peaks.

Inference benchmarking integrated at63a4360; 16 CPU-only tests passed in parent.
CLI scripts/v3/benchmark.py takes --family frozen|resnet|dinov2, --artifact,
--development-manifest metadata/v3/manifest.csv, --data-root ROOT, --output NEW,
--device cuda, --cpu-threads4, --batches1 16, --warmup5, --repetitions30 and
optional --cold-repetitions3 (process-warm reload, not OS-cold startup).
Run only after the training GPU owner releases the device. Warm timings include
preprocessing/model/probabilities but exclude disk read/decode. It reports
actual unique weight files and separates head-only artifacts from backbones.
The benchmark author confirmed the current fine-tuned DINO deployment loader
does NOT require the original pretrained cache; earlier uncertainty is resolved.

DINO milestone: first of six development fits (LR3e-6, fold0) completed at
epoch6 under fixed patience3; best epoch3, development macro-F1 0.8356045851.
No failures/AMP skips. This is one selected development fold, not a complete
comparison or test result. GPU owner continues remaining five fits unchanged.


## Completed locked evaluation and release staging — 2026-10-27
All GPU work is complete and the GPU is released. Full final_model_lock.json was committed at 909b81e before any fresh predictions (lock SHA256 4a5c02f5d9fc6c38b5de227db5520c89a29c0d75da94b689abc4e9f41f88d0bf). Primary remains frozen_selected. Fresh 630 images / 21 source groups: primary accuracy 80.1587%, macro-F1 .795203; frozen control 75.5556%/.756564; fixed ResNet 77.6190%/.766481; exploratory fine-tuned DINO 86.8254%/.863071. Primary accuracy source-group bootstrap 95% CI 66.03–93.65%; DINO 76.98–95.87%. The point target is achieved; population ≥80% and universal robustness are not established. Paired comparison intervals include zero.

All 14 final commands passed: four clean predictors, four comprehensive saved-prediction evaluations, primary corruption analysis, four development-only benchmarks, and an explicitly post hoc DINO corruption extension (amendment 249f3a7). No scoring reruns or model changes. Primary JPEG30 accuracy 59.05% (21.11pp drop); DINO 75.40%. Full evidence: reports/results/v3/final_execution_summary.json and fresh/, robustness/, benchmarks/. All model/checkpoint/dependency hashes unchanged. Fixed ResNet and DINO selection/final fits completed without OOM/failures/AMP skips; separate neural_execution_summary.json retains timing and launch-only error.

Combined local study audit ZIP now verified: artifacts/v3/release-staging/material-classification-v3-study-audit.zip, 1,005,048,843 bytes, SHA256 4f2bd057d66493ada1cc4b9d69d7932bcdfcb52798cb5aaf6f2887442a58d1da. 3,780 images, 126 material IDs, 122 known groups; 3,150 development / 630 test. Packaged copies remove private PNG metadata with identical decoded pixels and separate source/package hashes. All entries pass CRC/SHA checks. License unset, release_approved=false; no Kaggle upload. Code integrated as 9d606db.

Active final tasks: v3-final-report-20261027-r1 (PDF/Markdown/README/cards with all-page visual QA), v3-results-notebook-20261027-r1 (actually executed saved-results notebook), v3-final-spec-review-20261027-r1 and v3-final-standards-review-20261027-r1 (read-only through 9d606db). Final report/notebook integration, final tests/review, PR body/readiness and tracking closure remain. PR https://github.com/dianbrown/blender-material-classification/pull/1 remains draft and linked. No further model training or endpoint selection is planned.

Both final implementation reviews completed with no actionable findings; record
reports/V3_FINAL_REVIEW.md. Full local suite completed: 494 passed, two Windows
symlink-permission skips, one historical mocked-scheduler warning, 149.24 seconds.
The suite began before the provenance portability patch; its focused rerun passed
53 tests/one skip including LF and CRLF input plus content-change rejection.
GitHub CI previously failed two provenance tests on LF checkout against a CRLF
pin; fixed in e049381 with text-only bidirectional equivalence. No frozen bytes
or model dependencies changed. Latest pushed commit e049381 has CI pending;
local fa934d5 records the independent reviews. Report and notebook agents remain
active; final artifact QA/integration remains. Interactive results summary HTML
is staged in artifacts/v3/results-summary.html, previewed at widths728 and360;
publish with html_render only at final delivery. Linear PER-371 is Done,
PER-373 In Progress, all assigned only to HectoLogic.

Results notebook integrated as8882c69, text encoding corrected and all cells
re-executed in parent asfb06687: notebooks/05_v3_results.ipynb has30cells,
16executed code cells,11plots,61evidence hashes and no error outputs; nbformat
and Ruff pass. Optional notebook pins in requirements-notebook.txt preserve
the ML lock. Linux CI on e049381 passed in2m47s, resolving provenance portability.
Final report task remains active. Do not deliver the preliminary preview as final.

## Final delivery — 2026-10-27

Final report agent completed; source/docs integrated f5636f0 and measured
publication bundle aab557a. reports/v3/report.pdf is final:21pages,17body,
1references,3appendices; all pages rendered/visually checked,153evidencefiles
validated. Parent rechecked evidence loader and PDF/Markdownhashes and inspected
pages1,10,14,21. Notebook remains fully executed. Final claims review
v3-final-claims-review-20261027-r1 reports no actionable findings,61notebook
digests and112sourceinventoryentries verified. Review report V3_FINAL_REVIEW.md.

Final full local tests:499passed2Windows-symlinkskips1legacy mocked warning,
112.40seconds. Ruff anddiffchecks pass. Latest model/report artifacts unchanged;
GPU idle. Final PR description written; final GitHub checks pending at handoff
record time. PR #1 is to be marked ready and left open for user review, not
merged. Dataset publication remains deferred exactly as requested: complete
local studyZIP exists, licenseunset,release_approved=false,noKaggleupload.

Primary80.16%accuracyCI66.03-93.65%; exploratorycomparisonDINO86.83%CI76.98-95.87%.
ReportPDF,Markdown,notebook,README,model/datacards andevidencearecommitted.
Localcheckpointpaths remain inMODEL_CARD.md andfinal_model_lock.json; no public
weightsURLinvented. Inline results-summaryHTML inartifacts/v3 ispreviewedand
ready toshowatfinaldelivery. No active delegated tasks remain.

