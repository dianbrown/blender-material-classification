# V2 implementation plan: unseen-material generalisation

Status: design only. This document is an implementation specification; no
Optuna package has been installed, no V2 trial has run, and no V1 file is to be
changed by following it.

## 1. Goal and decision

The goal is a model that generalises to **new Poly Haven material assets**, not
one that merely achieves a higher score on the current validation images. V1
already provides a defensible assignment result and must remain reproducible:

| model | V1 TEST accuracy | V1 TEST Macro-F1 | grouped development CV Macro-F1 | V1 training time |
|---|---:|---:|---:|---:|
| CustomCNN | 0.498 | 0.490 | not run | 396 s |
| ResNet-50 | 0.698 | 0.683 | 0.717 ± 0.109 | 557 s |
| ConvNeXt-Tiny | 0.612 | 0.569 | 0.787 ± 0.113 | 905 s |
| ViT-B/16 | 0.600 | 0.562 | 0.745 ± 0.103 | 1,511 s |

The current ResNet training Macro-F1 is 0.9996 while its TEST Macro-F1 is
0.683. ConvNeXt's development-CV score is higher but its V1 TEST score is
lower. This reversal and the roughly 0.10 fold standard deviations show why a
single validation score is not an adequate V2 target. Regularisation and
augmentation may help, but a gain is not guaranteed.

### Recommended protocol: new holdout plus grouped development search

The recommended practical protocol is to add a genuinely new, untouched
material holdout and use **all 105 archived V1 material groups** for V2
development-only grouped CV and Optuna search. This deliberately changes the
V2 development boundary: the old V1 TEST groups are no longer held out in V2,
but their old TEST metrics/predictions remain sealed and are never used by the
search.

1. Preserve the original 105-material dataset, manifests, V1 checkpoints,
   methodology lock, TEST predictions, and report metrics unchanged.
2. Obtain/render **five new Poly Haven assets per class** (35 material groups,
   1,050 images), with source slugs not present in V1. Three per class (21
   groups) is the hard minimum if time or asset availability prevents five.
   The holdout is never visible to Optuna, augmentation selection, early
   stopping, or hard-negative design.
3. Treat all 105 archived V1 materials as V2 development data. Run five-fold
   `StratifiedGroupKFold` by `material_id` for the search and confirmation.
   Every image from one material stays in one fold.
4. Select a configuration on development folds, fit it on all 105 development
   groups, then evaluate once on the 35 new groups. Evaluate a predeclared V1
   baseline configuration (retrained from fresh weights on the same 105
   development groups) on the same new holdout for a paired comparison. Never
   evaluate either V2 model against the old V1 TEST split.

This is more useful for the stated scientific question than carving another
holdout out of the same 105 assets. It uses all archived data without leaking
the new-material endpoint. A full nested CV design remains the gold-standard
alternative if compute and time allow: 5 outer grouped folds, 3 inner grouped
folds, and all hyperparameter choices made inside each outer training portion.
It is substantially more expensive and is optional for this project.

If new materials cannot be cleared and rendered, use a new grouped-CV
implementation based on this plan (the older
`reports/OPTUNA_V2_PLAN.md` is a historical proposal and still describes the
91-group boundary). Label any no-new-holdout result development-only. Do not
promote a tuned score on the old V1 validation or TEST results as evidence of
new-material generalisation.

## 2. V1 protection and release boundaries

The V1 lock at `reports/results/methodology_lock.json` records the canonical
configs, manifests, source hashes, and V1 checkpoints. In V2, all 105 archived
groups (including the 14 old V1 TEST groups) are development groups; the old
TEST is therefore not an untouched V2 endpoint. It remains an immutable V1
archive and its metrics/predictions must never be opened by Optuna. V2 must use
new files and names:

- Never read old V1 TEST predictions, TEST confusion matrices, TEST error
  galleries, or TEST per-class metrics during V2 selection. V2 may copy the
  14 old TEST groups into its *development* view only after making clear that
  this is a new V2 protocol; do not use their V1 result files.
- Never edit `configs/data.yaml`, the four V1 model configs,
  `metadata/material_manifest_split.csv`, `metadata/render_manifest.csv`,
  `dataset/test`, `artifacts/checkpoints/{custom_cnn,resnet50,convnext_tiny,vit_b16}`,
  or V1 result directories.
- Do not call `evaluate_all.py --split test` for V2. The old TEST score remains
  a final V1 result, not a V2 tuning signal.
- V2 configuration files must live under `configs/v2/`; V2 checkpoints and
  studies under `artifacts/v2/`; V2 metrics/figures under
  `reports/results/v2/` and `reports/figures/v2/`.
- If a V2 process accidentally opens a V1 TEST path, stop the run, record the
  incident, quarantine its outputs as invalid, and retain V1 unchanged.

## 3. Poly Haven provenance and CC0 evidence

The project owner has confirmed that all 105 source textures came from Poly
Haven and that Poly Haven's official licence page states CC0, including
permission to redistribute and use commercially. The remaining requirement is
an auditable map from every local asset to its Poly Haven page. A filename such
as `Leather011` is not sufficient evidence.

### New tracked registry

Create `metadata/polyhaven_asset_registry.csv`, one row per source material
(105 V1 rows plus each V2 holdout row), with this schema:

```text
dataset_version,material_id,class,polyhaven_asset_slug,polyhaven_asset_name,
asset_page_url,download_url_or_api_record,source_map_files,local_relative_dir,
map_sha256_json,downloaded_utc,registry_checked_utc,license_name,
license_url,license_evidence_url,derivative_render_rights,attribution_text,
v1_split,v2_role,rights_status,verified_by,notes
```

Required values and checks:

- `polyhaven_asset_slug` is the canonical slug used by
  `https://polyhaven.com/a/<slug>`. It must be populated and unique for every
  material; maintain an explicit alias column if the local descriptive name
  differs from the slug.
- `asset_page_url` must resolve to the asset's page and
  `license_evidence_url` must be `https://polyhaven.com/license` (or a
  subsequently verified official URL). Store the evidence retrieval date and
  a SHA-256 hash of the saved evidence text/HTML or a stable archived copy.
- `source_map_files` lists every map actually used by Blender. Hash each map
  in `map_sha256_json`; hashes must match the local relative directory at audit
  time.
- `license_name` is `CC0 1.0` (or the exact current Poly Haven wording),
  `derivative_render_rights` is `yes`, and `rights_status` is `verified` only
  after a human checks the page-to-slug mapping.
- `v1_split` is `v1_train`, `v1_val`, `v1_test`, or `not_in_v1`; `v2_role` is
  `development` for all 105 archived V1 groups and `holdout` for new assets.
  Source slugs in the V2 holdout must be disjoint from every V1 split.
- `attribution_text` should credit Poly Haven even though CC0 does not require
  attribution; keep a polite source acknowledgement in the release.

Create `metadata/polyhaven_cc0_evidence.md` containing the official licence
URL, access/retrieval date, evidence hash, a concise rights interpretation,
and the registry version. Do not paste a long copyrighted page into the
repository. Link the evidence from `DATASET.md`, `ATTRIBUTION.md`, the Kaggle
data card, and the report.

Add `scripts/build_polyhaven_registry.py` to build/update the registry from the
official public Poly Haven API **once per auditable release**, not by scraping
HTML. It must use a descriptive unique `User-Agent` containing project/contact
information, include the required Poly Haven credit in its request/metadata,
cache the exact API JSON response under `artifacts/v2/provenance/api_cache/`,
and write a SHA-256 hash and retrieval timestamp for that response. The API
record is evidence for the asset-slug mapping; it does not replace the
official CC0 licence evidence. The builder must fail closed if the API returns
an unknown slug, a changed asset identifier, or a non-success response, and it
must never silently remap a local material. Distinguish (a) the CC0 asset
terms, which do not require attribution, from (b) Poly Haven's live API usage
and credit condition, which must be honoured in the tool's User-Agent and
release credit. Add `scripts/validate_polyhaven_registry.py` to reject missing
slugs, duplicate slugs, non-Poly-Haven page URLs, unverified rights, absent
hashes, absolute Windows paths, and V2/V1 slug overlap.

### New-material holdout acceptance

Target five assets per class (35 groups, 1,050 images); three per class (21
groups, 630 images) is the hard minimum and must be labelled as a lower-power
holdout with wider uncertainty. An asset is eligible only if its Poly Haven
slug is not in the V1 registry and its source maps are not byte-identical to a
V1 map. Also perform a perceptual-hash and asset-page duplicate check:
reuploaded variants, colour variants, and renamed copies must not enter the
holdout as “new” materials. Record the selection date and do not expose the
holdout image directory to the Optuna worker.

The V2 release may contain the holdout after evaluation, but the worker should
receive only development manifests until the study and final configuration
are frozen.

## 4. Dataset/version strategy

Do not append V2 rows to the V1 manifests. Create a separate `dataset_v2/`
working layout and a new manifest family:

```text
dataset_v2/
  development/{wood,metal,stone,concrete,ceramic,fabric,leather}/*.png
  holdout_v2/{wood,metal,stone,concrete,ceramic,fabric,leather}/*.png
metadata/v2_render_manifest.csv
metadata/v2_material_manifest.csv
metadata/v2_material_manifest_split.csv
metadata/v2_holdout_material_manifest.csv  # 35 rows target; 21 minimum
metadata/polyhaven_asset_registry.csv
metadata/polyhaven_cc0_evidence.md
```

Development rows reference all 105 archived V1 groups with a new
`dataset_version=v2.0.0` field, a `source_version=v1` field, and
`v2_role=development` (including rows whose V1 split was `test`). Holdout rows
reference only the 35 new groups (21 minimum) with `v2_role=holdout` and
`split=v2_holdout`. Preserve all V1
images by copying or linking them into a `dataset_v2/development` staging view,
not by rewriting V1 files; rewrite only the V2 manifest paths so old V1 TEST
images are never addressed through a `/test/` path by the V2 loader. Add
`image_sha256`, `source_asset_slug`, and
`dataset_version` to every V2 render row.

The V2 data validator must check: 7 classes; 105 development groups and 35
holdout groups (21 minimum); 30 images per material; exactly five holdout
materials per class (or the documented three-material minimum); no slug/map/image duplicates; all PNGs decode
as RGB 512×512; all development folds are group-disjoint; and no holdout row
is passed to the search or early-stopping loaders. The new holdout is a
scientific evaluation set, not a replacement for V1's 14-group TEST.

## 5. Concrete code changes

Implement in dependency order. V1 defaults must remain byte/behaviour
compatible, and all new code should be covered before GPU runs.

### Must-have changes

1. **`src/material_classifier/augmentations.py`**

   Change `build_train_transform` to accept an explicit augmentation config
   while retaining the current no-argument V1 defaults. Implement named
   policies (`baseline`, `texture_robust`) with crop scale/ratio, flip,
   brightness/contrast/saturation/hue, and optional low-probability
   `RandomErasing`. Keep the callable return type unchanged (a torchvision
   transform/`Compose` callable); expose a separate `resolve_augmentation_config`
   helper for logging rather than returning a tuple or metadata wrapper. Keep
   evaluation transforms deterministic and unchanged.

2. **`src/material_classifier/train.py`**

   Add an optional epoch callback such as
   `on_epoch_end(epoch, train_metrics, val_metrics, checkpoint_path)` so an
   Optuna trial can report `val_macro_f1` with a monotonic global step and raise
   `optuna.TrialPruned`. Keep the callback disabled for V1. The callback may
   record every epoch, but the objective must defer `trial.should_prune()` until
   at least two complete fixed-order folds have finished, preventing one hard
   material fold from killing a potentially good trial. Ensure pruning leaves
   a valid `trial_state.json` and does not overwrite a prior checkpoint.
   Add optional gradient accumulation and a config-controlled EMA/SWA hook,
   defaulting off. Record resolved config, seed, git commit, data manifest
   hash, peak VRAM, wall time, best epoch, and callback/pruning state.

3. **`src/material_classifier/data.py`**

   Add a manifest-driven loader that accepts an explicit list of allowed split
   names and a `dataset_version`. It must reject `test` for V2 search and
   reject `v2_holdout` until final evaluation. Add path-containment,
   image-hash, class-order, and material-to-source-slug checks. Preserve
   `material_id` and `source_asset_slug` in `return_metadata=True` batches.

4. **`src/material_classifier/cross_validate.py`**

   Add reusable `grouped_outer_folds` and `grouped_inner_folds` helpers with
   explicit assertions that group sets do not overlap. Add a hard assertion
   that neither V1 `test` nor V2 `v2_holdout` appears in a search dataframe.
   Return fold assignments and material counts for logging. Keep the existing
   V1 API and outputs unchanged.

5. **`src/material_classifier/models.py`**

   Retain the exact V1 constructors. Add config-controlled classifier dropout
   (default 0) and optional staged unfreezing (default current full-backbone
   unfreeze) for V2 only. Expose a safe `freeze_policy` for `head_only`,
   `last_stage`, and `full` fine-tuning. For implementation, insert dropout
   immediately before the final linear layer (`resnet.fc`,
   `convnext.classifier[2]`, or `vit.heads.head`) and keep the existing
   classifier-prefix naming contract. For `last_stage`, unfreeze ResNet
   `layer4`, ConvNeXt `features[-1]`, or the ViT final encoder block plus the
   head; document the exact parameter names in the resolved config. Verify
   parameter counts and state-dict compatibility in tests. Do not silently
   alter torchvision pretrained weights.

6. **New `scripts/run_optuna_v2.py`**

   Implement the persistent study, objective, fold loop, sampler/pruner,
   config materialisation, callback, OOM handling, and artifact retention.
   Required CLI options are `--model {resnet50,convnext_tiny}`,
   `--study-name`, `--storage`, `--trials`, `--folds`, `--seed`, `--protocol`,
   `--manifest`, `--device`, `--smoke`, and `--resume`. Refuse paths containing
   TEST/holdout rows before model construction. Never import the current TEST
   result files.

   The objective should be implemented as a small, testable function rather
   than a monolithic CLI: `suggest_params(trial, model_name)`,
   `build_v2_config(params, fold, seed)`, `run_trial_fold(config, train_rows,
   val_rows)`, and `objective(trial)`. The objective creates fresh model and
   optimizer instances for every fold, averages fold Macro-F1, and reports
   epoch-level values to the pruner. This API shape removes architectural
   guesswork for a later implementer and permits CPU fixture tests without
   importing the CLI.

7. **New `scripts/verify_v2_dataset.py` and
   `scripts/validate_polyhaven_registry.py`**

   Provide machine-readable pass/fail JSON plus a human-readable summary.
   These are the release gates for source provenance, group leakage, file
   integrity, duplicate detection, and V2 holdout isolation.

8. **New `scripts/confirm_v2.py` and `scripts/evaluate_v2_holdout.py`**

   `confirm_v2.py` reruns preselected finalists with fixed development folds.
   `evaluate_v2_holdout.py` accepts only a frozen model/config and the V2
   holdout manifest, writes a one-time evaluation marker, and has no Optuna
   dependency. It must not accept `metadata/render_manifest.csv` or
   `split=test` as input.

9. **New `scripts/train_v2_final.py` and `scripts/generate_v2_visuals.py`**

   `train_v2_final.py` consumes exactly one frozen resolved config and the 105
   development groups, writes a hash-locked final checkpoint, and refuses a
   holdout/test row. `generate_v2_visuals.py` consumes only V2 result files and
   writes the figures listed below; it must label development folds and the V2
   holdout separately.

   Add `scripts/run_v2_baseline.py` (or an explicit `--baseline` mode) to
   retrain the frozen V1 ResNet configuration from fresh weights on the same
   105-group V2 development folds and all-105 final fit. This baseline is
   created before holdout access and is the only comparator for the paired V2
   holdout delta; the existing V1 checkpoint is not a substitute because it
   was trained on only its original train split.

10. **Blender V2 rendering interface**

    Add an input-manifest argument to `blender/scripts/generate_dataset.py`
    and `blender/scripts/render_full_dataset.py` (for example
    `--input-manifest`), while retaining the current V1 default path and
    accepted `train/val/test` split behaviour. Permit `v2_holdout` only when a
    V2 flag is present and write to an explicit V2 output directory. Keep the
    existing `--manifest` output argument distinct from the input manifest so
    a V2 render cannot overwrite `metadata/render_manifest.csv`. Add a small
    `tests/test_v2_render_arguments.py` parser test; do not run Blender until
    registry and source-map checks pass.

11. **Configs**

   Add `configs/v2/base.yaml`, `configs/v2/resnet50.yaml`,
   `configs/v2/resnet50_v1_baseline.yaml`, and
   `configs/v2/convnext_tiny.yaml`. Include `dataset_version`, manifest paths,
   allowed split (`development`), holdout path (not passed to search), policy,
   search bounds, seed, trial budget, and artifact namespace. Never modify the
   existing V1 configs. Add a `tuning` extra containing exactly
   `optuna==4.9.0` to `pyproject.toml` (and regenerate the corresponding lock
   metadata) only when implementation starts; do not install it as part of
   this plan.

12. **Tests**

    Add `tests/test_v2_protocol.py` and `tests/test_polyhaven_registry.py` for
    test/holdout rejection, group disjointness, deterministic fold hashes,
    source-slug uniqueness, path/hash validation, objective aggregation,
    pruning, resume after interruption, and OOM retry. Add a two-trial CPU
    smoke fixture with tiny synthetic images; it must never use local dataset
    or V1 artifacts. Run the existing suite afterward.

### Optional changes (only after must-haves pass)

- `src/material_classifier/models.py`: head dropout in `{0, .1, .2, .3}` and
  last-stage-only unfreezing. These are regularisation ablations, not automatic
  improvements.
- `src/material_classifier/train.py`: EMA with decay `{.99, .995, .999}` or
  SWA over the last 3--5 non-stale epochs. Use one method at a time; report
  extra memory/time and do not combine EMA and SWA in the first search.
- Augmentation policy: crop lower scale `0.75--0.95`, brightness/contrast
  `0--.20`, saturation `0--.15`, hue `0--.03`, horizontal flip `0--.7`, and
  RandomErasing probability `0--.15`. Keep colour changes physically
  plausible; test MixUp/CutMix only as a labelled optional ablation because
  mixed materials may be unrealistic.
- A small classifier ensemble of the confirmed ResNet and ConvNeXt models,
  selected only after both individual models pass the same protocol. This is
  optional and likely costs more inference time.

Do not start with class weighting: V1 is perfectly balanced by class and
material. Test it only as a predeclared ablation if development diagnostics
show a repeated fold-level failure, never because of V1 TEST errors.

## 6. Search spaces and staged budget

Use `TPESampler(seed=42, n_startup_trials=8)` and
`MedianPruner(n_startup_trials=8, n_warmup_steps=4, interval_steps=1)`.
Report epoch callbacks with a monotonic global step such as
`fold_index * max_epochs + epoch`. Do not allow production pruning after the
first unusually hard fold: only call `trial.should_prune()` after at least two
complete, fixed-order folds have reported final Macro-F1 (or document and test
an equivalent robust rule). Objective: mean five-fold development validation
Macro-F1 over all 105 V2 development groups. Tie-break within 0.005 by lower
fold SD, then lower training time. Record accuracy,
balanced-accuracy, per-class F1, per-material accuracy, ECE, and loss as
secondary outputs.

### ResNet-50 first

Search 20 trials initially (expand to 24 only if the study is stable):

```text
head_learning_rate:       loguniform 1e-5 .. 3e-3
backbone_learning_rate:   loguniform 1e-6 .. 3e-4
weight_decay:             loguniform 1e-6 .. 1e-2
label_smoothing:          uniform 0.00 .. 0.20
freeze_head_epochs:       int 0 .. 8
warmup_epochs:            int 0 .. 6
max_epochs:               categorical [30, 40, 50]
early_stopping_patience:  int 5 .. 12
gradient_clip_norm:       categorical [0.5, 1.0, 2.0]
input_size:               categorical [224, 256]
effective_batch_size:     fixed 32
physical_batch_size:      32 (AMP; halve only on recorded OOM)
augmentation_policy:      categorical [baseline, texture_robust]
head_dropout:             categorical [0.0, 0.1, 0.2]   # optional phase
freeze_policy:            categorical [full, last_stage] # optional phase
```

Reject combinations with backbone LR ≥ head LR, a warm-up longer than
`max_epochs/2`. Do not optimize batch size as a performance hyperparameter:
keep effective batch size fixed at 32 and use gradient accumulation when the
physical batch must be reduced. A physical OOM adjustment must be recorded in
the trial and resolved config. Start
with the must-have parameters only; add dropout/freeze-policy policies as a
separate 8-trial ablation if the initial study still overfits.

### ConvNeXt-Tiny second

Run only after the ResNet study and confirmation artifacts are complete. Use
the same objective and folds but a model-safe batch size:

```text
head_learning_rate:       loguniform 1e-5 .. 2e-3
backbone_learning_rate:   loguniform 1e-6 .. 2e-4
weight_decay:             loguniform 1e-6 .. 1e-2
label_smoothing:          uniform 0.00 .. 0.20
freeze_head_epochs:       int 0 .. 8
warmup_epochs:            int 0 .. 6
max_epochs:               categorical [30, 40, 50]
early_stopping_patience:  int 5 .. 12
gradient_clip_norm:       categorical [0.5, 1.0, 2.0]
input_size:               categorical [224, 256]
effective_batch_size:     fixed 16
physical_batch_size:      16 (AMP; halve only on recorded OOM)
augmentation_policy:      categorical [baseline, texture_robust]
```

Run 12--16 trials, not a huge search that simply overfits five CV folds. The
V1 ConvNeXt CV/TEST reversal is a reason to test it, not evidence that it is
the right final model.

### Stages

1. **CPU/one-fold smoke:** two trials, three epochs, tiny fixture, then one
   CUDA trial with the real development manifest. Definition of done: no test
   or holdout rows, callback/pruner works, config hashes are stable, and an
   interrupted trial resumes.
2. **ResNet screening:** 20 trials × five grouped folds, with epoch pruning.
   Persist metrics for every fold; retain model weights only for the best five
   trials and retain histories/configs for all trials.
3. **ResNet confirmation:** rerun the best three configurations from fresh
   weights using the same five folds and fixed seeds. Freeze one winner before
   holdout access.
4. **ConvNeXt screening/confirmation:** 12--16 trials, then best two or three
   confirmation runs. Compare the confirmed winner against confirmed ResNet
   using the predeclared rule.
5. **Final V2 fit:** train the selected architecture/config on all 105
   development groups. Hash the checkpoint and freeze the study before
   evaluating the 35-group V2 holdout (21 groups at minimum).
6. **One-time holdout audit:** run the V2 baseline and frozen V2 model on
   `v2_holdout` only. No further search or model change is allowed afterward.

## 7. Leakage guards, caching, OOM, and reproducibility

Implement these assertions before each GPU trial:

- source asset slugs, map hashes, `material_id`s, image hashes, and groups are
  disjoint between development folds and V2 holdout;
- all 30 rows of a material share one class and one split;
- the search dataframe has no `split=test`, `split=v2_holdout`, or path whose
  normalized string contains `/test/` or `/holdout_v2/`;
- train augmentation is constructed only for train-fold rows; validation and
  holdout transforms are deterministic;
- pretrained weights are loaded once from a local cache, not downloaded in an
  objective; each trial starts from fresh model weights after that cache;
- DataLoader seeds are deterministic (`42 + fold` plus a recorded trial seed),
  and one GPU trial runs at a time on the RTX 4060;
- each trial records the exact manifest hash, source registry hash, code commit,
  resolved config hash, PyTorch/torchvision/CUDA versions, GPU, and seed.

Use the existing AMP and OOM-retry machinery, but make OOM a trial failure or
recorded batch-size adjustment rather than a silent change to the objective.
If a batch-size retry occurs, preserve the original requested value, the
transition, and effective batch size in the trial record. Do not run concurrent
trials on 8 GB VRAM.

Store Optuna in SQLite, e.g.
`artifacts/v2/studies/resnet50_development.db`, and resume with the same study
name/storage. Keep the top-five checkpoints only (about 270 MB each for
ResNet-50 and 319 MB each for ConvNeXt-Tiny based on V1); remove temporary
trial weights only after their metrics/config are durably recorded. A complete
all-trial checkpoint archive would consume tens of GB and is unnecessary.

## 8. RTX 4060 time and storage budget

The environment record reports an RTX 4060 with 8 GB VRAM. V1 measured 557 s
for ResNet-50, 905 s for ConvNeXt-Tiny, and 1,511 s for ViT-B/16, including
early stopping. Approximate sequential upper bounds (before pruning) are:

| phase | ResNet estimate | ConvNeXt estimate |
|---|---:|---:|
| ResNet screening: 20 trials × 5 folds | 18–20 h | — |
| ConvNeXt screening: 16 trials × 5 folds | — | 23–26 h |
| ResNet top 3 confirmation × 5 folds | 5.3–5.8 h | — |
| ConvNeXt top 2 confirmation × 5 folds | — | 2.9–3.3 h |
| final fit + holdout evaluation | 15–30 min | 20–40 min |

These start with V1's measured 557 s/ResNet and 905 s/ConvNeXt runs and add
approximately 15--25% because V2 trains on all 105 groups rather than the V1
train subset. Epoch pruning and early stopping should reduce them, but do not
budget on the reduction.
Screen ResNet first and stop before ConvNeXt if the assignment timeline cannot
absorb the cost. New holdout rendering at the five/class target is approximately
1,050 × 0.796 s = 13.9 minutes at the V1 measured render throughput (about 8.4
minutes for the three/class minimum), plus material download/audit and Blender
setup time (plan 45--120 minutes end-to-end).

The 1,050 additional PNGs are expected to be roughly 267 MB at the current mean
file size (about 160 MB at the three/class minimum). Keep SQLite, CSV/JSON histories, and figures in the repository's
ignored `artifacts/v2/`/`reports/results/v2/` paths. Budget 2--4 GB for retained
top-k checkpoints and logs; do not retain every trial's 270/319 MB checkpoint.

## 9. Metrics, uncertainty, and acceptance thresholds

Primary metric: seven-class Macro-F1 on the 35-material V2 holdout (21 groups
total is the hard-minimum fallback), with the material as the resampling unit.
Secondary metrics: accuracy, balanced
accuracy, per-class precision/recall/F1, per-material accuracy, log loss,
ROC-AUC where defined, expected calibration error, inference ms/image, and
checkpoint MB.

Compute five-fold development mean ± sample SD. For the holdout, compute a
paired bootstrap over material groups (at least 10,000 resamples) comparing
the predeclared V1 baseline and selected V2 model. Do not bootstrap individual
images as if the 30 correlated renders were independent. Report the 95%
interval and all holdout material-level scores.

Define `Δ Macro-F1` as selected V2 minus the frozen V1-hyperparameter baseline,
where both models are trained on the same 105 development groups and evaluated
on the same new holdout. Use these predeclared tiers:

- **Confirmed improvement:** `Δ Macro-F1 >= 0.03`, the material-group paired
  bootstrap 95% CI lower bound is strictly greater than zero, no class F1
  drops by more than 0.05, and development-CV Macro-F1 is no more than 0.01
  below the frozen-config baseline. Report the complete CI and all trade-offs.
- **Promising but uncertain:** point estimate `Δ Macro-F1 >= 0.03` and no
  unacceptable class collapse, but the bootstrap CI crosses zero. This is not
  a confirmed improvement; report it as evidence requiring more holdout
  materials.
- **Inconclusive/no gain:** point estimate below 0.03, non-positive point gain,
  unacceptable class degradation, or materially worse development-CV
  performance (more than 0.01 below baseline). Retain V1 ResNet-50 as the
  assignment recommendation.

Runtime/checkpoint increases must always be disclosed and remain within a
predeclared 2× latency/size budget unless accuracy is the explicit reason.
Never lower the 0.03 threshold after seeing holdout results. The three/class
minimum should normally be labelled promising/inconclusive unless its CI is
strongly positive; five/class is the target for a useful conclusion.

## 10. Artifacts and figures

Use this namespace and keep a manifest of every output:

```text
artifacts/v2/
  provenance/registry_validation.json
  studies/resnet50_development.db
  studies/convnext_tiny_development.db
  trials/<model>/<trial_id>/{config.json,history.csv,trial_state.json}
  checkpoints/<model>/top_k/{best_01.pt,...}
  confirmations/<model>/<candidate_id>/fold_<n>/
  baseline/resnet50/{fold_<n>,best.pt,resolved_config.json}
  final/<model>/{best.pt,resolved_config.json,run_state.json}
reports/results/v2/
  v2_dataset_validation.json
  fold_metrics.csv
  trial_summary.csv
  confirmation_metrics.csv
  v2_holdout_metrics.json
  v2_holdout_predictions.csv
  v2_provenance_summary.csv
reports/figures/v2/
  optuna_optimization_history.png
  optuna_parameter_importance.png
  fold_macro_f1_distribution.png
  development_per_class_f1.png
  v2_holdout_confusion_matrix_raw.png
  v2_holdout_confusion_matrix_normalized.png
  v2_holdout_per_material_accuracy.png
  v2_holdout_calibration.png
  v2_holdout_correct_examples.png
  v2_holdout_incorrect_examples.png
  v1_vs_v2_paired_material_delta.png
```

Include a source-slug/CC0 evidence coverage figure or table, development vs
holdout material counts, trial budget/pruning summary, and a V1/V2 metric table.
Label every plot as development CV or V2 holdout; never label V2 development
plots as V1 TEST performance.

## 11. Test strategy and execution commands

### Before any GPU work

```powershell
python scripts/validate_polyhaven_registry.py --registry metadata/polyhaven_asset_registry.csv
python scripts/verify_v2_dataset.py --manifest metadata/v2_material_manifest_split.csv
python -m pytest -q
```

The commands must pass and must not modify the V1 lock or V1 result files.
Create the new holdout and its registry before running the study:

```powershell
blender --background blender/material_dataset_master.blend `
  --python blender/scripts/render_full_dataset.py -- --v2 `
  --input-manifest metadata/v2_holdout_material_manifest.csv `
  --manifest metadata/v2_holdout_render_manifest.csv --out dataset_v2
python scripts/verify_v2_dataset.py --manifest metadata/v2_material_manifest_split.csv
```

The `--v2`/`--input-manifest` interface is one of the planned changes above; if
the implementation instead uses a V2-only wrapper, use that wrapper with the
same input/output separation. The renderer input is deliberately the
holdout-only 35-row manifest (21 rows at minimum), so its `v2_holdout` split cannot cause the
existing renderer's train/val/test assumptions to consume development rows.
Merge its output with the development manifest only after rendering and
validation. Do not pass the V1 manifest to a regeneration command. Record the
render command, Blender build, seed, and output hash.

### Install and smoke-test only when implementation is approved

```powershell
uv sync --extra dev --extra tuning
python scripts/run_optuna_v2.py --model resnet50 --protocol holdout `
  --manifest metadata/v2_render_manifest.csv --trials 2 --folds 1 --smoke `
  --study-name v2_smoke --storage sqlite:///artifacts/v2/studies/smoke.db
```

The real objective must reject `folds=1` unless `--smoke` is present.

### Development screening and confirmation

```powershell
python scripts/run_optuna_v2.py --model resnet50 --protocol holdout `
  --manifest metadata/v2_render_manifest.csv --trials 20 --folds 5 `
  --study-name v2_resnet50_dev --storage sqlite:///artifacts/v2/studies/resnet50_development.db `
  --resume --device cuda

python scripts/confirm_v2.py --model resnet50 --study v2_resnet50_dev `
  --top-k 3 --folds 5 --manifest metadata/v2_render_manifest.csv --device cuda

python scripts/run_optuna_v2.py --model convnext_tiny --protocol holdout `
  --manifest metadata/v2_render_manifest.csv --trials 16 --folds 5 `
  --study-name v2_convnext_dev --storage sqlite:///artifacts/v2/studies/convnext_tiny_development.db `
  --resume --device cuda
```

Freeze the winner and fit the final V2 model on development materials only:

```powershell
python scripts/run_v2_baseline.py --model resnet50 `
  --config configs/v2/resnet50_v1_baseline.yaml --manifest metadata/v2_render_manifest.csv `
  --folds 5 --fit-all-105 --output artifacts/v2/baseline/resnet50 --device cuda

python scripts/train_v2_final.py --model <winner> `
  --config configs/v2/<winner>.yaml --manifest metadata/v2_render_manifest.csv `
  --fit-all-105 --output artifacts/v2/final/<winner> --device cuda
python scripts/evaluate_v2_holdout.py --checkpoint artifacts/v2/baseline/resnet50/best.pt `
  --manifest metadata/v2_render_manifest.csv --split v2_holdout `
  --output reports/results/v2/v2_baseline_holdout_metrics.json --device cuda
python scripts/evaluate_v2_holdout.py --checkpoint artifacts/v2/final/<winner>/best.pt `
  --manifest metadata/v2_render_manifest.csv --split v2_holdout `
  --output reports/results/v2/v2_holdout_metrics.json --device cuda
python scripts/generate_v2_visuals.py --results reports/results/v2
```

There is deliberately no V2 command containing `--split test`. Verify the
holdout marker, hashes, and report outputs before any publication.

## 12. Stop, rollback, and failure rules

Stop the work and retain all artifacts as an aborted study if any of the
following occurs: an unverified or duplicate Poly Haven slug; an absent CC0
evidence hash; V1/holdout source-map overlap; any material appears in two
folds; a search loader opens TEST/holdout; the code modifies a locked V1 file;
repeated OOM or nondeterministic resume; or the holdout is viewed before the
configuration freeze. Do not delete evidence; write `status=aborted` and the
reason in the study metadata.

Rollback means selecting the unchanged V1 ResNet checkpoint and reporting its
existing metrics. Do not overwrite V1 directories, reset Git, or silently
replace a trial/checkpoint. If a new holdout is unusable, quarantine the V2
dataset version and stop; do not substitute current TEST images.

## 13. Report/GitHub integration

The assignment report should continue to present V1 as the locked headline
comparison. If V2 completes, add a clearly labelled “V2 unseen-material
generalisation” section with provenance coverage, new split counts, protocol,
search budget, development CV, one-time holdout results, uncertainty, and
negative findings. Add the V2 plan and registry links to the repository README;
do not rewrite the V1 results table or recommendation automatically.

For a public release, update `DATASET.md`, `MODEL_CARD.md`, and Kaggle's data
card with the Poly Haven asset registry, CC0 evidence URL, version, checksum,
and limitations. Publish only after the provenance gate in `kaggle.md` passes.

## 14. Milestones and definitions of done

| milestone | dependency | estimate | definition of done |
|---|---|---:|---|
| Freeze V1 snapshot | existing lock | 30 min | hashes recorded; no V1 files edited |
| Build Poly Haven registry/evidence | source asset access | 2--4 h | 140 exact slug/page mappings (126 at minimum), map hashes, CC0 evidence, validator passes |
| Select/render 35 new materials | registry complete | 1--2 h | 5/class target (3/class minimum), no duplicate maps/slugs, RGB 512×512, V2 manifest passes |
| Implement V2 loader/fold guards | V2 manifests | 3--5 h | test/holdout rejection and disjoint-fold tests pass |
| Implement callback/studies | loader + train engine | 4--8 h | resume, prune, OOM, config hashes, CPU smoke pass |
| ResNet screening/confirmation | all tests + GPU | 24--28 h elapsed | study complete; top-k confirmation; no protocol incidents |
| ConvNeXt screening/confirmation | ResNet decision | 26--30 h elapsed | same evidence, or documented stop decision |
| Final V2 fit/holdout | winner frozen | 1 h | one-time holdout metrics and hashes written |
| Report/publication integration | holdout + provenance | 2--4 h | labelled V2 section, data card, checksums, no V1 overwrite |

Must-have milestones are registry/evidence, new holdout or documented fallback,
V2 loader/fold guards, callback/study persistence, ResNet screening and
confirmation, final metrics, and report integration. EMA/SWA, MixUp/CutMix,
ensemble, ConvNeXt search, and Kaggle upload are optional and must never delay
or compromise the locked assignment result.
