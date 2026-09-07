# Version-2 hyperparameter-search plan (Optuna)

## Executive decision

Higher development-set performance is plausible, but it is not guaranteed to
translate to the sealed TEST split. The current evidence supports a cautious
V2 experiment rather than a last-minute replacement of the assignment result:

| model | TEST Macro-F1 | grouped development-CV Macro-F1 | CV SD |
|---|---:|---:|---:|
| ResNet-50 | 0.683 | 0.717 | 0.109 |
| ConvNeXt-Tiny | 0.569 | 0.787 | 0.113 |
| ViT-B/16 | 0.562 | 0.745 | 0.103 |

The large spread and the ConvNeXt reversal indicate material-level domain
heterogeneity, not simply an under-tuned learning rate. The current ResNet
training run also has a near-perfect training Macro-F1 (0.9996) and a much
lower TEST Macro-F1 (0.683), so regularisation, earlier stopping, and
augmentation may help. Search may reasonably recover a few Macro-F1 points on
development data; a large or reliable TEST gain cannot be promised.

**Do not run Optuna before assignment submission unless there is a separate,
well-documented V2 deadline.** The existing methodology lock and TEST
evaluation are strong evidence for the report. A hurried search risks hidden
selection, an apples-to-oranges comparison, and an unreviewed code change. If
run later, V2 must use a new namespace and report V1 and V2 as separate
experiments. This plan does not install Optuna, run trials, modify training
code, or read the current TEST predictions for model selection.

## Non-negotiable V1 protection

The V1 TEST split consists of 14 material groups (420 images) and was evaluated
after `reports/results/methodology_lock.json` was written. It is a sealed
artifact for this project. During V2:

- Do not load, score, plot, inspect, or rank any V1 TEST prediction,
  confusion-matrix cell, image, or per-class result to choose a hyperparameter.
- Do not change `dataset/test`, the split manifests, the locked configs, or
  V1 checkpoint directories.
- Do not use the TEST metrics in an Optuna objective, pruning rule, stopping
  criterion, or human decision. The V1 TEST profile can remain in the report as
  a final descriptive result, but it is not a V2 tuning signal.
- Store V2 trials under `artifacts/optuna_v2/` and V2 metrics under
  `reports/results/optuna_v2/`; retain trial data and resolved configs.
- If a V2 model is eventually evaluated on a new holdout, call it a new
  dataset/experiment version. Never relabel a V2 score as the original V1
  headline score.

## Recommended V2 protocol

### Option A: grouped nested cross-validation (preferred for a research result)

Use only the current development materials (`split in {train, val}`): 91
material groups, 2,730 images, and 13 material groups per class. Ignore the
V1 TEST rows entirely. Use `material_id` as the group and `class` for
stratification.

1. The outer loop is five-fold `StratifiedGroupKFold` with shuffle and seed 42.
   Each outer validation fold contains whole materials and no group overlap.
2. Inside each outer-training partition, run a three-fold grouped inner CV to
   score Optuna trials. Every trial is trained from fresh pretrained weights;
   no fold reuses a best checkpoint from another fold.
3. The trial objective is the mean inner validation **Macro-F1**. Aggregate
   image-level predictions within each fold before computing the seven-class
   macro average; also record per-class and per-material metrics for diagnosis.
4. Use the same preprocessing, class order, augmentations, seed policy, and
   early-stopping monitor for every trial. Keep a trial's full resolved config,
   fold assignments, checkpoint hash, elapsed time, and failure/OOM reason.
5. Select the search configuration by inner-CV mean Macro-F1, using the inner
   standard deviation as a tie-breaker only after a practically negligible
   mean difference (for example, within 0.005). Never use an outer score to
   choose a trial.
6. Retrain the selected configuration separately on each outer-training fold
   and evaluate once on its outer validation fold. Report outer mean, sample
   SD, 95% uncertainty interval if justified, and all per-class scores. This is
   the honest V2 estimate of generalisation to unseen materials.
7. For a final V2 research checkpoint, retrain on all 91 development groups
   using the selected configuration. Do **not** evaluate this checkpoint on the
   V1 TEST split in this project.

This nested protocol is expensive but prevents the optimistic bias of choosing
hyperparameters on the same folds used to report performance. The existing
`scripts/run_group_cv.py` is a development-only grouped CV runner, but it is
not an Optuna nested-search implementation. A V2 driver must explicitly reject
any row whose split is `test` before constructing a loader.

### Option B: new untouched V2 holdout (lower compute, simpler explanation)

If nested CV is too expensive, freeze a new group-level holdout from the 91
development materials before any search (for example, two materials per class,
14 groups / 420 images). Keep this file and its checksums private and never
read it during tuning. Run Optuna on the remaining 77 development groups with
three- or five-fold grouped CV, select one configuration, fit on those 77
groups, then evaluate once on the new holdout. The original V1 TEST remains
sealed and is not reused. This gives a clean V2 estimate, but the estimate is
based on only 14 material groups and must be reported with uncertainty.

Write a V2 split manifest with a new `dataset_version` and `holdout_name`; do
not overwrite `metadata/material_manifest_split.csv`. A new holdout is useful
if a final score is needed, but it must be treated as a new experiment rather
than an improvement to the V1 score.

## Search space

Start with one architecture at a time. ResNet-50 is the V1 recommended model;
ConvNeXt-Tiny is a useful V2 candidate because its development CV is high but
unstable on held-out material groups. Keep a second architecture search
separate so that architecture and hyperparameter selection are not confounded.

The following compact space is appropriate for an RTX 4060 (8 GB) and the
current AdamW/frozen-head-to-fine-tune engine:

| parameter | Optuna distribution | reason / guardrail |
|---|---|---|
| head learning rate | log-uniform `1e-5 .. 3e-3` | classifier adaptation |
| backbone learning rate | log-uniform `1e-6 .. 3e-4` | avoid destroying pretrained features |
| ratio head/backbone LR | report derived value | reject extreme unstable combinations |
| weight decay | log-uniform `1e-6 .. 1e-2` | overfitting control |
| label smoothing | uniform `0.0 .. 0.20` | useful for ambiguous material boundaries |
| frozen-head epochs | integer `0 .. 8` | test head-only warm start |
| warm-up epochs | integer `0 .. 6` | avoid early instability |
| maximum epochs | categorical `30, 40, 50` | retain a bounded budget |
| early-stop patience | integer `5 .. 12` | coupled to maximum epochs |
| gradient clip norm | categorical `0.5, 1.0, 2.0` | stabilise full fine-tuning |
| input size | categorical `224, 256` | 256 may help texture but costs VRAM |
| batch size | model-safe `8, 16, 32` | ResNet usually 16/32; ConvNeXt usually 8/16 |

The current configuration files already expose most optimisation and schedule
fields. Augmentation parameters are currently constructed by
`src/material_classifier/augmentations.py`; exposing them to a V2 config would
be a deliberate, separately tested code change. If that change is approved,
add only a small space: crop scale lower bound `0.80 .. 0.95`, brightness and
contrast `0.0 .. 0.20`, saturation `0.0 .. 0.15`, and horizontal-flip
probability `0.0 .. 0.7`. Keep transforms physically plausible for the sphere
renders. Do not add arbitrary vertical flips, severe perspective warps, or
texture erasure without a domain justification.

Do not begin with class weighting: classes are exactly balanced (450 images
per class and 15 materials per class), so weights can increase variance. Only
test weighted loss as a small ablation after the main search, and select it by
the same grouped development protocol.

## Optuna execution design

Use a seeded TPE sampler and an epoch-level pruner. A practical configuration
is `TPESampler(seed=42, n_startup_trials=8)` with
`MedianPruner(n_startup_trials=8, n_warmup_steps=4, interval_steps=1)`.
Report inner-fold validation Macro-F1 after every epoch and prune trials whose
intermediate score is clearly below completed trials. Do not prune from test
or outer-fold results.

Use a persistent SQLite study, for example
`artifacts/optuna_v2/optuna_resnet50.db`, with a study name containing the
dataset and protocol version. Persist `trial.number`, parameters, user attrs
(`model`, `fold`, `seed`, `config_hash`, `git_commit`, GPU), final objective,
intermediate values, runtime, peak VRAM, and exception/OOM information. Cache
ImageNet weights once outside the timing loop and use deterministic DataLoader
generators. Do not run concurrent GPU trials on an 8 GB card; sequential trials
make memory and timing interpretable. Resume the same SQLite study after an
interruption rather than deleting completed trials.

Suggested budgets:

- A smoke check: 2 trials, one inner fold, 3 epochs, never reported as a
  result.
- Screening: 20--30 trials for one model, three inner folds, 30--50 epoch
  ceiling with early stopping. Expect roughly hours on the RTX 4060; monitor
  actual times from V1 (ResNet V1 took about 557 seconds, ConvNeXt about 905
  seconds, and ViT about 1,511 seconds including early stopping).
- Confirmation: the best three to five configurations rerun across the full
  nested protocol or all five outer folds. Do not report the single best noisy
  trial without confirmation.

If the full nested budget is infeasible, use Option B's new holdout and a
20-trial grouped-CV screen. Reduce the search space before reducing the number
of folds; a very large search over one fold is less credible than a compact
search with group-aware validation.

## Diagnostics that can improve the model for the right reason

Derive all V2 diagnostics from development folds. For each outer fold, save a
confusion matrix, per-class precision/recall/F1, per-material accuracy,
confidence/calibration summaries, and representative correct/incorrect images.
Pay particular attention to metal/ceramic and other visually similar pairs,
but only promote a hard-negative or augmentation change when it is repeatedly
observed across development folds. The V1 TEST errors are useful context in
the existing report; they must not dictate a V2 change.

Potential interventions, each tested as its own ablation, are:

1. lower backbone LR and stronger weight decay or label smoothing to reduce
   memorisation of the 77 training materials;
2. modestly wider crop/lighting/color augmentation to improve robustness to
   render variation without destroying material cues;
3. add development-only hard negatives only if they are new, correctly labelled
   renders and remain grouped by base material;
4. compare the best ResNet and ConvNeXt configurations using identical outer
   folds and a predeclared Macro-F1 rule; and
5. calibrate probabilities after model selection on an inner validation fold,
   never on an outer holdout.

Report the tradeoff: a higher Macro-F1 can reduce accuracy for a strong class,
increase runtime, or enlarge the checkpoint. Preserve inference latency and
model-size measurements under the same batch-1/device protocol used for V1.

## Acceptance criteria and reporting

Before running, write a V2 preregistration containing the split, search space,
trial budget, sampler/pruner, objective, tie-break rule, seed, and stopping
rule. Accept V2 as a meaningful improvement only if the selected configuration
beats the V1 development-CV baseline under the predeclared outer estimate,
with no material leakage, and the gain is not confined to one fold or one
class. A new score should include mean ± sample SD, fold scores, per-class
metrics, confidence intervals where appropriate, compute cost, and all failed
trials. Do not claim that V2 improved V1 TEST performance because the V1 TEST
split was intentionally not reevaluated.

If the search is run after the assignment, add a short appendix stating that
V1 remained locked, V2 used development groups only (or a new holdout), how
many trials were run, and why the V1 headline recommendation remains the
reproducible assignment result. If the search is not run, list this plan under
future work; the current project already has a defensible leakage-safe design.
