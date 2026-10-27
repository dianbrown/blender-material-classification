# V3 research and delivery plan

Created 2026-10-27. Branch: `v3/material-classification`.
Linear project: https://linear.app/vaper/project/material-classification-v3-247f6c6b89f5
Sole owner/assignee: HectoLogic. Personal-And-Firends scope explicitly approved.

## Objective and evidence standard

Improve seven-class classification on the user's Blender dataset, targeting at
least 80% accuracy without manufacturing that result through split leakage or
test-driven selection. The published V1 best accuracy is 69.8%. V2 in the remote
branch contains plans, not a completed new benchmark. Synthetic-domain success
does not establish performance under arbitrary real-world circumstances.

Preserve the V1 manifests, configurations, results, figures and report. Put new
code in `src/material_classifier/v3/`, scripts in `scripts/v3/`, configurations in
`configs/v3/`, metadata in `metadata/v3/`, results in `reports/results/v3/`, figures
in `reports/figures/v3/`, and local weights/cache/data in `artifacts/v3/`.

## Phases and acceptance gates

1. **Audit and environment (PER-368).** Inventory archives, hash source assets,
   inspect corruption, duplicated materials, label ambiguity and old split
   assumptions. Reproduce a CUDA environment on the RTX 4070 Laptop GPU (8 GB).
   Record actual installed versions; historical pins may no longer resolve.
2. **Protocol and data (PER-369).** Define independent source groups and freeze
   candidate models, feature recipes, hyperparameter grids and split seeds before
   viewing v3 held-out predictions. Use nested grouped cross-validation on the
   existing corpus, or a new genuinely independent holdout. Published V1 test
   data cannot become an untouched v3 endpoint. Fit scaling, tuning and
   calibration only inside training partitions. Record every exclusion.
3. **Implementation (PER-370).** Compare pretrained texture representations with
   regularized heads and selected fine-tuning, subject to audit findings. Cache
   deterministic frozen features, support resume and image-level inference,
   record hashes/configs/environment, and unit test split isolation and artifact
   alignment. Use independent implementation worktrees and one GPU owner.
4. **Experiments (PER-371).** Run a finite, documented candidate matrix. Produce
   out-of-fold predictions and distinguish development estimates from untouched
   tests. Measure accuracy, balanced accuracy, macro/weighted/per-class F1,
   precision/recall, raw/normalized confusion matrices, ROC/PR where defined,
   log loss, Brier score, calibration, material-group uncertainty intervals,
   robustness to declared corruptions, and runtime/VRAM. Compare models on the
   same evaluation units. Preserve failed trials and honest negative outcomes.
5. **Report and release preparation (PER-372).** Build an illustrated v3 report
   with actual tables/plots, source citations, methods, ablations, uncertainty,
   limitations and a data collection roadmap. Update README and model/dataset
   cards. Prepare a validated local Kaggle package with source attribution and
   license checks. Publication to Kaggle is deferred as requested by the user.
6. **Review and delivery (PER-373).** Independent code/methodology/report review,
   fixes and regression checks, fresh-image inference smoke test, registered v3
   PR. Record all local artifact paths and reproduction commands.

## Agent ownership and memory

The parent thread coordinates integration and GPU scheduling. Auditors are
read-only. Implementers own disjoint modules in separate worktrees and return
commits, tests, artifact contracts and remaining risks. Reviews use a new task
for each round with earlier findings attached. Linear issue descriptions and
project documents store decisions, validation evidence and handoffs; this file
and `reports/V3_STATE.md` provide a repository-local recovery path.

No agent may relabel a development score as a test score, tune on outer-fold
predictions, overwrite V1 results, publish unverified source assets, or silently
change the protocol after results arrive. New data must add independent material
diversity, not merely more camera views of existing source textures.
