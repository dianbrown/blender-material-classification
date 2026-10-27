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
