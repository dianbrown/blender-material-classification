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
