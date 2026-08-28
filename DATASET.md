# Synthetic PBR material dataset

This dataset supports seven-way image classification of rendered PBR
materials: `wood`, `metal`, `stone`, `concrete`, `ceramic`, `fabric`, and
`leather`. It contains 105 independent base materials (15 per class), with
30 renders per base material for 3,150 RGB PNG images at 512×512 resolution.

## Split and leakage control

The locked split is by `material_id`, never by individual image:

| split | base materials | images | images/class |
| --- | ---: | ---: | ---: |
| train | 77 | 2,310 | 330 |
| val | 14 | 420 | 60 |
| test | 14 | 420 | 60 |

All 30 renders for a material remain in one split. The authoritative records
are in `metadata/material_manifest_split.csv` and
`metadata/render_manifest.csv`. The test split is locked and must not be used
for model selection or tuning.

## Generation and local layout

Images were generated with Blender EEVEE and the RenderSynth pipeline using
the deterministic generation seed 62642. The local, gitignored layout is:

```text
dataset/{train,val,test}/{class}/*.png
materials/{class}/{material_id}/        # source PBR texture sets
```

Existing generation scripts and the RenderSynth configuration are retained in
`blender/scripts/` and `blender/configs/`. The manifests and reports record
the exact generation provenance. Use the scripts only after confirming local
Blender/RenderSynth installation and the locked manifest state.

## Source and licensing

The source PBR assets and their licenses have not yet been normalized into a
machine-readable registry. Before redistribution, record each asset's source,
license, and attribution in the material manifest and comply with its terms.
The generated renders inherit any applicable restrictions of their source
assets and are not currently offered as a public download.

Current public dataset URL: **Not yet publicly released**

Future distribution placeholder: **Kaggle dataset: TBD**

## Packaging

`scripts/package_dataset.py` creates a local Kaggle-ready ZIP and a SHA-256
checksum when explicitly pointed at a dataset directory. It does not upload
anything. Keep the resulting archive outside Git (the default is under the
gitignored `artifacts/` directory).
