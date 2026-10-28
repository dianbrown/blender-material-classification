# Synthetic PBR Materials

**3,780 Blender renders for seven-class visual material classification.** Each
image is a 512 × 512 RGB PNG showing a material-covered sphere in a controlled
scene. The labels are **ceramic, concrete, fabric, leather, metal, stone, and
wood**, with 540 images per class.

The collection supports image classification, source-aware validation, and
experiments with synthetic material appearance. It contains 126 material IDs
with 30 views each, organized into 122 known source groups.

## What's included

| Partition | Images | Material IDs | Known source groups |
| --- | ---: | ---: | ---: |
| Development | 3,150 | 105 | 101 |
| Test | 630 | 21 | 21 |
| Total | 3,780 | 126 | 122 |

Each class has 450 development images and 90 test images. The test collection
contains three new source materials per class.

```text
images/development/{class}/  Development images
images/test/{class}/         Evaluated study test images
metadata/manifest.csv        Image paths, labels, source groups, splits, and hashes
metadata/                   Integrity inventory and release metadata
provenance/                 Source credits, license evidence, and generation records
DATASET_CARD.md              Dataset documentation
LICENSE                     Rendered-data license and scope
```

Use `metadata/manifest.csv` as the main image index. Its `image_path` values
are relative to the dataset root. Folder names also provide class labels.
Raw source textures, Blender scene binaries, model weights, and training logs
are excluded.

## Getting started

Run the public [starter notebook](https://www.kaggle.com/code/dianbrown/synthetic-pbr-materials-starter-notebook)
for image previews, source-group validation, and a small CPU baseline.

1. Attach this dataset to a Kaggle notebook using **Add Input**.
2. Read `metadata/manifest.csv` with pandas.
3. Filter `split == "development"` for model development.
4. Split by `source_group`, keeping related views and known source families
   together. The example notebook shows a stratified group split and a small
   CPU baseline.

```python
from pathlib import Path
import pandas as pd

manifests = [p for p in Path("/kaggle/input").rglob("metadata/manifest.csv")
             if (p.parent.parent / "images").is_dir()]
assert len(manifests) == 1, "Select the Synthetic PBR Materials dataset root"
root = manifests[0].parent.parent
images = pd.read_csv(manifests[0])
development = images.loc[images["split"].eq("development")].copy()
```

## How it was made

The images were rendered in Blender EEVEE using PBR surface maps from
[Poly Haven](https://polyhaven.com/) and [ambientCG](https://ambientcg.com/).
The generation pipeline varies camera, lighting, and object orientation while
retaining a controlled scene. Generation uses seed 62642; manifests retain
per-render settings and source identities. The test renders use Blender 5.1.1;
the historical development renderer version was not recorded.

Official provider records verify CC0 provenance for 316 development source
maps (315 used, one verified but unused) and 63 test source maps. Asset credits,
source-map hashes, license evidence, source-family assignments, and generation
records are included under `provenance/`. Private PNG ancillary metadata was
removed from release copies without changing decoded pixels.

## Using the partitions correctly

All 3,150 historical images are now **development data**. Historical
train/validation/test assignments survive only in `original_split` for
traceability; they are not a new evaluation protocol. Use `split` for the
current development/test partition.

The 630-image test set was frozen before the completed study's predictions
and **has now been evaluated**. Disclose further tuning against it; future
confirmatory evaluation requires a new independently frozen holdout. Thirty
views of one material are correlated observations, not 30 independent materials.

## Limitations

These labels describe synthetic visual appearance, not verified physical
composition. The historical ceramic class includes mosaic/terrazzo-like
materials, and concrete includes plaster/paint-like materials. Test definitions
are narrower. Renderer differences also introduce domain shift. Known source
groups capture documented relationships, but complete physical independence
and overlap with pretrained models' data have not been established. Performance
on these renders does not establish accuracy on real-world photographs.

## Credits and license

Published by **Dian Brown** for the Material Studio project. Rendered images and
project-authored dataset metadata are available under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), matching this
dataset's published license. Third-party surface assets are credited to their
respective authors in the included provenance records. Included provenance
code retains its existing notices; the dataset license does not relicense
third-party code or model weights.

This is a fixed study release; no recurring updates are scheduled. Documentation
corrections may be published as new versions.
