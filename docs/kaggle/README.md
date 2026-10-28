# Synthetic PBR Materials publication assets

The [dataset](https://www.kaggle.com/datasets/dianbrown/synthetic-pbr-materials)
and [starter notebook](https://www.kaggle.com/code/dianbrown/synthetic-pbr-materials-starter-notebook)
are public on Kaggle. Dataset version 2 contains 3,780 images and corrected
release documentation; the original version remains available.

| File | Purpose |
| --- | --- |
| [description.md](description.md) | Dataset page description, composition, use, credits, and limitations. |
| [provenance.md](provenance.md) | Generation methods and source-map attribution. |
| [file-descriptions.md](file-descriptions.md) | Descriptions of the published dataset's main files. |
| [column-descriptions.csv](column-descriptions.csv) | Descriptions of all 138 columns across the eight CSVs. |
| [dataset-cover-image.png](dataset-cover-image.png) | Master cover artwork matching Material Studio's visual style. |
| [dataset-cover-image.jpg](dataset-cover-image.jpg) | Smaller cover export. |

The cover is abstract artwork, separate from the dataset images. It was created
with imagegen using the app's charcoal acrylic styling and muted mint/lavender
ASCII jellyfish as references. Its title fits Kaggle's centered square crop.

The [notebook source](../../notebooks/06_kaggle_quickstart.ipynb) previews images,
checks sampled hashes, keeps source groups together in development validation,
and runs a small CPU baseline. Its public Kaggle run completed successfully.
Library versions can change the tutorial fold and result; the exported split
records the actual run. The 630-image study test is excluded from this tutorial's
training and predictions. See the [dataset card](../../DATASET.md) and
[model card](../../MODEL_CARD.md) for the study protocol and evaluation limits.
