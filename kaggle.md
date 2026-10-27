# Publish the Material Studio V3 dataset on Kaggle

Checked against official Kaggle documentation and CLI source on **27 October
2026**. This guide covers the planned upload; nothing has been uploaded.

## 1. Use the combined V3 archive

The prepared archive is:

```text
artifacts/v3/release-staging/material-classification-v3-study-audit.zip
```

It is **1,005,048,843 bytes** (about 1.01 GB / 958.5 MiB), with 3,927 file
entries, including **3,780 RGB PNG images at 512 × 512**. The accompanying local validation record,
`material-classification-v3-study-audit.validation.json`, reports verified
ZIP CRCs, entry hashes, source/pixel hashes, manifest contents,
and a passed private-path scan. Its SHA-256 is:

```text
4f2bd057d66493ada1cc4b9d69d7932bcdfcb52798cb5aaf6f2887442a58d1da
```

This checksum identifies the current audit archive, not a later edited release.
The other `data-audit.zip` and `original-corpus-verified-audit-20261027.zip`
files in staging do not contain the full combined study.

| Current partition | Images | Material IDs | Known source groups |
| --- | ---: | ---: | ---: |
| Development | 3,150 | 105 | 101 |
| Held-out test | 630 | 21 | 21 |
| Total | 3,780 | 126 | 122 |

Each of the seven labels—ceramic, concrete, fabric, leather, metal, stone,
wood—has 450 development and 90 test images, or 540 overall. Each material ID
has 30 views.

The archive opens directly into this layout, without a containing folder:

```text
DATASET_CARD.md
dataset-metadata.json
images/
  development/{class}/*.png
  test/{class}/*.png
metadata/
  manifest.csv
  archive_inventory.json
  input_hashes.json
  release_status.json
provenance/
  metadata/v3/                  # source groups, source maps, license evidence
  metadata/v3_holdout/          # test registry, identity audit, render manifests
  metadata/                    # historical material/render manifests
  blender/                     # generation scripts and configuration
  configs/v3/holdout.json
  artifacts/v3/holdout_dataset/ # data-only QC and run fingerprint
  reports/                     # generation and provenance verification records
```

There are 141 provenance files. Raw texture maps, `.blend` scene binaries,
model weights, and training logs are excluded. Use `metadata/manifest.csv` as
the image index: `image_path`, `class`, `material_id`, `source_group`, `split`,
`original_split`, `sha256`, `source_sha256`, `image_transformation`,
`source_provider`, `lineage_image_path`, and `pixel_sha256`.

**Split meaning:** all historical train/validation/test images now belong to
development. Their counts, 2,310/420/420, describe lineage only. The old test was
consulted and related families cross some historical splits. The separate 630
images have `split=test` and `original_split=holdout`. They were held out for the
completed study and have now been evaluated. Preserve the frozen `source_group`
assignments when splitting development data; random image splits leak related
views. Known groups do not prove complete physical independence.

Local evidence: [dataset card](DATASET.md),
[packaging implementation](scripts/v3/package_study.py), and the validation
record above.

## 2. Finish the release copy

Source provenance is verified: **316 development maps and 63 test maps** have
official CC0 evidence, covering all rendered materials. Of the development
maps, 315 are used and one additional verified map is unused. Both providers
publish their asset terms: [Poly Haven](https://polyhaven.com/license) and
[ambientCG](https://docs.ambientcg.com/license/). Detailed evidence is included
in `provenance/metadata/v3/license_registry.json`, its `license_evidence/`
directory, and the test source registry.

**The rendered dataset's own license is still your choice.** CC0 source maps
do not automatically assign a license to the project's renders. Two supported
Kaggle identifiers are `CC0-1.0` and `CC-BY-4.0`; neither is selected here.
CC0 dedicates applicable rights to the public domain; CC BY 4.0 permits reuse
with credit, a license link, and an indication of changes. Read the
[CC0 terms](https://creativecommons.org/publicdomain/zero/1.0/) and
[CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) before choosing.
Keep upstream credits and distinguish render creators from texture authors.

Extract the archive into a separate release working folder. Preserve the audit
ZIP and canonical images. Before uploading that release copy:

1. Choose the Kaggle owner/slug, public creator credit, rendered-data license,
   and repository URL for the description.
2. Replace the root `dataset-metadata.json`: its current title ends in
   `AUDIT ONLY`, `id` is `null`, `licenses` is empty, and its description says
   the destination/license are pending. Use the fields below. Its `isPrivate`
   value is not a substitute for Kaggle's sharing control.
3. Update `DATASET_CARD.md`: replace the leading `AUDIT ONLY / NOT YET
   PUBLISHED` stamp, `Dataset license: UNSET`, `release_approved=false`, and the
   opening destination/license-choice text with actual release details. State
   that the test has been evaluated. Add a `LICENSE` file identifying the
   chosen terms and their scope; retain applicable notices for included
   provenance code.
4. Update `metadata/release_status.json`: fill `dataset_license` and
   `kaggle_destination`, set `release_approved` only when the owner has actually
   decided to release, and replace its audit-only `status`. Keep
   `upload_performed=false` until an upload really happens. The
   `model_scores_accessed=false` field describes the packaging operation, not
   whether the study has evaluated the test set.
5. Rebuild `metadata/archive_inventory.json` for the final files, excluding
   the inventory itself. Repack as, for example,
   `material-classification-v3-study.zip`; generate a new ZIP SHA-256 sidecar
   and validation record. Check all inventory hashes, 3,780 manifest rows and
   images, partition/class counts, and private paths. The old validation and
   checksum do not certify edited files. The current packaging script always
   emits audit-only metadata, so rerunning it unchanged does not finalize a
   public release.

Keep frozen provenance records as historical evidence, including dated audit
statements; explain their original scope rather than rewriting history. Preserve
image paths, source groups, lineage fields, and image hashes when changing only
release documentation.

The packaged PNGs already have private ancillary metadata removed. All 3,780
have changed byte hashes but identical decoded pixels to their originals.
`sha256` identifies the packaged PNG; `source_sha256` identifies the untouched
original. `pixel_sha256` checks decoded content. `metadata/input_hashes.json`
records original versus normalized provenance-text hashes. Do not substitute
original PNGs when preparing the upload.

## 3. Fill in the Kaggle page

| Field | Suggested value |
| --- | --- |
| Title | Material Studio: Synthetic PBR Materials V3 |
| Subtitle | 3,780 Blender renders in seven classes, with source-group lineage |
| URL slug | `material-studio-synthetic-pbr-materials-v3` |
| Owner | Your chosen Kaggle username or organization |
| License | The rendered-data license you choose |
| Tags | Relevant available tags for computer vision, classification, and synthetic data |
| Update frequency | Never, if this is the fixed study release |

Copy this description, replacing the bracketed credit/license/URL fields:

> Material Studio V3 contains 3,780 synthetic RGB PNG images at 512 × 512
> pixels, rendered in Blender EEVEE. Seven visual material labels—ceramic,
> concrete, fabric, leather, metal, stone, and wood—have 540 images each.
> There are 126 material IDs with 30 views each, organized into 122 known
> source groups.
>
> **Partitions:** 3,150 development images cover 105 material IDs and 101 known
> source groups. The separate 630-image test collection covers 21 new sources,
> three per class. Historical train/validation/test assignments in
> `original_split` record lineage only; all 3,150 historical images are now
> development data. Use `split` for the current partition and `source_group`
> to keep related views and known source families together during development.
>
> **Files:** `images/development/{class}/` and `images/test/{class}/` contain the
> images. `metadata/manifest.csv` provides labels, groups, partition membership,
> lineage, and byte/pixel hashes. `DATASET_CARD.md` and `provenance/` document
> source attribution, license evidence, generation, and integrity checks. Raw
> texture maps, scene binaries, and model weights are not included.
>
> **Provenance and privacy:** source maps from Poly Haven and ambientCG have
> verified CC0 provenance. Private PNG metadata was removed from release
> copies without changing decoded pixels. Rendered images and dataset metadata:
> [CHOSEN LICENSE AND LINK], by [CREATOR CREDIT]. Upstream source terms and
> provenance-code notices remain separately identified in the package.
>
> **Evaluation limits:** the test set was frozen before the study's model
> predictions and has now been evaluated. Further tuning on it must be
> disclosed; new confirmatory evaluation needs a new independent holdout.
> Thirty views of one material are correlated observations. Known source
> groups do not establish complete physical independence, and overlap with
> model pretraining data has not been audited. Historical ceramic/concrete
> labels are broader than the new test definitions, and renderer differences
> introduce domain shift. These are synthetic appearance labels, not evidence
> of physical composition or accuracy on real-world photographs.
>
> Repository, study documentation, and citation: [PUBLIC REPOSITORY URL].

For CLI uploads, this is a template, not a finished metadata file.
Replace all placeholders and use the full description above:

```json
{
  "title": "Material Studio: Synthetic PBR Materials V3",
  "subtitle": "3,780 Blender renders in seven classes, with source-group lineage",
  "id": "YOUR_OWNER/material-studio-synthetic-pbr-materials-v3",
  "licenses": [{"name": "YOUR_CHOSEN_LICENSE_ID"}],
  "description": "REPLACE WITH THE COMPLETED DESCRIPTION ABOVE"
}
```

The CLI requires exactly one license entry. Its metadata reference documents
the identifiers and fields; titles are 6–50 characters and subtitles 20–80.
The suggested values fit. See the
[official metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).

## 4. Upload through the website

Once the release copy is complete:

1. Sign in, open [Kaggle Datasets](https://www.kaggle.com/datasets), and choose
   **New Dataset**.
2. Select the owner, enter the title, and check the generated URL slug.
3. Upload the finalized combined ZIP from your computer. Keep **Sharing** set
   to **Private** for the initial check.
4. Select the chosen license and create the dataset. Add the subtitle,
   description above, source credits, and relevant tags on its page.
5. After processing, inspect the files and download a copy. Check the manifest,
   all image counts/hashes, and representative images. Compare extracted files
   with `metadata/archive_inventory.json`, not Kaggle's whole-download ZIP
   checksum.
6. When satisfied, change the dataset's sharing visibility to **Public**.
   Record its canonical URL, version, publication date, and creator credit for
   the repository and report citation.

Kaggle documents private creation as the default and ZIP extraction with
directory structure preserved. An archive suits this image collection; inspect
the processed tree before publication. See
[Kaggle's dataset upload and sharing documentation](https://www.kaggle.com/docs/datasets).

## 5. Optional official CLI route

Use this instead of the web creation step. The current official CLI requires
Python 3.11+ and supports browser OAuth:

```powershell
python -m pip install --upgrade kaggle
kaggle auth login
```

Authentication source: [official CLI authentication reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md).
Keep generated credentials outside the dataset and repository.

Prepare a clean `kaggle-upload` directory containing only the finalized release
ZIP, its new checksum sidecar, and a completed `dataset-metadata.json` beside
them. The metadata inside the ZIP must also be finalized. Do not point the CLI
at the whole release-staging directory, which contains other archives and
inspection artifacts.

```powershell
# Uploads a new private dataset; run only when ready to upload.
kaggle datasets create -p kaggle-upload -t -r skip
kaggle datasets status YOUR_OWNER/material-studio-synthetic-pbr-materials-v3
kaggle datasets files YOUR_OWNER/material-studio-synthetic-pbr-materials-v3 --page-size 100

# Download for the private inspection described above.
kaggle datasets download YOUR_OWNER/material-studio-synthetic-pbr-materials-v3 -p kaggle-download-check --unzip
```

Here `-r skip` is appropriate because images and nested metadata are
already inside the top-level ZIP. With extracted directories instead, use
`-r zip` so folders are included. `-t` requests preservation of tabular formats.
Creation is private without `--public`; make it public through the website
after inspection. These commands and flags are documented in the
[official dataset CLI reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

The current [official API implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py)
sets privacy from the create command's `public` argument, not an `isPrivate`
metadata property. It also skips the loose root `dataset-metadata.json` as a
data upload and currently validates slugs at 6–50 characters; the suggested
slug fits both source validation and the documentation's stated range.

For later changes, prepare the complete revised upload folder and create a
version with clear notes:

```powershell
kaggle datasets version -p kaggle-upload -m "Corrected release documentation; images unchanged" -t -r skip
```

Preserve the version used for the study and cite it explicitly. A public Kaggle
dataset makes the test images available to everyone; future tuned results on
these images cannot be presented as a newly untouched test.
