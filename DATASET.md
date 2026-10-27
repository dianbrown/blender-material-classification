# Dataset card: Material Studio synthetic PBR renders

The complete study contains **3,780 images, 126 material IDs and 122 known
source families**: 3,150 development images / 105 IDs / 101 families plus
630 fresh test images / 21 IDs / 21 groups. There are 540 images per class
across the complete study. The holdout has been scored once per locked model;
it is now an evaluated benchmark, not an untouched future test.

[Project report](reports/project/report.pdf) | [Results notebook](notebooks/05_v3_results.ipynb) |
[Combined release audit](reports/V3_STUDY_DATASET_RELEASE.md)

## Original development corpus

The existing corpus contains **3,150 RGB PNG images at 512 x 512**, from **105
local material IDs**, 30 views each. Seven labels have 15 local materials and
450 images each. The frozen v3 family registry yields **101 source groups**,
not 105 independent physical materials.

| Label | Images | Local material IDs | Frozen source groups |
| --- | ---: | ---: | ---: |
| wood | 450 | 15 | 14 |
| metal | 450 | 15 | 15 |
| stone | 450 | 15 | 15 |
| concrete | 450 | 15 | 15 |
| ceramic | 450 | 15 | 15 |
| fabric | 450 | 15 | 15 |
| leather | 450 | 15 | 12 |
| Total | 3,150 | 105 | 101 |

The 101 groups merge three confirmed ambientCG leather generator families and
a conservative wood relationship. They encode known dependence, not proof that
all remaining families are unrelated. Multiple views share texture content;
uncertainty and splitting must respect source families.

## Generation, taxonomy and layout

Blender EEVEE and the retained RenderSynth pipeline render a material-covered
sphere, ground, camera and light using deterministic variation (generation seed
62642). Original scripts, configuration, manifests and retained scene are pinned
in [render provenance](metadata/v3/license_evidence/render_provenance.json).
The project authors the renders and experimental code, not the provider maps.
Local raw assets are gitignored:

```text
dataset/{train,val,test}/{class}/*.png
materials/{class}/{material_id}/
metadata/v3/manifest.csv
metadata/v3/source_groups.json
metadata/v3/source_family_overrides.json
metadata/v3/source_maps.csv
```

Labels are appearance-based archival categories. Ceramic includes mosaic and
terrazzo-like sources; concrete includes plaster and painted-wall-like sources.
The historic metallic shader convention is part of the rendering domain.
Retain archival labels and document exclusions before looking at predictions;
difficult images must not be relabelled or removed to improve a score.

## Splits and leakage control

The historical directory layout is train 2,310 images / 77 materials,
validation 420 / 14, and test 420 / 14. These are lineage records, **not an
untouched current test split**. The old test was consulted and some related
families cross old material-ID splits. All 3,150 existing images are now the
v3 development corpus.

V3 uses five outer and three inner class-stratified source-group folds, expanded
from group identities to image rows. Scalers, regularization selection and
heads stay inside the relevant training partitions. Preserve the frozen
`metadata/v3/manifest.csv`, family registry and recorded fold assignments.
Report image-weighted metrics with source-group bootstrap uncertainty; the
interval does not transform 30 correlated views into independent sources.

## Integrity and original-source provenance

The completed audit found 3,150 readable renders, 105 materials and 101 groups.
Release preparation rehashed all 3,150 images against the frozen manifest and
compared them with original `dataset.zip` members: all matched. Scene inspection
matched all 105 material graphs to the expected three maps and found no
additional external scene inputs needing resolution. Exact scope and
limitations are retained in [V3_KAGGLE_RELEASE.md](reports/V3_KAGGLE_RELEASE.md).

**All 316 local source maps are verified.** There are 315 manifest-referenced
maps (288 from 96 Poly Haven sources, 27 from nine ambientCG sources), plus one
verified unreferenced metal displacement map. No original source remains at
identifier-only or unknown-license status. The extra displacement map is not
referenced by the retained scene and is not included as a texture in the
rendered-image release.

Poly Haven verification compares official metadata MD5 and size with local
files; local SHA-256 pins the audited bytes. It is not a remote SHA-256 match.
ambientCG verification compares SHA-256 and size against members of official
downloaded archives. Provider terms, source URLs, attribution, timestamps and
verification evidence are recorded per asset in
[license_registry.json](metadata/v3/license_registry.json), compact evidence
under `metadata/v3/license_evidence/`, and
[the source audit](reports/V3_PROVENANCE.md). Unknown individual ambientCG map
authors do not mean unknown source permission. Earlier frozen audit warnings
about unresolved source licensing are historical, superseded by this evidence;
the frozen audit bytes themselves are preserved.

## Fresh collection and rendering-domain caveat

The separate `metadata/v3_holdout` registry specifies **21 new CC0 sources,
three per class, and 630 images**. It is excluded from the original-only release package and included in the
complete study audit ZIP described below. All 630 renders passed model-free integrity checks. The classifier
view is `metadata/v3_holdout/classification_manifest.csv`; the original renderer
manifest remains separate. Fresh evaluation is complete. The primary choice and all model artifacts were
locked before any fresh prediction. No model changed after the clean results.
A line-ending-only freeze-hash repair did not change the frozen
configuration, registry or model-selection policy.

Fresh ceramic/concrete definitions are restricted relative to the broad
archive labels. Blender 5.1.1 differs from the retained historical environment.
A seven-image old-TRAIN EEVEE pilot before bulk rendering visually matched
framing and texture placement, with softer ground shadows. Treat this as a
rendering-domain caveat, not a fresh evaluation result. Source novelty does not
remove renderer or taxonomy shift, and synthetic renders remain synthetic.

## Complete study package and release status

**The complete local audit ZIP is verified. Nothing is uploaded.** Original
source permissions are resolved; the owner's rendered-dataset license and
publication decisions remain pending. `dataset_license` is unset and
`release_approved=false`. Third-party CC0 maps are attributed; project authorship
concerns rendered images and code, not those maps. Provider terms do not
license model weights, websites or unrelated project assets.

- Local archive: `artifacts/v3/release-staging/material-classification-v3-study-audit.zip`
- Size: **1,005,048,843 bytes**; 3,927 entries.
- SHA-256: `4f2bd057d66493ada1cc4b9d69d7932bcdfcb52798cb5aaf6f2887442a58d1da`
- Companion: `material-classification-v3-study-audit.validation.json` in the same directory.
- Split: 3,150 development images and 630 fresh test images, with 30 views per ID.

The validation checks every entry's CRC/hash, the manifest, source/pixel hashes
and private paths. Archive paths are `images/development/{class}/{filename}` and
`images/test/{class}/{filename}`. `metadata/manifest.csv` records V3 split,
historical `original_split`, source group, lineage path, source/package byte
hashes and decoded-pixel hashes. Old test membership remains lineage only.

All 316 original and **63 fresh source maps** were verified against official
CC0 provider evidence. The package contains rendered images and compact source,
rendering and audit records, not raw textures, scene binaries or model weights.
Known ancestry checks cover aliases, normalized names, procedural families,
recipe/map identities and raw map reuse; they do not prove every physical source
unrelated. The original-only audit ZIP remains preserved separately.

Private Windows paths in embedded PNG metadata were stripped from **copies**.
All packaged images have identical decoded pixels to their canonical sources;
original files were not modified. `sha256` in the combined manifest hashes the
packaged PNG; `source_sha256` hashes the original PNG. Do not compare them as
if metadata sanitization preserved byte identity. Both hashes and pixel identity
are retained, with published text normalized to UTF-8/LF.

See [V3_STUDY_DATASET_RELEASE.md](reports/V3_STUDY_DATASET_RELEASE.md) for complete
verification and the original [release audit](reports/V3_KAGGLE_RELEASE.md).
There is no public download URL. Publishing or choosing CC0/CC-BY-4.0 for the
rendered dataset remains the owner's decision, as does Kaggle publication.
Once scored, and especially after publication, further tuning on this test set
must be disclosed; confirmatory claims need a newly frozen independent holdout.

## Reuse and reproduction

Use the frozen v3 manifests and the reproduction commands in [README.md](README.md).
Keep the old split directories for traceability while grouping experiments by
`source_group`. Audit changed image/map bytes and preserve a separate output
directory for each extraction, training, evaluation and report assembly run.
The canonical validated cache/run are `artifacts/v3/features_final`
and `artifacts/v3/run_final`; initial outputs remain available for comparison.

Raw Windows-generated training files may use CRLF and their run records hash
those original bytes. Published report evidence is normalized to UTF-8/LF before
manifest hashing, while its inventory preserves both original and staged
hashes. Never compare the raw training-output hash to a Git-normalized copy as
if the bytes were unchanged.

This dataset supports controlled synthetic appearance research. It does not
establish physical composition, real-world coverage or synthetic-to-real
performance. Source-provider concentration, finite families, ambiguous labels,
pretraining overlap and shared scene cues remain limitations.
