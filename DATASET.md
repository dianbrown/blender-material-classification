# Dataset card: Material Studio

Material Studio uses **3,780 synthetic PBR renders**, covering seven material
labels, 126 material IDs, and 122 known source families. Each material ID has
30 views. The dataset supports evaluation of visual material recognition under
controlled rendering conditions.

**The full dataset is not published. A Kaggle release is planned; no dataset
upload has been made.** The Windows installer includes 21 sample images, not the
full dataset. The rendered dataset's license has not yet been assigned.

## Composition

| Partition | Images | Material IDs | Known source groups |
| --- | ---: | ---: | ---: |
| Development | 3,150 | 105 | 101 |
| Held-out test | 630 | 21 | 21 |
| Total | 3,780 | 126 | 122 |

There are 540 images per label across both partitions. Development images are
RGB PNGs at 512 × 512 pixels. The test collection contributes 90 images from
three new sources per label.

| Label | Development images | Development IDs | Development source groups |
| --- | ---: | ---: | ---: |
| wood | 450 | 15 | 14 |
| metal | 450 | 15 | 15 |
| stone | 450 | 15 | 15 |
| concrete | 450 | 15 | 15 |
| ceramic | 450 | 15 | 15 |
| fabric | 450 | 15 | 15 |
| leather | 450 | 15 | 12 |

The 101 development groups merge three confirmed ambientCG leather generator
families and a conservative wood relationship. Groups encode known dependence;
they do not prove that every remaining source is physically unrelated.

## Rendering and labels

Blender EEVEE and the retained RenderSynth pipeline render a material-covered
sphere with a ground plane, camera, and light. View variation is deterministic
with generation seed 62642. The
[render provenance](metadata/v3/license_evidence/render_provenance.json) records
the original scripts, configuration, manifests, and retained scene.

Labels describe visual appearance. The development archive's ceramic category
includes mosaic and terrazzo-like sources; concrete includes plaster and
painted-wall-like sources. The historical metallic shader convention is part of
the rendering domain. Preserve the archived labels and define exclusions before
examining predictions.

The fresh test collection uses narrower ceramic and concrete definitions.
Blender 5.1.1 also differs from the historical rendering environment. A seven-image
pilot using old training sources matched framing and texture placement visually,
with softer ground shadows. New source identities therefore do not eliminate
renderer or taxonomy shift.

## Partitions and leakage control

The historical directory layout has 2,310 training, 420 validation, and 420 test
images. These directories record lineage: the old test was consulted, and some
related source families cross the old material-ID splits. All 3,150 images now
form the development corpus, not an untouched test set.

Frozen-feature selection uses five outer and three inner source-group folds,
stratified by label. Scalers, regularization selection, and fitted heads remain
inside their respective training partitions. Preserve the
[development manifest](metadata/v3/manifest.csv),
[source groups](metadata/v3/source_groups.json), and
[family overrides](metadata/v3/source_family_overrides.json).

The separate [test registry](metadata/v3_holdout/source_registry.json) contains
21 new sources. All 630 renders passed model-free integrity checks; the
[classification manifest](metadata/v3_holdout/classification_manifest.csv) is
separate from the renderer manifest. The primary model choice and all four
checkpoints were locked before test prediction. No model changed after the
clean test results.

The test set has now been evaluated. Further tuning on it must be disclosed;
new confirmatory claims require a newly frozen independent holdout. Metrics are
image-weighted, with uncertainty resampled over source groups. Thirty correlated
views do not constitute 30 independent materials. See [MODEL_CARD.md](MODEL_CARD.md)
for confidence intervals and evaluation limits.

## Source provenance

The source maps come from Poly Haven and ambientCG. Recorded official provider
evidence verifies CC0 provenance for **316 development maps and 63 test maps**.
The project creates the rendered images and code and attributes the third-party
maps separately.

Of the development maps, 315 are referenced by the manifest: 288 from 96 Poly
Haven sources and 27 from nine ambientCG sources. One additional metal
displacement map is verified but unused by the retained scene. Scene inspection
matched all 105 material graphs to their expected three maps and found no
additional external scene inputs requiring resolution.

Poly Haven verification compares official metadata MD5 and size with local
files; local SHA-256 pins the audited bytes. ambientCG verification compares
SHA-256 and size with members of official downloaded archives. Source URLs,
attribution, timestamps, and verification evidence are retained in the
[license registry](metadata/v3/license_registry.json) and
[per-asset evidence](metadata/v3/license_evidence). The test collection retains
its own [source identity audit](metadata/v3_holdout/source_identity_audit.json).

Known ancestry checks cover aliases, normalized names, procedural families,
recipe/map identities, and raw map reuse. They do not establish complete physical
ancestry. Public texture overlap with backbone pretraining has not been audited.

## Release preparation

A local archive has been prepared and verified for the planned Kaggle release.
It contains all 3,780 renders and compact provenance records, excluding raw
textures, scene binaries, and model weights. Source-map permissions are distinct
from the pending license for the rendered dataset and from model-weight terms.

The archive layout is `images/development/{class}/{filename}` and
`images/test/{class}/{filename}`. Its `metadata/manifest.csv` records the current
partition, historical `original_split`, source group, lineage path, byte hashes,
and decoded-pixel hashes. Historical test membership remains lineage only.

Private Windows paths in embedded PNG metadata were stripped from archive copies.
All packaged images retain identical decoded pixels to their canonical sources;
the original files were not modified. In the combined manifest, `sha256` hashes
the packaged PNG and `source_sha256` hashes the original PNG. Metadata cleanup
can change byte hashes without changing pixels.

The development audit found 3,150 readable renders, rehashed them against the
frozen manifest, and matched them to the original `dataset.zip` members. Archive
validation checks entry CRCs/hashes, manifests, source/pixel hashes, and private
paths. Local preparation does not constitute publication or assign a license.

## Reuse and limitations

Full image and texture collections are separate from Git. Retain the historical
directories for traceability while grouping experiments by `source_group`.
Audit changed image or map bytes and keep separate outputs for each extraction,
training, and evaluation run. The canonical local cache and fitted run are
`artifacts/v3/features_final` and `artifacts/v3/run_final`.

This dataset measures synthetic appearance recognition. It does not establish
physical composition or performance on arbitrary real-world photographs.
Source-provider concentration, few independent families, ambiguous labels,
shared scene cues, rendering differences, and possible pretraining overlap limit
transfer. The [results notebook](notebooks/05_v3_results.ipynb) and
[raw test outputs](reports/results/v3/fresh) document the evaluated benchmark;
[README.md](README.md) covers application use and development setup.
