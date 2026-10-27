# Dataset card: synthetic PBR material renders (V3 draft)

## Composition and scope

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
three per class, and 630 images**. It is excluded from the original release
package. Fresh scoring is pending. Production rendering/integrity work and
pre-prediction model locking must finish before it can support an independent
endpoint. A line-ending-only freeze-hash repair did not change the frozen
configuration, registry or model-selection policy.

Fresh ceramic/concrete definitions are restricted relative to the broad
archive labels. Blender 5.1.1 differs from the retained historical environment.
A seven-image old-TRAIN EEVEE pilot before bulk rendering visually matched
framing and texture placement, with softer ground shadows. Treat this as a
rendering-domain caveat, not a fresh evaluation result. Source novelty does not
remove renderer or taxonomy shift, and synthetic renders remain synthetic.

## Release status and permissions

**Local release ZIP and source proof are prepared. Nothing is published.**
The original sources' CC0 permissions have been verified, and image-to-source
lineage/scene checks are documented. The remaining decisions are the owner's
**dataset license** for the rendered collection and **later publication**.
`dataset_license` remains empty and `release_approved` remains false. Do not
describe the 105 sources as unresolved simply because the owner has not yet
chosen a dataset license.

The prepared package contains original-corpus renders and reproducibility/
provenance records, not provider texture maps or the external RenderSynth
add-on. No public download URL exists yet. The release workflow and exact
archive hashes live in [V3_KAGGLE_RELEASE.md](reports/V3_KAGGLE_RELEASE.md).
Provider CC0 terms do not automatically license project code, pretrained model
weights, provider websites, or unrelated future assets.

## Reuse and reproduction

Use the frozen v3 manifests and the reproduction commands in [README.md](README.md).
Keep the old split directories for traceability while grouping experiments by
`source_group`. Audit changed image/map bytes and preserve a separate output
directory for each extraction, training, evaluation and report assembly run.
The planned canonical validated cache/run are `artifacts/v3/features_final`
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
