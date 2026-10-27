# Original-corpus Kaggle preparation

Local preparation completed on 2026-10-27. The archive and extracted directory
are ready for the owner's license/publication decisions. **Nothing was uploaded.**
`dataset_license` remains empty and `release_approved` remains `false`.
The original 105 sources now have verified source permissions; none remains
unresolved merely because the dataset license has not been chosen.

This report updates the release status from [V3_PROVENANCE.md](V3_PROVENANCE.md)
and the earlier audit/contract's historical unverified-license statements. Those
frozen audit files were not rewritten. The original corpus alone is in scope;
the new holdout remains separate and excluded.

## Evidence and unchanged inputs

The completed [original-source verification](results/v3/provenance_verification.json)
was ingested offline into [the registry](../metadata/v3/license_registry.json).
All 105 entries retain their existing source fingerprints. Each entry records
`verified_by: agent` and `human_review_claimed: false`, a provider asset ID and
source URL, attribution, CC0 terms, retrieval date, verification tier, and the
SHA-256 of a compact evidence file. `redistribution_allowed: true` records the
verified source terms; it does not approve publication of the rendered dataset.

The 315 manifest-referenced maps comprise 288 Poly Haven MD5-and-size matches
and 27 ambientCG official ZIP-member SHA-256-and-size matches. Local SHA-256
values bind every map to the existing audit. The extra metal displacement map
is separately verified and remains part of its original source fingerprint.
No source map is included in the release package, and the retained scene does
not reference that extra displacement. Full provider response bodies are not
bundled; the compact records retain their response hashes, URLs, retrieval
timestamps, excerpts and comparison values. Poly Haven's remote checksum tier
is MD5 plus size, not a claimed remote SHA-256 comparison.

All 3,150 rendered images were rehashed against the frozen v3 manifest and
compared with their original `dataset.zip` members: 3,150/3,150 match in both
checks. Historical render rows match the image paths, classes, original splits,
material names, recorded material assignments and deterministic child seeds.
There are 30 images per material. The original render manifest has no image
hash column; the frozen v3 audit supplies those image hashes.

[Render provenance](../metadata/v3/license_evidence/render_provenance.json)
pins the original manifests, generation scripts, pipeline configuration and
retained `.blend` by SHA-256. Read-only Blender 5.1.1 inspection used background
mode, factory startup, disabled auto-execution and one CPU thread, with no
rendering or saving. All 105 material node graphs refer to the expected three
maps. The only scene objects are the local sphere, untextured ground plane,
camera and area light. There are no linked libraries, packed assets, HDRI/world
texture, external fonts, sounds, movie clips, cache files or additional image
inputs. The ground uses an untextured Principled BSDF; the world uses a constant
Background node. The author states that these are their own rendered images.

Blender warned: `File written by newer Blender binary (502.44), expect loss of
data!` The inspection found the expected objects/materials/maps and saved
nothing. This is a documented inspection limitation, not a new render
reproduction. Historical execution is supported by the retained records and
scene, rather than a cryptographic execution attestation. No concrete
unresolved external scene input was found. The external RenderSynth add-on is
an execution dependency and is not distributed in this image archive.

The frozen 101 source groups/family decisions and all manifest bytes are
unchanged. The canonical v3 manifest SHA-256 remains
`8ffafa1709c36d78fb7215535c33ea04ee42ffc08f71719b5ad941ba58d4501f`;
the source-map CSV SHA-256 remains
`33176f631ad6314af03ec9c0d79dce890b97963dc7a9ac0e82d65c5435f6e3e9`.
Before/after hashes matched for six protected parent metadata files and ten
existing feature-cache files. No cache, original archive, original source
texture/render, grouping or audit manifest was modified by this task.

## Local staging artifacts

Parent staging root:
`C:/Users/dianb/blender-material-classification/artifacts/v3/release-staging/`

- ZIP: `original-corpus-verified-audit-20261027.zip`
- Size: **841,991,438 bytes**
- SHA-256: `0b19b667d48150c9b272a28c26a685b6519c96fd2830a9922e5aaad882f39681`
- Extracted directory: `original-corpus-verified-audit-20261027/`
- Checksum sidecar: `original-corpus-verified-audit-20261027.zip.sha256`
- Validation results: `original-corpus-verified-audit-20261027.validation.json`

The directory and ZIP contain 3,269 files: 3,150 images, 108 evidence files and
11 other metadata/card files. Their layout is:

```text
dataset/{train,val,test}/{class}/*.png
metadata/manifest.csv
metadata/source_maps.csv
metadata/data_audit.json
metadata/source_groups.json
metadata/source_family_overrides.json
metadata/license_registry.json
metadata/license_evidence/{material_id}.json  # 105 source records
metadata/license_evidence/render_provenance.json
metadata/license_evidence/scene_inspection.json
metadata/license_evidence/inspect_scene.py
DATASET_CARD.md
SOURCES.md
image_hashes.sha256
STAGING_STATUS.json
dataset-metadata.json
```

Every registry `evidence_path` resolves inside the archive relative to
`metadata/license_registry.json`, and every evidence SHA-256 was verified.
`SOURCES.md` carries source/author attribution and license URLs, including the
ambientCG distinction between publisher operator and unknown individual author.
It credits provenance metadata retrieval as Powered by Poly Haven. No raw
textures, model weights, repository generation code, `.blend`, private paths
or holdout data are included. The bundled inspection script is the small
read-only evidence collector, not the generation pipeline.

ZIP CRC checks, the exact member allowlist, all image hashes, all evidence
hashes, the metadata private-path scan, and the extracted directory's image
hashes passed on the actual corpus. The package preserves historical splits;
those splits are not an untouched v3 test set. Use the frozen source groups
for grouped evaluation.

## Publication gates and remaining decisions

Exactly two release blockers remain:

1. The owner must choose the rendered dataset license. The supported choices
   are `CC0-1.0` and `CC-BY-4.0`; verified CC0 inputs support either choice.
2. The owner must explicitly authorize publication, including the Kaggle
   account/slug and visibility. The current `local-audit/material-classification-v3`
   identifier is a staging placeholder.

No choice or approval was inferred from source verification. The archive is
explicitly `audit_only: true`, `publishable: false`, `uploaded: false`, with
an empty Kaggle `licenses` list. It must be regenerated after the decisions;
do not upload this audit ZIP unchanged. Selecting CC-BY for the dataset does
not relabel the underlying CC0 source evidence. The compatibility gate also
rejects a CC-BY source under a CC0 dataset choice and requires carried CC-BY
attribution. There is no upload code in this preparation path.

## Reproduction and tests

From this checkout, using the parent's prepared Python environment:

```powershell
& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe scripts/v3/record_provenance.py
& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe scripts/v3/package_dataset.py --data-root C:/Users/dianb/blender-material-classification --output C:/Users/dianb/blender-material-classification/artifacts/v3/release-staging/original-corpus-verified-audit-rerun.zip --audit-only
& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe -m pytest tests/test_v3_data.py -q
```

The optional ingestion helper checks official comparison values, material
scope, the original manifest pin and all existing source fingerprints before
writing the registry/evidence. It resets publication defaults to empty/false
and does not change grouping. Packaging rehashes the real source/render bytes.
Existing staged archives are never overwritten.

Targeted data tests: **52 passed, 1 skipped** (Windows does not permit the
unprivileged symlink fixture). Ruff passed for the changed Python files.
Tests cover compatibility, explicit publication defaults, archive path
containment, metadata collisions, missing/changed evidence, attribution,
evidence inclusion/hashes, stale corpus rejection and deterministic ZIPs.
Synthetic fixture tests and metadata-only ingestion tests are separate from
the actual-corpus staging validations described above.
