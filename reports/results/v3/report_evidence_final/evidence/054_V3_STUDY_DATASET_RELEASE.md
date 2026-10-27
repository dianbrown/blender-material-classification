# V3 combined study dataset: local audit package

**AUDIT ONLY / NOT YET PUBLISHED — 2026-10-27**

The combined package is staged locally in the parent checkout. No upload was
performed, no model scores were accessed, and no dataset license was selected.
`dataset_license` remains empty and `release_approved=false`. The user still needs
to choose the Kaggle destination and whether the rendered dataset uses CC0 or
CC-BY-4.0. The included Kaggle metadata is an intentionally incomplete skeleton:
`id=null`, `licenses=[]`, `isPrivate=true`; there is no invented public URL.

## Actual artifact and validation reference

All three outputs are under the parent's ignored `artifacts/v3/release-staging/`:

- `material-classification-v3-study-audit.zip`
- `material-classification-v3-study-audit.zip.sha256`
- `material-classification-v3-study-audit.validation.json`

The archive is **1,005,048,843 bytes** (3,927 entries).

```text
ZIP SHA256
4f2bd057d66493ada1cc4b9d69d7932bcdfcb52798cb5aaf6f2887442a58d1da

Packaged metadata/manifest.csv SHA256
1de0d87eaae8b4f29b80fa25776f65f0c71b553b1dcd9671d4eaa55f378e17b2
```

The small validation sidecar records `crc_verified=true`,
`all_entry_hashes_verified=true`, `manifest_verified=true`,
`source_and_pixel_hashes_verified=true`, `private_path_scan_passed=true`,
`model_scores_accessed=false`, and `upload_performed=false`.

The archive contains 3,780 images: 540 in each of ceramic, concrete, fabric,
leather, metal, stone and wood. There are 126 material IDs and 122 known source
groups. Exactly 3,150 original images and 101 original source families are
`development`; exactly 630 fresh images and 21 fresh source groups are `test`.
There are 30 views per material. The original 105 materials and fresh 21 materials
retain their frozen identities.

## Split and source interpretation

Archive image paths are `images/development/{class}/{filename}` and
`images/test/{class}/{filename}`. `metadata/manifest.csv` includes class,
material ID, source group, provider, V3 split, historical `original_split`,
`lineage_image_path`, packaged/source SHA256 values and decoded-pixel SHA256.
Historical original train/val/test membership is lineage only: every original
image is development data in V3. The original test subset is not represented as
untouched evaluation. Fresh `original_split=holdout` becomes V3 `split=test`.

The source-family overrides are preserved, including related original materials
that cross historical splits. The package validates class purity and development
versus test disjointness. Fresh selection validation checks original IDs, aliases,
conservative normalized-name families, procedural ancestry, official recipe/map
identities, and raw source-map reuse. Differing source-ID strings alone do not
establish independence. The frozen identity and leather lineage audits are
included. Physical lineage remains inferred, not universally proven: provider
records and exact hashes cannot exclude every undisclosed capture, crop or recolor.

The fresh endpoint uses restricted ceramic/concrete semantics and only three
source groups per class. Correlated views are not independent source samples.
Fresh Blender 5.1.1 EEVEE settings do not prove an identical domain to historical
renders with an unrecorded engine version. Metal coating/rust, suede appearance,
map resolutions and shader approximations remain caveats. The dataset is entirely
synthetic and cannot establish real-world performance. Once scored, especially
after publication, the test set is a public benchmark: later tuning must be
disclosed and confirmatory claims require a newly frozen independent holdout.

## Privacy, source hashes and canonical bytes

The original PNGs contain private Windows paths in Blender's embedded `File`
metadata. Distributing the original PNG bytes would violate the privacy requirement.
The packager therefore strips ancillary PNG metadata from archive copies only.
It preserves exact IHDR/PLTE/IDAT/IEND and supported fixed color-chunk bytes,
checks every original PNG chunk CRC and source hash, and verifies identical
decoded RGBA content before packaging. No pixels are rendered or re-encoded.
Unknown critical chunks and ICC profiles fail closed for explicit review.

All 3,780 packaged PNG byte hashes changed through metadata removal, including
fresh images whose metadata also contained dates and render details. In the
combined manifest, **`sha256` hashes the packaged PNG; `source_sha256` hashes the
untouched source PNG**. Frozen provenance manifests retain the original hashes.
Consumers must not compare packaged PNGs directly with the source-byte hashes.

Published text is UTF-8/LF. Known historical absolute material-folder paths in
CSV copies become relative `materials/...` paths. Unrecognized private absolute
paths fail closed. `metadata/input_hashes.json` separately records actual source
bytes, source LF-normalized bytes, exact canonical packaged bytes, and accepted
frozen hash references. Historical embedded checksums are preserved, rather than
silently rewritten to pretend normalized copies are original files. Compact sorted
JSON fingerprints used by renderer/QC records are a separate hash convention.

The archive contains 141 allowlisted source text files: complete compact original
license evidence, original/fresh source attribution, source maps and hashes, frozen
lineage and identity audits, download receipt, original renderer provenance,
fresh renderer manifest/QC/run identity, retained generation scripts/configuration,
and the original generation report. Provider records distinguish verified asset
authors from publisher/operator credits. Source textures are CC0; that does not
select a license for the author's rendered dataset. Earlier audit limitations
remain historical statements and are contextualized by later provenance evidence.

There are 381 verified binary inputs recorded but not distributed: 316 original
source maps, 63 fresh source maps, the retained original scene and `dataset.zip`.
No raw texture files, scene binaries, weights, training logs or model-score outputs
are in the archive. There was no new network lookup. The package includes an audit
dataset card and release-status record without changing repository README/cards.

## Implementation and checks

`scripts/v3/package_study.py` uses only CPU image decoding, streaming hashes,
data-only integrity helpers and local file I/O. Its CLI enforces the full frozen
population, stages only in the parent release directory, refuses existing ZIP or
sidecar paths, and removes only its own incomplete outputs on failure. ZIP entries
are sorted, stored without compression and use fixed 1980 timestamps and file
permissions. Every completed entry is streamed back through ZIP CRC and SHA256
validation. `metadata/archive_inventory.json` pins all other entries; the external
sidecar pins the entire archive. Synthetic repeated builds produced identical bytes.

```powershell
& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe `
  -m pytest tests/test_v3_package_study.py -q

& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe `
  -m ruff check scripts/v3/package_study.py tests/test_v3_package_study.py

& C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe `
  scripts/v3/package_study.py `
  --parent-root C:/Users/dianb/blender-material-classification `
  --output C:/Users/dianb/blender-material-classification/artifacts/v3/release-staging/material-classification-v3-study-audit.zip
```

The last command intentionally refuses to overwrite the artifact already built.
The synthetic suite passed **54 tests**, with **one OS-dependent symlink test
skipped** because Windows denied symlink creation; traversal, drive, separator,
case/path conflict and containment checks otherwise ran. Ruff passed. Tests cover
hash mismatches, duplicate bytes/pixels, split/family conflicts, counts, source
exclusions/ancestry, renamed map reuse, receipt/QC identities, metadata privacy,
source preservation, archive exclusions, CRC corruption, deterministic output,
overwrite refusal and cleanup after source changes during packaging.

A separate final inventory check reconciled all image names with the combined
manifest, all archive entries with the inventory, and all 141 packaged provenance
files with both their canonical hashes and unchanged input-byte hashes.
Before/after SHA256 comparison confirmed **128 protected files unchanged**,
including frozen V3 manifests/metadata, the holdout configuration and the previous
original-corpus audit ZIP. Its SHA256 remains:

```text
original-corpus-verified-audit-20261027.zip
0b19b667d48150c9b272a28c26a685b6519c96fd2830a9922e5aaad882f39681
```

Only the new packager, its tests and this report are committed. Training,
inference, configurations, README/cards, frozen inputs and existing packages
remain untouched. No GPU work, agents, Linear operation, PR or upload was used.
