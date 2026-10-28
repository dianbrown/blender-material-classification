# File descriptions

The `images/` folder contains **3,780 RGB PNG renders at 512 × 512**, organized
by current partition and visual class. Use `metadata/manifest.csv` as the image
index. Paths below refer to files in the published Kaggle dataset.

**`metadata/manifest.csv`**

Primary index: 3,780 packaged images, labels, groups, current splits, lineage, and integrity hashes.

**`metadata/archive_inventory.json`**

SHA-256 inventory of every other archive entry; excludes itself to avoid a circular hash.

**`metadata/input_hashes.json`**

Original versus normalized provenance-text hashes from the audit packaging step.

**`metadata/release_status.json`**

Public dataset URL, CC0 license, owner approval, and build-time release scope.

**`DATASET_CARD.md`**

Dataset composition, generation, source attribution, partition use, license, and limitations.

**`LICENSE`**

CC0 rendered-image and project-authored metadata terms; retained upstream notices have separate scope.

**`dataset-metadata.json`**

Kaggle metadata snapshot for this existing dataset.

**`provenance/metadata/material_manifest.csv`**

Historical Blender material import and source-map audit records for 105 materials.

**`provenance/metadata/material_manifest_split.csv`**

Historical material audit with original partition assignments; lineage only.

**`provenance/metadata/render_manifest.csv`**

Historical development render settings and image identities; original paths and splits.

**`provenance/metadata/v3/manifest.csv`**

Frozen V3 development audit: 3,150 original image identities and known source families.

**`provenance/metadata/v3/source_maps.csv`**

Audited development source texture identities, PBR roles, dimensions, and hashes; maps excluded.

**`provenance/metadata/v3_holdout/classification_manifest.csv`**

Frozen 630-image study test index with original hashes and 21 source groups; now evaluated.

**`provenance/metadata/v3_holdout/render_manifest.csv`**

Test rendering parameters, source IDs, engine version, and provenance fingerprints.
