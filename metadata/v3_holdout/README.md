# Frozen independent-source holdout preparation

This is a **restricted semantic source holdout**, not a directly interchangeable
replacement for the original test split. It contains **21 source groups: three per
class, 30 deterministic views each (630 planned images)**. No v3 model metrics or
predictions were opened during selection, implementation, or QC. No fresh images
have been rendered by this implementation task.

## Frozen selection and provenance

`source_registry.json` is the frozen list, with a canonical selection SHA-256,
UTC freeze timestamp, original-manifest hash, all 105 exclusions, canonical asset
URLs, author/publisher credits, CC0 URLs, source/recipe lineage, label rationale,
download URLs and provider hashes. `download_receipt.json` seals SHA-256 hashes of
all **63 downloaded maps**. No textures, archives, renders or blend files belong in
git.

| Class | Canonical source IDs |
|---|---|
| wood | bamboo_veneer, hinoki_planks, rough_pine_door |
| metal | container_side, Metal063, MetalWalkway014 |
| stone | cliff_side, dry_riverbed_rock, slate_floor |
| concrete | brushed_concrete, ribbed_concrete_wall, scuffed_cement |
| ceramic | brown_floor_tiles, dirty_tiles, rounded_square_tiled_wall |
| fabric | rough_linen, ribbed_corduroy, crepe_satin |
| leather | Leather037, Leather038, Leather039 |

Capitalized IDs are ambientCG; the remaining IDs are Poly Haven.

Official live sources used:

- [Poly Haven API](https://polyhaven.com/our-api), `/assets?t=textures`,
  `/info/{id}` and `/files/{id}`; [CC0 license](https://polyhaven.com/license).
- [ambientCG v3 API documentation](https://docs.ambientcg.com/api/v3/assets/),
  `/api/v3/assets?…&include=technique,relations,downloads,…`;
  [CC0 license](https://docs.ambientcg.com/license/).
- [Leather037](https://ambientcg.com/a/Leather037),
  [Leather038](https://ambientcg.com/a/Leather038), and
  [Leather039](https://ambientcg.com/a/Leather039) explicitly identify photometric
  stereo. v3 API relations list no parents or variations for these scans.

The v2 API returned empty lineage fields for procedural sources, so it was **not**
accepted as lineage evidence. `leather_lineage_audit.json` preserves the useful v3
records. Procedural leather candidates share substance parents with originals;
roots LeatherSubstance001 through 006 are all excluded. This also covers the
parent audit's discovery that LeatherSubstance002 derives from root001. Catalog
`next`/`previous` links are navigation, not ancestry. Leather001–007 share one
unrepresented root001 recipe, not seven independent source groups. Approximated
Leather014 and Leather033A–C lack equally strong physical independence evidence
relative to the original approximated leather; they were not used to inflate the
quota to five.

`source_identity_audit.json` compares each selected Poly Haven 1K color, roughness
and normal MD5 with all available same-role 1K JPG/PNG/EXR hashes for **all 96
original Poly Haven IDs**. There are no matches. Whole-recipe `files_hash` and
normalized source names are checked too. This detects exact reuse even under a
different ID. Alias/family checks reject wood_table, wood_table_worn, numbered and
recolored variants. Poly Haven does not publish a complete physical-capture graph;
distinct descriptive sources and hashes support independence but cannot establish
that an undisclosed crop/recolor or common physical origin is impossible.

Only 1K RGB color, roughness and OpenGL normal maps were retained. Poly Haven
normals use PNG where available; ambientCG maps use its 1K JPG tier. HTTP range
reads extract only the three required ambientCG ZIP members, with member CRC
verification; no AO, displacement, DX normals, opacity, metalness, project files
or previews are downloaded. Nominal tiers retain original aspect ratios (linen
is 1024x1026); no resampling/recoloring was performed.

## Endpoint limitations and pre-score rules

Ceramic excludes marble/terrazzo; concrete excludes plaster/paint. These eligibility
rules deliberately narrow the old ambiguous taxonomy. Metal remains a substrate
label: painted container steel and rusted walkway steel use metallic=0; oxidized
polished Metal063 uses metallic=1. No mixed-metal mask is synthesized. Walkway
perforations are represented by color/normal only, without opacity or geometry.
Fabrics receive no extra sheen/anisotropy; scanned suede may resemble textiles.
These are fixed representation limitations, not reasons to drop difficult images.

The quota, no-replacement policy, technical retry rules and numeric data-only QC
thresholds are frozen in `configs/v3/holdout.json`. There are no reserve swaps.
Source unavailability, corruption, or failed QC blocks a complete endpoint. Any
replacement requires a new documented pre-score registry/version and family audit.
Never replace a source, tune appearance, or select views using model outcomes.
Only three source groups per class are independent sampling units; 30 views of one
source do not provide 30 independent material observations.

## Scene and renderer

The standalone renderer constructs a new scene without opening or saving a blend.
It requires **Blender 5.1.1**, build `b70da489d7f4` on this machine (the executable
hash and runtime build are recorded per run). Production/development use EEVEE,
256 render samples, screen ray tracing, AgX / look None / exposure 0 / gamma 1,
sRGB display and 512x512 8-bit RGB PNGs. All paths must be explicit and absolute.
No V1 output, material, scene or report writes are possible through its CLI.

The old generation report and read-only master-scene inspection supplied:

- Unit UV sphere at (0,0,1), 96 segments / 48 rings, smooth shading; 1000 m neutral
  floor, base color (0.25,0.25,0.25), roughness 0.75.
- Camera 50 mm / 36 mm sensor; azimuth -20..20 degrees, elevation 21..37 degrees,
  distance 4.3146..4.8654 m. Old observed sphere height was 58.3–66.2%; the frozen
  acceptable framing band is 55–75%.
- Square 5 m area light, power 1600..2400 W, x=4.5..6.5, y=-2.5..-0.5, z=3.8..5.0,
  aimed at the sphere. World gray 0.05, strength 2.
- Object rotations x/y=-10..10 and z=0..360 degrees. Independent uniforms use the
  V1 parameter ordering and SHA-256 child-seed recipe, master seed 62642.
- `build_materials.py` shader policy: UV → neutral Mapping (scale 1) → linear,
  repeating texture nodes; sRGB color and Non-Color roughness/GL tangent normals,
  normal strength 1. No displacement, AO, added coating or appearance tuning.

The master blend reports saved version 5.2.44; the old generation report does not
state the rendering binary version. Therefore an exact old-engine-version match
is **not claimed**. Run and inspect the old-TRAIN EEVEE pilot before fresh scoring.
The CPU smoke test is engineering verification, not a substitute for that pilot.

The CSV uses `image_path` relative to the explicitly supplied **dataset root**
(`artifacts/v3/holdout_dataset`), e.g. `wood/polyhaven_bamboo_veneer_r01.png`.
Both `split` and `original_split` are `holdout`; `material_id` is the source group.
The committed render manifest contains only its header until production is run.

Resume validates all existing PNG CRCs, complete decompressed scanlines, dimensions,
SHA-256 and deterministic CSV fields before rendering anything. Sources, selection,
configuration, code, realized scene, executable/build and GPU runtime are hashed.
Changed fingerprints, orphan PNGs, missing rows, malformed images and duplicate
rows fail closed. There is no force/overwrite/partial-view flag. A crash between a
PNG and its CSV row creates an orphan; quarantine that incomplete run and rerun the
unchanged protocol. Do not silently import the orphan.

## Exact parent commands (PowerShell, after applying this commit)

The sources already exist at
`C:/Users/dianb/blender-material-classification/artifacts/v3/holdout_sources`.
Use the parent's existing environment; no packages were installed into it.

```powershell
$holdoutProject = 'C:/Users/dianb/blender-material-classification'
$holdoutBlender = 'C:/Program Files/Blender Foundation/Blender 5.1/blender.exe'
$holdoutPython = "$holdoutProject/.venv/Scripts/python.exe"

# Verify the sealed downloads (no network or source changes).
& $holdoutPython "$holdoutProject/scripts/v3/prepare_holdout.py" `
  --registry "$holdoutProject/metadata/v3_holdout/source_registry.json" `
  --original-manifest "$holdoutProject/metadata/material_manifest.csv" `
  --identity-audit "$holdoutProject/metadata/v3_holdout/source_identity_audit.json" `
  --sources-root "$holdoutProject/artifacts/v3/holdout_sources" `
  --receipt "$holdoutProject/metadata/v3_holdout/download_receipt.json"

# Schedule first: seven EEVEE pilot images, one old TRAIN source per class.
& $holdoutBlender --background --factory-startup --python-exit-code 1 `
  --python "$holdoutProject/blender/scripts/v3/render_holdout.py" -- `
  --mode development --config "$holdoutProject/configs/v3/holdout.json" `
  --original-manifest "$holdoutProject/metadata/material_manifest.csv" `
  --development-manifest "$holdoutProject/metadata/material_manifest_split.csv" `
  --development-materials-root "$holdoutProject/materials" `
  --out "$holdoutProject/artifacts/v3/holdout_development" `
  --manifest "$holdoutProject/artifacts/v3/holdout_development/render_manifest.csv"

# After old-pilot data-only review, schedule the 630-image fresh run once.
& $holdoutBlender --background --factory-startup --python-exit-code 1 `
  --python "$holdoutProject/blender/scripts/v3/render_holdout.py" -- `
  --mode holdout --config "$holdoutProject/configs/v3/holdout.json" `
  --original-manifest "$holdoutProject/metadata/material_manifest.csv" `
  --registry "$holdoutProject/metadata/v3_holdout/source_registry.json" `
  --receipt "$holdoutProject/metadata/v3_holdout/download_receipt.json" `
  --identity-audit "$holdoutProject/metadata/v3_holdout/source_identity_audit.json" `
  --sources-root "$holdoutProject/artifacts/v3/holdout_sources" `
  --out "$holdoutProject/artifacts/v3/holdout_dataset" `
  --manifest "$holdoutProject/metadata/v3_holdout/render_manifest.csv"

# Complete, model-free QC before the parent opens fresh scores.
& $holdoutPython "$holdoutProject/blender/scripts/v3/qc_holdout.py" `
  --config "$holdoutProject/configs/v3/holdout.json" `
  --registry "$holdoutProject/metadata/v3_holdout/source_registry.json" `
  --receipt "$holdoutProject/metadata/v3_holdout/download_receipt.json" `
  --identity-audit "$holdoutProject/metadata/v3_holdout/source_identity_audit.json" `
  --original-manifest "$holdoutProject/metadata/material_manifest.csv" `
  --sources-root "$holdoutProject/artifacts/v3/holdout_sources" `
  --dataset-root "$holdoutProject/artifacts/v3/holdout_dataset" `
  --manifest "$holdoutProject/metadata/v3_holdout/render_manifest.csv" `
  --report "$holdoutProject/artifacts/v3/holdout_dataset/qc.json"
```

Check `$LASTEXITCODE` after each command. Do not run these jobs concurrently on the
single GPU. The rendering task was deliberately left to the parent scheduler.

The preparation command can fetch missing sources on a new machine with
`--download` and a new receipt path having the required canonical suffix. Preserve
the committed receipt separately and compare every re-downloaded hash before
accepting any regenerated receipt. An existing sealed receipt is verification-only:
the tool intentionally refuses missing/changed bytes instead of silently updating
the frozen hashes.
