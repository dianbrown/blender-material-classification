# Final synthetic material dataset - generation report

**Status: COMPLETE - all checks passed**

| | |
|---|---|
| Total images | **3150** |
| Manifest rows | 3150 |
| Base materials | 105 x 30 renders |
| Render engine | BLENDER_EEVEE |
| Resolution | 512x512, 100%, PNG RGB |
| Master seed | 62642 (SHA-256 child seed per render) |
| Wall-clock | 41 min 48 s |
| Avg per image | 0.796 s |
| Throughput | 75.3 images/min |
| Completed in one pass | yes - single pass, 3150 rendered / 0 skipped / 0 failed |

## Counts per split

| split | images | expected | ok |
|---|---|---|---|
| train | 2310 | 2310 | yes |
| val | 420 | 420 | yes |
| test | 420 | 420 | yes |
| **total** | **3150** | **3150** | yes |

## Counts per class and split

| class | train | val | test | total |
|---|---|---|---|---|
| wood | 330 | 60 | 60 | 450 |
| metal | 330 | 60 | 60 | 450 |
| stone | 330 | 60 | 60 | 450 |
| concrete | 330 | 60 | 60 | 450 |
| ceramic | 330 | 60 | 60 | 450 |
| fabric | 330 | 60 | 60 | 450 |
| leather | 330 | 60 | 60 | 450 |
| **expected** | **330** | **60** | **60** | **450** |

## Validation

| check | result | detail |
|---|---|---|
| exactly 3150 PNG files on disk | PASS | 3150 |
| manifest has 3150 rows | PASS | 3150 |
| no missing files | PASS | 0 |
| no zero-byte files | PASS | 0 |
| no corrupt PNG headers | PASS | 0 |
| all images 512x512 | PASS | 0 |
| no filename collisions | PASS | 0 |
| every material has exactly 30 renders | PASS | 0 |
| no missing render indices | PASS | 0 |
| 105 base materials present | PASS | 105 |
| split inheritance correct | PASS | 0 |
| class inheritance correct | PASS | 0 |
| no material crosses splits | PASS | 0 |
| file paths match split/class | PASS | 0 |
| all seeds unique | PASS | 0 |
| no all-black images (sampled) | PASS | 0 |
| no all-white images (sampled) | PASS | 0 |
| no blank images (sampled) | PASS | 0 |
| no missing-texture magenta (sampled) | PASS | 0 |
| sphere never cropped (all rows) | PASS | 0 |
| sphere 55-75% of frame height (all rows) | PASS | 0 |

Pixel sanity was measured on a stratified sample of 420 images (4 per base material); geometric framing was checked on all 3150 rows.

## Observed randomization envelope

| parameter | observed min | observed max | configured |
|---|---|---|---|
| camera_azimuth_deg | -19.991 | 19.97 | -20 .. +20 (baseline 0) |
| camera_elevation_deg | 21.004 | 37.0 | 21 .. 37 (baseline 25, -4/+12) |
| camera_distance | 4.315 | 4.865 | 4.315 .. 4.865 (0.94-1.06x) |
| area_light_power | 1600.022 | 2399.041 | 1600 .. 2400 (0.80-1.20x) |
| area_light_location_x | 4.502 | 6.499 | 4.5 .. 6.5 (+/-1.0) |
| area_light_location_y | -2.499 | -0.5 | -2.5 .. -0.5 (+/-1.0) |
| area_light_location_z | 3.8 | 5.0 | 3.8 .. 5.0 (+/-0.6) |
| object_rotation_x | -9.995 | 9.994 | -10 .. +10 |
| object_rotation_y | -9.992 | 9.988 | -10 .. +10 |
| object_rotation_z | 0.064 | 359.922 | 0 .. 360 |

Sphere occupied 58.3%-66.2% of frame height across all renders (target 55-75%).

## Failed / repaired renders

None. No render failed, and no image required a retry or repair.

The job ran to completion in a single uninterrupted headless pass; no render was retried or repaired.

## Material / texture mismatches

Preflight verified, for all 105 manifest rows: the Blender material exists, its `base_material_id` custom property matches the manifest `material_id`, every Image Texture node resolves to a file on disk, class and split are valid, and the destination directory is creatable.
`metal_13` had been replaced with `worn_corrugated_iron` before this run; it resolved cleanly under its existing `MAT_Metal_13` name, so no mapping repair was required.

## Contact sheets

- `reports/dataset_generation/contact_sheets/split_train.png`
- `reports/dataset_generation/contact_sheets/split_val.png`
- `reports/dataset_generation/contact_sheets/split_test.png`
- `reports/dataset_generation/contact_sheets/class_wood.png`
- `reports/dataset_generation/contact_sheets/class_metal.png`
- `reports/dataset_generation/contact_sheets/class_stone.png`
- `reports/dataset_generation/contact_sheets/class_concrete.png`
- `reports/dataset_generation/contact_sheets/class_ceramic.png`
- `reports/dataset_generation/contact_sheets/class_fabric.png`
- `reports/dataset_generation/contact_sheets/class_leather.png`
- `reports/dataset_generation/contact_sheets/overview.png`
