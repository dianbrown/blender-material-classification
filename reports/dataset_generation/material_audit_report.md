# Base-material audit

One 512x512 baseline render per base material. No randomization: validated camera, light position/power, sphere transform, texture mapping and material parameters all untouched.

- Materials audited: **105**
- Clean: **105**
- Flagged: **0**

Measurements are taken inside the sphere silhouette only (central disc at 80% of the projected radius), so the neutral background never dilutes them. `detail` is mean |pixel - local mean|, which separates real texture from smooth shading.

## Thresholds

| Flag | Condition |
|---|---|
| `almost_uniform` | luminance std < 4.0 |
| `nearly_black` | mean < 18 |
| `nearly_white` | mean > 238 |
| `no_visible_texture` | detail < 1.2 |
| `texture_load_failed` | a map file is missing or failed to load |

## Result

No material tripped any flag.

## Lowest-detail materials (texture visibility ranking)

`detail` is absolute, so it shrinks on dark materials; `detail_rel` normalises by mean luminance and is the fairer cross-class comparison. Both rankings are given.

### By absolute detail

| material_id | class | detail | detail_rel | std | mean |
|---|---|---|---|---|---|
| wood_13 | wood | 1.292 | 2.544 | 50.34 | 50.8 |
| metal_03 | metal | 1.323 | 3.795 | 52.77 | 34.85 |
| concrete_11 | concrete | 1.446 | 1.013 | 54.6 | 142.86 |
| leather_03 | leather | 1.553 | 2.174 | 50.15 | 71.45 |
| concrete_06 | concrete | 1.571 | 1.21 | 55.23 | 129.8 |
| concrete_07 | concrete | 1.704 | 1.244 | 54.39 | 136.96 |
| fabric_01 | fabric | 1.822 | 1.181 | 53.82 | 154.22 |
| concrete_13 | concrete | 1.85 | 1.361 | 55.12 | 135.92 |
| concrete_14 | concrete | 2.0 | 2.029 | 53.19 | 98.57 |
| ceramic_04 | ceramic | 2.123 | 1.825 | 54.8 | 116.34 |
| leather_05 | leather | 2.203 | 1.661 | 55.92 | 132.63 |
| wood_09 | wood | 2.227 | 1.724 | 55.54 | 129.19 |

### By brightness-normalised detail

| material_id | class | detail_rel | detail | std | mean |
|---|---|---|---|---|---|
| concrete_11 | concrete | 1.013 | 1.446 | 54.6 | 142.86 |
| fabric_01 | fabric | 1.181 | 1.822 | 53.82 | 154.22 |
| concrete_06 | concrete | 1.21 | 1.571 | 55.23 | 129.8 |
| concrete_07 | concrete | 1.244 | 1.704 | 54.39 | 136.96 |
| concrete_13 | concrete | 1.361 | 1.85 | 55.12 | 135.92 |
| leather_05 | leather | 1.661 | 2.203 | 55.92 | 132.63 |
| wood_09 | wood | 1.724 | 2.227 | 55.54 | 129.19 |
| ceramic_04 | ceramic | 1.825 | 2.123 | 54.8 | 116.34 |
| fabric_08 | fabric | 1.843 | 2.253 | 53.44 | 122.24 |
| fabric_13 | fabric | 1.855 | 2.862 | 53.92 | 154.29 |
| concrete_01 | concrete | 1.874 | 3.174 | 51.03 | 169.38 |
| concrete_15 | concrete | 1.978 | 2.944 | 54.32 | 148.79 |
