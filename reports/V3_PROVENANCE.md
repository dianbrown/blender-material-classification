# Original-source provenance verification

Verified on **2026-10-27 (UTC)**. Scope: the original 105 sources in `metadata/material_manifest.csv`; new holdout assets are excluded.

All **315 manifest-referenced maps** match official source files by hash and size: **288 Poly Haven maps** match provider-published MD5 values; **27 ambientCG maps** match SHA-256 values computed from freshly downloaded official archive members. There are **zero mismatches, missing referenced files, or unmatched filenames**. An additional unreferenced displacement map was discovered and independently verified; the directory therefore contains **316 verified maps**, while the manifest still references 315.

Source-asset provenance is sufficient to support staging these verified CC0 inputs. This is **not a blanket release clearance** for rendered images, other scene inputs, repository code, or new assets. Render-to-source lineage and final release contents remain for the integrating agents to check.

## Verification and reproducibility

The machine-readable evidence is [provenance_verification.json](results/v3/provenance_verification.json). It records every material, provider, official asset URL, author evidence, license evidence and retrieval time, provisional or confirmed family, generator ancestry, map path, size, local SHA-256/MD5, official comparison values, and verification tier. Original absolute `source_folder` paths refer to an older checkout; actual files were resolved as `materials/{class}/{material_id}/{filename}`. The manifest SHA-256 is `ecee6bb357ab348d7d71f07f4073f7059834bf23c77419043c023b74c0ecb654`.

| Tier | Sources | Manifest maps | Evidence |
|---|---:|---:|---|
| T2: official metadata hash match | 96 | 288 | Local MD5 and size equal same-filename entries in `https://api.polyhaven.com/files/{slug}`. |
| T3: official download SHA-256 match | 9 | 27 | Local SHA-256 and size equal decompressed members of official 2K JPG ZIPs. |
| Identifier-only | 0 | 0 | No source is counted as verified merely from its name. |

Poly Haven source bytes were not downloaded again. MD5 is not collision-resistant; it is the checksum the official API publishes. Local SHA-256 independently pins every audited file but is not a remotely published SHA-256 comparison. ambientCG supplied archive names, sizes and official links, but no per-member cryptographic hashes in the retrieved metadata (`zipContent` was empty). The ZIP member hashes were calculated here after fetching official HTTPS downloads. ZIP CRC32 values and archive SHA-256/MD5 values are also recorded; CRC alone was not treated as cryptographic verification.

The nine ambientCG archives total **229,169,660 bytes (229.17 MB)**, below the 500 MB cap. They were held in memory and discarded; no source assets were replaced or added. Retained metadata responses total about 4.30 MB, plus small exploratory metadata/page requests. No multi-GB archives, GPU work, package installations, or commits were used. Metadata requests used an identifying user-agent, at most three workers, a 0.4-second global interval, a 25-second timeout and at most two attempts. Archive requests were sequential with a 40-second socket timeout.

To repeat the verification: pin the manifest hash; hash the resolved local bytes; fetch the official catalog and each Poly Haven `/files/{slug}` response; match filenames, size and MD5; fetch only the exact ambientCG `*_2K-JPG.zip` archives; compare decompressed named members by SHA-256 and size. Preserve retrieval timestamps and exact URLs. A changed local map requires a new comparison. Response digests pin the metadata responses retrieved at this time; full HTML/API bodies are not archived in these reports.

## License and attribution evidence

[Poly Haven's license page](https://polyhaven.com/license) states that its assets are original or directly contributed works, dedicates the assets under CC0, and allows redistribution and commercial use. [ambientCG's license page](https://docs.ambientcg.com/license/) applies CC0 1.0 Universal to downloadable assets and permits modification and redistribution. Both point to the [CC0 1.0 dedication](https://creativecommons.org/publicdomain/zero/1.0/). This report links those statements to the actual local file bytes; it does not manufacture a license from a matching identifier.

Poly Haven's per-asset author names and roles are recorded from its [official texture catalog](https://api.polyhaven.com/assets?t=textures). ambientCG's retrieved asset API/pages did not expose an individual author credit. Its [official imprint](https://docs.ambientcg.com/legal/) names **Lennart Demes as operator**; the JSON deliberately leaves individual author unknown rather than treating operator identity as proof of authorship. Publisher attribution is ambientCG. This authorship detail does not negate the provider's express CC0 dedication.

Poly Haven asset CC0 and [live API terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md) are separate. Provenance metadata retrieval was **Powered by Poly Haven**; no endorsement is implied.

## ambientCG archives and generator lineage

The [official asset API](https://ambientcg.com/api/v2/full_json?id=Leather011,Leather016,Leather034B,Leather022,Leather008,Leather036B,Leather009,Leather030,Leather032&include=displayData,relationshipData,downloadData) provides the child `createdUsing` relationships; the [official generator API](https://ambientcg.com/api/v2/full_json?id=LeatherSubstance001,LeatherSubstance002,LeatherSubstance003,LeatherSubstance004,LeatherSubstance005,LeatherSubstance006&include=relationshipData) independently confirms them through parent `basedOnThis` lists. Exact URLs, dates, archive redirect URLs, response hashes, member hashes and sizes are retained in the JSON.

| Material | Official source | Immediate parent | Archive bytes | Verified maps |
|---|---|---|---:|---:|
| leather_07 | [Leather011](https://ambientcg.com/a/Leather011) | LeatherSubstance002 | 18,612,864 | 3/3 |
| leather_08 | [Leather016](https://ambientcg.com/a/Leather016) | LeatherSubstance003 | 26,142,681 | 3/3 |
| leather_09 | [Leather034B](https://ambientcg.com/a/Leather034B) | LeatherSubstance006 | 28,397,378 | 3/3 |
| leather_10 | [Leather022](https://ambientcg.com/a/Leather022) | LeatherSubstance004 | 23,838,474 | 3/3 |
| leather_11 | [Leather008](https://ambientcg.com/a/Leather008) | None reported | 27,974,224 | 3/3 |
| leather_12 | [Leather036B](https://ambientcg.com/a/Leather036B) | LeatherSubstance006 | 31,599,460 | 3/3 |
| leather_13 | [Leather009](https://ambientcg.com/a/Leather009) | LeatherSubstance002 | 16,934,182 | 3/3 |
| leather_14 | [Leather030](https://ambientcg.com/a/Leather030) | LeatherSubstance005 | 29,531,483 | 3/3 |
| leather_15 | [Leather032](https://ambientcg.com/a/Leather032) | LeatherSubstance005 | 26,138,914 | 3/3 |

The confirmed shared families are:

- **Leather011 + Leather009 -> LeatherSubstance002 -> LeatherSubstance001.** Keep `leather_07` and `leather_13` together. Future holdouts must account for the older ancestor and its other descendants, not only the immediate parent.
- **Leather030 + Leather032 -> LeatherSubstance005.** Keep `leather_14` and `leather_15` together.
- **Leather034B + Leather036B -> LeatherSubstance006.** Keep `leather_09` and `leather_12` together.

Leather016 derives from LeatherSubstance003; Leather022 derives from LeatherSubstance004. They have no second original-set member with those parents, but future additions must be checked against them. Leather008 reports `PBRApproximated` and no parent; the absence of a relationship is not proof of independent origin. The API's human-readable creation-method label incorrectly reports "Height field photogrammetry" even for some different method codes. The audit preserves that discrepancy and uses the explicit code and relationship fields.

Collapsing only the three confirmed pairs produces **102 provisional family roots**, not evidence of 102 independent physical captures. `asset_family` preserves immediate-parent groupings; `family_root_for_split` follows known ancestor chains. Keep generator relatives and all rendered derivatives in the same split.

## Wood-table lineage and other possible families

[wood_table](https://api.polyhaven.com/info/wood_table) credits Dimitrios Savva, reports 1000 x 1000 mm and a current publication date of 2025-10-14. [wood_table_worn](https://api.polyhaven.com/info/wood_table_worn) credits Savva for photography and Rico Cilliers for processing, reports approximately 550 x 550 mm and 2022-07-08. Neither retrieved record provides a parent/derived link. Local diffuse maps show different surface layouts. These observations neither establish a common source nor prove independent capture. **Lineage remains unresolved.** Conservatively keep `wood_02` and `wood_03` together if an independence-sensitive split is required, documenting that as a precaution.

Other obvious naming/creator candidates were checked in official catalog metadata: shutter and corrugated-iron pairs; metal-plate and rusty-metal variants; clay-roof-tile variants; concrete_wall_008/009; denim variants (including the official spelling `denmin_fabric_02`); hessian_230/380; fabric-leather/brown-leather and red/white leather groups. No explicit common parent was found. Eleven fabric sources share colormass/Rico Cilliers credits and a current publication timestamp; that supports a shared production cohort, not an assertion that they share one physical sample or generator.

There are no identical SHA-256 map bytes among the 315 manifest maps. A supplemental CPU SIFT screen of 21 candidate diffuse-map pairs found no strong positive geometric overlap (at most five homography inliers; wood-table pair: three ratio matches, zero inliers). These negative screens have limited recall on repetitive/low-texture surfaces and transformed derivatives. They do not establish independence. Candidate lists, method settings and pair results are retained separately from confirmed lineage in the JSON.

## Extra map and release gaps

`materials/metal/metal_13/worn_corrugated_iron_disp_2k.png` is present but absent from the manifest map columns. Its **5,793,719 bytes** match the [official map metadata](https://api.polyhaven.com/files/worn_corrugated_iron), with MD5 `0da53ea44a76b756f1697bd7669ad504` and local SHA-256 `59ef04b29ae9f5a58c6c1869c37386d32e3b55a6dc473d2e5e7c16782ff0f844`. It is verified separately under `unreferenced_asset_verification`. Its actual render usage was not established; staging should explicitly include or exclude it rather than silently changing the manifest count.

Remaining gaps are concrete:

1. Historical download receipts and acquisition dates were not available. This audit establishes equality with current official files on the retrieval date. Provider publication dates are not local acquisition or capture dates.
2. Poly Haven's checksum tier is MD5 plus size; no fresh Poly Haven source-byte SHA-256 comparison was performed.
3. Individual ambientCG authorship and most physical-capture/generator lineage remain unreported. Confirmed generator families must be grouped; unverified singleton labels must not be described as independent-source guarantees.
4. This source audit does not prove which source map produced each rendered image, inventory other external Blender inputs, or inspect the final Kaggle package. The integrating agent must verify that lineage and preserve the evidence for exactly the shipped assets.
5. CC0 rights evidence here covers these verified provider assets. It does not itself dedicate the user's rendered dataset or license repository code, website copy, previews from other sources, or future holdout assets. The release owner must select the dataset license within the rights they hold.

**Staging handoff:** retain the JSON and this report; ingest exact provider IDs, source URLs, author-status fields, license evidence/retrieval dates, hashes and confirmed ancestor-aware split groups. Reverify changed maps. Source-byte/license provenance passes for all original maps; overall Kaggle readiness remains conditional on render lineage and final package validation.

Only the two requested report paths were written. Source assets, metadata, code, repository history and release state were left untouched.
