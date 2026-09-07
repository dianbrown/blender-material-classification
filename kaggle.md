# Kaggle release guide

This guide describes a possible public release of the synthetic Blender PBR
material-render dataset. It is intentionally a release checklist, not a claim
that the current local corpus is publishable.

## Decision: do not publish the current corpus yet

The current answer is **no-go for a public Kaggle upload**. The images are
derived from 105 local PBR texture sets, but the repository does not contain a
source URL, creator, licence, permission record, or attribution for each asset.
`DATASET.md` explicitly says that provenance has not been normalised, and the
authoritative `metadata/material_manifest_split.csv` only contains local
absolute paths and filenames. The build record
`reports/dataset_generation/_build_records.json` has the same limitation.

Do not select CC0, CC-BY, or another Kaggle licence merely because it is
convenient. A Kaggle licence describes the package you are offering; it cannot
override restrictions in upstream textures. Generated renders may be
derivative works and may inherit attribution, non-commercial, share-alike, or
no-redistribution conditions. Blender and any RenderSynth add-on are separate
software dependencies and should not be bundled without checking their terms.

Publication becomes a go when every source asset has a documented licence that
allows redistribution of the proposed renders, all required attribution is in
the package, and any asset that fails this test is removed and the affected
renders regenerated. If rights cannot be established, publish only the code,
manifests with sensitive paths removed, and a regeneration recipe, or release
a separately regenerated set made only from assets with clear redistribution
rights.

## 1. Licensing and provenance gate

Complete this gate before creating a public archive:

1. Freeze the exact candidate package and record a release version (for
   example `v1.0.0-provenance-audit`). Do not change the locked assignment
   experiment or its TEST split while doing the audit.
2. For every `material_id` in `metadata/material_manifest.csv`, identify the
   original provider page or download URL, creator, asset name, download date,
   licence name and version, and the licence URL. Record a stable file hash of
   each source map (base colour, roughness, normal, metallic, displacement,
   AO, and any other map actually used).
3. Save a copy or archived link to the licence terms as allowed by those terms.
   Note whether the terms permit redistribution, commercial use, derivatives,
   sublicensing, and redistribution inside a dataset. Do not rely on a search
   result or a filename as proof.
4. Check attribution requirements, notices, trademark restrictions, and any
   account or access conditions. Preserve exact attribution text where the
   provider requires it.
5. Check the generated output separately. Record whether the provider treats
   rendered images as derivatives and whether a notice must accompany every
   render or only the dataset README.
6. Remove or replace any asset for which the answer is unknown, contradictory,
   non-redistributable, or limited to personal/non-commercial use. Regenerate
   all 30 renders for the affected material and update both manifests.
7. Ask the lecturer/group members to confirm that public release is permitted
   by the course and that no contributor's private data or credentials are in
   the archive. Keep the decision and sign-off in the release notes.
8. Only then choose a single package licence that is compatible with all
   upstream conditions. If the assets have mixed licences, state the per-asset
   licences and do not imply that one permissive licence covers the restricted
   components.

### Source asset registry template

Create a tracked `metadata/asset_provenance.csv` (or equivalent JSON) with one
row per source map. Replace every placeholder; blank provenance fields are a
release blocker.

```text
material_id,class,map_type,local_relative_path,sha256,source_title,
source_provider,source_url,asset_page_url,creator,downloaded_utc,
license_name,license_version,license_url,redistribution_allowed,
derivatives_allowed,commercial_use,attribution_required,attribution_text,
share_alike_required,notice_required,access_terms,verification_date,
verified_by,notes
wood_01,wood,base_color,materials/wood/wood_01/file.jpg,<hash>,<title>,
<provider>,<URL>,<URL>,<creator>,<YYYY-MM-DD>,<licence>,<version>,<URL>,
yes,yes,yes,no,,no,<notice>,<YYYY-MM-DD>,<name>,
```

Also add a material-level registry (one row per `material_id`) with
`render_rights_decision`, `render_licence_basis`, `required_credit`,
`excluded_maps`, and `audit_status`. The source-folder paths currently written
to the manifests are machine-specific and must be converted to relative paths
before release.

### Derived-render rights record

For each material, retain a short decision record:

```text
material_id | source asset(s) | licence(s) | render redistribution? |
required credit/notice | allowed package licence | reviewer | evidence URL
```

A “yes” answer must be supported by the actual licence terms, not by the fact
that the file was downloadable. If a licence requires attribution, include it
in `ATTRIBUTION.md` and the dataset description. If a licence is share-alike,
check whether the generated images and metadata must carry the same terms. If
the terms prohibit redistribution, keep that material private and do not put
its raw maps, renders, thumbnails, or validation images in the upload.

### Privacy and ethics

The current renders contain no known people, faces, medical records, or
personal identifiers, and the scene is a synthetic sphere. Still check source
pages, EXIF/text metadata, filenames, Blender files, notebook outputs, and
logs for usernames, local paths, API keys, or accidentally embedded assets.
Remove absolute Windows paths and any `.env`, credentials, private checkpoints,
or irrelevant source files. Describe the synthetic-only scope and state that
the data should not be interpreted as a validated real-world material or
safety dataset. Honour takedown requests and keep a contact address for the
maintainer.

## 2. Proposed dataset identity and content

Suggested title: **Blender Synthetic PBR Material Renders**.

Suggested subtitle: **Leakage-safe seven-class EEVEE material benchmark**.

Suggested description (edit after the licence audit):

> A reproducible synthetic image-classification benchmark generated with
> Blender EEVEE and a RenderSynth-style randomisation pipeline. It contains
> 3,150 RGB PNG renders at 512 x 512 from 105 independent PBR base materials,
> with 30 renders per material and seven classes: wood, metal, stone,
> concrete, ceramic, fabric, and leather. All renders from a material remain
> in one split: 2,310 train images (77 materials), 420 validation images (14
> materials), and 420 test images (14 materials). The manifests document the
> material identity, render index, child seed, camera/light parameters,
> resolution, class, and split. The test split is intended for final
> evaluation only. This is a synthetic benchmark; performance does not imply
> generalisation to photographs.

Do not call the package “CC0” or “open” until the provenance gate is complete.
List the exact package licence, all upstream licences, attribution, Blender /
RenderSynth versions, generation seed `62642`, and the repository/report URL.

## 3. Recommended public package layout

Use a clean staging directory outside the repository (the source dataset is
currently about 802 MB before packaging). Keep raw source PBR maps out unless
their licences explicitly allow redistribution and you intend to maintain
them.

```text
blender-pbr-material-renders-v1/
  dataset-metadata.json
  README.md
  LICENSE                         # only the package licence actually granted
  ATTRIBUTION.md                  # per-source credit and licence notices
  DATA_CARD.md                    # purpose, composition, limitations, ethics
  checksums.sha256
  images/
    train/{wood,metal,stone,concrete,ceramic,fabric,leather}/*.png
    val/{wood,metal,stone,concrete,ceramic,fabric,leather}/*.png
    test/{wood,metal,stone,concrete,ceramic,fabric,leather}/*.png
  metadata/
    render_manifest.csv           # relative image paths only
    material_manifest.csv         # relative asset IDs, no local paths
    material_manifest_split.csv
    dataset_split_summary.csv
    asset_provenance.csv
  generation/
    pipeline_config.json
    generation_notes.md
```

If retaining the existing directory convention (`train/...` at the archive
root), document it consistently and ensure the manifest paths match it. Do not
upload `materials/`, `renders/`, `.blend` files, checkpoints, TensorBoard logs,
or notebooks containing embedded local images unless each is intentionally
licensed and needed. A compact release can provide the PNG renders and
metadata while leaving source maps and Blender regeneration dependencies
outside the public package.

### Manifest requirements

The public render manifest should contain at least:

`image_path`, `filename`, `class`, `material_id`, `split`, `render_index`,
`random_seed`, `render_engine`, `resolution_width`, `resolution_height`,
`camera_azimuth_deg`, `camera_elevation_deg`, `camera_distance`,
`area_light_power`, object rotations, and `image_sha256`.

The public material manifest should contain `class`, `material_id`,
`source_material_name`, map availability/default flags, `split`, provenance
record ID, and `provenance_status`. Replace `source_folder` and map filenames
that expose a workstation path with relative paths or stable asset IDs. Never
include a path beginning with `C:/Users/`.

Preserve the locked split exactly: each `material_id` and all 30 of its
renders must occur in one and only one split. Keep the original counts (77/14/14
materials; 2,310/420/420 images; 15 materials per class). Publish the split
manifest so users can reproduce the leakage-safe protocol. If you regenerate
or remove an asset, increment the dataset version and create a new split
manifest; do not silently replace files in version 1.

## 4. Prepare and validate locally

1. Create a staging folder with the package layout and copy only files cleared
   by the licence audit. Keep the current local dataset untouched.
2. Run the repository validator from the project root:

   ```powershell
   python scripts/verify_dataset.py
   ```

   It should report 3,150 decodable RGB PNGs, 105 materials, 30 renders per
   material, correct class/split counts, and no material crossing splits.
3. Generate a local archive and checksum using the existing upload-free helper
   (after choosing the correct staging layout):

   ```powershell
   python scripts/package_dataset.py `
     --dataset-root dataset `
     --output artifacts/blender-pbr-material-renders-v1.zip `
     --metadata-root metadata/public
   Get-FileHash artifacts/blender-pbr-material-renders-v1.zip -Algorithm SHA256
   ```

   The helper writes a `.sha256` sidecar and never uploads. Inspect the ZIP
   before sharing it. Do not pass the unreviewed `metadata/` directory if it
   still contains absolute local paths or records with uncleared provenance.
4. For a Kaggle staging directory, copy/rename the release `images/` and
   `metadata/` folders and place `dataset-metadata.json` beside them. If using
   the helper's archive, extract it into staging and verify that extraction
   does not introduce an extra unwanted directory level.
5. Run dry-run checks before an upload. The first command checks filenames and
   extensions in the staged *relative* package contents; it does not mistake a
   normal staging path such as `C:\\Users\\...\\kaggle-staging` for leaked
   metadata. The second command searches text files for embedded workstation
   paths and common credentials:

   ```powershell
   python -m json.tool kaggle-staging/dataset-metadata.json
   $forbidden = Get-ChildItem kaggle-staging -Recurse -File |
     Where-Object { $_.Name -match '(^|\.)(env|pem|key|secret|credentials)$' -or $_.Extension -in '.pt','.pth','.ckpt','.onnx','.safetensors','.blend' }
   $forbidden | ForEach-Object { $_.FullName }
   rg -n -i -S --hidden --glob '!*.png' --glob '!*.jpg' --glob '!*.exr' --glob '!*.zip' `
     --glob '*.json' --glob '*.csv' --glob '*.md' --glob '*.txt' --glob '*.yaml' --glob '*.yml' `
     'C:[\\/]+Users[\\/]+|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|api[_-]?key|secret|token|password' kaggle-staging
   Get-FileHash kaggle-staging\metadata\render_manifest.csv -Algorithm SHA256
   kaggle datasets list --search "Blender Synthetic PBR Material Renders"
   ```

   Both scans should be empty or explicitly reviewed before release. Compare
   file counts, class counts, split counts, and every image checksum to the
   release manifest. Inspect representative images from every class and split
   after extraction.

## 5. Kaggle account and CLI setup

1. Create/sign in to a Kaggle account, verify the account as required by
   Kaggle, and confirm that public dataset publication is permitted for your
   account. Use an account/organisation slug that will remain stable in the
   citation.
2. Prefer the current authenticated CLI flow:

   ```powershell
   kaggle auth login
   ```

   Complete the browser/device flow and keep the resulting credential private.
   The documented legacy alternative is an API token saved at
   `%USERPROFILE%\.kaggle\kaggle.json`; do not commit it or paste it into a
   notebook. Restrict that file to your Windows user account. If a token has
   ever been exposed, revoke it and issue a new one.
3. Install and test the official CLI in the project environment:

   ```powershell
   python -m pip install --upgrade kaggle
   kaggle --version
   kaggle datasets list --search "titanic" --page-size 1
   ```

   CLI syntax can change, so check `kaggle datasets --help` before a release.
   The current official CLI reference documents dataset initialization,
   creation, versioning, directory modes, status, and metadata commands:
   [Kaggle CLI datasets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
   The metadata schema and licence field are documented in the companion
   [Kaggle dataset metadata reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).

## 6. Metadata file

Run `kaggle datasets init -p kaggle-staging` once to obtain a valid template,
then edit it. Use exactly one licence entry and replace the username/slug.
The licence value below is a placeholder: do not use CC0 unless the upstream
audit proves that it is valid for every included item.

```json
{
  "title": "Blender Synthetic PBR Material Renders",
  "subtitle": "Leakage-safe seven-class EEVEE material benchmark",
  "description": "Synthetic 512x512 Blender EEVEE renders from 105 independent PBR materials. Seven classes: wood, metal, stone, concrete, ceramic, fabric, and leather. Grouped train/validation/test split by material_id; see README.md, DATA_CARD.md, and ATTRIBUTION.md for provenance, limitations, and exact terms.",
  "id": "YOUR_KAGGLE_USERNAME/blender-synthetic-pbr-material-renders",
  "licenses": [{"name": "LICENSE_IDENTIFIER_AFTER_AUDIT"}],
  "keywords": ["computer vision", "image classification", "blender", "pbr", "synthetic data", "materials"],
  "resources": [
    {"path": "metadata/render_manifest.csv", "description": "Image-level render parameters, labels, groups, splits, and checksums."},
    {"path": "metadata/material_manifest_split.csv", "description": "One row per base material with the locked split and provenance record."},
    {"path": "DATA_CARD.md", "description": "Dataset composition, intended use, limitations, ethics, and provenance."}
  ]
}
```

Kaggle's schema may call the file `dataset-metadata.json`; use that exact name.
The title and subtitle length rules are enforced by the CLI/API. Let Kaggle
parse the PNG directories with the documented directory mode, or upload a
tested archive if the current file-size rules require it.

## 7. Create, version, and validate the dataset

First create privately if the account/API supports that workflow. This allows
one last inspection before making the page public:

```powershell
kaggle datasets create -p kaggle-staging -t -r zip
kaggle datasets status YOUR_KAGGLE_USERNAME/blender-synthetic-pbr-material-renders
kaggle datasets files YOUR_KAGGLE_USERNAME/blender-synthetic-pbr-material-renders --page-size 100
```

Use `-r zip` here because the staging layout contains nested `images/` and
`metadata/` directories. In the current CLI, `-r skip` ignores directories and
would upload only top-level files, silently omitting the dataset contents.

After checking the private page, make the first release public using the
Kaggle web UI or the currently supported CLI option. Some CLI versions support
`--public` directly during creation:

```powershell
kaggle datasets create -p kaggle-staging --public -t -r zip
```

Do not run both create commands for the same slug. Confirm the result on the
web page, including the licence, description, file tree, preview images,
credits, and citation. A later release is a new immutable version:

```powershell
kaggle datasets version -p kaggle-staging -m "v1.1.0: corrected provenance and checksums" -r zip
kaggle datasets status YOUR_KAGGLE_USERNAME/blender-synthetic-pbr-material-renders
```

The same `-r zip` directory mode is required for versioning while the package
remains nested. If you intentionally prepare one prebuilt archive as a
top-level upload file, `-r skip` is appropriate only when all required content
is inside that archive and the metadata points to it; verify the downloaded
page before making it public.

Keep the first public version if it has already been cited. For material or
licence changes, explain the change, increment the version, regenerate the
checksum file, and update the data card. Do not overwrite history by replacing
files manually on the web page.

### Size, checksums, and versioning

The current 3,150 PNG files occupy approximately 802 MB locally. Measure the
actual staged size and compare it with Kaggle's current upload and per-file
limits before release; limits and CLI behaviour can change. Prefer the
repository's deterministic ZIP/checksum process, split a release into clearly
documented files only when necessary, and avoid recompressing already
compressed PNGs repeatedly. Keep one `checksums.sha256` covering each archive
or every file, plus the manifest's `image_sha256` column. Record the generation
commit, seed, package version, date, and source-registry version in
`generation/generation_notes.md`.

## 8. Post-publication checks and report citation

After Kaggle reports the upload as complete:

1. Download the dataset to a clean temporary directory with `kaggle datasets
   download ... --unzip` and rerun the structural checks. Compare counts,
   checksums, class labels, paths, and split inheritance to the local release.
2. Open representative files from every class and split. Check that Kaggle did
   not convert, omit, or rename anything unexpectedly.
3. Confirm the page shows the intended licence, attribution, data card,
   version, and contact information. Record the canonical dataset URL and
   publication date in `DATASET.md`, `README.md`, and the report.
4. Cite the dataset by owner, title, version, Kaggle URL, and access date. Add
   the version DOI if Kaggle or a linked archival service provides one. Example
   report text:

   > The rendered benchmark is publicly archived as *Blender Synthetic PBR
   > Material Renders*, version 1.0.0, Kaggle, `<canonical URL>`, accessed
   > `<date>`, DOI `<DOI if available>`. Source-material credits and licence
   > restrictions are listed in `ATTRIBUTION.md`.

5. Add the URL/DOI to the report references and dataset section, not only to a
   GitHub README. Preserve the exact version used for the reported metrics.

## 9. Maintenance and takedown plan

Maintain a changelog and provenance registry. Review source licences and
Kaggle/API terms periodically. If a source provider changes terms or requests
removal, immediately mark the affected material, stop redistribution, publish
an explanatory version, and remove/rebuild affected files as appropriate. Keep
the original local TEST artifacts private and immutable for assignment
reproducibility. Never add tuned models or revised labels to the original
dataset version without a new version and a clear changelog.

Until the asset registry is complete, the safe public release is the code,
documentation, manifests with private paths removed, and a regeneration guide;
the current renders and source textures should remain private.
