# Material Studio v0.5

User-authorized scope: publish a GitHub release tagged `v0.5` with a Windows
installer, bundle models and samples, add a calm minimal cube icon/favicon,
and update the README with a looping showcase of the real native application.

Base: `ca9efdf`; branch: `release/v0.5`. Native package version: `0.5.0`.
Repository visibility remains unchanged. Do not publish the full Kaggle study
archive or change its unresolved dataset licence/release approval.

## Work and ownership

- Installer agent: Inno Setup recipe, validated build wrapper, installation guide
  and focused tests. Per-user install, shortcuts and safe uninstall. GitHub
  release assets must each remain below 2 GiB.
- Icon agent: original vector cube identity, PNG/ICO/favicon exports, Windows
  PE icon/version resources, app version and locked build. Sole Cargo owner.
- Notices agent: source-bound native notices, CRT redist provenance and applicable
  bundled runtime terms. No invented copyright or compliance certification.
- Parent: integration, actual GUI captures, GIF/video encoding, final package,
  installer install/inference/uninstall validation, reviews, README and release.

## Gates

1. Validate icon at small and large sizes and in the executable resources.
2. Validate exact packaged inputs and record hashes before compiling installer.
3. Install into an isolated local test destination; verify bundled resources,
   native startup and actual offline prediction. Test uninstall without deleting
   unrelated files. Do not alter the trained models or sealed study results.
4. Inspect showcase frames and loop. Use real application results, crop only
   window chrome if useful, and disclose shortened pauses rather than implying
   benchmark timing. No private dialogs or unrelated desktop content.
5. Independent review; relevant local and hosted checks; publish tag and installer
   assets with SHA-256 checksums and truthful Windows/signing/size instructions.

Tracking: Linear PER-378, assigned only to HectoLogic. Existing app/report work
is complete; this release phase adds distribution and presentation assets.
