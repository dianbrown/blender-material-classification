WINDOWS NATIVE NOTICE BUNDLE — LOCAL DEMO

Deliver app/THIRD_PARTY_NOTICES.txt via the packager's --native-notices FILE.
That UTF-8 file is self-contained, with complete license/notice text bodies
deduplicated by SHA-256 and explicit package-to-text mappings. The entire
app/licenses directory may also be included as supplementary audit evidence.
There is no need to distribute a second copy of every crate's license file.

Scope
-----
The Windows application graph was supplied by the native app agent on
2026-10-27 using cargo metadata --locked --offline --format-version 1
--filter-platform x86_64-pc-windows-msvc. No Cargo command or build was run
by this collector. It reads that snapshot, the actual application Cargo.lock,
cached package manifests, license files, source comments, and the installed
Rust 1.99.0 standard-library copyright collection.

311 third-party Cargo packages are accounted for: 248 normal runtime
dependencies and 63 build-only tools/proc macros, retained conservatively
for generated code. Dev edges are excluded. 216 other Cargo.lock entries
are excluded because they are outside the Windows root dependency closure.
This is a dependency-level inventory, not linker-level dead-code analysis.

All 14 selected Zed crates declare Apache-2.0 individually, at revision
aed37547d8543000494fb69101c05daec5891068. Windows materializes their license
symlinks as small relative-path files; the collector resolves those paths
to the actual full Apache license. Zed's root application license is not
used for these framework crates. No unrelated Zed application is included.

Rust's installed COPYRIGHT-library.html is converted into text with its
file/package mappings intact and repeated notice bodies deduplicated. That
upstream standard-library collection conservatively covers other library
targets too. Compiler-only COPYRIGHT.html and compiler/documentation-only
licenses are not included. Rust is additional to the 311 Cargo packages.

Declared SPDX and the permissive option used are recorded separately. The
legacy Cargo slash notation is normalized to OR. Original alternate terms,
source-level notices, and rav1e's AOM patent license are preserved. Authors
are copied from manifests and are not presented as invented copyright claims.

Known unresolved evidence
-------------------------
* seahash 4.1.0 (runtime), MIT: the published archive has no license file or
  copyright notice. Its official GitLab raw license URL at archive revision
  94b632aeac099031c373599313d5b5f0acbbaec0 returned HTTP 403.
  Actual manifest authors: ticki <ticki@users.noreply.github.com>;
  Tom Almeida <tom@tommoa.me>.
* simd_helpers 0.1.0 (build proc macro), MIT: no license file or copyright
  notice in the archive. The exact upstream Git tree at
  ca1a2f84aa386d758e98f8a609d990263932fb85 also has no license file;
  LICENSE and LICENSE-MIT raw paths returned HTTP 404.
  Actual manifest author: Luca Barbato <lu_zero@gentoo.org>.

The full standard SPDX MIT template is included for these two packages;
its placeholder is intentionally NOT replaced with a fabricated copyright.
This documents their declared terms but does not supply the missing exact
copyright/permission notices. The local demo can retain this transparent
inventory; do not describe it as a complete compliance certification.

taffy 0.13.0 and pulp-wasm-simd-flag 0.1.1 also omit license files in their
archives. Their actual MIT notices were obtained from the exact published
upstream revisions recorded in .cargo_vcs_info.json. For Apache alternatives
whose archives omit license files, the full official Apache 2.0 text is
included with package authors and captured source-level notices. All fetched
text URLs and raw SHA-256 digests appear in upstream/sources.json. The
collector itself is offline and uses those retained primary-source files.
Rendered notice bodies normalize line endings and trailing line whitespace;
original source bytes are fingerprinted separately. Validation uses LF hashes
for generated notices, fetched text and application inputs so Git's Windows
CRLF checkout conversion does not produce a false mismatch.

Packaging boundary
------------------
The parent reports VCRUNTIME140.dll adjacent to the executable; other PE
imports are Windows system libraries, including ICU, D3D11, DComp, and
DirectWrite. There is no fxc/dxcompiler runtime. Microsoft CRT distribution
terms and the final binary file inventory belong to the packaging agent,
as do Python/runtime packages, ML models, textures and other application
assets. They are not silently certified by the Cargo inventory. This bundle
does not select a license for this project's own code. No private repository
content was read and nothing was uploaded or published.

Evidence and validation
-----------------------
inventory.json records package/version/source/checksum, SPDX, features,
role, actual manifest authors, evidence paths/digests, upstream lookup gaps,
Rust provenance, and the complete notice file's SHA-256.
windows-dependency-graph.json records the metadata snapshot hash, lockfile
and app-manifest hashes, dependency edges, and excluded lockfile packages.
The original metadata file is not copied: it includes machine-local paths
and much redundant Cargo package data. Its SHA-256 pins the supplied input.

From a checkout containing these files:

  python scripts/demo/collect_native_notices.py --validate

To also verify the current app's inputs:

  python scripts/demo/collect_native_notices.py --validate --lock app/Cargo.lock --manifest app/Cargo.toml

To regenerate (obtain metadata only at an app-agent-approved Cargo idle
boundary; never block an active build):

  python scripts/demo/collect_native_notices.py --metadata PATH_TO_METADATA.json --lock PATH_TO_APP/Cargo.lock --manifest PATH_TO_APP/Cargo.toml --rust-doc PATH_TO_RUST_1.99/share/doc/rust

Validation checks graph closure, exact package coverage, mapped text IDs,
body hashes, presence of selected license terms, Rust mappings, and optional
input hashes. A successful validation reports unresolved exceptions; it
does not turn them into resolved legal evidence. Regenerate after changing
the lockfile, manifest, target/features or Rust toolchain. Inspect the final
package separately for newly bundled runtime DLLs and assets.
