# v0.5 release-notice evidence

Audited 2026-10-27 for the v0.5 Windows app, 21 sample images, and bundled
models. This record covers dependency notices and source evidence. The full
rendered dataset has separate licensing terms and is not included in the
installer.

## Native MIT notice gaps

Both exact MIT notice gaps remain unresolved. The generic MIT template and
manifest authors already retained in the native bundle are not replacements
for an upstream copyright notice. The checked archives' SHA-256 values match
their Cargo.lock checksums:

- `seahash 4.1.0`: `1c107b6f4780854c8b126e228ea8869f4d7b71260f962fefb57b996b8959ba6b`.
  All 16 archive members were inventoried; no license file or copyright/permission
  notice appears in its Rust sources. The exact revision is
  `94b632aeac099031c373599313d5b5f0acbbaec0`. The earlier raw-license attempt was
  HTTP 403; this audit received HTTP 404 at the same raw URL. The exact-revision
  GitLab tree API initially returned HTTP 403, then succeeded. All 17 tree
  entries were inspected (`x-total-pages: 1`, empty `x-next-page`): no license
  file is present. Archive source bytes match the upstream Git blob hashes.
- `simd_helpers 0.1.0`: `95890f873bec569a0362c235787f3aca6e1e887302ba4840839bcc6459c42da6`.
  All seven archive members were inventoried; there is no license file or
  source copyright/permission notice. The complete, non-truncated official
  Git tree at `ca1a2f84aa386d758e98f8a609d990263932fb85` likewise has no license
  file. `LICENSE` and `LICENSE-MIT` raw requests returned HTTP 404.

The primary-source URLs, fresh HTTP outcomes, archive file hashes and retained
Git trees are in [native-notice-gaps.json](../app/licenses/release-evidence/native-notice-gaps.json),
[seahash-4.1.0-tree.json](../app/licenses/release-evidence/seahash-4.1.0-tree.json)
and [simd_helpers-0.1.0-tree.json](../app/licenses/release-evidence/simd_helpers-0.1.0-tree.json).
Neither gap was resolved by borrowing a copyright from a different project,
revision, publisher-generated attribution list or guessed author/year.

## Microsoft runtime provenance

The installed x64 REDIST candidate is:

```text
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT\vcruntime140.dll
File version: 14.44.35211.0
SHA-256: d5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066
```

Authenticode verified as valid, signed by Microsoft Windows / Microsoft
Corporation. The previous package's app-adjacent DLL is byte-identical to
System32's `14.50.35719.0` copy, hash
`184146852727a9db4eea06178716bec3cdbb1015c911f6b0f915b184ad7775b2`.
The native build used MSVC toolset `14.44.35207`, linker
`14.44.35228` and PE linker version `14.44`; the candidate is the matching
14.44 runtime family. The REDIST directory provides explicit redistributable
provenance. Its use remains conditional on applicable terms.
Microsoft supports a runtime at least as recent as the toolset; the folder
version and actual DLL version differ. See Microsoft's
[toolset compatibility guidance](https://learn.microsoft.com/en-us/cpp/windows/determining-which-dlls-to-redistribute).

The installed `Licenses/1033/Redist.txt` links the official
[VS2022 distribution list](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution).
That list covers unmodified files under `VC/redist`, excludes debug runtimes,
and is conditional on Visual Studio license terms. The
[Community terms](https://visualstudio.microsoft.com/license-terms/vs2022-ga-community/)
include requirements for application functionality and protective downstream
terms. The [Build Tools terms](https://visualstudio.microsoft.com/license-terms/vs2022-ga-diagnosticbuildtools/)
are supplemental. Installed files do not by themselves establish the
publisher's entitlement. This audit neither chooses an edition nor claims
that all these conditions are satisfied.

[MICROSOFT_CRT.txt](../app/licenses/MICROSOFT_CRT.txt) is the packaged notice;
[crt-provenance.json](../app/licenses/release-evidence/crt-provenance.json)
records candidates and previous package copies. Python's separate runtime
CRT copies are `14.42.34438.0`; they are inventoried separately. Final package
identities and checksums are recorded in each release's build receipt.

## Python, GPU libraries, models and assets

The bounded inspection used the main checkout's existing
`dist/MaterialStudio-release`. Python 3.13.5 has `runtime/LICENSE.txt`; all 38
`*.dist-info` directories contain named license files. Complete notice file
paths and SHA-256 values are recorded in
[packaged-runtime-audit.json](../app/licenses/release-evidence/packaged-runtime-audit.json).
The packager copies these distribution metadata files, native notices and
the complete `app/licenses` directory. Presence is evidence of retained text,
not certification of every bundled dependency's obligations.

PyTorch `2.11.0+cu128` reports CUDA `12.8` and cuDNN `91900` (9.19.0). The
cuDNN DLL confirms `9.19.0.56`. The requested version-only probe made no
tensors or inference calls, but `torch.cuda.is_initialized()` unexpectedly
reported true afterward; the process exited and no further runtime probes
were made. This is not a GPU performance or inference test.

Official [CUDA 12.8.0](https://docs.nvidia.com/cuda/archive/12.8.0/eula/index.html)
and [12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html) EULA
article text is identical; both are retained because the runtime version
string does not establish its patch release. The full SDK agreement, CUDA
supplement, redistribution list and third-party notices are included.
The matching [cuDNN 9.19.0 EULA](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.19.0/reference/eula.html),
[acknowledgements](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.19.0/reference/acknowledgements.html)
and [notices](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.19.0/reference/notices.html)
are also retained under `app/licenses/upstream`. These terms condition
redistribution on identified components and application distribution terms;
retaining them does not certify the final DLL-to-grant mapping. PyTorch's
existing LICENSE/NOTICE remain intact alongside these supplemental texts.

The existing model attribution contains DINOv2 Apache-2.0 and the exact
[ConvNeXt MIT notice](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE),
with the Apache body, timm and torchvision notices also present. Pinned
pretrained revisions and existing weight hashes are recorded from the runtime
manifest. Large weights were not rehashed in this bounded notice inspection.
No new license was assigned to the study-fitted/fine-tuned checkpoints.

All 21 sample files exist and retain attribution/source URLs plus `CC0-1.0`
source-texture declarations in `app/assets/study.json`. These sample records
do not assign a license to the full rendered dataset. Keep `study.json`,
`DATASET.md`, model notices, Python distribution notices and this supplemental
bundle in the installer.

## Reproduction and validation

`upstream/sources.json` records official URLs, retrieval date, original response
hashes and LF-normalized text hashes. HTML article text omits site navigation;
Microsoft DOCX terms are extracted paragraph text with automatic list markers
not expanded. Sources are retained offline and no entitlement is inferred.

The notice collector accepts a saved Windows Cargo metadata snapshot, the
app manifest and lockfile, and the installed Rust library notices. Use
`--validate` to check graph closure, notice hashes and mapped clauses; use
`--lock` and `--manifest` to match the app build inputs. The v0.5 collection used
`native-metadata-v05.json` and the integrated app manifest and lockfile.
Both MIT notice exceptions listed above remain recorded in the validation.

Final installer and package hashes are separate packaging outputs.

Validation completed: 311 Cargo dependencies, 445 mapped native notice texts,
12 upstream/supplementary files with matching LF hashes, and two explicitly
unresolved MIT exceptions. `--validate` also passed against the integrated
`app/Cargo.lock` and `app/Cargo.toml`. Official archive source bytes were
matched to both recovered Git trees; all saved evidence JSON parses. No
Cargo build/metadata command, UI action or inference test was run here.
