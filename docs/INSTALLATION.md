# Material Studio v0.5 for Windows

Download all three files from the v0.5 release assets into the same folder:

- `MaterialStudio-0.5.0-windows-x64-Setup.exe`
- `MaterialStudio-0.5.0-windows-x64-Setup-1.bin`
- `MaterialStudio-0.5.0-windows-x64-Setup-2.bin`

Keep the filenames unchanged and run the EXE. The BIN files are installation
data, not separate programs. The split keeps each download below GitHub's
per-file limit; installation combines them automatically.

Setup installs for the current user, by default in
`%LOCALAPPDATA%\Programs\Material Studio`, without requesting administrator
rights. It creates a Start Menu folder and offers an optional desktop shortcut
(unchecked by default). The final page offers to launch Material Studio.
Silent installations never launch the app.

The installer includes the native app, isolated Python runtime, four models,
sample images, and notices. Installation and inference need no host Python,
runtime downloads, model downloads, or network connection. The installer adds
no updater, telemetry, service, or PATH entry. Close the app before reinstalling.

The OS gate is **Windows 10 version 1903 (build 18362) or later, x64 Intel/AMD**.
The native UI needs a DirectX 11 capable graphics adapter and a suitable driver;
CPU inference does not remove the UI's graphics requirement. CUDA inference
additionally needs compatible NVIDIA hardware and an installed NVIDIA driver.
Drivers are not installed by Setup. This OS floor follows
[upstream GPUI/Zed requirements](https://zed.dev/docs/installation#system-requirements),
not Inno Setup's much older default. It is a compatibility gate, not a claim
that every older Windows build has been tested with this release. ARM64 and
32-bit Windows are outside this x64 release's validated scope.

Allow room for the approximately 4.8 GiB installed payload, the downloaded
installer assets, and temporary installation files. Setup checks destination
space. Keep additional headroom for your own images and application output.

Uninstall through Windows Settings > Apps > Material Studio, or its Start Menu
uninstall shortcut. Inno Setup removes the files and shortcuts it recorded
during installation, and directories only when empty. User-added files,
including images placed in the installation folder, remain. There is no
recursive deletion rule. Files originally installed by Setup are still owned
by the installer, so keep personal work under different filenames.

## Build the release installer

Build on Windows with Python 3.13 and Inno Setup 7.1.0 or newer (7.1.0 tested).
Python is needed on the **build host only**. The builder uses only the Python
standard library. The portable package is built separately; see
[DEMO_PACKAGING.md](DEMO_PACKAGING.md). Do not rebuild an installer from the old
`dist/MaterialStudio-release` candidate. Use the completed final
`dist/MaterialStudio-v0.5` package and the final icon from `app/assets/icon.ico`.

The builder requires explicit, independently established EXE and icon SHA-256
values. It checks the native PE architecture and fixed file version `0.5.0.0`,
the icon structure, every manifest file size/hash, and the exact package file
inventory before compiling. Versioned manifests with `release_version=0.5.0`
are accepted, including `release_status=prepared-release`; no approval flag is
required. The original manifest schema is also accepted when paired with the
explicit final identity checks. No source files are changed or regenerated.

Run from the integrated repository in PowerShell:

```powershell
$python = 'C:/Users/dianb/blender-material-classification/.venv/Scripts/python.exe'
$source = 'C:/Users/dianb/blender-material-classification/dist/MaterialStudio-v0.5'
$icon = (Resolve-Path 'app/assets/icon.ico').Path
$compiler = 'C:/Users/dianb/.t3/worktrees/blender-material-classification/t3-7d26daf1/artifacts/installer-tools/Inno Setup 7/ISCC.exe'
# Identity of the final native release supplied by its builder, not inferred from an old package:
$exeHash = '34c5257c74fcc77ef37b3b6f3bfd0ea0e14ba4897bc6e178dd090deeb44805d2'
# The reviewed icon in the integrated release checkout:
$iconHash = (Get-FileHash -LiteralPath $icon -Algorithm SHA256).Hash.ToLowerInvariant()
& $python scripts/demo/build_installer.py `
  --source $source --output dist/installer-v0.5 `
  --icon $icon --exe-sha256 $exeHash --icon-sha256 $iconHash `
  --compiler $compiler --layout auto
if ($LASTEXITCODE -ne 0) { throw 'Installer build failed; inspect the compiler log' }
```

All input/output/compiler paths are configurable. Output must be a new directory
outside the package. Compiler discovery checks `INNO_SETUP_COMPILER`, PATH,
standard Program Files and per-user locations, and
`artifacts/installer-tools/Inno Setup 7/ISCC.exe`. An explicit `--compiler` takes
precedence. The compiler progress log is under `output/build/<layout>/compiler.log`.

`--layout auto` first uses solid LZMA2 compression to build one Setup EXE. If
that file reaches GitHub's limit, it recompiles using disk spanning with
2,000,000,000-byte slices. `--layout single` fails if its output is too large;
`--layout spanned` goes directly to slices. If Inno itself refuses an oversized
single file before finishing, rerun with `--layout spanned` and a new output
directory. A generic compiler failure is never silently hidden by a fallback.
No compression ratio is assumed from the approximately 5.15 GB input size.

[GitHub requires each release asset to be strictly under 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).
Inno's [disk spanning](https://jrsoftware.org/ishelp/topic_setup_diskspanning.htm)
produces a Setup EXE plus numbered data slices, with the maximum size controlled
by [DiskSliceSize](https://jrsoftware.org/ishelp/topic_setup_diskslicesize.htm).
Every generated release binary is size-checked and SHA-256 hashed. Inputs are
revalidated after compression before any success receipt is emitted.

Only the fresh `output/release/` directory is the deliverable:

- The Setup EXE and any matching `.bin` slices.
- `build-receipt.json`: package-manifest hash, native/icon identity, compiler
  identity, version, source sizes, layout, compilation attempts and binary hashes.
- `SHA256SUMS.txt`: hashes of all binaries and the receipt itself.

Do not publish `output/build/`; it may retain an oversized rejected single EXE
and contains local build paths/logs. The builder does not upload, tag, commit,
sign, or run the application. A success receipt is build evidence, not evidence
of GPU testing, visual QA, or Authenticode signing.

## Inno Setup compiler installation

The implementation was tested with the official x64 Inno Setup 7.1.0 compiler
installed per-user under the ignored `artifacts/installer-tools` directory.
The downloaded `innosetup-7.1.0-x64.exe` had SHA-256
`0362a383ed217d4c4239b5933866dd96d3eb2102737da92f80f6057a4b40df2f`
and a valid Authenticode signature from **Pyrsys B.V.**

For another build machine, use the [official downloads page](https://jrsoftware.org/isdl.php)
and [verification instructions](https://jrsoftware.org/isdl-verify.php). This
bootstrap is a separate developer step; the release installer does not download
the compiler. Example PowerShell for the pinned tool:

```powershell
$tools = Join-Path (Get-Location).Path 'artifacts/installer-tools'
New-Item -ItemType Directory -Path $tools -Force | Out-Null
$download = Join-Path $tools 'innosetup-7.1.0-x64.exe'
Invoke-WebRequest 'https://github.com/jrsoftware/issrc/releases/download/is-7_1_0/innosetup-7.1.0-x64.exe' -OutFile $download
$signature = Get-AuthenticodeSignature -LiteralPath $download
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Pyrsys B\.V\.') {
  throw 'Unexpected or invalid Inno Setup signer'
}
$target = Join-Path $tools 'Inno Setup 7'
$process = Start-Process -FilePath $download -WindowStyle Hidden -Wait -PassThru `
  -ArgumentList @('/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
    '/SP-', '/NOICONS', '/TASKS=', '/MERGETASKS=!fileassoc', ('/DIR="' + $target + '"'))
if ($process.ExitCode -ne 0) { throw "Inno installation failed: $($process.ExitCode)" }
```

## Tests and isolated installation QA

Synthetic integrity tests need no model runtime, app build, Cargo, or GPU:

```powershell
& $python -m pytest tests/test_installer.py -q
# Also compile tiny fixtures using the actual compiler (single EXE and spanning):
$env:MATERIAL_STUDIO_TEST_ISCC = $compiler
& $python -m pytest tests/test_installer.py -q
# Optional Windows CLI-only install/uninstall test with a separate fixture AppId:
$env:MATERIAL_STUDIO_TEST_INSTALL = '1'
& $python -m pytest tests/test_installer.py -q
```

For parent QA of the **real** release, use a fresh isolated directory, not a
directory containing another installation or personal files. `/NOICONS` disables
Start Menu shortcuts; `/TASKS=""` explicitly clears the optional desktop task.
These [standard Inno command-line options](https://jrsoftware.org/ishelp/topic_setupcmdline.htm)
do not change the normal interactive default. Silent install still creates the
current user's Windows uninstall registration for Material Studio, so use a
test Windows account or VM if another Material Studio installation exists.

```powershell
$setup = (Resolve-Path 'dist/installer-v0.5/release/MaterialStudio-0.5.0-windows-x64-Setup.exe').Path
$qa = Join-Path $env:LOCALAPPDATA ('MaterialStudio-QA-' + [guid]::NewGuid())
$installLog = Join-Path $env:TEMP ('MaterialStudio-install-' + [guid]::NewGuid() + '.log')
$p = Start-Process -FilePath $setup -WindowStyle Hidden -Wait -PassThru `
  -ArgumentList @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/NOICONS',
    '/TASKS=""', ('/DIR="' + $qa + '"'), ('/LOG="' + $installLog + '"'))
if ($p.ExitCode -ne 0) { throw "Installation failed: $($p.ExitCode)" }
# Verify installed payload hashes; then separately run the intended native/offline QA.
& $python scripts/demo/package_app.py verify --package $qa
$sentinel = Join-Path $qa 'user-added-image.txt'
Set-Content -LiteralPath $sentinel -Value 'This personal file must survive uninstall.'
$uninstaller = Join-Path $qa 'unins000.exe'
$p = Start-Process -FilePath $uninstaller -WindowStyle Hidden -Wait -PassThru `
  -ArgumentList @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')
if ($p.ExitCode -ne 0) { throw "Uninstall failed: $($p.ExitCode)" }
if (-not (Test-Path -LiteralPath $sentinel)) { throw 'User-added file was removed' }
if (Test-Path -LiteralPath (Join-Path $qa 'MaterialStudio.exe')) { throw 'App remains installed' }
```

No wildcard or recursive deletion is needed to run QA. Keep the logs and the
preserved sentinel as evidence. The uninstaller follows Inno's file log; its
documentation explicitly advises against [recursive uninstall cleanup](https://jrsoftware.org/ishelp/topic_uninstalldeletesection.htm).

## Signing and redistribution

The builder does not Authenticode-sign the application, Setup, or uninstaller.
Windows may show an unknown publisher or SmartScreen warning for an unsigned
release. SHA-256 files establish integrity against a trusted release listing;
they are not a publisher signature. If a separate signing process changes any
deliverable, regenerate the relevant receipt/checksums from the signed bytes.
Sign the native EXE **before** the portable manifest is sealed.

The compiler vendor's signature does not sign Material Studio. Inno Setup's
current terms distinguish commercial licensing; consult its
[official license guidance](https://jrsoftware.org/isdl.php). Retain the bundled
native dependency notices, Python/distribution license texts, model attribution,
sample attribution and dataset documentation when distributing the payload.
The installer preserves these files, but does not create additional rights to
redistribute third-party assets. The final package uses the VS 2022 Build Tools
MSVC redistributable `vcruntime140.dll` selected by the parent build, not an
arbitrary DLL copied from System32; redistribution remains subject to the
applicable Microsoft toolchain license. No machine-wide VC runtime or NVIDIA
driver installer is run.
