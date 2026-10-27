# Material Studio portable Windows folder

The output is a **folder**, with `MaterialStudio.exe` at its top level. Keep the
whole folder together when moving it to the RTX 4070 machine. Python, the saved
models, pretrained backbones, and CUDA runtime libraries are bundled. A separate
Python installation, the development repository, and an internet connection are
not needed for inference. The GPU uses the machine's NVIDIA driver; the worker's
`--device auto` selects CUDA when available and otherwise uses CPU.

This packager does not build Rust, install packages, change the validated Python
environment, download weights, publish, or upload anything. Run it using the
existing validated **Windows x64 Python 3.13.5** environment. Python/model versions
are checked against the saved model metadata, not an unpinned requirements solve.

## Prepare the runtime before the app is ready

From the integrated repository, in PowerShell:

```powershell
$repo = (Get-Location).Path
$python = Join-Path $repo '.venv/Scripts/python.exe'
$hub = Join-Path $env:USERPROFILE '.cache/huggingface/hub'
$prepared = Join-Path $repo 'artifacts/material-studio/package-runtime'

& $python scripts/demo/package_app.py validate-inputs --root $repo --hub-cache $hub
& $python scripts/demo/package_app.py prepare-runtime --root $repo --hub-cache $hub --output $prepared
& $python scripts/demo/package_app.py smoke --package $prepared
```

`--hub-cache` is the **hub** directory, not its parent. It must already contain
these exact files from the frozen metadata:

| Backbone | Revision | Weights SHA-256 |
| --- | --- | --- |
| `timm/vit_small_patch14_reg4_dinov2.lvd142m` | `c04b5193082a8d5b0c4856c7937384a48136c5de` | `dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d` |
| `timm/convnext_tiny.fb_in22k_ft_in1k` | `bc48a87f119bae1b06418f5340bfa97ef75d4609` | `77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a` |

Only their `model.safetensors` snapshot files are copied, materializing HF symlinks
as ordinary files. The neural deployment checkpoints contain their complete state
and need no additional ResNet/DINO training cache. The final lock plus four
checkpoint/sidecar pairs and these two caches total **387,873,049 bytes** in the
validated source inputs. Gallery assets are additional.

The runtime seeds are torch, torchvision, timm, NumPy, pandas, scikit-learn, SciPy,
Pillow, PyYAML, safetensors, huggingface-hub, joblib, threadpoolctl, and packaging.
Declared required dependencies are followed transitively with environment markers,
version constraints and dependency extras. Installed wheel `RECORD` files drive the
copy, and their content hashes are checked while copying. Shipped package data,
native DLLs, `.libs` directories, `.dist-info` metadata and licenses are retained.
Wheel console launchers, `.pth` files, editable-install provenance, and bytecode
are omitted. The source venv, Jupyter, training data and unrelated caches are not
copied. Package-internal files are retained so runtime imports do not depend on
guessing which files a third-party module might need.

## Assemble after integration

Prerequisites are the integrated worker/source, study assets, four model artifacts,
the prepared runtime, a built native release executable, and the native dependency
notice bundle generated from its actual `Cargo.lock`.

```powershell
$exe = Join-Path $repo 'artifacts/material-studio/cargo-target/release/material-studio.exe'
$notices = Join-Path $repo 'app/THIRD_PARTY_NOTICES.txt'
$output = Join-Path $repo 'dist/MaterialStudio-release'
# Use the reviewed, redistributable MSVC runtime selected during PE inspection.
$vcRuntime = Join-Path $env:SystemRoot 'System32/vcruntime140.dll'
$exeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()

& $python scripts/demo/package_app.py build --root $repo --runtime $prepared `
    --exe $exe --exe-sha256 $exeHash --native-notices $notices `
    --native-dll $vcRuntime --output $output
& $python scripts/demo/package_app.py verify --package $output
& $python scripts/demo/package_app.py smoke --package $output
```

Use the final, UI-verified executable path/hash reported by the native build. The
packager renames it to `MaterialStudio.exe`. The reviewed Windows release imports
`VCRUNTIME140.dll`, so an explicit copy is mandatory alongside the app executable.
If PE import inspection identifies another app-local DLL,
pass `--native-dll C:/path/to/dependency.dll` for each one. This copies only explicit
dependencies alongside the executable. It does not sweep the Rust build directory
or guess that system DLLs should be redistributed. Python's own VC runtime DLLs
are already alongside `runtime/python.exe`.

`--native-notices` must be a nonempty file. It is copied as
`app/THIRD_PARTY_NOTICES` with its original suffix. If present, `app/licenses/**`
is copied at its original relative location, preserving notice links. Python
license/notice files remain in the installed distributions; `runtime/LICENSE.txt`
contains the Python license. `licenses/MODEL_ATTRIBUTION.txt` records the model
sources, with MIT and Apache-2.0 texts bundled. Sample attributions and original
source links remain in `app/assets/study.json`; `DATASET.md` preserves the dataset
scope. These local artifacts do not grant publication approval or choose a new
license for the study's renders or fitted weights.

## Layout and launch contract

```text
MaterialStudio/
  MaterialStudio.exe
  START_HERE.txt
  package-manifest.json
  package-smoke.json
  runtime-manifest.json
  runtime/
    python.exe, python3.dll, python313.dll, vcruntime140*.dll
    python313._pth
    Lib/, DLLs/
    runtime-info.json
    huggingface/hub/models--.../snapshots/<pinned revision>/model.safetensors
  scripts/demo/worker.py
  scripts/v3/finetune.py
  scripts/v3/predict_finetuned.py
  src/material_classifier/
  configs/v3/
  reports/results/v3/final_model_lock.json
  artifacts/v3/run_final/{model.joblib,model_metadata.json,
                          control_model.joblib,control_model_metadata.json}
  artifacts/v3/resnet-final/final/{model.pt,model.json}
  artifacts/v3/finetune/candidates/finetuned/{model.pt,model.json}
  app/assets/study.json
  app/assets/samples/
  app/THIRD_PARTY_NOTICES.txt
  app/licenses/
  licenses/
```

The app resolves `ROOT` as the executable's parent by default, selects
`ROOT/runtime/python.exe`, and launches `ROOT/scripts/demo/worker.py --root ROOT
--device auto`. App overrides are `--resources ROOT --python PATH` (and any
documented app environment override). A portable launch must not fall back to the
developer's venv or rely on the current directory. The manifest's sample paths
are relative to `ROOT/app/assets`.

`python313._pth` lists only relative runtime paths and `../src`, enabling isolated
module lookup and the bundled `sitecustomize.py`. That startup hook forces the
portable interpreter's `HF_HOME` and both Hub cache variables to
`ROOT/runtime/huggingface`, enables offline mode, and disables telemetry. This
deliberately wins over inherited host cache variables. The ordinary development
worker may honor an environment override; the packaged interpreter always uses
its own pinned cache. Python's application directory, its normal extension loader,
and the libraries' own DLL loaders resolve native dependencies; the startup hook
does not modify global or process `PATH`.

The two `scripts/v3` files are required even though the worker does not execute
them: `finetune.load_deployed` hashes them as part of the exported source identity.
All reviewed `src/material_classifier` files are copied unchanged. The final
lock stays at its original relative path so its `../../../artifacts/...` paths
resolve inside the portable root. The packager verifies source identities using
the same text normalization as the deployed loader, while copying original bytes.

Python's Windows `._pth` behavior is documented in the
[CPython Windows guide](https://docs.python.org/3.13/using/windows.html#finding-modules).
Backbone attribution follows the [DINOv2 source license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
and [ConvNeXt source license](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE).

## Integrity, size and verification

Every command refuses an existing output folder. Choose a fresh output instead of
overwriting a working demo. Assembly writes to a unique sibling stage, verifies
hashes during copy, revalidates copied model/source identities, and runs the CPU
import smoke before publishing the completed folder with a rename. On failure it
cleans only its own contained stage; it never deletes an existing package or
source. No hardlinks connect output files to source weights or installed packages.

`runtime-manifest.json` and `package-manifest.json` list each payload file's relative
path, byte count and SHA-256. `payload_bytes` excludes that manifest itself; the
command prints its GiB equivalent (1 GiB = 1,073,741,824 bytes). Report this measured
value, not the small executable size. A CUDA-enabled Torch installation alone is
about 4 GiB here, so a complete offline package is several GiB. These manifests
detect accidental corruption, not malicious replacement of both files and their
manifest; they are not digital signatures.

On 27 October 2026, runtime preparation completed with **38 distributions,
22,713 payload files and 4,937,506,288 bytes (4.598 GiB)**, including both pretrained
backbone caches. The complete runtime directory including its manifest is
**4,941,894,394 bytes (4.602 GiB)**. Preparation and a second
smoke after renaming the stage to `package-runtime` both passed: Python 3.13.5,
Torch 2.11.0+cu128 / CUDA build 12.8, CPU tensor and torchvision native operations,
two offline cache hits, isolated local import paths, and empty diagnostic stderr.
Evidence is staged at `artifacts/material-studio/package-runtime/runtime-smoke.json`
and `artifacts/material-studio/runtime-relocation-smoke.json`. This measured size
is the prepared runtime, before adding the app, fitted checkpoints, and samples.

The first assembled **candidate** measured **5,151,637,452 bytes (4.798 GiB)**
including its manifest, in 22,798 files (payload alone 5,147,237,062 bytes).
Its packaged worker passed health, eight CPU predictions (cold/warm for all four
models), and clean shutdown in 21.39 seconds, from an external working directory
with `PYTHONPATH` removed, a system-only child `PATH`, and deliberately conflicting
host HF cache settings. All probabilities were valid, all devices were `cpu`, all
warm calls reported cached models, and stderr was empty. Evidence:
`artifacts/material-studio/package-worker-cpu-smoke.json`.

That candidate contains executable SHA-256
`ef4500156073a798e32cceb79c927ede2a40f6cf66669b268d6698b8591298cf` and app-local
`vcruntime140.dll` version 14.50.35719.0, SHA-256
`184146852727a9db4eea06178716bec3cdbb1015c911f6b0f915b184ad7775b2`.
Native UI verification was pending; subsequent app fixes require a fresh build
with the final executable hash and refreshed inventory. This candidate is not
the final UI-approved release. Use a new output folder rather than silently
replacing files covered by the candidate manifest. GPU validation is separate.

The automatic smoke launches the **bundled** interpreter from a fresh external
temporary working directory, removes `PYTHON*` variables, restricts the child
`PATH` to Windows system directories, and sets `CUDA_VISIBLE_DEVICES=-1`. It checks
isolated/local module paths, imports the inference distributions, runs CPU tensor
and torchvision native operations, and resolves both pinned HF cache entries
offline. With the full package present it also imports the reviewed inference
modules. It records Python/distribution/CUDA-build versions, paths and stderr.
DLL-loader/import errors fail assembly with captured diagnostics. This is CPU
runtime validation, not GPU validation or proof that the native UI launches.

After moving the folder, run `smoke --package <moved-folder>` again, then launch
the app from outside the checkout:

```powershell
Push-Location $env:TEMP
& (Join-Path $output 'MaterialStudio.exe')
Pop-Location
```

Confirm double-click startup, sample rendering, and one live prediction for each
of the four models. On the 4070 machine confirm the worker reports `cuda`; also
exercise the documented CPU mode. Keep these demo timings separate from sealed
study results. The native app, worker and assets must be integrated before those
end-to-end checks. Building the runtime alone does not mark them complete.

Small synthetic tests (no model loading, no large copies):

```powershell
& $python -m pytest tests/test_demo_package.py -q
& $python -m ruff check scripts/demo/package_app.py tests/test_demo_package.py
```

## Verified local release

Use `dist/MaterialStudio-release/MaterialStudio.exe`. The initial assembled
monochrome package contained 22,798 files and 5,151,660,569 bytes (about 4.8 GiB).
All payload hashes and source/model identities passed assembly verification.
The final packaged worker passed all four models cold/warm on CPU and CUDA,
from an external temporary directory with restricted PATH, no PYTHONPATH and
conflicting host cache variables. Evidence:
`artifacts/material-studio/final-package-worker-smoke.json`.

The packaged native executable also opened outside the checkout without root or
Python overrides, rendered its bundled example and completed a live prediction.
The latest UI-only refresh reduces neutral panel opacity to expose more native
Acrylic blur. Its executable SHA-256 is
`35fac15c99305572a1a33baf55c2acd1ce495ef5fab155b685861019cb4e2cc2`.
The refresh verifies the entire previous payload, retains its original binary
and manifest in local QA artifacts, replaces only the executable and updates its
manifest entry. Models, runtime, licenses and sample pixels are unchanged.
Final refresh evidence: `artifacts/material-studio/glass-refresh.json`.

Earlier `dist/MaterialStudio` and `dist/MaterialStudio-final` folders are retained
candidates with the superseded interface. They are not the recommended demo.
