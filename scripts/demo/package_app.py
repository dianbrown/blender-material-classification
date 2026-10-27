"""Assemble an offline Windows folder from the validated local environment.

No installers, pip, network access, GPU work, or source mutations. Run this with
the project's validated Python 3.13 environment, not the packaged interpreter.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

LOCK = "reports/results/v3/final_model_lock.json"
MODEL_IDS = {"frozen_selected", "frozen_control", "resnet_fixed", "dinov2_finetuned"}
SEEDS = (
    "torch",
    "torchvision",
    "timm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "Pillow",
    "PyYAML",
    "safetensors",
    "huggingface-hub",
    "joblib",
    "threadpoolctl",
    "packaging",
)
IGNORED = {"__pycache__", ".git"}
MANIFEST = "package-manifest.json"
RUNTIME_MANIFEST = "runtime-manifest.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    """Reject traversal and Windows aliases even when tests run on another OS."""
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError(f"Not a relative file path: {value}")
    if any(
        part in {"", ".", ".."}
        or part.rstrip(" .") != part
        or any(char in part for char in ':<>"|?*')
        or part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL"}
        or re.fullmatch(r"(?:COM|LPT)[1-9](?:\..*)?", part, re.I)
        for part in value.split("/")
    ):
        raise ValueError(f"Unsafe relative file path: {value}")
    return path.as_posix()


def contained_file(root, relative):
    root = Path(root).resolve()
    path = root / relative_path(relative)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Missing file or path escapes root: {path}")
    return resolved


def checked_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid SHA-256 for {path}")
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


@dataclass(frozen=True)
class Item:
    source: Path
    destination: str
    expected: str | None = None
    record_hash: tuple[str, str] | None = None


def tree_items(root, prefix, *, excluded=frozenset()):
    root = Path(root).resolve(strict=True)
    items = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if set(relative.parts) & (IGNORED | set(excluded)) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Links are not allowed in source trees: {path}")
        if path.is_file():
            source = contained_file(root, relative.as_posix())
            items.append(Item(source, f"{prefix}/{relative.as_posix()}"))
    return items


def validate_plan(items):
    names = set()
    for item in items:
        name = relative_path(item.destination).casefold()
        if name in names:
            raise ValueError(f"Duplicate destination: {item.destination}")
        names.add(name)
        if not item.source.is_file():
            raise FileNotFoundError(item.source)
    for name in names:
        if any(parent.as_posix() in names for parent in PurePosixPath(name).parents):
            raise ValueError(f"File/directory collision: {name}")


def model_inputs(root, hub):
    """Validate sealed artifacts before loading pickle or importing torch."""
    root, hub = Path(root).resolve(), Path(hub).resolve()
    lock_path = contained_file(root, LOCK)
    lock = read_json(lock_path)
    if set(lock["models"]) != MODEL_IDS:
        raise ValueError("Final model lock must contain exactly all four demo models")
    items = [Item(lock_path, LOCK, sha256(lock_path))]
    specs, environments = {}, []
    for model_id, entry in lock["models"].items():
        paths = {}
        for kind in ("artifact", "metadata"):
            # The existing lock intentionally uses ../../../artifacts. Resolve
            # against its own directory, then enforce the resources boundary.
            path = (lock_path.parent / entry[f"{kind}_path"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(f"Locked {kind} escapes resources root: {model_id}")
            checked_hash(path, entry[f"{kind}_sha256"])
            paths[kind] = path
            items.append(Item(path, path.relative_to(root).as_posix(), entry[f"{kind}_sha256"]))
        info = read_json(paths["metadata"])
        if info["class_order"] != lock["class_order"]:
            raise ValueError(f"Class order mismatch: {model_id}")
        for key in ("environment", "feature_environment", "learning_environment"):
            if key in info:
                environments.append(info[key])
        if model_id.startswith("frozen_"):
            for spec in info["protocol"]["models"].values():
                key = (spec["repository"], spec["revision"], spec["weights_file"])
                if key in specs and specs[key] != spec:
                    raise ValueError(f"Conflicting frozen model specs: {key}")
                specs[key] = spec
        for relative, expected in info.get("source_hashes", {}).items():
            path = contained_file(root, relative)
            normalized = json.dumps(
                path.read_text(encoding="utf-8"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            if hashlib.sha256(normalized).hexdigest() != expected:
                raise ValueError(f"Pinned inference source changed: {relative}")
    caches = []
    for (repository, revision, filename), spec in sorted(specs.items()):
        if not re.fullmatch("[0-9a-f]{40}", revision):
            raise ValueError(f"Pretrained revision is not pinned: {repository}")
        relative = relative_path(
            f"models--{repository.replace('/', '--')}/snapshots/{revision}/{filename}"
        )
        path = contained_file(hub, relative)  # HF snapshot symlinks may resolve to hub/blobs.
        checked_hash(path, spec["weights_sha256"])
        caches.append(Item(path, f"runtime/huggingface/hub/{relative}", spec["weights_sha256"]))
    if len(caches) != 2:
        raise ValueError("Expected exactly the pinned DINOv2 and ConvNeXt backbone caches")
    return items, caches, list(specs.values()), environments


def distribution_closure(seeds=SEEDS):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    selected, extras, pending = {}, {}, [Requirement(name) for name in seeds]
    while pending:
        request = pending.pop()
        name = canonicalize_name(request.name)
        dist = metadata.distribution(request.name)
        if request.url or (request.specifier and dist.version not in request.specifier):
            raise ValueError(f"Installed dependency does not satisfy {request}: {dist.version}")
        requested_extras = set(request.extras)
        if name in selected and requested_extras <= extras[name]:
            continue
        selected[name] = dist
        extras.setdefault(name, set()).update(requested_extras)
        for value in dist.requires or []:
            dependency = Requirement(value)
            if dependency.marker is None or any(
                dependency.marker.evaluate({"extra": extra}) for extra in {"", *extras[name]}
            ):
                pending.append(dependency)
    return selected


def distribution_items(distributions):
    items, seen = [], set()
    for name, dist in sorted(distributions.items()):
        base = Path(dist.locate_file("")).resolve()
        if not dist.files:
            raise ValueError(f"Distribution has no RECORD: {name}")
        for record in dist.files:
            relative = str(record).replace("\\", "/")
            # Console launchers live outside site-packages and are unnecessary.
            if relative.startswith("../"):
                if relative.startswith("../../share/"):
                    shared = relative.removeprefix("../../")
                    file_hash = (record.hash.mode, record.hash.value) if record.hash else None
                    items.append(
                        Item(
                            contained_file(base.parent.parent, shared),
                            f"runtime/{shared}",
                            record_hash=file_hash,
                        )
                    )
                    continue
                if not relative.startswith("../../Scripts/"):
                    raise ValueError(
                        f"Unexpected distribution file outside site-packages: {record}"
                    )
                continue
            relative = relative_path(relative)
            if "__pycache__" in PurePosixPath(relative).parts or relative.endswith(
                (".pyc", ".pyo")
            ):
                continue
            # Never execute an installed .pth or leak editable checkout paths.
            if relative.endswith(".pth") or relative.endswith("/direct_url.json"):
                continue
            destination = f"runtime/Lib/site-packages/{relative}"
            if destination.casefold() in seen:
                raise ValueError(f"Distributions share a file: {relative}")
            seen.add(destination.casefold())
            file_hash = (record.hash.mode, record.hash.value) if record.hash else None
            items.append(Item(contained_file(base, relative), destination, record_hash=file_hash))
    return items


SITE_CUSTOMIZE = '''"""Material Studio portable interpreter: process-local, offline resources."""
import os
import sys
from pathlib import Path

_runtime = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
# Override inherited caches so a copied folder cannot silently use host weights.
os.environ["HF_HOME"] = str(_runtime / "huggingface")
os.environ["HF_HUB_CACHE"] = str(_runtime / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TORCH_HOME"] = str(_runtime / "torch-cache")
'''


def runtime_items(environments):
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13) or platform.machine() != "AMD64":
        raise ValueError("Run prepare-runtime with validated Windows x64 Python 3.13")
    distributions = distribution_closure()
    from packaging.utils import canonicalize_name

    for env in environments:
        if env.get("python") != platform.python_version():
            raise ValueError("Python version differs from the locked inference environment")
        for name, version in env.get("packages", {}).items():
            if version == "not installed":
                continue
            dist = distributions.get(canonicalize_name(name))
            if dist is None or dist.version != version:
                raise ValueError(f"Runtime differs from saved model environment: {name}=={version}")
    base = Path(sys.base_prefix)
    names = (
        "python.exe",
        "python3.dll",
        "python313.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "LICENSE.txt",
    )
    items = [Item(contained_file(base, name), f"runtime/{name}") for name in names]
    items += tree_items(base / "DLLs", "runtime/DLLs")
    items += tree_items(
        base / "Lib",
        "runtime/Lib",
        excluded={
            "site-packages",
            "test",
            "tests",
            "idlelib",
            "tkinter",
            "turtledemo",
            "ensurepip",
        },
    )
    items += distribution_items(distributions)
    versions = {name: dist.version for name, dist in sorted(distributions.items())}
    return items, versions


def safe_cleanup(stage, parent):
    """Delete only a newly allocated, contained stage; never a caller's output."""
    stage, parent = Path(stage), Path(parent).resolve(strict=True)
    if (
        stage.is_symlink()
        or (hasattr(stage, "is_junction") and stage.is_junction())
        or stage.resolve().parent != parent
        or ".stage-" not in stage.name
    ):
        raise ValueError(f"Refusing cleanup outside owned staging directory: {stage}")
    # Refuse directory reparse points anywhere in the stage before recursion.
    for path in stage.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Refusing cleanup through link: {path}")
    shutil.rmtree(stage)


def copy_item(item, stage):
    target = stage / relative_path(item.destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    record_digest = hashlib.new(item.record_hash[0]) if item.record_hash else None
    with item.source.open("rb") as source, target.open("xb") as dest:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            dest.write(chunk)
            digest.update(chunk)
            if record_digest:
                record_digest.update(chunk)
    if item.expected and digest.hexdigest() != item.expected:
        raise ValueError(f"SHA-256 mismatch while copying: {item.source}")
    if record_digest:
        actual = base64.urlsafe_b64encode(record_digest.digest()).rstrip(b"=").decode()
        if actual != item.record_hash[1]:
            raise ValueError(f"Installed RECORD hash mismatch: {item.source}")
    return {"sha256": digest.hexdigest(), "bytes": target.stat().st_size}


def assemble(output, items, generated, manifest_name, details, *, smoke=False, validate=None):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a fresh directory: {output}")
    validate_plan(items)
    all_names = [item.destination for item in items] + list(generated) + [manifest_name]
    if len({relative_path(name).casefold() for name in all_names}) != len(all_names):
        raise ValueError("Generated file collides with copied input")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    # Refuse writing inside a source tree or over an input file.
    for item in items:
        if item.source.resolve().is_relative_to(output):
            raise ValueError("Output contains package inputs")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        records = {}
        for index, item in enumerate(items, start=1):
            records[item.destination] = copy_item(item, stage)
            if index % 2000 == 0:
                print(f"Verified and copied {index}/{len(items)} files", file=sys.stderr)
        for name, content in generated.items():
            target = stage / relative_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            records[name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
        if validate:
            validate(stage)
        if smoke:
            result = smoke_runtime(stage)
            smoke_name = (
                "runtime-smoke.json" if manifest_name == RUNTIME_MANIFEST else "package-smoke.json"
            )
            write_json(stage / smoke_name, result)
            records[smoke_name] = {
                "sha256": sha256(stage / smoke_name),
                "bytes": (stage / smoke_name).stat().st_size,
            }
        manifest = {
            "schema_version": 1,
            **details,
            "files": dict(sorted(records.items())),
            "payload_bytes": sum(row["bytes"] for row in records.values()),
        }
        write_json(stage / manifest_name, manifest)
        # Windows rename fails if output appeared while staging; never replace it.
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
        return manifest
    except BaseException:
        safe_cleanup(stage, output.parent)
        raise


def verify_manifest(root, name):
    root = Path(root).resolve(strict=True)
    document = read_json(contained_file(root, name))
    items = []
    for relative, entry in document["files"].items():
        path = contained_file(root, relative)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        checked_hash(path, entry["sha256"])
        items.append(Item(path, relative, entry["sha256"]))
    validate_plan(items)
    if document["payload_bytes"] != sum(row["bytes"] for row in document["files"].values()):
        raise ValueError("Manifest payload size mismatch")
    return document, items


SMOKE_CODE = """
import importlib, importlib.metadata as md, json, os, pathlib, platform, sys
root = pathlib.Path(sys.executable).resolve().parent.parent
assert sys.flags.isolated and sys.flags.ignore_environment
assert all(pathlib.Path(p).resolve().is_relative_to(root) for p in sys.path), sys.path
assert os.environ['HF_HOME'] == str(root / 'runtime' / 'huggingface')
modules = ['torch', 'torchvision', 'timm', 'numpy', 'pandas', 'sklearn', 'scipy',
           'PIL', 'yaml', 'safetensors', 'huggingface_hub', 'joblib', 'threadpoolctl']
for name in modules:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root), module.__file__
import torch, torchvision
assert (torch.ones(2) + torch.ones(2)).tolist() == [2., 2.]
assert torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), .5).tolist() == [0]
from huggingface_hub import hf_hub_download
specs = json.loads(sys.argv[1])
for spec in specs:
    path = hf_hub_download(spec['repository'], spec['weights_file'],
                           revision=spec['revision'], local_files_only=True)
    assert pathlib.Path(path).resolve().is_relative_to(root / 'runtime' / 'huggingface')
if (root / 'src' / 'material_classifier').is_dir():
    for name in ['features', 'learning', 'inference', 'resnet_baseline', 'finetune']:
        importlib.import_module('material_classifier.v3.' + name)
    for name in ['runtime', 'service']:
        importlib.import_module('material_classifier.demo.' + name)
print(json.dumps({'python': platform.python_version(), 'executable': sys.executable,
    'isolated': sys.flags.isolated, 'sys_path': sys.path, 'hf_home': os.environ['HF_HOME'],
    'versions': {name: md.version(name) for name in json.loads(sys.argv[2])},
    'cuda_build': torch.version.cuda, 'cpu_tensor_and_torchvision_ops': 'passed',
    'offline_cache_hits': len(specs), 'gpu_tested': False}))
"""


def smoke_runtime(root):
    root = Path(root).resolve(strict=True)
    config = read_json(contained_file(root, "runtime/runtime-info.json"))
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    # A minimal child PATH rules out accidentally resolving DLLs from the venv,
    # CUDA toolkit or base Python. This never changes the host/global PATH.
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env["PATH"] = str(Path(system_root) / "System32") + os.pathsep + system_root
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    with tempfile.TemporaryDirectory(prefix="material-studio-external-cwd-") as cwd:
        try:
            result = subprocess.run(
                [
                    str(root / "runtime/python.exe"),
                    "-I",
                    "-B",
                    "-c",
                    SMOKE_CODE,
                    json.dumps(config["model_specs"]),
                    json.dumps(list(config["distributions"])),
                ],
                cwd=cwd,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Portable Python launch failed: {exc}") from exc
    if result.returncode:
        raise RuntimeError(
            f"Portable import/DLL smoke failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return {
        "result": json.loads(result.stdout),
        "stderr": result.stderr,
        "cwd_outside_package": True,
        "pythonpath_removed": True,
        "system_only_path": True,
    }


def prepare_runtime(root, hub, output):
    _, caches, specs, environments = model_inputs(root, hub)
    items, versions = runtime_items(environments)
    info = {"python": platform.python_version(), "distributions": versions, "model_specs": specs}
    generated = {
        "runtime/python313._pth": ".\nLib\nDLLs\nLib/site-packages\n../src\nimport site\n",
        "runtime/Lib/sitecustomize.py": SITE_CUSTOMIZE,
        "runtime/runtime-info.json": json.dumps(info, indent=2) + "\n",
    }
    return assemble(
        output,
        items + caches,
        generated,
        RUNTIME_MANIFEST,
        {"kind": "prepared-runtime", **info},
        smoke=True,
    )


def application_inputs(root, model_items):
    root = Path(root).resolve()
    items = list(model_items)
    for relative in (
        "scripts/demo/worker.py",
        "scripts/v3/finetune.py",
        "scripts/v3/predict_finetuned.py",
        "app/assets/study.json",
        "DATASET.md",
    ):
        items.append(Item(contained_file(root, relative), relative))
    items += tree_items(root / "src/material_classifier", "src/material_classifier")
    items += tree_items(root / "configs/v3", "configs/v3")
    study = read_json(root / "app/assets/study.json")
    if not study.get("samples"):
        raise ValueError("Study assets contain no samples")
    for sample in study["samples"]:
        for field in ("id", "path", "attribution", "source_url", "license"):
            if not sample.get(field):
                raise ValueError(f"Sample lacks required {field}")
        relative = relative_path(sample["path"])
        path = contained_file(root / "app/assets", relative)
        items.append(Item(path, f"app/assets/{relative}"))
    return items


def build_package(
    root,
    runtime,
    exe,
    native_notices,
    output,
    *,
    native_dlls=(),
    exe_sha256=None,
    smoke=True,
    release_version=None,
):
    if release_version is not None and not re.fullmatch(r"\d+\.\d+\.\d+", release_version):
        raise ValueError("Release version must use major.minor.patch")
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    document, runtime_files = verify_manifest(runtime, RUNTIME_MANIFEST)
    if document.get("kind") != "prepared-runtime":
        raise ValueError("Not a prepared Material Studio runtime")
    model_files, caches, specs, environments = model_inputs(
        root, runtime / "runtime/huggingface/hub"
    )
    if specs != document["model_specs"]:
        raise ValueError("Prepared runtime's pretrained specs differ from locked models")
    from packaging.utils import canonicalize_name

    for env in environments:
        if env["python"] != document["python"]:
            raise ValueError("Prepared Python differs from model environment")
        for name, version in env["packages"].items():
            if (
                version != "not installed"
                and document["distributions"].get(canonicalize_name(name)) != version
            ):
                raise ValueError(f"Prepared distribution differs from model environment: {name}")
    exe, native_notices = Path(exe).resolve(strict=True), Path(native_notices).resolve(strict=True)
    with exe.open("rb") as stream:
        signature = stream.read(2)
    if signature != b"MZ":
        raise ValueError("Native application is not a Windows executable")
    executable_hash = exe_sha256 or sha256(exe)
    checked_hash(exe, executable_hash)
    if native_notices.stat().st_size == 0:
        raise ValueError("Native dependency notices must not be empty")
    items = (
        application_inputs(root, model_files)
        + runtime_files
        + [
            Item(exe, "MaterialStudio.exe", executable_hash),
            Item(native_notices, "app/THIRD_PARTY_NOTICES" + native_notices.suffix),
        ]
    )
    for dll in native_dlls:
        dll = Path(dll).resolve(strict=True)
        if dll.suffix.lower() != ".dll":
            raise ValueError(f"Native dependency must be a DLL: {dll}")
        items.append(Item(dll, dll.name))
    if not any(item.destination.casefold() == "vcruntime140.dll" for item in items):
        raise ValueError("Material Studio requires --native-dll <MSVC redist>/vcruntime140.dll")
    # Preserve the prepared inventory too; no source .venv is needed at assembly.
    items.append(Item(runtime / RUNTIME_MANIFEST, RUNTIME_MANIFEST))
    if (root / "app/licenses").is_dir():
        items += tree_items(root / "app/licenses", "app/licenses")
    apache = next(
        (
            item
            for item in runtime_files
            if item.destination.endswith(".dist-info/licenses/LICENSE.APACHE")
            and "/packaging-" in item.destination
        ),
        None,
    )
    if apache is None:
        raise ValueError("Prepared runtime lacks the Apache-2.0 license text")
    items.append(Item(apache.source, "licenses/APACHE-2.0.txt", apache.expected))
    build_label = (
        f"Material Studio {release_version} - offline Windows application\n\n"
        if release_version
        else "Material Studio - local Windows demo\n\n"
        "Candidate build; native UI verification is recorded separately.\n"
    )
    generated = {
        "START_HERE.txt": build_label
        + "Double-click MaterialStudio.exe. Keep this entire folder together.\n"
        "All four locked models, Python and CUDA libraries are bundled.\n"
        "Inference is offline. Device auto uses CUDA when available, otherwise CPU.\n"
        "This is a portable folder, not a single-file executable.\n"
        "Stored study results are distinct from live predictions.\n"
        "See DATASET.md, app/assets/study.json, app/THIRD_PARTY_NOTICES, licenses/\n"
        "and the licenses in Python dist-info directories.\n"
        "This packaging command does not upload or publish files.\n",
        "licenses/MODEL_ATTRIBUTION.txt": model_notices(),
    }
    return assemble(
        output,
        items,
        generated,
        MANIFEST,
        {
            "kind": "MaterialStudio-portable",
            "release_status": "prepared-release" if release_version else "candidate",
            "release_version": release_version,
            "models": sorted(MODEL_IDS),
            "python": document["python"],
            "distributions": document["distributions"],
            "native_ui_tested": False,
            "gpu_tested": False,
        },
        smoke=smoke,
        validate=lambda stage: model_inputs(stage, stage / "runtime/huggingface/hub"),
    )


def model_notices():
    return """Local research demo. No upload/publication is performed by this packager.
Study-fitted heads and fine-tuned checkpoints: this project's sealed V3 study.
DINOv2: Meta Platforms, Inc. and affiliates; Apache-2.0.
https://github.com/facebookresearch/dinov2
https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m
Apache-2.0 full text: licenses/APACHE-2.0.txt
ConvNeXt: Meta Platforms, Inc. and affiliates; MIT license below.
https://github.com/facebookresearch/ConvNeXt
https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k
ResNet50 / torchvision: see torchvision dist-info/LICENSE.
timm adaptations: see timm dist-info/licenses/LICENSE (Apache-2.0).
Python: runtime/LICENSE.txt. PyTorch and its bundled libraries: torch dist-info
LICENSE and NOTICE. Other Python distributions retain their metadata/licenses
in runtime/Lib/site-packages. Native dependencies: app/THIRD_PARTY_NOTICES.
Sample-specific attribution and source URLs: app/assets/study.json.
Source texture licenses do not assign a license to the rendered study dataset.

MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-runtime", "validate-inputs", "build"):
        sub = commands.add_parser(command)
        sub.add_argument("--root", type=Path, required=True)
        if command != "build":
            sub.add_argument(
                "--hub-cache",
                type=Path,
                required=True,
                help="Existing Hugging Face hub directory; never downloaded",
            )
        if command != "validate-inputs":
            sub.add_argument("--output", type=Path, required=True)
        if command == "build":
            sub.add_argument("--release-version", help="Versioned release payload, e.g. 0.5.0")
            sub.add_argument("--runtime", type=Path, required=True)
            sub.add_argument("--exe", type=Path, required=True)
            sub.add_argument("--native-notices", type=Path, required=True)
            sub.add_argument(
                "--exe-sha256", help="Expected SHA-256 of the UI-verified release binary"
            )
            sub.add_argument(
                "--native-dll",
                type=Path,
                action="append",
                default=[],
                help="Explicit app-local native DLL; repeat when needed",
            )
    for command in ("verify", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-runtime":
        result = prepare_runtime(args.root, args.hub_cache, args.output)
    elif args.command == "validate-inputs":
        items, caches, specs, _ = model_inputs(args.root, args.hub_cache)
        result = {
            "model_files": len(items),
            "backbone_caches": len(caches),
            "model_names": [spec["name"] for spec in specs],
            "bytes": sum(item.source.stat().st_size for item in items + caches),
        }
    elif args.command == "build":
        result = build_package(
            args.root,
            args.runtime,
            args.exe,
            args.native_notices,
            args.output,
            native_dlls=args.native_dll,
            exe_sha256=args.exe_sha256,
            release_version=args.release_version,
        )
    elif args.command == "smoke":
        result = smoke_runtime(args.package)
    else:
        name = MANIFEST if (args.package / MANIFEST).exists() else RUNTIME_MANIFEST
        document, items = verify_manifest(args.package, name)
        result = {"verified_files": len(items), "payload_bytes": document["payload_bytes"]}
    if "files" in result:
        result = {
            "files": len(result["files"]),
            "payload_bytes": result["payload_bytes"],
            "payload_gib": round(result["payload_bytes"] / 1024**3, 3),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
