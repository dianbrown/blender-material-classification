"""Small synthetic inputs exercise packaging integrity without copying CUDA GBs."""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.demo import package_app as pkg


def put(root, relative, value=b"fixture"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        pkg.write_json(path, value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_bytes(value)
    return path


@pytest.fixture
def resources(tmp_path):
    root, hub = tmp_path / "source", tmp_path / "hub"
    specs = {}
    for name in ("dino", "convnext"):
        content = name.encode()
        spec = {
            "name": name,
            "repository": f"timm/{name}",
            "revision": "a" * 40,
            "weights_file": "model.safetensors",
            "weights_sha256": hashlib.sha256(content).hexdigest(),
        }
        specs[name] = spec
        put(hub, f"models--timm--{name}/snapshots/{'a' * 40}/model.safetensors", content)
    source = put(root, "scripts/v3/finetune.py", "# unchanged\n")
    source_hash = hashlib.sha256(json.dumps(source.read_text()).encode()).hexdigest()
    models = {}
    for name in sorted(pkg.MODEL_IDS):
        artifact = put(root, f"artifacts/{name}/model.bin", name.encode())
        info = {"class_order": ["wood", "metal"]}
        if name.startswith("frozen_"):
            info["protocol"] = {"models": specs}
        if name == "dinov2_finetuned":
            info["source_hashes"] = {"scripts/v3/finetune.py": source_hash}
        sidecar = put(root, f"artifacts/{name}/model.json", info)
        models[name] = {
            "artifact_path": "../../../" + artifact.relative_to(root).as_posix(),
            "metadata_path": "../../../" + sidecar.relative_to(root).as_posix(),
            "artifact_sha256": pkg.sha256(artifact),
            "metadata_sha256": pkg.sha256(sidecar),
        }
    put(root, pkg.LOCK, {"models": models, "class_order": ["wood", "metal"]})
    return root, hub


def test_all_locked_inputs_and_only_two_snapshot_files(resources):
    root, hub = resources
    put(hub, "unrelated/download.bin")
    files, caches, specs, environments = pkg.model_inputs(root, hub)
    assert len(files) == 9
    assert len(caches) == len(specs) == 2
    assert all(item.destination.startswith("runtime/huggingface/hub/models--") for item in caches)
    assert environments == []


@pytest.mark.parametrize("kind", ["artifact", "metadata"])
def test_corrupt_locked_input_rejected(resources, kind):
    root, hub = resources
    lock = pkg.read_json(root / pkg.LOCK)
    relative = lock["models"]["frozen_selected"][f"{kind}_path"]
    (root / pkg.LOCK).parent.joinpath(relative).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pkg.model_inputs(root, hub)


def test_locked_path_cannot_escape_root(resources):
    root, hub = resources
    outside = put(root.parent, "outside.bin")
    lock = pkg.read_json(root / pkg.LOCK)
    lock["models"]["frozen_selected"]["artifact_path"] = "../../../../outside.bin"
    lock["models"]["frozen_selected"]["artifact_sha256"] = pkg.sha256(outside)
    pkg.write_json(root / pkg.LOCK, lock)
    with pytest.raises(ValueError, match="escapes resources root"):
        pkg.model_inputs(root, hub)


def test_missing_model_and_corrupt_cache_fail_closed(resources):
    root, hub = resources
    cache = next(hub.rglob("*.safetensors"))
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pkg.model_inputs(root, hub)
    cache.unlink()
    with pytest.raises(FileNotFoundError):
        pkg.model_inputs(root, hub)


def test_four_models_required(resources):
    root, hub = resources
    lock = pkg.read_json(root / pkg.LOCK)
    lock["models"].pop("dinov2_finetuned")
    pkg.write_json(root / pkg.LOCK, lock)
    with pytest.raises(ValueError, match="all four"):
        pkg.model_inputs(root, hub)


def test_source_identity_normalizes_line_endings_but_rejects_changes(resources):
    root, hub = resources
    source = root / "scripts/v3/finetune.py"
    source.write_bytes(b"# unchanged\r\n")
    pkg.model_inputs(root, hub)
    source.write_bytes(b"# changed\r\n")
    with pytest.raises(ValueError, match="Pinned inference source changed"):
        pkg.model_inputs(root, hub)


@pytest.mark.parametrize(
    "relative",
    [
        "../outside",
        "/absolute",
        "C:/absolute",
        "C:relative",
        "folder/../x",
        "a//b",
        "a./b",
        "a /b",
        "file:stream",
        "NUL.txt",
        "COM1",
        "folder/LPT9.log",
        "./x",
    ],
)
def test_unsafe_destination_paths(relative):
    with pytest.raises(ValueError):
        pkg.relative_path(relative)


def test_assemble_verifies_bytes_and_preserves_source(tmp_path):
    source = put(tmp_path / "source", "a.bin", b"original")
    output = tmp_path / "output with spaces"
    manifest = pkg.assemble(
        output,
        [pkg.Item(source, "nested/a.bin", pkg.sha256(source))],
        {"notes.txt": "hello"},
        pkg.MANIFEST,
        {"kind": "test"},
    )
    assert manifest["payload_bytes"] == len(b"originalhello")
    assert source.read_bytes() == b"original"
    pkg.verify_manifest(output, pkg.MANIFEST)
    (output / "nested/a.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pkg.verify_manifest(output, pkg.MANIFEST)


def test_copy_hash_failure_cleans_only_owned_stage(tmp_path):
    source = put(tmp_path, "input.bin", b"changed after planning")
    sibling = put(tmp_path / "unrelated", "keep.bin")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pkg.assemble(
            tmp_path / "output", [pkg.Item(source, "a.bin", "0" * 64)], {}, pkg.MANIFEST, {}
        )
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob("*.stage-*"))
    assert sibling.read_bytes() == b"fixture"
    assert source.read_bytes() == b"changed after planning"


def test_existing_output_never_overwritten(tmp_path):
    sentinel = put(tmp_path / "output", "keep.txt")
    with pytest.raises(FileExistsError):
        pkg.assemble(sentinel.parent, [], {}, pkg.MANIFEST, {})
    assert sentinel.read_bytes() == b"fixture"


def test_cleanup_refuses_unowned_paths(tmp_path):
    sentinel = put(tmp_path / "keep", "file")
    with pytest.raises(ValueError, match="Refusing cleanup"):
        pkg.safe_cleanup(sentinel.parent, tmp_path)
    assert sentinel.exists()


@pytest.mark.parametrize("names", [("a", "A"), ("a", "a/b"), ("a/b", "A/B")])
def test_destination_collisions_rejected(tmp_path, names):
    source = put(tmp_path, "input")
    with pytest.raises(ValueError, match="collision|Duplicate"):
        pkg.validate_plan([pkg.Item(source, name) for name in names])


def test_record_hash_checked_during_copy(tmp_path):
    source = put(tmp_path, "input", b"installed wheel file")
    expected = base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest()).rstrip(b"=")
    item = pkg.Item(source, "copied", record_hash=("sha256", expected.decode()))
    pkg.copy_item(item, tmp_path / "good")
    source.write_bytes(b"mutated wheel file")
    with pytest.raises(ValueError, match="RECORD hash mismatch"):
        pkg.copy_item(item, tmp_path / "bad")


def test_dependency_closure_markers_extras_and_version_checks(monkeypatch):
    fake = {
        "app": SimpleNamespace(
            version="1",
            requires=[
                "core>=2",
                'optional; extra == "feature"',
                'never; python_version < "3"',
            ],
        ),
        "core": SimpleNamespace(version="2", requires=["leaf"]),
        "leaf": SimpleNamespace(version="1", requires=None),
        "optional": SimpleNamespace(version="1", requires=None),
    }
    monkeypatch.setattr(pkg.metadata, "distribution", lambda name: fake[name])
    assert set(pkg.distribution_closure(["app"])) == {"app", "core", "leaf"}
    assert set(pkg.distribution_closure(["app[feature]"])) == set(fake)
    with pytest.raises(ValueError, match="does not satisfy"):
        pkg.distribution_closure(["core>=3"])


@pytest.mark.parametrize("release_version", [None, "0.5.0"])
def test_full_assembly_preserves_layout_and_attribution(resources, tmp_path, release_version):
    root, hub = resources
    for relative in (
        "scripts/demo/worker.py",
        "scripts/v3/predict_finetuned.py",
        "src/material_classifier/v3/finetune.py",
        "configs/v3/protocol.json",
        "DATASET.md",
        "app/licenses/example/LICENSE.txt",
    ):
        put(root, relative)
    put(root, "src/material_classifier/__pycache__/ignore.pyc")
    sample = {
        "id": "wood",
        "path": "samples/wood.png",
        "attribution": "Creator",
        "source_url": "https://example.test/source",
        "license": "CC0-1.0",
    }
    put(root, "app/assets/study.json", {"samples": [sample]})
    put(root, "app/assets/samples/wood.png")
    _, caches, specs, _ = pkg.model_inputs(root, hub)
    runtime = tmp_path / "runtime-prepared"
    pkg.assemble(
        runtime,
        caches,
        {
            "runtime/Lib/site-packages/packaging-1.dist-info/licenses/LICENSE.APACHE": "Apache",
            "runtime/python.exe": "fake Python",
        },
        pkg.RUNTIME_MANIFEST,
        {"kind": "prepared-runtime", "model_specs": specs, "python": "3.13.5", "distributions": {}},
    )
    exe = put(tmp_path, "native.exe", b"MZfake executable")
    dll = put(tmp_path, "vcruntime140.dll", b"fake runtime")
    notices = put(tmp_path, "NOTICE.txt", "Native third-party licenses")
    output = tmp_path / "MaterialStudio"
    with pytest.raises(ValueError, match="requires --native-dll"):
        pkg.build_package(root, runtime, exe, notices, output, smoke=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pkg.build_package(
            root, runtime, exe, notices, output, native_dlls=[dll], exe_sha256="0" * 64, smoke=False
        )
    assert not output.exists()
    manifest = pkg.build_package(
        root,
        runtime,
        exe,
        notices,
        output,
        native_dlls=[dll],
        exe_sha256=pkg.sha256(exe),
        smoke=False,
        release_version=release_version,
    )
    assert manifest["release_version"] == release_version
    assert manifest["release_status"] == ("prepared-release" if release_version else "candidate")
    if release_version:
        assert "Material Studio 0.5.0" in (output / "START_HERE.txt").read_text()
        assert "Candidate build" not in (output / "START_HERE.txt").read_text()
    assert (output / "MaterialStudio.exe").read_bytes() == b"MZfake executable"
    assert (output / "scripts/v3/finetune.py").read_bytes() == (
        root / "scripts/v3/finetune.py"
    ).read_bytes()
    assert (output / "app/licenses/example/LICENSE.txt").is_file()
    assert (output / "licenses/APACHE-2.0.txt").is_file()
    assert all("__pycache__" not in name for name in manifest["files"])
    assert all(not Path(name).is_absolute() for name in manifest["files"])
    pkg.verify_manifest(output, pkg.MANIFEST)


def test_portable_startup_uses_relative_paths_and_process_local_cache(tmp_path, monkeypatch):
    runtime = tmp_path / "relocated" / "runtime"
    startup = put(runtime, "Lib/sitecustomize.py", pkg.SITE_CUSTOMIZE)
    monkeypatch.setenv("HF_HOME", "Z:/host-cache")
    monkeypatch.setenv("HF_HUB_CACHE", "Z:/host-cache/hub")
    for name in (
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "TORCH_HOME",
    ):
        monkeypatch.setenv(name, pkg.os.environ.get(name, ""))
    namespace = {"__file__": str(startup)}
    # Restore global interpreter bytecode policy after executing the startup hook.
    monkeypatch.setattr(pkg.sys, "dont_write_bytecode", pkg.sys.dont_write_bytecode)
    exec(compile(pkg.SITE_CUSTOMIZE, str(startup), "exec"), namespace)
    assert pkg.os.environ["HF_HOME"] == str(runtime / "huggingface")
    assert pkg.os.environ["HF_HUB_CACHE"] == str(runtime / "huggingface/hub")
    assert pkg.os.environ["HF_HUB_OFFLINE"] == "1"
