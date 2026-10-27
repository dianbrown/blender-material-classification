"""Integrity and installer policy tests; no large runtime or model loading."""

import json
import os
import struct
import subprocess
import uuid
from pathlib import Path

import pytest
from scripts.demo import build_installer as ins


def put(root, relative, data=b"fixture"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def pe_fixture():
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, 0x8664)
    struct.pack_into("<H", data, 88, 0x20B)
    return bytes(data)


def ico_fixture():
    # One 1x1, 32-bit DIB icon, only for installer fixture compilation.
    image = struct.pack("<IiiHHIIiiII", 40, 1, 2, 1, 32, 0, 4, 0, 0, 0, 0)
    image += b"\x80\x80\x80\xff" + b"\0" * 4
    return struct.pack("<HHHBBBBHHII", 0, 1, 1, 1, 1, 0, 0, 1, 32, len(image), 22) + image


def seal(root, **metadata):
    records = {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": ins.sha256(p)}
        for p in root.rglob("*")
        if p.is_file() and p.name != ins.MANIFEST
    }
    document = {
        "schema_version": 1,
        "kind": "MaterialStudio-portable",
        "models": sorted(ins.MODELS),
        "files": records,
        "payload_bytes": sum(row["bytes"] for row in records.values()),
        **metadata,
    }
    ins.write_json(root / ins.MANIFEST, document)
    return document


@pytest.fixture
def payload(tmp_path):
    source = tmp_path / "portable"
    put(source, ins.EXE, pe_fixture())
    for name in (
        "runtime/python.exe",
        "runtime/python313._pth",
        "vcruntime140.dll",
        "app/assets/study.json",
        "app/THIRD_PARTY_NOTICES.txt",
    ):
        put(source, name)
    seal(source, release_version="0.5.0", release_status="release")
    icon = put(tmp_path, "icon.ico", ico_fixture())
    return source, icon


def test_accepts_versioned_release_and_original_candidate(payload):
    source, _ = payload
    assert ins.verify_package(source)["release_version"] == "0.5.0"
    seal(source, release_status="candidate")
    assert ins.verify_package(source)["release_status"] == "candidate"


@pytest.mark.parametrize("change", ["hash", "size", "extra", "missing", "total", "version"])
def test_manifest_integrity(payload, change):
    source, _ = payload
    document = json.loads((source / ins.MANIFEST).read_text())
    if change == "hash":
        (source / "runtime/python.exe").write_bytes(b"changed")
    elif change == "size":
        (source / "runtime/python.exe").write_bytes(b"different length")
    elif change == "extra":
        put(source, "user-image.png")
    elif change == "missing":
        (source / "runtime/python.exe").unlink()
    elif change == "total":
        document["payload_bytes"] += 1
    else:
        document["release_version"] = "0.1.0"
    ins.write_json(source / ins.MANIFEST, document)
    with pytest.raises(ValueError):
        ins.verify_package(source)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "C:/escape",
        "/escape",
        "a\\b",
        "a/../b",
        "a//b",
        "CON.txt",
        "dir/NUL",
        "a:x",
        "bad\nname",
        "a.",
        "a/ b ",
        "x/*",
        'x/"',
        "COM¹.txt",
    ],
)
def test_unsafe_manifest_paths(name):
    with pytest.raises(ValueError):
        ins.safe_relative(name)


def test_duplicate_json_key_and_windows_collision(payload):
    source, _ = payload
    path = source / ins.MANIFEST
    text = path.read_text().replace(
        '"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'
    )
    path.write_text(text)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        ins.verify_package(source)
    document = seal(source)
    document["files"]["MATERIALSTUDIO.EXE"] = document["files"][ins.EXE]
    ins.write_json(path, document)
    with pytest.raises(ValueError, match="Duplicate"):
        ins.verify_package(source)


@pytest.mark.parametrize(
    "name", ["MaterialStudio.ico", "unins000.exe", "materialstudio-installation.json"]
)
def test_reserved_destinations(payload, name):
    source, _ = payload
    put(source, name)
    seal(source)
    with pytest.raises(ValueError, match="Reserved"):
        ins.verify_package(source)


def test_symlink_rejected_before_directory_descent(payload, tmp_path):
    source, _ = payload
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (source / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks require privileges on this Windows configuration")
    with pytest.raises(ValueError, match="Reparse"):
        ins.verify_package(source)


def test_native_and_icon_identity(payload, monkeypatch):
    source, icon = payload
    monkeypatch.setattr(ins, "file_version", lambda _: "0.5.0.0")
    exe_hash, icon_hash = ins.sha256(source / ins.EXE), ins.sha256(icon)
    assert ins.verify_identity(source, icon, exe_hash, icon_hash) == "0.5.0.0"
    with pytest.raises(ValueError, match="SHA-256"):
        ins.verify_identity(source, icon, "0" * 64, icon_hash)
    with pytest.raises(ValueError, match="SHA-256"):
        ins.verify_identity(source, icon, exe_hash, "0" * 64)
    monkeypatch.setattr(ins, "file_version", lambda _: "0.1.0.0")
    with pytest.raises(ValueError, match="Rebuild"):
        ins.verify_identity(source, icon, exe_hash, icon_hash)


def test_rejects_wrong_architecture_and_bad_icon(payload):
    source, icon = payload
    data = bytearray(pe_fixture())
    struct.pack_into("<H", data, 68, 0x14C)
    (source / ins.EXE).write_bytes(data)
    with pytest.raises(ValueError, match="x64"):
        ins.verify_x64_pe(source / ins.EXE)
    icon.write_bytes(ico_fixture()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        ins.verify_icon(icon)


def test_recipe_and_file_list_are_exact_and_safe(payload, tmp_path):
    source, icon = payload
    put(source, "data/literal{app}.txt")
    document = seal(source)
    script, _ = ins.prepare_script(source, document, icon, tmp_path / "build", spanning=True)
    text = script.read_text(encoding="utf-8-sig")
    lines = (script.parent / "package-files.iss").read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == len(document["files"]) + 3
    assert not any("*" in line or "recursesubdirs" in line for line in lines)
    assert any("literal{{app}.txt" in line for line in lines)
    assert "PrivilegesRequired=lowest" in text
    assert "MinVersion=10.0.18362" in text
    assert "ArchitecturesAllowed=x64os" in text
    assert "postinstall skipifsilent" in text
    assert "Flags: unchecked" in text
    assert "AllowNoIcons=yes" in text
    assert not any(
        line.startswith(("[UninstallDelete]", "[InstallDelete]")) for line in text.splitlines()
    )


def test_strict_github_asset_limit(tmp_path, monkeypatch):
    path = put(tmp_path, "setup.exe", b"1234")
    monkeypatch.setattr(ins, "ASSET_LIMIT", 4)
    with pytest.raises(ValueError, match="strictly below"):
        ins.asset_records([path])
    path.write_bytes(b"123")
    assert ins.asset_records([path])[path.name]["bytes"] == 3


def test_output_must_be_new_and_separate(payload, tmp_path):
    source, icon = payload
    for output in (source / "installer", source.parent, tmp_path / "existing"):
        if output.name == "existing":
            output.mkdir()
        with pytest.raises((ValueError, FileExistsError)):
            ins.build_installer(source, output, icon, "0" * 64, "0" * 64)


def fake_build(payload, tmp_path, monkeypatch, *, oversized=False, mutate=False):
    source, icon = payload
    compiler = put(tmp_path, "ISCC.exe")
    monkeypatch.setattr(ins, "compiler_version", lambda _: "7.1.0")
    monkeypatch.setattr(ins, "file_version", lambda p: "7.1.0.0" if p == compiler else "0.5.0.0")
    if oversized:
        monkeypatch.setattr(ins, "ASSET_LIMIT", 4000)

    def compile_script(_, script):
        directory = script.parent / "assets"
        put(
            directory,
            "Setup.exe",
            b"x" * (4000 if oversized and script.parent.name == "single" else 20),
        )
        if script.parent.name == "spanned":
            put(directory, "Setup-1.bin", b"y" * 30)
        if mutate:
            (source / "runtime/python.exe").write_bytes(b"changed")
        return [str(compiler), str(script)]

    monkeypatch.setattr(ins, "compile_script", compile_script)
    return ins.build_installer(
        source,
        tmp_path / "output",
        icon,
        ins.sha256(source / ins.EXE),
        ins.sha256(icon),
        compiler=compiler,
    )


def test_build_receipt_and_single_preference(payload, tmp_path, monkeypatch):
    receipt = fake_build(payload, tmp_path, monkeypatch)
    assert receipt["layout"] == "single"
    assert len(receipt["attempts"]) == 1
    release = tmp_path / "output/release"
    assert set(p.name for p in release.iterdir()) == {
        "Setup.exe",
        "build-receipt.json",
        "SHA256SUMS.txt",
    }
    assert ins.sha256(release / "Setup.exe") == receipt["assets"]["Setup.exe"]["sha256"]
    assert ins.sha256(release / "build-receipt.json") in (release / "SHA256SUMS.txt").read_text()


def test_auto_falls_back_when_single_is_too_large(payload, tmp_path, monkeypatch):
    receipt = fake_build(payload, tmp_path, monkeypatch, oversized=True)
    assert receipt["layout"] == "spanned"
    assert len(receipt["attempts"]) == 2
    assert set(receipt["assets"]) == {"Setup.exe", "Setup-1.bin"}


def test_source_changed_during_compile_is_not_promoted(payload, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="SHA-256"):
        fake_build(payload, tmp_path, monkeypatch, mutate=True)
    assert not (tmp_path / "output/release").exists()


@pytest.mark.skipif(
    not os.environ.get("MATERIAL_STUDIO_TEST_ISCC"),
    reason="Opt-in tiny Inno compilation: set MATERIAL_STUDIO_TEST_ISCC",
)
@pytest.mark.parametrize("spanning", [False, True])
def test_real_compiler_tiny_fixture(payload, tmp_path, spanning):
    source, icon = payload
    document = ins.verify_package(source)
    script, assets = ins.prepare_script(
        source, document, icon, tmp_path / "build", spanning=spanning
    )
    ins.compile_script(Path(os.environ["MATERIAL_STUDIO_TEST_ISCC"]), script)
    assert len(list(assets.glob("*.exe"))) == 1
    assert bool(list(assets.glob("*.bin"))) == spanning
    assert ins.asset_records(list(assets.iterdir()))


@pytest.mark.skipif(
    not (
        os.environ.get("MATERIAL_STUDIO_TEST_ISCC")
        and os.environ.get("MATERIAL_STUDIO_TEST_INSTALL")
    ),
    reason="Opt-in Windows fixture install/uninstall requires MATERIAL_STUDIO_TEST_INSTALL=1",
)
def test_real_silent_install_uninstall_preserves_user_file(payload, tmp_path, monkeypatch):
    source, icon = payload
    csc = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if not csc.is_file():
        pytest.skip("Windows .NET Framework C# compiler not available")
    code = tmp_path / "Fixture.cs"
    code.write_text(
        "using System.IO; using System.Reflection; "
        '[assembly: AssemblyFileVersion("0.5.0.0")] '
        "class Fixture { static void Main() { File.WriteAllText("
        'Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, "launched.txt"), "bad"); } }'
    )
    subprocess.run(
        [
            str(csc),
            "/nologo",
            "/target:winexe",
            "/platform:x64",
            "/out:" + str(source / ins.EXE),
            str(code),
        ],
        check=True,
        capture_output=True,
    )
    seal(source, release_version="0.5.0")
    prepare = ins.prepare_script

    def isolated_script(*args, **kwargs):
        script, assets = prepare(*args, **kwargs)
        text = script.read_text(encoding="utf-8-sig")
        text = text.replace("C0673919-301E-4B52-AD89-A3960836BB2C", str(uuid.uuid4()))
        text = text.replace(
            "UninstallDisplayName=Material Studio",
            "UninstallDisplayName=Material Studio Installer Test",
        )
        script.write_text(text, encoding="utf-8-sig")
        return script, assets

    monkeypatch.setattr(ins, "prepare_script", isolated_script)
    output = tmp_path / "installer"
    ins.build_installer(
        source,
        output,
        icon,
        ins.sha256(source / ins.EXE),
        ins.sha256(icon),
        compiler=os.environ["MATERIAL_STUDIO_TEST_ISCC"],
    )
    setup = next((output / "release").glob("*.exe"))
    destination = tmp_path / "isolated-install"
    installed = subprocess.run(
        [
            str(setup),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/SP-",
            "/NOICONS",
            "/TASKS=",
            "/DIR=" + str(destination),
            "/LOG=" + str(tmp_path / "install.log"),
        ],
        check=False,
        timeout=60,
    )
    try:
        assert installed.returncode == 0
        assert ins.sha256(destination / ins.EXE) == ins.sha256(source / ins.EXE)
        assert not (destination / "launched.txt").exists()
        put(destination, "my-images/user-added.png", b"user image must survive")
        assert (destination / "MaterialStudio.ico").is_file()
    finally:
        uninstaller = destination / "unins000.exe"
        if uninstaller.exists():
            subprocess.run(
                [
                    str(uninstaller),
                    "/VERYSILENT",
                    "/SUPPRESSMSGBOXES",
                    "/NORESTART",
                    "/LOG=" + str(tmp_path / "uninstall.log"),
                ],
                check=True,
                timeout=60,
            )
    assert not (destination / ins.EXE).exists()
    assert not (destination / "runtime/python.exe").exists()
    assert (destination / "my-images/user-added.png").read_bytes() == b"user image must survive"
