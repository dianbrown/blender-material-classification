"""Build an offline Inno Setup installer from a final, hash-verified portable folder.

Build-host Python only; the installed app uses its own bundled interpreter.
No downloads, package mutation, signing, publishing, or application execution.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "package-manifest.json"
EXE = "MaterialStudio.exe"
VERSION = "0.6.0"
ASSET_LIMIT = 2 * 1024**3  # GitHub requires strictly less than this.
MODELS = {"dinov2_finetuned", "frozen_control", "frozen_selected", "resnet_fixed"}
RESERVED = {"materialstudio.ico", "materialstudio-installation.json"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Expected a lowercase SHA-256 for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return actual


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid manifest path: {value!r}")
    if PureWindowsPath(value).drive or PurePosixPath(value).is_absolute():
        raise ValueError(f"Absolute manifest path: {value}")
    for part in value.split("/"):
        if (
            part in {"", ".", ".."}
            or part.rstrip(" .") != part
            or any(ord(c) < 32 or c in ':<>"|?*' for c in part)
            or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", part, re.I)
        ):
            raise ValueError(f"Unsafe manifest path: {value!r}")
    return value


def inventory(root):
    """Refuse links/reparse points, including directory junctions, before descent."""
    result = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(directory) / name
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError(f"Reparse point is not a package input: {path}")
        for name in files:
            result.add((Path(directory) / name).relative_to(root).as_posix())
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def verify_package(source):
    source = Path(source).resolve(strict=True)
    actual = inventory(source)
    document = json.loads(
        (source / MANIFEST).read_text(encoding="utf-8"), object_pairs_hook=unique_object
    )
    if document.get("schema_version") != 1 or document.get("kind") != "MaterialStudio-portable":
        raise ValueError("Expected a schema 1 MaterialStudio-portable package manifest")
    if set(document.get("models", [])) != MODELS:
        raise ValueError("Package must contain all four selected models")
    # The original packager has no release version; newer versioned releases are accepted.
    for key in ("version", "app_version", "release_version"):
        if key in document and document[key] != VERSION:
            raise ValueError(f"Package {key} is not {VERSION}")
    records = document.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("Empty or invalid package inventory")
    names = set()
    for relative in records:
        normalized = safe_relative(relative).casefold()
        if normalized in names or normalized == MANIFEST.casefold():
            raise ValueError(f"Duplicate/self-referencing manifest entry: {relative}")
        if normalized in RESERVED or re.fullmatch(r"unins\d+\.(?:exe|dat|msg)", normalized):
            raise ValueError(f"Reserved installer destination: {relative}")
        names.add(normalized)
    if actual != set(records) | {MANIFEST}:
        missing = sorted(set(records) - actual)[:5]
        extra = sorted(actual - set(records) - {MANIFEST})[:5]
        raise ValueError(f"Manifest inventory mismatch: missing={missing}, extra={extra}")
    required = {
        EXE,
        "runtime/python.exe",
        "runtime/python313._pth",
        "vcruntime140.dll",
        "app/assets/study.json",
        "app/THIRD_PARTY_NOTICES.txt",
    }
    if not required <= set(records):
        raise ValueError(f"Missing offline payload inputs: {sorted(required - set(records))}")
    total = 0
    for index, (relative, entry) in enumerate(sorted(records.items()), start=1):
        path = source / relative
        if not isinstance(entry, dict) or type(entry.get("bytes")) is not int:
            raise ValueError(f"Invalid file record: {relative}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        checked_hash(path, entry.get("sha256"))
        total += entry["bytes"]
        if index % 2000 == 0:
            print(f"Verified {index}/{len(records)} payload files", flush=True)
    if document.get("payload_bytes") != total:
        raise ValueError("Manifest payload size mismatch")
    return document


def verify_x64_pe(path):
    with Path(path).open("rb") as stream:
        header = stream.read(64)
        if len(header) < 64 or header[:2] != b"MZ":
            raise ValueError("Application is not a PE executable")
        stream.seek(struct.unpack_from("<I", header, 60)[0])
        pe = stream.read(26)
    if (
        len(pe) != 26
        or pe[:4] != b"PE\0\0"
        or struct.unpack_from("<H", pe, 4)[0] != 0x8664
        or struct.unpack_from("<H", pe, 24)[0] != 0x20B
        or struct.unpack_from("<H", pe, 22)[0] & 0x2000
    ):
        raise ValueError("Application must be a Windows x64 PE32+ executable, not a DLL")


def file_version(path):
    """Read fixed PE version metadata with Windows APIs; never run the application."""
    if sys.platform != "win32":
        raise ValueError("Native version validation and installer compilation require Windows")
    from ctypes import wintypes

    api = ctypes.WinDLL("version", use_last_error=True)
    api.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    api.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    api.GetFileVersionInfoW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    api.GetFileVersionInfoW.restype = wintypes.BOOL
    api.VerQueryValueW.argtypes = [
        wintypes.LPCVOID,
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.UINT),
    ]
    api.VerQueryValueW.restype = wintypes.BOOL
    size = api.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        raise ValueError(f"Missing native file version resource: {path}")
    data = ctypes.create_string_buffer(size)
    if not api.GetFileVersionInfoW(str(path), 0, size, data):
        raise ctypes.WinError(ctypes.get_last_error())
    pointer, length = ctypes.c_void_p(), wintypes.UINT()
    if not api.VerQueryValueW(data, "\\", ctypes.byref(pointer), ctypes.byref(length)):
        raise ValueError("Missing VS_FIXEDFILEINFO")
    if length.value < 52:
        raise ValueError("Truncated VS_FIXEDFILEINFO")
    fields = struct.unpack("<13I", ctypes.string_at(pointer, 52))
    if fields[0] != 0xFEEF04BD:
        raise ValueError("Invalid VS_FIXEDFILEINFO")
    return ".".join(
        str(n) for n in (fields[2] >> 16, fields[2] & 65535, fields[3] >> 16, fields[3] & 65535)
    )


def verify_icon(path):
    data = Path(path).read_bytes()
    if len(data) < 6:
        raise ValueError("Icon is not an ICO file")
    reserved, kind, count = struct.unpack_from("<HHH", data)
    if reserved or kind != 1 or not count or len(data) < 6 + 16 * count:
        raise ValueError("Icon is not an ICO file")
    for index in range(count):
        size, offset = struct.unpack_from("<II", data, 6 + 16 * index + 8)
        if not size or offset < 6 + 16 * count or offset + size > len(data):
            raise ValueError("Truncated ICO image")


def verify_identity(source, icon, exe_hash, icon_hash):
    checked_hash(source / EXE, exe_hash)
    checked_hash(icon, icon_hash)
    verify_x64_pe(source / EXE)
    version = file_version(source / EXE)
    if version != VERSION + ".0":
        raise ValueError(f"Native file version {version}; expected {VERSION}.0. Rebuild the app.")
    verify_icon(icon)
    return version


def find_compiler(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = []
        if os.environ.get("INNO_SETUP_COMPILER"):
            candidates.append(Path(os.environ["INNO_SETUP_COMPILER"]))
        if shutil.which("ISCC.exe"):
            candidates.append(Path(shutil.which("ISCC.exe")))
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            if os.environ.get(variable):
                root = Path(os.environ[variable])
                for name in ("Inno Setup 7", "Inno Setup 7 (x64)", "Inno Setup 6"):
                    candidates.extend(
                        [root / name / "ISCC.exe", root / "Programs" / name / "ISCC.exe"]
                    )
        candidates.append(REPO / "artifacts/installer-tools/Inno Setup 7/ISCC.exe")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("ISCC.exe not found; pass --compiler. See docs/INSTALLATION.md.")


def compiler_version(compiler):
    # Official ISCC.exe/ISCmplr.dll fixed PE versions can be 0.0.0.0. Ask the engine.
    result = subprocess.run(
        [str(compiler), "--version"], capture_output=True, text=True, check=False, timeout=30
    )
    version = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Cannot identify compiler engine; use Inno Setup 7.1.0 or newer")
    if tuple(int(n) for n in version.split(".")) < (7, 1, 0):
        raise ValueError("Inno Setup 7.1.0 or newer is required")
    return version


def iss_string(value, *, constants=False):
    value = str(value)
    if any(ord(c) < 32 for c in value):
        raise ValueError("Control character in Inno Setup value")
    if constants:
        value = value.replace("{", "{{")
    return '"' + value.replace('"', '""') + '"'


def file_entry(source, relative):
    relative = safe_relative(relative)
    parent = PurePosixPath(relative).parent.as_posix()
    suffix = "" if parent == "." else "\\" + parent.replace("/", "\\").replace("{", "{{")
    return (
        f"Source: {iss_string(source, constants=True)}; "
        f'DestDir: "{{app}}{suffix}"; '
        f"DestName: {iss_string(PurePosixPath(relative).name, constants=True)}; "
        "Flags: ignoreversion"
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_script(source, document, icon, build, *, spanning):
    build.mkdir(parents=True)
    output = build / "assets"
    output.mkdir()
    shutil.copyfile(REPO / "installer/material-studio.iss", build / "material-studio.iss")
    shutil.copyfile(icon, build / "icon.ico")
    write_json(
        build / "installation.json",
        {
            "app": "Material Studio",
            "version": VERSION,
            "package_manifest_sha256": sha256(source / MANIFEST),
            "icon_sha256": sha256(icon),
        },
    )
    definitions = {
        "AppVersion": VERSION,
        "AppIcon": str(build / "icon.ico"),
        "BuildOutput": str(output),
        "Spanning": "yes" if spanning else "no",
    }
    (build / "build-config.iss").write_text(
        "\n".join(f"#define {key} {iss_string(value)}" for key, value in definitions.items())
        + "\n",
        encoding="utf-8-sig",
    )
    entries = [file_entry(source / name, name) for name in sorted(document["files"])]
    entries += [
        file_entry(source / MANIFEST, MANIFEST),
        file_entry(build / "icon.ico", "MaterialStudio.ico"),
        file_entry(build / "installation.json", "materialstudio-installation.json"),
    ]
    (build / "package-files.iss").write_text("\n".join(entries) + "\n", encoding="utf-8-sig")
    return build / "material-studio.iss", output


def compile_script(compiler, script):
    command = [str(compiler), "/Qp", str(script)]
    log = script.parent / "compiler.log"
    # Stream compiler progress into a log; retain it even after a failed compile.
    print(f"Compiling: {script}; progress log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(
            f"ISCC exited {result.returncode}; see {log}. "
            "If a single installer exceeds Inno's size limit, use --layout spanned."
        )
    return command


def asset_records(paths):
    result = {}
    for path in paths:
        size = path.stat().st_size
        if size >= ASSET_LIMIT or size <= 0:
            raise ValueError(f"Release asset must be nonempty and strictly below 2 GiB: {path}")
        result[path.name] = {"bytes": size, "sha256": sha256(path)}
    return result


def build_installer(source, output, icon, exe_hash, icon_hash, *, compiler=None, layout="auto"):
    source, icon = Path(source).resolve(strict=True), Path(icon).resolve(strict=True)
    output = Path(output).resolve()
    if (
        output.is_relative_to(source)
        or source.is_relative_to(output)
        or icon.is_relative_to(output)
    ):
        raise ValueError("Output must not overlap package or icon inputs")
    if output.exists():
        raise FileExistsError(f"Choose a fresh build output directory: {output}")
    if layout not in {"auto", "single", "spanned"}:
        raise ValueError(f"Unknown layout: {layout}")
    native_version = verify_identity(source, icon, exe_hash, icon_hash)
    document = verify_package(source)
    manifest_hash = sha256(source / MANIFEST)
    compiler = find_compiler(compiler)
    compiler_hash, engine_version = sha256(compiler), compiler_version(compiler)
    output.mkdir(parents=True, exist_ok=False)
    attempts = []
    modes = [False, True] if layout == "auto" else [layout == "spanned"]
    for spanning in modes:
        mode = "spanned" if spanning else "single"
        script, assets = prepare_script(
            source, document, icon, output / "build" / mode, spanning=spanning
        )
        command = compile_script(compiler, script)
        paths = sorted(assets.glob("*"))
        if not paths or len(list(assets.glob("*.exe"))) != 1:
            raise ValueError("Compiler did not produce exactly one setup executable")
        if any(p.suffix.lower() not in {".exe", ".bin"} or not p.is_file() for p in paths):
            raise ValueError("Unexpected compiler outputs")
        attempts.append(
            {
                "layout": mode,
                "command": command,
                "script_sha256": sha256(script),
                "file_list_sha256": sha256(script.parent / "package-files.iss"),
            }
        )
        if not spanning and paths[0].stat().st_size >= ASSET_LIMIT and layout == "auto":
            print(
                "Single EXE exceeds GitHub limit; rebuilding as 2,000,000,000-byte slices.",
                flush=True,
            )
            continue
        records = asset_records(paths)
        break
    # Detect source changes during compression before producing a success receipt.
    checked_hash(source / MANIFEST, manifest_hash)
    verify_identity(source, icon, exe_hash, icon_hash)
    verify_package(source)
    checked_hash(compiler, compiler_hash)
    release = output / "release"
    release.mkdir()
    for path in paths:
        shutil.move(str(path), release / path.name)
    receipt = {
        "schema_version": 1,
        "status": "built",
        "app": "Material Studio",
        "version": VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "layout": mode,
        "architecture": "x64",
        "min_windows_version": "10.0.18362",
        "source": str(source),
        "package_manifest_sha256": manifest_hash,
        "payload_files": len(document["files"]),
        "payload_bytes": document["payload_bytes"],
        "native_exe_sha256": exe_hash,
        "native_file_version": native_version,
        "icon_sha256": icon_hash,
        "compiler": {"path": str(compiler), "version": engine_version, "sha256": compiler_hash},
        "attempts": attempts,
        "assets": records,
        "signing": "No signing performed by this builder; hashes describe these exact bytes.",
    }
    receipt_path = release / "build-receipt.json"
    write_json(receipt_path, receipt)
    checksums = {**records, **asset_records([receipt_path])}
    (release / "SHA256SUMS.txt").write_text(
        "".join(f"{entry['sha256']}  {name}\n" for name, entry in sorted(checksums.items())),
        encoding="utf-8",
    )
    print(f"Verified installer assets and receipt: {release}", flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Final portable folder")
    parser.add_argument("--output", type=Path, required=True, help="Fresh build directory")
    parser.add_argument("--icon", type=Path, required=True, help="Final app/assets/icon.ico")
    parser.add_argument("--exe-sha256", required=True, help="Reviewed final native EXE SHA-256")
    parser.add_argument("--icon-sha256", required=True, help="Reviewed final icon SHA-256")
    parser.add_argument("--compiler", type=Path, help="Path to Inno Setup ISCC.exe")
    parser.add_argument("--layout", choices=("auto", "single", "spanned"), default="auto")
    args = parser.parse_args(argv)
    try:
        build_installer(
            args.source,
            args.output,
            args.icon,
            args.exe_sha256,
            args.icon_sha256,
            compiler=args.compiler,
            layout=args.layout,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Installer build failed: {exc}\n")


if __name__ == "__main__":
    main()
