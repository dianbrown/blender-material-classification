"""Collect an auditable Windows native notice bundle without invoking Cargo.

Requires a caller-supplied `cargo metadata --locked --offline --format-version 1
--filter-platform x86_64-pc-windows-msvc` snapshot. The caller must obtain that
snapshot while Cargo is idle. No downloads, builds, lockfile writes, or license
choices for the application itself are performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import tomllib
from pathlib import Path

TARGET = "x86_64-pc-windows-msvc"
ZED_REV = "aed37547d8543000494fb69101c05daec5891068"
NOTICE_NAME = re.compile(
    r"^(?:unlicen[sc]e|licen[sc]e|copying|copyright|notices?|thirdpartynotices|patents|authors)(?:$|[._-])",
    re.I,
)
SOURCE_SUFFIXES = {
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".s",
    ".asm",
    ".wgsl",
    ".hlsl",
    ".metal",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    return (
        "\n".join(
            line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        ).strip()
        + "\n"
    )


def lf_digest(data: bytes) -> str:
    """Portable fingerprint across Git's Windows CRLF checkout conversion."""
    return digest(data.replace(b"\r\n", b"\n"))


def dump(path: Path, obj: object) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def html_text(value: str) -> str:
    value = re.sub(r"</?(?:p|div|h[1-6]|li|ul|details|summary|pre)\b[^>]*>", "\n", value)
    value = html.unescape(re.sub(r"<[^>]+>", "", value))
    return re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", value).strip()


def selected_graph(metadata: dict) -> tuple[dict, dict, set, set]:
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    root = metadata["resolve"]["root"]
    # A normal edge into a proc-macro is a build tool, including its dependencies.
    states = set()
    pending = [(root, "runtime")]
    while pending:
        key, role = pending.pop()
        if (key, role) in states:
            continue
        states.add((key, role))
        for dep in nodes[key]["deps"]:
            for kind in dep["dep_kinds"]:
                if kind["kind"] == "dev":
                    continue
                proc = any("proc-macro" in t["kind"] for t in packages[dep["pkg"]]["targets"])
                next_role = (
                    "build" if role == "build" or kind["kind"] == "build" or proc else "runtime"
                )
                pending.append((dep["pkg"], next_role))
    selected = {k for k, _ in states} - {root}
    runtime = {k for k, role in states if role == "runtime"} - {root}
    return packages, nodes, selected, runtime


def collect(args: argparse.Namespace) -> None:
    metadata_bytes = args.metadata.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
    lock_bytes = args.lock.read_bytes()
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    locked = {(p["name"], p["version"], p.get("source")): p for p in lock["package"]}
    packages, nodes, selected, runtime = selected_graph(metadata)
    root = packages[metadata["resolve"]["root"]]
    if root["name"] != "material-studio":
        raise ValueError("Metadata must resolve material-studio as its root")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    upstream_sources = json.loads((out / "upstream/sources.json").read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    text_sources: dict[str, list[str]] = {}

    def add(text: str, source: str) -> str:
        text = normalized(text)
        key = digest(text.encode("utf-8"))
        texts[key] = text
        text_sources.setdefault(key, [])
        if source not in text_sources[key]:
            text_sources[key].append(source)
        return key

    def upstream(name: str) -> str:
        path = out / "upstream" / name
        if "sha256_lf" in upstream_sources[name]:
            assert lf_digest(path.read_bytes()) == upstream_sources[name]["sha256_lf"], (
                f"Upstream file changed: {name}"
            )
        return add(path.read_text(encoding="utf-8"), upstream_sources[name]["url"])

    apache = upstream("Apache-2.0.txt")
    inventory = []
    exceptions = []
    for key in sorted(selected, key=lambda k: (packages[k]["name"], packages[k]["version"])):
        p = packages[key]
        locked_p = locked.get((p["name"], p["version"], p["source"]))
        if not locked_p:
            raise ValueError(f"Metadata/lock mismatch: {key}")
        directory = Path(p["manifest_path"]).parent
        raw_manifest = Path(p["manifest_path"]).read_bytes()
        source = p["source"]
        if source.startswith("git+") and (
            "zed-industries/zed" not in source or not source.endswith("#" + ZED_REV)
        ):
            raise ValueError(f"Unexpected git revision: {source}")
        entry = {
            "name": p["name"],
            "version": p["version"],
            "id": key,
            "source": source,
            "repository": p["repository"],
            "license_declared": p["license"],
            "authors": p["authors"],
            "role": "runtime" if key in runtime else "build-tool",
            "features": nodes[key]["features"],
            "checksum": locked_p.get("checksum"),
            "manifest_sha256": digest(raw_manifest),
            "evidence": [],
            "exceptions": [],
        }
        vcs = directory / ".cargo_vcs_info.json"
        if vcs.exists():
            entry["vcs"] = json.loads(vcs.read_text(encoding="utf-8"))
        declared = p["license"] or "NOASSERTION"
        # Legacy Cargo slash syntax denotes OR; do not silently rewrite AND.
        spdx = re.sub(r"\s*/\s*", " OR ", declared)
        entry["spdx"] = spdx
        chosen = (
            "Apache-2.0"
            if "Apache-2.0" in spdx
            else ("MIT" if re.search(r"\bMIT\b", spdx) else spdx)
        )
        if " AND " in spdx:
            if spdx != "(MIT OR Apache-2.0) AND Unicode-3.0":
                raise ValueError(f"Unreviewed conjunctive license: {spdx}")
            chosen = "Apache-2.0 AND Unicode-3.0"
        entry["license_option_used"] = chosen
        if p["name"] == "seahash":
            entry["upstream_lookup"] = {
                "date": "2026-10-27",
                "url": "https://gitlab.redox-os.org/redox-os/seahash/-/raw/94b632aeac099031c373599313d5b5f0acbbaec0/LICENSE",
                "result": (
                    "Earlier lookup HTTP 403; release audit raw URL HTTP 404. "
                    "Exact-revision GitLab tree API succeeded on retry: all "
                    "17 entries inspected, no license file. Archive sources "
                    "match upstream blobs and contain no copyright notice. See "
                    "release-evidence/native-notice-gaps.json for archive hashes "
                    "and fresh requests. No copyright statement invented."
                ),
            }
        elif p["name"] == "simd_helpers":
            entry["upstream_lookup"] = {
                "date": "2026-10-27",
                "url": "https://api.github.com/repos/lu-zero/simd_helpers/git/trees/ca1a2f84aa386d758e98f8a609d990263932fb85?recursive=1",
                "result": (
                    "Exact published revision tree inspected: no LICENSE, COPYING"
                    " or NOTICE file. LICENSE and LICENSE-MIT raw paths returned "
                    "HTTP 404."
                ),
            }
        found = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            rel = path.relative_to(directory).as_posix()
            is_notice = bool(NOTICE_NAME.match(path.name))
            if not is_notice:
                continue
            data = path.read_bytes()
            encoding = "utf-8"
            try:
                text = data.decode(encoding)
            except UnicodeDecodeError:
                encoding = "windows-1252"
                text = data.decode(encoding)
            resolved = path
            # Git symlinks materialized as small text files on Windows.
            if len(text.strip()) < 200 and re.fullmatch(r"(?:\.\./)+[\w./-]+", text.strip()):
                resolved = (path.parent / text.strip()).resolve()
                text = resolved.read_text(encoding="utf-8")
            if len(text.strip()) < 100 and not path.name.upper().startswith("AUTHORS"):
                raise ValueError(f"Unresolved or abbreviated license: {path}")
            text_id = add(text, f"{source} :: {p['name']}-{p['version']}/{rel}")
            found.append(text_id)
            evidence = {
                "path": rel,
                "sha256": digest(data),
                "text": text_id,
                "kind": "license-file",
                "encoding": encoding,
            }
            if resolved != path:
                evidence["resolved_pointer"] = path.read_text(encoding="utf-8").strip()
                evidence["resolved_sha256"] = digest(resolved.read_bytes())
            entry["evidence"].append(evidence)
        # Retain full comment blocks containing source-level legal notices,
        # including assembly/C code vendored inside otherwise Rust packages.
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            rel_parts = path.relative_to(directory).parts
            if any(x in {"tests", "benches", "examples", ".git"} for x in rel_parts):
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            if not re.search(
                (
                    "copyright|SPDX-License-Identifier|redistribution and "
                    "use|permission is hereby granted"
                ),
                content,
                re.I,
            ):
                continue
            blocks = re.findall(r"/\*.*?\*/", content, re.S)
            blocks.extend(re.findall(r"(?:^[ \t]*(?://|;)[^\n]*(?:\n|$))+", content, re.M))
            for block in blocks:
                if not re.search(
                    (
                        "copyright|SPDX-License-Identifier|redistribution and "
                        "use|permission is hereby granted"
                    ),
                    block,
                    re.I,
                ):
                    continue
                text_id = add(
                    block,
                    f"{source} :: {p['name']}-{p['version']}/"
                    f"{path.relative_to(directory).as_posix()}",
                )
                entry["evidence"].append(
                    {
                        "path": path.relative_to(directory).as_posix(),
                        "kind": "source-notice",
                        "text": text_id,
                    }
                )
        if "Apache-2.0" in chosen:
            entry["evidence"].append({"kind": "standard-license", "text": apache})
        if p["name"] in {"taffy", "pulp-wasm-simd-flag"}:
            text_id = upstream(p["name"] + "-MIT.txt")
            entry["evidence"].append({"kind": "pinned-upstream-license", "text": text_id})
        elif not found and "Apache-2.0" not in chosen:
            text_id = upstream("MIT.txt") if chosen == "MIT" else None
            if text_id:
                entry["evidence"].append(
                    {"kind": "standard-license-template-only", "text": text_id}
                )
            issue = (
                "Published archive has no license file; exact upstream "
                "copyright/permission notice could not be obtained. Manifest SPDX"
                " and authors are retained; the standard template does not supply"
                " the missing copyright notice."
            )
            entry["exceptions"].append(issue)
            exceptions.append(
                {"package": p["name"] + " " + p["version"], "role": entry["role"], "issue": issue}
            )
        if not entry["evidence"]:
            raise ValueError(f"No license evidence at all: {key}")
        inventory.append(entry)
        if len(inventory) % 50 == 0:
            print(f"Collected {len(inventory)}/{len(selected)} packages", flush=True)

    # The installed toolchain supplies a library-only notice collection. Preserve
    # its mappings, but deduplicate its many repeated license bodies with crates.
    rust_html = (args.rust_doc / "COPYRIGHT-library.html").read_text(encoding="utf-8")
    rust_references = []

    def replace_pre(match: re.Match) -> str:
        text_id = add(
            html.unescape(match.group(1)),
            "Rust 1.99.0 distribution: share/doc/rust/COPYRIGHT-library.html",
        )
        rust_references.append(text_id)
        return "<p>Full notice text: " + text_id + "</p>"

    rust_mapping = re.sub(r"<pre>(.*?)</pre>", replace_pre, rust_html, flags=re.S)
    rust_mapping = html_text(rust_mapping)
    rust_mapping_id = add(
        rust_mapping,
        (
            "Rust 1.99.0 distribution: share/doc/rust/COPYRIGHT-library.html "
            "(HTML rendered as plain text; pre blocks deduplicated)"
        ),
    )
    rust_terms = []
    # Include only licenses referenced by the library collection, not compiler
    # or documentation-only GPL/GCC/OFL/CC-BY-SA license files.
    for name in [
        "Apache-2.0",
        "MIT",
        "Unicode-3.0",
        "BSD-2-Clause",
        "ISC",
        "LLVM-exception",
        "NCSA",
    ]:
        path = args.rust_doc / "licenses" / (name + ".txt")
        if name in rust_mapping and path.exists():
            rust_terms.append(
                add(
                    path.read_text(encoding="utf-8"),
                    "Rust 1.99.0 distribution: share/doc/rust/licenses/" + path.name,
                )
            )
    toolchain_manifest = args.rust_doc.parents[2] / "lib/rustlib/multirust-channel-manifest.toml"
    toolchain = tomllib.loads(toolchain_manifest.read_text(encoding="utf-8"))
    rust_version = toolchain["pkg"]["rust"]["version"]
    if not rust_version.startswith("1.99.0 "):
        raise ValueError("Expected installed Rust 1.99.0 notice collection")

    graph = {
        "target": TARGET,
        "command": "cargo metadata --locked --offline --format-version 1 --filter-platform "
        + TARGET,
        "raw_metadata_sha256": digest(metadata_bytes),
        "cargo_lock_sha256": digest(lock_bytes),
        "app_manifest_sha256": digest(args.manifest.read_bytes()),
        "cargo_lock_lf_sha256": lf_digest(lock_bytes),
        "app_manifest_lf_sha256": lf_digest(args.manifest.read_bytes()),
        "root": root["name"] + " " + root["version"],
        "root_id": root["id"],
        "zed_revision": ZED_REV,
        "selection": (
            "Transitive closure of root resolve node, normal and build edges; dev"
            " edges excluded. Input must already be filtered by Cargo for "
            "Windows. Build tools/proc macros retained conservatively, not "
            "claimed to be linked."
        ),
        "nodes": [
            {
                "id": k,
                "features": nodes[k]["features"],
                "deps": [
                    d for d in nodes[k]["deps"] if any(t["kind"] != "dev" for t in d["dep_kinds"])
                ],
            }
            for k in sorted(selected | {root["id"]})
        ],
        "excluded_lock_packages": [
            {"name": p["name"], "version": p["version"], "source": p.get("source")}
            for p in lock["package"]
            if (p["name"], p["version"], p.get("source"))
            not in {(q["name"], q["version"], q["source"]) for q in inventory}
            and p["name"] != root["name"]
        ],
    }
    bundle = {
        "schema_version": 1,
        "target": TARGET,
        "dependency_count": len(inventory),
        "runtime_dependency_count": len(runtime),
        "build_only_dependency_count": len(selected - runtime),
        "rust": {
            "version": rust_version,
            "source": "https://github.com/rust-lang/rust",
            "copyright_library_sha256": digest(
                (args.rust_doc / "COPYRIGHT-library.html").read_bytes()
            ),
            "mapping_text": rust_mapping_id,
            "license_texts": sorted(set(rust_references + rust_terms)),
            "scope": (
                "Upstream toolchain library notice collection is intentionally "
                "conservative across supported library platforms; compiler-only "
                "COPYRIGHT.html is excluded."
            ),
        },
        "exceptions": exceptions,
        "release_checks": [
            (
                "This release inventory transparently retains the exact notice"
                " gaps listed in exceptions; it is not a certification of "
                "complete licensing compliance or approval for external binary "
                "publication."
            ),
            (
                "Match Cargo.lock, Cargo.toml, target, features and Rust "
                "toolchain to these hashes; regenerate after dependency changes."
            ),
            (
                "Microsoft CRT provenance and conditional redistribution terms "
                "are documented separately in app/licenses/MICROSOFT_CRT.txt "
                "and app/licenses/release-evidence/crt-provenance.json. Match "
                "the final DLL and build toolset; recheck PE imports after "
                "executable changes."
            ),
            (
                "Python/runtime and model notices remain in the packaged "
                "runtime and licenses directories. Supplemental official CUDA "
                "12.8 and cuDNN 9.19.0 terms are in app/licenses/upstream; see "
                "app/licenses/RELEASE_COMPONENTS.txt for scope and evidence. "
                "No application-code or rendered-dataset license is chosen here."
            ),
        ],
        "packages": inventory,
        "texts": {k: {"sha256": k, "sources": text_sources[k]} for k in sorted(texts)},
    }
    lines = [
        "MATERIAL STUDIO — WINDOWS NATIVE THIRD-PARTY NOTICES",
        "",
        f"Target: {TARGET}",
        f"Cargo dependencies: {len(inventory)} "
        f"({len(runtime)} normal runtime dependencies; "
        f"{len(selected - runtime)} build-only tools/proc macros)",
        f"GPUI/Zed framework revision: {ZED_REV}",
        f"Rust standard library: {rust_version}",
        "",
        (
            "This file is self-contained: each package maps to SHA-256 text IDs "
            "whose complete contents appear below. Identical texts are included "
            "once. Supplementary JSON records graph selection, checksums, "
            "features, source paths and validation evidence."
        ),
        (
            "The graph is Windows-specific and contains only the application "
            "dependency closure, not unrelated Zed applications. The 14 selected "
            "Zed framework crates declare Apache-2.0 in their individual "
            "manifests; the Zed root application license is not substituted."
        ),
        (
            "Declared SPDX expressions and the permissive option used for this "
            "notice bundle are both recorded. Supplied alternate license texts "
            "and embedded notices are retained. Build tools are included "
            "conservatively because generated code may enter the binary. A "
            "runtime dependency entry is not a claim that every function survives"
            " linker dead-code elimination."
        ),
        (
            "Rust's supplied standard-library collection also covers some other "
            "library targets; it is retained conservatively and does not assert "
            "that all those components are present on Windows."
        ),
        "",
        "UNRESOLVED NOTICE EVIDENCE / RELEASE CHECKS",
        "",
    ]
    lines.extend(f"{e['package']} ({e['role']}): {e['issue']}" for e in exceptions)
    lines.extend(bundle["release_checks"])
    lines += ["", "PACKAGE-TO-TEXT MAPPING", ""]
    for p in inventory:
        lines += [
            f"{p['name']} {p['version']} [{p['role']}]",
            f"Source: {p['source']}",
            f"SPDX: {p['spdx']}",
            f"License option used: {p['license_option_used']}",
            "Manifest authors (not a substitute for copyright notices): "
            + ("; ".join(p["authors"]) or "not specified"),
        ]
        for text_id in sorted({e["text"] for e in p["evidence"]}):
            lines.append("Text: " + text_id)
        lines.extend("EXCEPTION: " + e for e in p["exceptions"])
        lines.append("")
    lines += ["Rust 1.99.0 standard-library notice mapping: " + rust_mapping_id]
    lines += ["Rust supplemental terms: " + t for t in rust_terms]
    lines += ["", "COMPLETE LICENSE AND NOTICE TEXTS", ""]
    for key in sorted(texts):
        lines += ["=" * 78, "TEXT " + key, "=" * 78, texts[key]]
    notices = "\n".join(lines)
    args.notices.write_text(notices, encoding="utf-8", newline="\n")
    bundle["notices_sha256"] = digest(args.notices.read_bytes())
    bundle["notices_lf_sha256"] = lf_digest(args.notices.read_bytes())
    bundle["text_normalization"] = (
        "UTF-8, LF line endings, trailing line whitespace removed; original "
        "source bytes separately fingerprinted. Git CRLF checkout conversion "
        "is accepted for bundle and input validation."
    )
    dump(out / "inventory.json", bundle)
    dump(out / "windows-dependency-graph.json", graph)
    validate(args)


def validate(args: argparse.Namespace) -> None:
    sources = json.loads((args.output / "upstream/sources.json").read_text(encoding="utf-8"))
    for name, source in sources.items():
        assert lf_digest((args.output / "upstream" / name).read_bytes()) == source["sha256_lf"], (
            f"Upstream or supplemental text changed: {name}"
        )
    bundle = json.loads((args.output / "inventory.json").read_text(encoding="utf-8"))
    graph = json.loads((args.output / "windows-dependency-graph.json").read_text(encoding="utf-8"))
    raw = args.notices.read_bytes()
    assert lf_digest(raw) == bundle["notices_lf_sha256"], "Notices digest mismatch"
    notice_text = raw.decode("utf-8").replace("\r\n", "\n")
    blocks = dict(
        re.findall(
            r"={78}\nTEXT ([a-f0-9]{64})\n={78}\n(.*?)(?=\n={78}\nTEXT |\Z)", notice_text, re.S
        )
    )
    assert set(blocks) == set(bundle["texts"]), "Text index mismatch"
    for key, value in blocks.items():
        assert digest(normalized(value).encode("utf-8")) == key, f"Text digest mismatch: {key}"
    ids = {p["id"] for p in bundle["packages"]}
    assert len(ids) == bundle["dependency_count"]
    assert ids == {n["id"] for n in graph["nodes"]} - {graph["root_id"]}
    for node in graph["nodes"]:
        assert all(d["pkg"] in ids or d["pkg"] == graph["root_id"] for d in node["deps"])
    for p in bundle["packages"]:
        assert p["spdx"] != "NOASSERTION" and p["evidence"]
        assert all(e["text"] in blocks for e in p["evidence"])
        assert p["exceptions"] or any(
            e["kind"] in {"license-file", "pinned-upstream-license", "standard-license"}
            for e in p["evidence"]
        )
        bodies = "\n".join(blocks[e["text"]] for e in p["evidence"])
        clauses = {
            "Apache-2.0": r"TERMS AND CONDITIONS FOR USE",
            "MIT": r"permission is hereby granted",
            "BSD-2-Clause": r"redistribution and use",
            "BSD-3-Clause": r"redistribution and use",
            "Unicode-3.0": r"UNICODE.*LICENSE",
            "Zlib": r"altered source versions",
        }
        for term in p["license_option_used"].split(" AND "):
            assert term in clauses and re.search(clauses[term], bodies, re.I | re.S), (
                f"Missing license body for {p['name']}: {term}"
            )
    reachable = set()
    pending = [graph["root_id"]]
    nodes = {n["id"]: n for n in graph["nodes"]}
    while pending:
        key = pending.pop()
        if key not in reachable:
            reachable.add(key)
            pending.extend(d["pkg"] for d in nodes[key]["deps"])
    assert reachable == ids | {graph["root_id"]}, "Orphan or missing graph package"
    assert bundle["rust"]["mapping_text"] in blocks
    assert all(t in blocks for t in bundle["rust"]["license_texts"])
    if args.lock:
        assert lf_digest(args.lock.read_bytes()) == graph["cargo_lock_lf_sha256"], (
            "Cargo.lock changed"
        )
    if args.manifest:
        assert lf_digest(args.manifest.read_bytes()) == graph["app_manifest_lf_sha256"], (
            "Cargo.toml changed"
        )
    print(
        json.dumps(
            {
                "dependencies_accounted": len(ids),
                "unique_texts": len(blocks),
                "upstream_files_verified": len(sources),
                "unresolved_notice_exceptions": len(bundle["exceptions"]),
                "notices_bytes": len(raw),
                "validation": "passed; exceptions remain explicitly unresolved",
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rust-doc", type=Path)
    parser.add_argument("--output", type=Path, default=Path("app/licenses"))
    parser.add_argument("--notices", type=Path, default=Path("app/THIRD_PARTY_NOTICES.txt"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate(args)
    else:
        if not all([args.metadata, args.lock, args.manifest, args.rust_doc]):
            parser.error("collection requires --metadata, --lock, --manifest and --rust-doc")
        collect(args)


if __name__ == "__main__":
    main()
