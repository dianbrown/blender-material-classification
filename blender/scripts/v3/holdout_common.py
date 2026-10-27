"""Pure-Python integrity rules shared by the isolated v3 holdout tools."""

import argparse
import csv
import hashlib
import json
import re
import struct
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

CLASSES = ("wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather")
MAPS = ("color", "roughness", "normal")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches_text_digest(path, expected):
    """Accept only LF/CRLF checkout differences for explicitly textual frozen inputs."""
    raw = Path(path).read_bytes()
    raw.decode("utf-8")  # Never apply newline equivalence to binary source assets.
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {
        hashlib.sha256(content).hexdigest()
        for content in (raw, lf, lf.replace(b"\n", b"\r\n"))
    }


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("an explicit absolute path is required")
    return path.resolve()


def require_suffix(path, suffix):
    path = Path(path).resolve()
    parts = tuple(part.lower() for part in path.parts)
    if parts[-len(suffix) :] != tuple(part.lower() for part in suffix):
        raise ValueError(f"isolated path must end with {'/'.join(suffix)}: {path}")
    return path


def inside(root, relative):
    """Reject traversal, absolute/drive paths, and symlinks outside the source root."""
    posix = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or posix.is_absolute()
        or any(p in ("..", ".") for p in relative.split("/"))
    ):
        raise ValueError(f"unsafe relative path: {relative}")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"path escapes root: {relative}")
    return path


def family(provider, asset_id):
    """Conservative variant screen; explicit lineage is checked in addition."""
    if provider == "ambientcg":
        return re.sub(r"(?<=\d)[a-z]+$", "", asset_id.lower())
    words = re.sub(r"[_\d]+", " ", asset_id.lower()).split()
    variants = {
        "red",
        "white",
        "black",
        "blue",
        "green",
        "grey",
        "gray",
        "brown",
        "worn",
        "clean",
        "weathered",
        "painted",
        "rusty",
        "rusted",
    }
    return "_".join(w for w in words if w not in variants)


def official_url(url, provider, download=False):
    parsed = urlparse(url)
    hosts = (
        {"api.polyhaven.com", "polyhaven.com", "dl.polyhaven.org"}
        if provider == "polyhaven"
        else {"ambientcg.com", "docs.ambientcg.com", "acg-download.struffelproductions.com"}
    )
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError(f"not an official HTTPS URL: {url}")
    if download and provider == "polyhaven" and parsed.hostname != "dl.polyhaven.org":
        raise ValueError("Poly Haven maps must use its official download host")


def validate_registry(registry, original_manifest):
    if registry.get("schema_version") != 1 or registry.get("split") != "holdout":
        raise ValueError("registry must be schema 1, holdout only")
    payload = {k: v for k, v in registry.items() if k != "selection_sha256"}
    if registry.get("selection_sha256") != fingerprint(payload):
        raise ValueError("frozen selection fingerprint mismatch")
    if not matches_text_digest(original_manifest, registry["original_manifest_sha256"]):
        raise ValueError("original source manifest changed")
    with Path(original_manifest).open(encoding="utf-8-sig", newline="") as stream:
        originals = list(csv.DictReader(stream))
    old = {r["source_material_name"].casefold() for r in originals}
    if old != {s.casefold() for s in registry["excluded_source_ids"]}:
        raise ValueError("source exclusion inventory does not match original manifest")
    old_families = {
        (p, family(p, r["source_material_name"]))
        for r in originals
        for p in (
            "ambientcg" if re.match(r"Leather\d", r["source_material_name"]) else "polyhaven",
        )
    }
    old_lineage = set(registry["excluded_lineage_ids"])
    assets = registry["assets"]
    counts = Counter(a["class"] for a in assets)
    if set(counts) != set(CLASSES) or set(counts.values()) != {registry["groups_per_class"]}:
        raise ValueError("holdout must be balanced across exactly seven classes")
    if registry["groups_per_class"] not in (3, 4, 5) or registry["views_per_material"] != 30:
        raise ValueError("holdout requires 3-5 independent groups/class and 30 views/group")
    seen, lineages, paths = set(), set(), set()
    for asset in assets:
        provider, aid = asset["provider"], asset["asset_id"]
        if provider not in ("polyhaven", "ambientcg"):
            raise ValueError("unapproved asset provider")
        if not re.fullmatch(r"[A-Za-z0-9_]+", aid):
            raise ValueError("unsafe asset ID")
        key = (provider, family(provider, aid))
        identities = {aid.casefold(), *(x.casefold() for x in asset["aliases"])}
        if (
            identities & old
            or key in old_families
            or key in seen
            or any((provider, family(provider, x)) in old_families for x in asset["aliases"])
        ):
            raise ValueError(f"duplicate/original source family: {aid}")
        seen.add(key)
        lineage = set(asset["lineage_ids"])
        relations = asset.get("provider_relations", {})
        lineage.update(f"{provider}:{x}" for x in relations.get("parents", []))
        lineage.update(f"{provider}:{x}" for x in relations.get("variations", []))
        if any(
            x.casefold() in old
            for x in relations.get("parents", []) + relations.get("variations", [])
        ):
            raise ValueError(f"original source in provider ancestry: {aid}")
        if not lineage or lineage & (lineages | old_lineage):
            raise ValueError(f"shared/original lineage: {aid}")
        lineages.update(lineage)
        if asset["split"] != "holdout" or asset["license"] != "CC0-1.0":
            raise ValueError("all assets must be CC0 holdout sources")
        if (
            not asset["authors"]
            or not asset["label_rationale"]
            or not asset["independence_evidence"]
        ):
            raise ValueError("missing source attribution or review evidence")
        official_url(asset["canonical_url"], provider)
        official_url(asset["license_url"], provider)
        if asset["material_id"] != f"{provider}_{aid}":
            raise ValueError("material ID must encode the provider and canonical source")
        if set(asset["maps"]) != set(MAPS) or asset["normal_convention"] != "OpenGL":
            raise ValueError("exactly color, roughness and OpenGL normal maps required")
        if not 0 <= asset["metallic_value"] <= 1 or not asset["render_assumptions"]:
            raise ValueError("missing PBR assumptions")
        for role, spec in asset["maps"].items():
            official_url(spec["url"], provider, download=True)
            relative = spec["path"]
            inside(Path.cwd(), relative)
            if not relative.startswith(f"{provider}/{aid}/") or relative.casefold() in paths:
                raise ValueError("map path does not belong to a unique source")
            paths.add(relative.casefold())
            if spec["resolution"] not in (1024, 2048):
                raise ValueError("only 1K/2K maps are permitted")
            if spec["colorspace"] != ("sRGB" if role == "color" else "Non-Color"):
                raise ValueError("invalid map color space")


def validate_identity_audit(registry, audit):
    if audit["selection_sha256"] != registry["selection_sha256"] or audit["collisions"]:
        raise ValueError("source identity audit mismatch or collision")
    originals = audit["original_sources"]
    expected = {x for x in registry["excluded_source_ids"] if not re.match(r"Leather\d", x)}
    if {x["asset_id"] for x in originals} != expected or len(originals) != len(expected):
        raise ValueError("source identity audit omits original sources")
    old_hashes = {x["files_hash"] for x in originals}
    for asset in registry["assets"]:
        if asset["provider"] == "polyhaven":
            if asset["provider_files_hash"] in old_hashes:
                raise ValueError("original source recipe hash reused")
            for role, spec in asset["maps"].items():
                if any(spec["provider_md5"] in x["map_md5_1k"][role].values() for x in originals):
                    raise ValueError("original map identity reused under a new name")


def validate_receipt(registry, receipt, sources_root):
    if receipt["selection_sha256"] != registry["selection_sha256"]:
        raise ValueError("download receipt belongs to another selection")
    expected = {m["path"] for a in registry["assets"] for m in a["maps"].values()}
    actual = {m["path"] for m in receipt["files"]}
    if expected != actual or len(actual) != len(receipt["files"]):
        raise ValueError("download receipt must contain exactly the selected maps")
    by_path = {m["path"]: m for m in receipt["files"]}
    for asset in registry["assets"]:
        for role, spec in asset["maps"].items():
            rec = by_path[spec["path"]]
            path = inside(sources_root, spec["path"])
            if not path.is_file() or digest(path) != rec["sha256"]:
                raise ValueError(f"source hash mismatch or missing source: {path}")
            if rec["url"] != spec["url"] or rec["role"] != role:
                raise ValueError("download URL or role mismatch")
            if rec["size"] != path.stat().st_size:
                raise ValueError("source file size mismatch")


def child_seed(master, cls, material_id, index):
    key = f"{master}|{cls}|{material_id}|{index:02d}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31 - 1)


def validate_png(path, width, height):
    """Check every CRC, complete IDAT decompression and every scanline (8-bit RGB/A)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
    offset, compressed, ended, channels = 8, bytearray(), False, None
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        size = struct.unpack_from(">I", data, offset)[0]
        tag = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + size]
        if offset + size + 12 > len(data):
            raise ValueError("truncated PNG payload")
        crc = struct.unpack_from(">I", data, offset + 8 + size)[0]
        if zlib.crc32(tag + body) & 0xFFFFFFFF != crc:
            raise ValueError("PNG CRC mismatch")
        if offset == 8 and tag != b"IHDR":
            raise ValueError("missing first PNG IHDR")
        if tag == b"IHDR":
            w, h, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
            if (
                (w, h) != (width, height)
                or bits != 8
                or color not in (2, 6)
                or compression
                or filtering
                or interlace
                or channels is not None
            ):
                raise ValueError("unsupported or stale PNG format/dimensions")
            channels = 3 if color == 2 else 4
        elif tag == b"IDAT":
            compressed.extend(body)
        elif tag == b"IEND":
            if size or offset + 12 != len(data):
                raise ValueError("invalid PNG ending")
            ended = True
            break
        offset += size + 12
    if not ended or channels is None:
        raise ValueError("incomplete PNG")
    row_bytes = width * channels + 1
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(compressed, row_bytes * height + 1)
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(pixels) != row_bytes * height
        or any(pixels[i] > 4 for i in range(0, len(pixels), row_bytes))
    ):
        raise ValueError("invalid PNG pixel stream")
    return digest(path)
