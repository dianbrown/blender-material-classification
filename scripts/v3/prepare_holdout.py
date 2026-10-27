"""Download/verify ONLY the frozen holdout's three maps; never select or evaluate models.

ambientCG ZIPs are read using HTTP byte ranges, so unused displacement, AO, DX
normals, previews and project files are not downloaded or extracted.
"""

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender/scripts/v3"))
from holdout_common import (  # noqa: E402
    absolute_path,
    digest,
    inside,
    official_url,
    read_json,
    require_suffix,
    validate_identity_audit,
    validate_receipt,
    validate_registry,
    write_json,
)

USER_AGENT = "MaterialClassificationHoldout/1.0 (CC0 academic provenance audit)"


class RemoteZip(io.RawIOBase):
    """Seekable official archive; reject servers that ignore byte-range requests."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.transferred = 0
        self.resolved_url = url

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= self.position <= self.size:
            raise ValueError("archive seek out of bounds")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end == self.position:
            return b""
        start = self.position
        request = urllib.request.Request(
            self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end - 1}"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            official_url(response.url, "ambientcg", download=True)
            expected = f"bytes {start}-{end - 1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("official archive does not honor exact HTTP byte ranges")
            data = response.read(end - start + 1)
            if len(data) != end - start:
                raise ValueError("truncated/oversized archive range")
            self.resolved_url = response.url
        self.transferred += len(data)
        self.position = end
        return data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--registry", type=absolute_path, required=True)
    parser.add_argument("--original-manifest", type=absolute_path, required=True)
    parser.add_argument("--sources-root", type=absolute_path, required=True)
    parser.add_argument("--receipt", type=absolute_path, required=True)
    parser.add_argument("--identity-audit", type=absolute_path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="fetch missing maps; default verifies"
    )
    return parser.parse_args(argv)


def inspect_image(path, spec, role):
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        # Provider resolution tiers preserve a scanned surface's aspect ratio.
        # e.g. Poly Haven rough_linen 1K is 1024x1026, not a square image.
        if min(image.size) != spec["resolution"] or max(image.size) > 2 * spec["resolution"]:
            raise ValueError(f"wrong map resolution: {path}: {image.size}")
        if image.mode not in (
            ("RGB", "RGBA") if role in ("color", "normal") else ("L", "RGB", "RGBA")
        ):
            raise ValueError(f"wrong map channels: {path}: {image.mode}")
        return {"width": image.width, "height": image.height, "mode": image.mode}


def main(argv=None):
    args = parse_args(argv)
    require_suffix(args.sources_root, ("artifacts", "v3", "holdout_sources"))
    require_suffix(args.receipt, ("metadata", "v3_holdout", "download_receipt.json"))
    registry = read_json(args.registry)
    validate_registry(registry, args.original_manifest)
    validate_identity_audit(registry, read_json(args.identity_audit))
    receipt = read_json(args.receipt) if args.receipt.exists() else None
    if receipt is not None:
        # Once sealed, never silently replace a source with new remote content.
        validate_receipt(registry, receipt, args.sources_root)
        for asset in registry["assets"]:
            for role, spec in asset["maps"].items():
                inspect_image(inside(args.sources_root, spec["path"]), spec, role)
        print(f"Verified sealed receipt: {len(receipt['files'])} source maps")
        return
    if not args.download:
        raise ValueError("no sealed download receipt; explicitly pass --download")
    files, bytes_downloaded = [], 0
    for asset in registry["assets"]:
        archive, remote = None, None
        try:
            if asset["provider"] == "ambientcg":
                first = asset["maps"]["color"]
                remote = RemoteZip(first["url"], first["archive_size"])
                archive = zipfile.ZipFile(remote)
            for role, spec in asset["maps"].items():
                target = inside(args.sources_root, spec["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                # Unsealed partial files are re-fetched, never trusted as provenance.
                if archive:
                    member = archive.getinfo(spec["archive_member"])
                    if member.file_size > 32 * 1024 * 1024:
                        raise ValueError("unexpectedly large map in archive")
                    data = archive.read(member)  # also checks the member CRC
                    resolved_url = remote.resolved_url
                elif (
                    target.exists()
                    and hashlib.md5(target.read_bytes()).hexdigest() == spec["provider_md5"]
                ):
                    data, resolved_url = target.read_bytes(), spec["url"]
                else:
                    request = urllib.request.Request(
                        spec["url"], headers={"User-Agent": USER_AGENT}
                    )
                    with urllib.request.urlopen(request, timeout=90) as response:
                        official_url(response.url, asset["provider"], download=True)
                        data = response.read(32 * 1024 * 1024 + 1)
                        resolved_url = response.url
                    if len(data) != spec["size"]:
                        raise ValueError("map download size mismatch")
                    if hashlib.md5(data).hexdigest() != spec["provider_md5"]:
                        raise ValueError("map no longer matches the frozen provider MD5")
                    bytes_downloaded += len(data)
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(data)
                details = inspect_image(temporary, spec, role)
                temporary.replace(target)
                files.append(
                    {
                        "path": spec["path"],
                        "role": role,
                        "url": spec["url"],
                        "resolved_url": resolved_url,
                        "archive_member": spec.get("archive_member"),
                        "sha256": digest(target),
                        "size": len(data),
                        **details,
                    }
                )
            print(f"Prepared {asset['material_id']}", flush=True)
        finally:
            if archive:
                archive.close()
                bytes_downloaded += remote.transferred
    receipt = {
        "schema_version": 1,
        "selection_sha256": registry["selection_sha256"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "bytes_transferred": bytes_downloaded,
        "files": files,
    }
    validate_receipt(registry, receipt, args.sources_root)
    write_json(args.receipt, receipt)
    print(f"Sealed {len(files)} map hashes; transferred {bytes_downloaded} bytes")


if __name__ == "__main__":
    main()
