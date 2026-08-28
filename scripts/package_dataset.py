"""Build a local dataset archive and SHA-256 checksum.

This utility is deliberately upload-free. It packages a supplied dataset
directory (and optionally selected metadata) into a ZIP suitable for a future
distribution workflow. The archive is written outside the source directory
by default under the gitignored ``artifacts/`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(dataset_root: Path, output_zip: Path, metadata_root: Path | None = None) -> Path:
    dataset_root = dataset_root.resolve()
    output_zip = output_zip.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
    if output_zip.parent == dataset_root or dataset_root in output_zip.parents:
        raise ValueError("output archive must be outside the dataset directory")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"dataset directory is empty: {dataset_root}")

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(dataset_root).as_posix())
        if metadata_root is not None:
            metadata_root = metadata_root.resolve()
            if not metadata_root.is_dir():
                raise FileNotFoundError(f"metadata directory does not exist: {metadata_root}")
            for path in sorted(p for p in metadata_root.rglob("*") if p.is_file()):
                archive.write(path, Path("metadata") / path.relative_to(metadata_root))

    checksum_path = output_zip.with_suffix(output_zip.suffix + ".sha256")
    checksum_path.write_text(f"{sha256(output_zip)}  {output_zip.name}\n", encoding="utf-8")
    return checksum_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset.zip"))
    parser.add_argument("--metadata-root", type=Path, default=None)
    args = parser.parse_args()
    checksum = package(args.dataset_root, args.output, args.metadata_root)
    print(f"archive: {args.output.resolve()}")
    print(f"sha256:  {checksum.resolve()}")


if __name__ == "__main__":
    main()
