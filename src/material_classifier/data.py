"""Leakage-safe loading of the rendered material dataset.

The render manifest is the image-level source of truth.  The split manifest is
used to verify that every image inherits the locked material-level split.  No
split assignment is performed by this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # Keep metadata helpers usable in environments without PyTorch.
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only in minimal installs

    class Dataset:  # type: ignore[no-redef]
        """Small fallback so manifest utilities remain importable."""


CLASS_ORDER: tuple[str, ...] = (
    "wood",
    "metal",
    "stone",
    "concrete",
    "ceramic",
    "fabric",
    "leather",
)
SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_RENDER_MANIFEST = Path("metadata/render_manifest.csv")
DEFAULT_SPLIT_MANIFEST = Path("metadata/material_manifest_split.csv")


@dataclass(frozen=True)
class RenderRecord:
    """Typed view of a render-manifest row."""

    image_path: str
    filename: str
    class_name: str
    material_id: str
    split: str
    render_index: int

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "RenderRecord":
        return cls(
            image_path=row["image_path"].replace("\\", "/"),
            filename=row["filename"],
            class_name=row["class"],
            material_id=row["material_id"],
            split=row["split"],
            render_index=int(row["render_index"]),
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_render_manifest(path: str | Path = DEFAULT_RENDER_MANIFEST) -> list[dict[str, str]]:
    """Load the canonical image manifest without changing row order."""

    rows = _read_csv(Path(path))
    required = {"image_path", "filename", "class", "material_id", "split", "render_index"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"render manifest {path} is missing columns: {sorted(missing)}")
    return rows


def load_material_split_manifest(
    path: str | Path = DEFAULT_SPLIT_MANIFEST,
) -> list[dict[str, str]]:
    """Load the locked material-level split manifest."""

    rows = _read_csv(Path(path))
    required = {"class", "material_id", "split"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"split manifest {path} is missing columns: {sorted(missing)}")
    return rows


def class_to_idx(classes: Sequence[str] = CLASS_ORDER) -> dict[str, int]:
    """Return a deterministic class mapping and reject duplicate labels."""

    values = tuple(classes)
    if len(values) != len(set(values)):
        raise ValueError("class labels must be unique")
    unknown = set(values).difference(CLASS_ORDER)
    if unknown:
        raise ValueError(f"unknown material classes: {sorted(unknown)}")
    return {name: index for index, name in enumerate(values)}


def records_from_manifest(rows: Iterable[Mapping[str, str]]) -> list[RenderRecord]:
    """Convert raw rows to typed records while retaining material identity."""

    return [RenderRecord.from_row(row) for row in rows]


def _resolve_image(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"image path escapes dataset root: {relative_path}") from exc
    return candidate


class MaterialRenderDataset(Dataset):
    """PyTorch dataset backed by canonical manifest records.

    ``material_id`` is retained in :attr:`records` and can be returned with
    each sample by setting ``return_metadata=True``.  The default tuple shape
    remains ``(image, class_index)`` for compatibility with training loops.
    """

    def __init__(
        self,
        records: Sequence[RenderRecord | Mapping[str, str]],
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        transform: Callable[[Any], Any] | None = None,
        classes: Sequence[str] = CLASS_ORDER,
        return_metadata: bool = False,
    ) -> None:
        self.records = [
            record if isinstance(record, RenderRecord) else RenderRecord.from_row(record)
            for record in records
        ]
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.class_to_idx = class_to_idx(classes)
        self.return_metadata = return_metadata
        unknown = sorted({record.class_name for record in self.records} - set(self.class_to_idx))
        if unknown:
            raise ValueError(f"records contain unknown classes: {unknown}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int] | dict[str, Any]:
        from PIL import Image

        record = self.records[index]
        path = _resolve_image(self.dataset_root.parent, record.image_path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            value: Any = image.copy()
        if self.transform is not None:
            value = self.transform(value)
        target = self.class_to_idx[record.class_name]
        if self.return_metadata:
            return {
                "image": value,
                "label": target,
                "material_id": record.material_id,
                "split": record.split,
                "image_path": record.image_path,
            }
        return value, target


def records_for_split(
    rows: Sequence[Mapping[str, str]], split: str, *, strict: bool = True
) -> list[RenderRecord]:
    """Select records for an existing split; never creates a new partition."""

    if split not in SPLIT_ORDER:
        raise ValueError(f"split must be one of {SPLIT_ORDER}, got {split!r}")
    selected = records_from_manifest(row for row in rows if row["split"] == split)
    if strict and not selected:
        raise ValueError(f"canonical manifest has no records for split {split!r}")
    return selected
