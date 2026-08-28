"""Reproducibility, environment capture, and durable artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and (when installed) all PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older PyTorch
                torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initializer derived from PyTorch's initial seed."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_environment(root: Path | None = None) -> dict[str, Any]:
    """Capture versions and GPU metadata without requiring CUDA Python bindings."""
    result: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "git_commit": git_commit(root),
        "captured_at": now_utc(),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "torchvision": None,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        try:
            import torchvision

            result["torchvision"] = torchvision.__version__
        except Exception as exc:  # pragma: no cover - broken optional import
            result["torchvision_error"] = repr(exc)
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        result.update(
            {"torch": None, "torchvision": None, "cuda": None, "cuda_available": False, "gpu": None}
        )
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        result["nvidia_smi"] = smi.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        result["nvidia_smi"] = None
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Save a complete checkpoint atomically, avoiding partial files on interruption."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_torch_checkpoint(path: Path, map_location: Any = "cpu") -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location=map_location)


def count_parameters(model: Any) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    """Hash a JSON-compatible resolved configuration deterministically."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
