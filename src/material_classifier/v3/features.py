"""Deterministic frozen extraction and content-addressed, row-aligned caches.

Models are loaded only by an explicit extraction/inference call. Tests inject a
model factory and never download weights. Resolutions and models run sequentially.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from . import CLASS_NAMES, SCHEMA_VERSION

MANIFEST_COLUMNS = ("image_path", "class", "material_id", "source_group", "original_split")


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Atomic replacement, used only inside an owned run/cache directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(path):
    protocol = read_json(path)
    if protocol["class_order"] != list(CLASS_NAMES):
        raise ValueError("Protocol class order must match the seven v3 classes")
    if (protocol["outer_folds"], protocol["inner_folds"]) != (5, 3):
        raise ValueError("V3 requires five outer and three inner source-group folds")
    if protocol["group_column"] != "source_group":
        raise ValueError("V3 grouping must use source_group")
    if protocol["C"] != [0.01, 0.1, 1.0, 10.0]:
        raise ValueError("V3 C grid must be [0.01, 0.1, 1, 10]")
    names = [candidate["name"] for candidate in protocol["candidates"]]
    if len(names) != len(set(names)) or protocol["control"] not in names:
        raise ValueError("Candidate names must be unique and include the fixed control")
    return protocol


def environment():
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "Pillow",
        "safetensors",
        "huggingface-hub",
        "joblib",
        "threadpoolctl",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
    }


def implementation_hash():
    # Normalize checkout line endings, never hide substantive code changes.
    return json_hash(
        {
            name: Path(__file__).with_name(name).read_text(encoding="utf-8")
            for name in ("features.py", "__init__.py")
        }
    )


def read_manifest(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    if frame.empty or frame["image_path"].duplicated().any():
        raise ValueError("Manifest must be nonempty with unique image_path values")
    for column in MANIFEST_COLUMNS:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Manifest has empty {column}")
    if not frame["class"].isin(CLASS_NAMES).all():
        raise ValueError("Manifest contains unknown class labels")
    if frame.groupby("source_group")["class"].nunique().gt(1).any():
        raise ValueError("Each source_group must belong to exactly one class")
    return frame.reset_index(drop=True)


def image_file(data_root, image_path):
    root = Path(data_root).resolve()
    path = (root / image_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Image missing or outside --data-root: {image_path}")
    return path


def source_identity(manifest, frame, data_root):
    records = [
        {"image_path": path, "sha256": file_hash(image_file(data_root, path))}
        for path in frame["image_path"]
    ]
    return {
        "manifest_sha256": file_hash(manifest),
        "rows": len(frame),
        "image_paths": frame["image_path"].tolist(),
        "sources": records,
        "source_content_sha256": json_hash(records),
    }


@contextmanager
def directory_lock(directory):
    lock = Path(directory) / ".writer.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(
            f"Output is locked: {lock}. If a writer was killed, verify it has stopped "
            "before removing the stale lock."
        ) from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def configure_torch(device, threads=4):
    if threads < 1:
        raise ValueError("CPU threads must be positive")
    torch.set_num_threads(min(threads, os.cpu_count() or 1, 8))
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose --device cpu explicitly")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    # This setting must precede CUDA context creation in the extraction process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def transform_image(image, spec, size):
    """Explicit PIL equivalent of evaluation resize + center crop; no augmentation."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    short = int(size / spec["crop_pct"])
    width, height = image.size
    if width <= height:
        target = (short, int(short * height / width))
    else:
        target = (int(short * width / height), short)
    image = image.resize(target, Image.Resampling.BICUBIC)
    left, top = round((target[0] - size) / 2), round((target[1] - size) / 2)
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32).copy() / np.float32(255)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(spec["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(spec["std"], dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def load_frozen_model(spec, size, device="cpu", offline=False):
    """Load hash-verified pinned weights, adapting DINO position embeddings explicitly."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights = hf_hub_download(
        spec["repository"],
        spec["weights_file"],
        revision=spec["revision"],
        local_files_only=offline,
    )
    if file_hash(weights) != spec["weights_sha256"]:
        raise ValueError(f"Weights SHA-256 mismatch for {spec['name']}")
    kwargs = {"img_size": size} if "patch_size" in spec else {}
    model = timm.create_model(spec["name"], pretrained=False, num_classes=0, **kwargs)
    state = load_file(weights, device="cpu")
    if "patch_size" in spec:
        from timm.models.vision_transformer import checkpoint_filter_fn

        state = checkpoint_filter_fn(state, model, interpolation="bicubic", antialias=True)
        if tuple(model.patch_embed.img_size) != (size, size):
            raise ValueError("DINO input resolution override failed")
        if model.num_prefix_tokens != spec["num_prefix_tokens"]:
            raise ValueError("Unexpected DINO prefix-token count")
    else:
        # The source classifier is deliberately discarded, not used as an embedding.
        for key in ("head.fc.weight", "head.fc.bias"):
            state.pop(key, None)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def pool_dino_tokens(tokens, num_prefix_tokens, expected_patches=None, width=None):
    if tokens.ndim != 3 or num_prefix_tokens < 1 or tokens.shape[1] <= num_prefix_tokens:
        raise ValueError("Expected B x (prefix + patches) x width DINO tokens")
    patches = tokens[:, num_prefix_tokens:, :].float()
    if expected_patches is not None and patches.shape[1] != expected_patches:
        raise ValueError(f"Unexpected patch count: {patches.shape[1]} != {expected_patches}")
    if width is not None and tokens.shape[2] != width:
        raise ValueError(f"Unexpected DINO token width: {tokens.shape[2]} != {width}")
    return {
        "cls": tokens[:, 0, :].float(),
        "mean": patches.mean(dim=1),
        "std": patches.std(dim=1, correction=0),
    }


@torch.inference_mode()
def extract_batch(model, batch, spec, size):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Extractor model must be eval() with all parameters frozen")
    features = model.forward_features(batch)
    if "patch_size" in spec:
        pooled = pool_dino_tokens(
            features, model.num_prefix_tokens, (size // spec["patch_size"]) ** 2, spec["width"]
        )
    else:
        pooled = {"pooled": model.forward_head(features, pre_logits=True).float()}
    result = {}
    for name, tensor in pooled.items():
        if tensor.shape != (len(batch), spec["width"]) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid {name} feature shape or nonfinite features: {tensor.shape}")
        result[name] = tensor.cpu().numpy().astype(np.float32, copy=False)
    return result


def jobs_for_protocol(protocol, parts=None):
    jobs = []
    for family, spec in protocol["models"].items():
        for size in spec["resolutions"]:
            prefix = f"{family}_{size}"
            if parts is None or any(part.startswith(prefix + "_") for part in parts):
                jobs.append((prefix, spec, size))
    if not jobs:
        raise ValueError("No extraction jobs requested")
    return jobs


def extract_features(
    manifest,
    data_root,
    output,
    protocol,
    *,
    batch_size=16,
    device="cuda",
    threads=4,
    resume=False,
    offline=False,
    parts=None,
    model_factory=None,
):
    """Extract base arrays; resume only identical data, recipe, code, runtime and batching.

    A committed batch has a verified file hash. A crash before committing a batch
    merely recomputes that batch. Completed caches are never rewritten.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = configure_torch(device, threads)
    frame = read_manifest(manifest)
    source = source_identity(manifest, frame, data_root)
    jobs = jobs_for_protocol(protocol, parts)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "protocol_sha256": json_hash(protocol),
        "protocol": protocol,
        "implementation_sha256": implementation_hash(),
        "environment": environment(),
        "jobs": [job[0] for job in jobs],
        "device": str(device),
        "hardware": hardware,
        "batch_size": batch_size,
        "torch_threads": torch.get_num_threads(),
    }
    identity_hash = json_hash(identity)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f"Cache output exists: {output}; use --resume for an identical cache")
    output.mkdir(parents=True, exist_ok=True)
    with directory_lock(output):
        record_path = output / "cache.json"
        if record_path.exists():
            record = read_json(record_path)
            if record.get("identity_sha256") != identity_hash or record.get("identity") != identity:
                raise ValueError("Stale cache: identity mismatch; choose a new --output directory")
            if record["complete"]:
                load_feature_cache(
                    output, manifest=manifest, data_root=data_root, protocol=protocol
                )
                return record
        else:
            if any(path.name != ".writer.lock" for path in output.iterdir()):
                raise ValueError("Cache directory contains files without an identity record")
            record = {
                "identity": identity,
                "identity_sha256": identity_hash,
                "complete": False,
                "jobs": {},
                "arrays": {},
            }
            write_json(record_path, record)
            write_json(output / "protocol.json", protocol)
            frame.to_csv(output / "manifest.csv", index=False)
        factory = model_factory or load_frozen_model
        for prefix, spec, size in jobs:
            job = record["jobs"].setdefault(prefix, {"batches": [], "complete": False})
            if job["complete"]:
                _verify_arrays(output, job["arrays"], len(frame))
                continue
            job_dir = output / prefix
            job_dir.mkdir(exist_ok=True)
            completed = 0
            for shard in job["batches"]:
                if (
                    shard["start"] != completed
                    or file_hash(output / shard["file"]) != shard["sha256"]
                ):
                    raise ValueError(f"Corrupt resume batch in {prefix}")
                completed = shard["end"]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            model = factory(spec, size, device=device, offline=offline)
            model.eval().requires_grad_(False)
            for first in range(completed, len(frame), batch_size):
                last = min(first + batch_size, len(frame))
                tensors = []
                for path in frame["image_path"].iloc[first:last]:
                    with Image.open(image_file(data_root, path)) as image:
                        tensors.append(transform_image(image, spec, size))
                batch = torch.stack(tensors).to(device)
                try:
                    values = extract_batch(model, batch, spec, size)
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "CUDA OOM: batching is immutable for this cache. Use a new output "
                        "with a smaller --batch-size; no silent retry changes extraction."
                    ) from exc
                shard_path = job_dir / f"batch_{first:08d}_{last:08d}.npz"
                temporary = shard_path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, **values)
                os.replace(temporary, shard_path)
                job["batches"].append(
                    {
                        "file": shard_path.relative_to(output).as_posix(),
                        "start": first,
                        "end": last,
                        "sha256": file_hash(shard_path),
                    }
                )
                write_json(record_path, record)
                del batch, tensors, values
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            metrics = {
                "session_seconds": elapsed,
                "session_rows": len(frame) - completed,
                "session_images_per_second": (len(frame) - completed) / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else 0,
            }
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            components = ("cls", "mean", "std") if "patch_size" in spec else ("pooled",)
            arrays = {}
            for component in components:
                key = prefix + "_" + component
                path = output / (key + ".npy")
                matrix = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(len(frame), spec["width"])
                )
                for shard in job["batches"]:
                    with np.load(output / shard["file"], allow_pickle=False) as archive:
                        matrix[shard["start"] : shard["end"]] = archive[component]
                matrix.flush()
                del matrix
                arrays[key] = {
                    "file": path.name,
                    "sha256": file_hash(path),
                    "shape": [len(frame), spec["width"]],
                    "dtype": "float32",
                }
            job.update({"complete": True, "arrays": arrays, "metrics": metrics})
            record["arrays"].update(arrays)
            write_json(record_path, record)
        # Detect files changed while extraction was in progress.
        if source_identity(manifest, frame, data_root) != source:
            raise ValueError("Source content changed during extraction; discard this cache")
        record["complete"] = True
        write_json(record_path, record)
        return record


def _verify_arrays(directory, arrays, rows):
    result = {}
    for key, info in arrays.items():
        path = Path(directory) / info["file"]
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"Feature cache checksum mismatch: {key}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            list(array.shape) != info["shape"]
            or array.shape[0] != rows
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"Invalid feature array: {key}")
        result[key] = array
    return result


def load_feature_cache(directory, *, manifest, data_root=None, protocol=None):
    record = read_json(Path(directory) / "cache.json")
    identity = record["identity"]
    if not record["complete"] or json_hash(identity) != record["identity_sha256"]:
        raise ValueError("Incomplete or corrupt feature cache identity")
    frame = read_manifest(manifest)
    if (
        file_hash(manifest) != identity["source"]["manifest_sha256"]
        or frame["image_path"].tolist() != identity["source"]["image_paths"]
    ):
        raise ValueError("Stale cache: manifest/image_path alignment mismatch")
    if protocol is not None and json_hash(protocol) != identity["protocol_sha256"]:
        raise ValueError("Stale cache: protocol mismatch")
    if identity["implementation_sha256"] != implementation_hash():
        raise ValueError("Stale cache: feature implementation changed")
    if data_root is not None and source_identity(manifest, frame, data_root) != identity["source"]:
        raise ValueError("Stale cache: source image content changed")
    return _verify_arrays(directory, record["arrays"], len(frame)), record


def assemble_features(base_arrays, parts):
    missing = set(parts) - set(base_arrays)
    if missing:
        raise ValueError(f"Feature cache is missing recipe parts: {sorted(missing)}")
    arrays = [base_arrays[part] for part in parts]
    if not arrays or len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("Feature parts have incompatible row counts")
    return np.concatenate(arrays, axis=1)
