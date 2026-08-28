"""Benchmark the four completed headline checkpoints on synthetic tensors only.

This command never opens a dataset manifest or image. It requires completed
headline run metadata under ``artifacts/checkpoints/<model>/`` and writes
resumable efficiency metrics under ``reports/results/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from material_classifier.inference import benchmark_inference, load_model
from material_classifier.models import MODEL_NAMES
from material_classifier.utils import (
    atomic_json_dump,
    capture_environment,
    count_parameters,
    load_torch_checkpoint,
    now_utc,
    sha256_file,
    stable_hash,
)

SCHEMA_VERSION = "1.0"
BATCH_SIZES = (1, 32)
CSV_FIELDS = (
    "schema_version",
    "model",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_mb",
    "total_parameters",
    "trainable_parameters",
    "best_epoch",
    "total_training_time_seconds",
    "peak_vram_bytes",
    "peak_vram_mb",
    "effective_batch_size",
    "oom_adjustments_json",
    "batch_1_available",
    "batch_1_reason",
    "batch_1_latency_ms_batch",
    "batch_1_latency_ms_per_image",
    "batch_1_throughput_images_per_second",
    "batch_32_available",
    "batch_32_reason",
    "batch_32_latency_ms_batch",
    "batch_32_latency_ms_per_image",
    "batch_32_throughput_images_per_second",
    "cache_key",
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid {label}: expected a JSON object at {path}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Invalid completed-run metadata: {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Invalid completed-run metadata: {label} must be finite")
    if positive and result <= 0:
        raise RuntimeError(f"Invalid completed-run metadata: {label} must be positive")
    return result


def validate_prerequisites(root: Path, model_name: str) -> dict[str, Any]:
    """Load and cross-check one completed headline checkpoint and its metadata."""
    directory = root / "artifacts" / "checkpoints" / model_name
    checkpoint = directory / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {model_name} headline checkpoint: {checkpoint}")
    state = _read_json(directory / "run_state.json", f"{model_name} run state")
    resolved = _read_json(directory / "resolved_config.json", f"{model_name} resolved config")
    if state.get("status") != "completed":
        raise RuntimeError(f"{model_name} run is not completed (status={state.get('status')!r})")
    if state.get("model") not in (None, model_name):
        raise RuntimeError(f"{model_name} run_state model does not match its directory")
    config_hash = state.get("config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        raise RuntimeError(f"{model_name} completed run has no config_hash")
    if resolved.get("config_hash") != config_hash:
        raise RuntimeError(f"{model_name} run_state/resolved_config hash mismatch")

    payload = load_torch_checkpoint(checkpoint, "cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise RuntimeError(f"{model_name} checkpoint has no model_state mapping")
    if payload.get("config_hash") != config_hash:
        raise RuntimeError(f"{model_name} checkpoint/run_state config hash mismatch")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{model_name} checkpoint has no config mapping")
    checkpoint_model = config.get("name", config.get("architecture"))
    aliases = {"CustomCNN": "custom_cnn", "vit_b_16": "vit_b16"}
    checkpoint_model = aliases.get(checkpoint_model, checkpoint_model)
    if checkpoint_model != model_name:
        raise RuntimeError(f"{model_name} checkpoint config identifies {checkpoint_model!r}")

    best = state.get("best")
    if not isinstance(best, dict):
        raise RuntimeError(f"{model_name} completed run has no best metadata")
    best_epoch = int(_number(best.get("best_epoch"), "best.best_epoch", positive=True))
    checkpoint_epoch = int(_number(payload.get("epoch"), "checkpoint.epoch", positive=True))
    if best_epoch != checkpoint_epoch:
        raise RuntimeError(f"{model_name} run_state/checkpoint best epoch mismatch")
    training_time = _number(
        state.get("total_training_time_seconds"),
        "total_training_time_seconds",
        positive=True,
    )
    checkpoint_training_time = payload.get("total_training_time_seconds")
    if checkpoint_training_time is not None:
        checkpoint_training_time = _number(
            checkpoint_training_time,
            "checkpoint.total_training_time_seconds",
            positive=True,
        )
        if not math.isclose(training_time, checkpoint_training_time):
            raise RuntimeError(f"{model_name} run_state/checkpoint total training time mismatch")
    peak_vram = state.get("peak_vram_bytes", payload.get("peak_vram_bytes"))
    if peak_vram is not None:
        peak_vram = int(_number(peak_vram, "peak_vram_bytes"))
    state_effective_batch = state.get("effective_batch_size")
    checkpoint_effective_batch = payload.get("effective_batch_size")
    if state_effective_batch is not None and checkpoint_effective_batch is not None:
        if int(_number(state_effective_batch, "effective_batch_size", positive=True)) != int(
            _number(
                checkpoint_effective_batch,
                "checkpoint.effective_batch_size",
                positive=True,
            )
        ):
            raise RuntimeError(f"{model_name} run_state/checkpoint effective batch size mismatch")
    effective_batch = (
        state_effective_batch if state_effective_batch is not None else checkpoint_effective_batch
    )
    if effective_batch is None:
        raise RuntimeError(f"{model_name} completed run has no effective_batch_size")
    effective_batch = int(_number(effective_batch, "effective_batch_size", positive=True))

    state_has_adjustments = "oom_adjustments" in state
    checkpoint_has_adjustments = "oom_adjustments" in payload
    if (
        state_has_adjustments
        and checkpoint_has_adjustments
        and state.get("oom_adjustments") != payload.get("oom_adjustments")
    ):
        raise RuntimeError(f"{model_name} run_state/checkpoint OOM adjustments mismatch")
    adjustments = (
        state.get("oom_adjustments")
        if state_has_adjustments
        else payload.get("oom_adjustments", {})
    )
    if adjustments is None:
        adjustments = {}
    if not isinstance(adjustments, (dict, list)):
        raise RuntimeError(f"{model_name} oom_adjustments must be an object or list")
    input_size = int(_number(config.get("input_size", 224), "config.input_size", positive=True))
    return {
        "directory": directory,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024 * 1024),
        "payload": payload,
        "best_epoch": best_epoch,
        "total_training_time_seconds": training_time,
        "peak_vram_bytes": peak_vram,
        "effective_batch_size": effective_batch,
        "oom_adjustments": adjustments,
        "input_size": input_size,
    }


def _device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking requested but torch.cuda.is_available() is false")
    return device


def benchmark_settings(device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "device": str(device),
        "dtype": "float32",
        "batch_sizes": list(BATCH_SIZES),
        "warmup": warmup,
        "repeats": repeats,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        settings["gpu_name"] = torch.cuda.get_device_name(index)
        settings["gpu_vram_bytes"] = torch.cuda.get_device_properties(index).total_memory
    return settings


def _unavailable(batch_size: int, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "batch_size": batch_size,
        "latency_ms_batch": None,
        "latency_ms_per_image": None,
        "throughput_images_per_second": None,
    }


def _benchmark_batch(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    tensor = torch.randn(batch_size, 3, input_size, input_size, device=device)
    result = benchmark_inference(model, [tensor], device, warmup=warmup, repeats=repeats)
    return {"available": True, "reason": None, **result}


def benchmark_one(
    root: Path,
    model_name: str,
    device: torch.device,
    warmup: int,
    repeats: int,
    settings: dict[str, Any],
    prerequisites: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prerequisites = prerequisites or validate_prerequisites(root, model_name)
    checkpoint = prerequisites["checkpoint"]
    cache_key = stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "model": model_name,
            "checkpoint_sha256": prerequisites["checkpoint_sha256"],
            "input_size": prerequisites["input_size"],
            "settings": settings,
        }
    )
    model, _payload, loaded_device = load_model(checkpoint, device)
    if loaded_device != device:
        raise RuntimeError(f"{model_name} loaded on {loaded_device}, expected {device}")
    total_parameters, trainable_parameters = count_parameters(model)
    batches: dict[str, dict[str, Any]] = {}
    try:
        batches["1"] = _benchmark_batch(
            model, device, prerequisites["input_size"], 1, warmup, repeats
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(f"{model_name} batch-1 benchmark ran out of memory") from exc
        raise

    try:
        batches["32"] = _benchmark_batch(
            model, device, prerequisites["input_size"], 32, warmup, repeats
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        batches["32"] = _unavailable(32, f"CUDA OOM at fixed batch size 32: {exc}")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    peak_vram = prerequisites["peak_vram_bytes"]
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "checkpoint_path": str(checkpoint.relative_to(root).as_posix()),
        "checkpoint_sha256": prerequisites["checkpoint_sha256"],
        "checkpoint_size_mb": prerequisites["checkpoint_size_mb"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "best_epoch": prerequisites["best_epoch"],
        "total_training_time_seconds": prerequisites["total_training_time_seconds"],
        "peak_vram_bytes": peak_vram,
        "peak_vram_mb": None if peak_vram is None else peak_vram / (1024 * 1024),
        "effective_batch_size": prerequisites["effective_batch_size"],
        "oom_adjustments": prerequisites["oom_adjustments"],
        "input_size": prerequisites["input_size"],
        "benchmark": batches,
        "cache_key": cache_key,
    }


def _valid_cached(entry: Any, model_name: str, cache_key: str) -> bool:
    if not isinstance(entry, dict):
        return False
    if (
        entry.get("schema_version") != SCHEMA_VERSION
        or entry.get("model") != model_name
        or not isinstance(entry.get("cache_key"), str)
        or not entry["cache_key"]
        or entry.get("cache_key") != cache_key
    ):
        return False
    digest = entry.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    try:
        int(digest, 16)
    except ValueError:
        return False
    positive_fields = (
        "checkpoint_size_mb",
        "total_parameters",
        "best_epoch",
        "total_training_time_seconds",
        "effective_batch_size",
        "input_size",
    )
    if not all(_finite_positive(entry.get(field)) for field in positive_fields):
        return False
    trainable = entry.get("trainable_parameters")
    if (
        isinstance(trainable, bool)
        or not isinstance(trainable, (int, float))
        or not math.isfinite(float(trainable))
        or trainable < 0
        or trainable > entry["total_parameters"]
    ):
        return False
    benchmark = entry.get("benchmark")
    if not isinstance(benchmark, dict) or set(benchmark) != {"1", "32"}:
        return False
    return _valid_benchmark_result(benchmark["1"], 1, unavailable_allowed=False) and (
        _valid_benchmark_result(benchmark["32"], 32, unavailable_allowed=True)
    )


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def _valid_benchmark_result(result: Any, batch_size: int, *, unavailable_allowed: bool) -> bool:
    if not isinstance(result, dict) or result.get("batch_size") != batch_size:
        return False
    available = result.get("available")
    if not isinstance(available, bool):
        return False
    timing_fields = (
        "latency_ms_batch",
        "latency_ms_per_image",
        "throughput_images_per_second",
    )
    if available:
        return result.get("reason") is None and all(
            _finite_positive(result.get(field)) for field in timing_fields
        )
    return (
        unavailable_allowed
        and isinstance(result.get("reason"), str)
        and bool(result["reason"].strip())
        and all(result.get(field) is None for field in timing_fields)
    )


def _csv_row(entry: dict[str, Any]) -> dict[str, Any]:
    row = {key: entry.get(key) for key in CSV_FIELDS}
    row["oom_adjustments_json"] = json.dumps(entry.get("oom_adjustments", {}), sort_keys=True)
    for batch_size in BATCH_SIZES:
        result = entry["benchmark"][str(batch_size)]
        prefix = f"batch_{batch_size}_"
        row[prefix + "available"] = result.get("available")
        row[prefix + "reason"] = result.get("reason")
        row[prefix + "latency_ms_batch"] = result.get("latency_ms_batch")
        row[prefix + "latency_ms_per_image"] = result.get("latency_ms_per_image")
        row[prefix + "throughput_images_per_second"] = result.get("throughput_images_per_second")
    return row


def _atomic_write_csv(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(_csv_row(entry) for entry in entries)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv_matches(entries: list[dict[str, Any]], path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return False
    if tuple(reader.fieldnames or ()) != CSV_FIELDS or len(rows) != len(entries):
        return False
    expected = [
        {field: "" if value is None else str(value) for field, value in _csv_row(entry).items()}
        for entry in entries
    ]
    return rows == expected


def _completed_cache_entries(
    document: dict[str, Any], csv_path: Path
) -> dict[str, dict[str, Any]] | None:
    if document.get("schema_version") != SCHEMA_VERSION or document.get("status") != "completed":
        return None
    entries = document.get("models")
    if not isinstance(entries, list) or len(entries) != len(MODEL_NAMES):
        return None
    if any(not isinstance(entry, dict) for entry in entries):
        return None
    names = [entry.get("model") for entry in entries]
    if names != list(MODEL_NAMES) or len(set(names)) != len(MODEL_NAMES):
        return None
    if any(
        not _valid_cached(entry, model_name, entry.get("cache_key"))
        for model_name, entry in zip(MODEL_NAMES, entries, strict=True)
    ):
        return None
    if not _csv_matches(entries, csv_path):
        return None
    return {entry["model"]: entry for entry in entries}


def _write_outputs(
    entries: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
    settings: dict[str, Any],
    environment: dict[str, Any],
    generated_at: str,
    status: str,
) -> None:
    atomic_json_dump(
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "generated_at": generated_at,
            "environment": environment,
            "benchmark_settings": settings,
            "models": entries,
        },
        json_path,
    )
    _atomic_write_csv(entries, csv_path)


def run(
    root: Path,
    device_arg: str = "cuda",
    warmup: int = 10,
    repeats: int = 50,
    force: bool = False,
) -> tuple[Path, Path, list[str]]:
    if warmup <= 0 or repeats <= 0:
        raise ValueError("warmup and repeats must both be positive")
    root = root.resolve()
    device = _device(device_arg)
    settings = benchmark_settings(device, warmup, repeats)
    json_path = root / "reports" / "results" / "efficiency_metrics.json"
    csv_path = root / "reports" / "results" / "efficiency_metrics.csv"
    cached_by_model: dict[str, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    completed_pair_valid = False
    if json_path.is_file() and not force:
        previous = _read_json(json_path, "efficiency metrics cache")
        completed_entries = _completed_cache_entries(previous, csv_path)
        if completed_entries is not None:
            cached_by_model = completed_entries
            completed_pair_valid = True
        elif (
            previous.get("schema_version") == SCHEMA_VERSION
            and previous.get("status") == "in_progress"
        ):
            partial_entries = previous.get("models")
            if not isinstance(partial_entries, list):
                partial_entries = []
            cached_by_model = {
                item.get("model"): item
                for item in partial_entries
                if isinstance(item, dict) and isinstance(item.get("model"), str)
            }

    entries: list[dict[str, Any]] = []
    reused: list[str] = []
    changed = force or previous is None
    generated_at = now_utc()
    environment = capture_environment(root)
    for model_name in MODEL_NAMES:
        prerequisites = validate_prerequisites(root, model_name)
        expected_key = stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "model": model_name,
                "checkpoint_sha256": prerequisites["checkpoint_sha256"],
                "input_size": prerequisites["input_size"],
                "settings": settings,
            }
        )
        cached = cached_by_model.get(model_name)
        if not force and _valid_cached(cached, model_name, expected_key):
            entries.append(cached)
            reused.append(model_name)
            continue
        entry = benchmark_one(
            root, model_name, device, warmup, repeats, settings, prerequisites=prerequisites
        )
        entries.append(entry)
        changed = True
        _write_outputs(
            entries,
            json_path,
            csv_path,
            settings,
            environment,
            generated_at,
            "in_progress",
        )

    if not changed and completed_pair_valid and len(reused) == len(MODEL_NAMES):
        return json_path, csv_path, reused
    _write_outputs(
        entries,
        json_path,
        csv_path,
        settings,
        environment,
        generated_at,
        "completed",
    )
    return json_path, csv_path, reused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda; use cpu only for development checks)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up forwards per batch")
    parser.add_argument("--repeats", type=int, default=50, help="Timed forwards per batch")
    parser.add_argument("--force", action="store_true", help="Ignore valid cached measurements")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path, csv_path, reused = run(args.root, args.device, args.warmup, args.repeats, args.force)
    if reused:
        print(f"reused cached models: {', '.join(reused)}")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
