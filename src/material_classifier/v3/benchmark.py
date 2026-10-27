"""Offline, development-only efficiency measurements of already frozen artifacts.

No fitting, scoring, feature caches, downloads, AMP, or device fallback. Low-level
backends accept model objects for offline parity tests; only ``benchmark`` writes
report results, and it always uses the project's validated deployment loaders.
"""

from __future__ import annotations

import csv
import gc
import os
import platform
import random
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from threadpoolctl import threadpool_info, threadpool_limits

from . import CLASS_NAMES, features, inference
from .learning import class_probabilities

DEFAULT_SEED = 20261027
WARM_SCOPE = (
    "Predecoded, EXIF-oriented in-memory PIL RGB images -> exact family preprocessing "
    "-> tensor stacking and device transfer -> resident eval models -> saved head "
    "-> class-order probabilities in host NumPy memory. Includes feature assembly "
    "and CPU classifier for frozen recipes. Excludes disk I/O, decoding/EXIF orientation, "
    "artifact loading, hashing, result validation/serialization and thread-pool setup. "
    "Float32 network inference; family-specific probability precision preserved. "
    "Same declared image batch repeated; no feature or tensor cache."
)
COLD_SCOPE = (
    "Same-process artifact reload plus one predecoded image through probabilities; "
    "includes existing loader metadata/runtime/hash checks and model construction. "
    "The artifact was already loaded once for preflight; Python imports and library "
    "caches remain warm, and OS file caches are not flushed. Models are "
    "released and CUDA allocator cache emptied between trials. Not full CLI startup, "
    "not an OS-cold or fresh-process measurement."
)


def _validate_probabilities(values, rows):
    if (
        values.shape != (rows, len(CLASS_NAMES))
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.allclose(values.sum(axis=1), 1.0)
    ):
        raise ValueError("Invalid benchmark probabilities")


class FrozenBackend:
    """Resident models keyed by jobs_for_protocol prefix; never fits a head."""

    def __init__(self, bundle, models, device):
        self.metadata = bundle["metadata"]
        self.pipeline = bundle["pipeline"]
        self.device = torch.device(device)
        self.jobs = features.jobs_for_protocol(
            self.metadata["protocol"], self.metadata["recipe"]["parts"]
        )
        if set(models) != {prefix for prefix, _, _ in self.jobs}:
            raise ValueError("Resident models must match exactly the saved recipe jobs")
        self.models = models
        for model in self.models.values():
            model.eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, images):
        base = {}
        for prefix, spec, size in self.jobs:
            batch = torch.stack([features.transform_image(img, spec, size) for img in images])
            values = features.extract_batch(self.models[prefix], batch.to(self.device), spec, size)
            base.update({prefix + "_" + key: value for key, value in values.items()})
        matrix = features.assemble_features(base, self.metadata["recipe"]["parts"])
        if matrix.shape[1] != self.metadata["input_features"]:
            raise ValueError("Extracted feature width differs from final head")
        return class_probabilities(self.pipeline, matrix)


class NeuralBackend:
    """Use exactly the public ResNet/DINO deployment transforms and softmax order."""

    def __init__(self, family, model, metadata, device):
        if family not in ("resnet", "dinov2"):
            raise ValueError("Neural family must be resnet or dinov2")
        self.family, self.metadata, self.device = family, metadata, torch.device(device)
        self.model = model.eval().requires_grad_(False)
        if family == "resnet":
            from .resnet_baseline import build_transform

            self.transform = build_transform(metadata["config"], training=False)
        else:
            self.transform = lambda img: features.transform_image(
                img, metadata["architecture"], metadata["config"]["input_size"]
            )

    @torch.inference_mode()
    def predict(self, images):
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        logits = self.model(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite prediction logits")
        if self.family == "resnet":
            return logits.double().softmax(1).cpu().numpy()
        values = logits.softmax(1).double().cpu().numpy()
        return values / values.sum(axis=1, keepdims=True)


def load_backend(family, artifact, device):
    """Production path: no injectable factories or network-enabled weight loading."""
    if family == "frozen":
        bundle = inference.load_artifact(artifact)
        inference._check_feature_runtime(bundle["metadata"])
        models = {
            prefix: features.load_frozen_model(spec, size, device=device, offline=True)
            for prefix, spec, size in features.jobs_for_protocol(
                bundle["metadata"]["protocol"], bundle["metadata"]["recipe"]["parts"]
            )
        }
        return FrozenBackend(bundle, models, device)
    if family == "resnet":
        from . import resnet_baseline as resnet

        model, metadata = resnet.load_model(artifact, device)
        if metadata["training_stage"] != "final":
            raise ValueError("ResNet benchmark requires an all-development final artifact")
        if metadata["implementation"] != resnet.implementation_identity():
            raise ValueError("ResNet deployment implementation changed")
        current = features.environment()["packages"]
        if any(
            current[name] != metadata["environment"]["packages"][name]
            for name in ("torch", "torchvision", "numpy", "Pillow")
        ):
            raise ValueError("ResNet deployment runtime differs from exported environment")
    elif family == "dinov2":
        from . import finetune

        model, metadata = finetune.load_deployed(artifact, device)
        if metadata.get("kind") != "exploratory-partial-dinov2":
            raise ValueError("DINO benchmark requires a final exported partial-DINO artifact")
        # Require provenance even though older inference artifacts may omit it.
        if metadata.get("source_hashes") != finetune.source_hashes():
            raise ValueError("DINO deployment source provenance missing or changed")
    else:
        raise ValueError("Choose family frozen, resnet or dinov2 explicitly")
    return NeuralBackend(family, model, metadata, device)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def statistics(seconds, batch_size):
    """Latency is per batch, in seconds; throughput is total images / total time."""
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Timing samples must be finite positive seconds")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return {
        "repetitions": len(values),
        "batch_size": batch_size,
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95, method="linear")),
        "mean_seconds": float(values.mean()),
        "std_seconds": float(values.std(ddof=0)),
        "images_per_second": float(batch_size / values.mean()),
    }


def _memory(device):
    if torch.device(device).type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def measure_warm(backend, images, *, warmup=5, repetitions=30):
    """Low-level unverified backend measurement, also usable with CPU test doubles."""
    if warmup < 1 or repetitions < 1 or not images:
        raise ValueError("Warmup, repetitions and image batch must be positive")
    for _ in range(warmup):
        _validate_probabilities(backend.predict(images), len(images))
    synchronize(backend.device)
    baseline = {"allocated_bytes": None, "reserved_bytes": None}
    if backend.device.type == "cuda":
        baseline = {
            "allocated_bytes": torch.cuda.memory_allocated(backend.device),
            "reserved_bytes": torch.cuda.memory_reserved(backend.device),
        }
        torch.cuda.reset_peak_memory_stats(backend.device)
    seconds = []
    for _ in range(repetitions):
        synchronize(backend.device)
        start = time.perf_counter()
        values = backend.predict(images)
        synchronize(backend.device)
        seconds.append(time.perf_counter() - start)
        _validate_probabilities(values, len(images))
    return {
        "measurement_kind": "backend_only_unverified",
        "scope": "warm",
        **statistics(seconds, len(images)),
        "warmup": warmup,
        "samples_seconds": seconds,
        "memory_before_trials": baseline,
        **_memory(backend.device),
    }


def _file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": features.file_hash(path)}


def _weight_record(spec):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        spec["repository"], spec["weights_file"], revision=spec["revision"], local_files_only=True
    )
    record = _file_record(path)
    if record["sha256"] != spec["weights_sha256"]:
        raise ValueError("Pinned pretrained weights checksum mismatch")
    return {
        **record,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "weights_file": spec["weights_file"],
    }


def deployment_inventory(family, artifact, backend):
    """Physical bytes are measured, unique dependencies deduplicated by SHA-256.

    Runtime counts use unique Parameter objects. Frozen source parameter counts
    use original checkpoint shapes for named parameters used by resident models;
    position embeddings can differ after resolution adaptation. Discarded source
    classifiers/buffers are included in file bytes, not backbone parameter counts.
    """
    artifact = Path(artifact)
    sidecar = (
        artifact.with_name(artifact.stem + "_metadata.json")
        if family == "frozen"
        else artifact.with_suffix(".json")
    )
    files = {"artifact": _file_record(artifact), "sidecar": _file_record(sidecar)}
    dependencies, instances, unique_source_parameters = {}, [], {}
    if family == "frozen":
        from safetensors import safe_open

        all_parameters = {}
        for prefix, spec, size in backend.jobs:
            record = _weight_record(spec)
            digest = record["sha256"]
            dependencies.setdefault(digest, {**record, "runtime_instances": []})
            dependencies[digest]["runtime_instances"].append(prefix)
            model = backend.models[prefix]
            parameters = dict(model.named_parameters())
            with safe_open(record["path"], framework="pt", device="cpu") as weights:
                for name in parameters:
                    if name not in weights.keys():
                        raise ValueError(f"Cannot map runtime parameter to pinned weights: {name}")
                    unique_source_parameters[(digest, name)] = int(
                        np.prod(weights.get_slice(name).get_shape(), dtype=np.int64)
                    )
            instances.append(
                {
                    "name": prefix,
                    "model": spec["name"],
                    "resolution": size,
                    "parameters": sum(p.numel() for p in parameters.values()),
                    "weights_sha256": digest,
                }
            )
            all_parameters.update({id(p): p.numel() for p in parameters.values()})
        classifier = backend.pipeline.named_steps["classifier"]
        head_count = int(classifier.coef_.size + classifier.intercept_.size)
        scaler = backend.pipeline.named_steps["scaler"]
        scaler_elements = sum(
            np.asarray(getattr(scaler, key)).size
            for key in ("mean_", "scale_", "var_")
            if getattr(scaler, key, None) is not None
        )
        runtime_total = sum(all_parameters.values()) + head_count
        unique_total = sum(unique_source_parameters.values()) + head_count
        dependency_note = (
            "Pinned cached pretrained safetensors required; joblib is only scaler/head."
        )
    else:
        model = backend.model
        head = model.fc if family == "resnet" else model.head
        head_ids = {id(p) for p in head.parameters()}
        head_count = sum(p.numel() for p in head.parameters())
        backbone_count = sum(p.numel() for p in model.parameters() if id(p) not in head_ids)
        runtime_total = sum(p.numel() for p in model.parameters())
        unique_total, scaler_elements = runtime_total, 0
        instances.append({"name": family, "parameters": backbone_count})
        dependency_note = (
            "Exported .pt contains network state. Current deployment loader constructs with "
            "pretrained=False (DINO) or weights=None (ResNet); no original pretrained file "
            "dependency. See recorded loader source hashes; training initial weights are "
            "not counted as deployment dependencies."
        )
    pretrained_bytes = sum(record["bytes"] for record in dependencies.values())
    return {
        **files,
        "artifact_role": "classifier_and_scaler_only" if family == "frozen" else "network_state",
        "pretrained_weight_files": list(dependencies.values()),
        "unique_pretrained_weight_bytes": pretrained_bytes,
        "total_deployment_bytes": files["artifact"]["bytes"]
        + files["sidecar"]["bytes"]
        + pretrained_bytes,
        "dependency_note": dependency_note,
        "backbone_instances": instances,
        "runtime_backbone_instances": len(instances),
        "head_parameters": head_count,
        "scaler_state_elements_excluded_from_parameters": scaler_elements,
        "runtime_unique_parameters": runtime_total,
        "unique_source_parameters_including_head": unique_total,
        "parameter_scope": (
            "Runtime: unique resident Parameter objects plus fitted head coef/intercept. "
            "Unique source: each (weight-file SHA256, used parameter name) once using original "
            "checkpoint shapes, plus fitted head; neural: exported model parameters. "
            "Excludes buffers/scaler state and discarded pretrained classifier parameters. "
            "File bytes include complete files and are not inferred from parameter counts."
        ),
    }


def development_images(manifest, data_root, metadata, count, seed):
    """Require the exact training development corpus, including all content hashes.

    Historical split labels may include 'test'; membership is established from
    artifact provenance, never guessed from filenames or original_split labels.
    """
    frame = features.read_manifest(manifest)
    source = features.source_identity(manifest, frame, data_root)
    training = metadata["training_source"]
    if any(training.get(key) != value for key, value in source.items()):
        raise ValueError("Benchmark requires the exact development manifest/content, never holdout")
    if len(frame) < count:
        raise ValueError("Development corpus is smaller than the largest requested batch")
    indices = np.random.default_rng(seed).permutation(len(frame))[:count]
    selected, images = [], []
    for index in indices:
        entry = source["sources"][int(index)]
        with Image.open(features.image_file(data_root, entry["image_path"])) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.load()
        images.append(rgb)
        selected.append({**entry, "manifest_row": int(index), "rgb_size": list(rgb.size)})
    return images, {"source": source, "selected": selected}


def _provenance():
    root = Path(__file__).resolve().parents[3]
    paths = [
        Path(__file__).with_name(name)
        for name in (
            "benchmark.py",
            "features.py",
            "inference.py",
            "learning.py",
            "resnet_baseline.py",
            "finetune.py",
            "__init__.py",
        )
    ]
    paths += [
        root / "scripts/v3/benchmark.py",
        root / "pyproject.toml",
        root / "requirements-v3-lock.txt",
    ]
    paths += sorted((root / "configs/v3").glob("*.json"))
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "files": {path.relative_to(root).as_posix(): features.file_hash(path) for path in paths},
    }


def _hardware(device):
    record = {
        "hostname": socket.gethostname(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_pools": threadpool_info(),
        "execution": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "HF_HOME",
                    "HF_HUB_CACHE",
                    "TORCH_HOME",
                )
            },
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record["cuda"] = {
            "index": device.index if device.index is not None else torch.cuda.current_device(),
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
        }
    return record


def _release(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _measure_reload(family, artifact, device, image, repetitions):
    seconds, peaks = [], []
    for _ in range(repetitions):
        _release(device)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        backend = load_backend(family, artifact, device)
        probabilities = backend.predict([image])
        synchronize(device)
        seconds.append(time.perf_counter() - start)
        _validate_probabilities(probabilities, 1)
        peaks.append(_memory(device))
        del backend
    _release(device)
    return {
        "measurement_kind": "validated_artifact",
        "scope": "process_warm_reload",
        **statistics(seconds, 1),
        "warmup": 0,
        "samples_seconds": seconds,
        **{
            key: max(p[key] for p in peaks) if device.type == "cuda" else None
            for key in ("peak_allocated_bytes", "peak_reserved_bytes")
        },
    }


def benchmark(
    family,
    artifact,
    development_manifest,
    data_root,
    output,
    *,
    device,
    cpu_threads=4,
    batches=(1, 16),
    warmup=5,
    repetitions=30,
    seed=DEFAULT_SEED,
    cold_repetitions=0,
):
    """Write benchmark.json and benchmark.csv in a NEW directory, or fail explicitly."""
    batches = tuple(batches)
    if family not in ("frozen", "resnet", "dinov2"):
        raise ValueError("Choose family frozen, resnet or dinov2 explicitly")
    if (
        not batches
        or any(n < 1 for n in batches)
        or len(set(batches)) != len(batches)
        or warmup < 1
        or repetitions < 1
        or cpu_threads < 1
        or cold_repetitions not in (0, 3)
        or not 0 <= seed < 2**32
    ):
        raise ValueError(
            "Invalid batches/warmup/repetitions/threads/seed; cold repetitions is 0 or 3"
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError("Benchmark output must be a new directory")
    device = features.configure_torch(device, cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "status": "running",
        "family": family,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Efficiency only; no fitting, accuracy evaluation or model selection",
        "class_order": list(CLASS_NAMES),
        "environment": features.environment(),
        "provenance": _provenance(),
        "settings": {
            "device": str(device),
            "cpu_threads_requested": cpu_threads,
            "batches": list(batches),
            "warmup": warmup,
            "repetitions": repetitions,
            "seed": seed,
            "cold_repetitions": cold_repetitions,
        },
        "scopes": {"warm": WARM_SCOPE, "process_warm_reload": COLD_SCOPE},
        "statistics": "Per-batch seconds; p95 linear interpolation; std population ddof=0; "
        "images/sec = total images / total timed seconds; no FLOPs measured.",
        "memory_scope": "CUDA process allocator peaks, including resident model weights; "
        "reserved memory can retain earlier batch allocations. Not total board usage. "
        "CPU VRAM fields are null; no CPU RSS claim.",
        "results": [],
    }
    features.write_json(output / "benchmark.json", record)
    try:
        with threadpool_limits(limits=torch.get_num_threads()):
            backend = load_backend(family, artifact, device)
            metadata = backend.metadata
            images, selection = development_images(
                development_manifest, data_root, metadata, max(batches), seed
            )
            record["development"] = {
                "manifest": str(Path(development_manifest).resolve()),
                "data_root": str(Path(data_root).resolve()),
                "manifest_sha256": selection["source"]["manifest_sha256"],
                "source_content_sha256": selection["source"]["source_content_sha256"],
                "rows": selection["source"]["rows"],
                "selected": selection["selected"],
                "selection_rule": "NumPy default_rng(seed).permutation(manifest rows), first "
                "max(batches); each batch uses the first batch_size images every trial.",
            }
            record["deployment"] = deployment_inventory(family, artifact, backend)
            record["artifact_metadata"] = {
                key: metadata[key]
                for key in (
                    "recipe",
                    "protocol",
                    "protocol_sha256",
                    "config",
                    "config_sha256",
                    "architecture",
                    "input_features",
                    "training_stage",
                    "kind",
                )
                if key in metadata
            }
            record["hardware"] = _hardware(device)
            # Persist the input declaration before collecting any timings.
            features.write_json(output / "benchmark.json", record)
            if cold_repetitions:
                del backend
                _release(device)
                record["results"].append(
                    _measure_reload(family, artifact, device, images[0], cold_repetitions)
                )
                backend = load_backend(family, artifact, device)
            for batch_size in batches:
                result = measure_warm(
                    backend, images[:batch_size], warmup=warmup, repetitions=repetitions
                )
                result["measurement_kind"] = "validated_artifact"
                record["results"].append(result)
            del backend
            _release(device)
        # Detect changes to the supplied corpus, classifier or dependencies during the run.
        frame = features.read_manifest(development_manifest)
        if features.source_identity(development_manifest, frame, data_root) != selection["source"]:
            raise ValueError("Development content changed during benchmark")
        inventory = record["deployment"]
        for item in [
            inventory["artifact"],
            inventory["sidecar"],
            *inventory["pretrained_weight_files"],
        ]:
            if _file_record(item["path"]) != {k: item[k] for k in ("path", "bytes", "sha256")}:
                raise ValueError("Deployment file changed during benchmark")
        if _provenance() != record["provenance"]:
            raise ValueError("Benchmark source/config provenance changed during run")
        rows = [
            {
                "family": family,
                "artifact_sha256": inventory["artifact"]["sha256"],
                **{
                    key: result[key]
                    for key in (
                        "scope",
                        "batch_size",
                        "warmup",
                        "repetitions",
                        "median_seconds",
                        "p95_seconds",
                        "mean_seconds",
                        "std_seconds",
                        "images_per_second",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                    )
                },
                "artifact_bytes": inventory["artifact"]["bytes"],
                "sidecar_bytes": inventory["sidecar"]["bytes"],
                **{
                    key: inventory[key]
                    for key in (
                        "unique_pretrained_weight_bytes",
                        "total_deployment_bytes",
                        "head_parameters",
                        "runtime_unique_parameters",
                        "unique_source_parameters_including_head",
                        "runtime_backbone_instances",
                    )
                },
            }
            for result in record["results"]
        ]
        with (output / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        record.update(status="complete", csv_sha256=features.file_hash(output / "benchmark.csv"))
        features.write_json(output / "benchmark.json", record)
        return record
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        features.write_json(output / "benchmark.json", record)
        raise
