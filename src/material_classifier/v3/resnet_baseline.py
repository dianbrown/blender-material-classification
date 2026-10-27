"""Preregistered assessment-only ResNet family; never participates in v3 selection.

OOF and all-development fitting are separate explicit jobs. Only the terminal
epoch is exported/scored. Tests replace the model factory; importing is offline.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from . import CLASS_NAMES
from .features import (
    configure_torch,
    environment,
    file_hash,
    image_file,
    json_hash,
    read_json,
    read_manifest,
    source_identity,
    write_json,
)
from .learning import (
    fold_record,
    grouped_folds,
    labels_for,
    prediction_frame,
    probability_metrics,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/v3/resnet_baseline.json"


def load_config(path=CONFIG_PATH, *, physical_batch_size=None):
    """Only the preregistered memory choice is configurable; no training search."""
    config = read_json(path)
    reference = read_json(CONFIG_PATH)
    if config != reference:
        raise ValueError("Config differs from the preregistered supplemental recipe")
    if physical_batch_size is not None:
        if physical_batch_size not in config["permitted_physical_batch_sizes"]:
            raise ValueError("Physical batch size must be explicitly 16 or 8")
        config["physical_batch_size"] = physical_batch_size
        config["accumulation_steps"] = config["effective_batch_size"] // physical_batch_size
    return config


def validate_config(config, protocol=None):
    reference = load_config(physical_batch_size=config["physical_batch_size"])
    if config != reference:
        raise ValueError("Effective configuration differs from the fixed recipe")
    if protocol is not None and any(
        protocol[key] != config[key] for key in ("seed", "outer_folds", "class_order")
    ):
        raise ValueError("Supplemental split seed/folds/classes differ from primary protocol")


def preregistration(config_path=CONFIG_PATH):
    """Refuse uncommitted configuration, including runtime custom config files."""
    path = Path(config_path).resolve()
    relative = path.relative_to(ROOT).as_posix()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    commit = git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise ValueError("Commit the supplemental config before executing any training")
    committed = git("show", f"{commit}:{relative}")
    if committed != path.read_text(encoding="utf-8").strip():
        raise ValueError("Supplemental config has uncommitted changes")
    return {"config_commit": commit, "execution_commit": git("rev-parse", "HEAD")}


def implementation_identity():
    files = [
        Path(__file__),
        Path(__file__).with_name("learning.py"),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("__init__.py"),
    ]
    return {path.name: file_hash(path) for path in files}


def seed_epoch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # manual_seed also seeds CUDA when it is initialized; no model/GPU work here.


def build_transform(config, *, training):
    spec = config["preprocessing"]
    interpolation = transforms.InterpolationMode.BILINEAR
    if training:
        operations = [
            transforms.RandomResizedCrop(
                spec["input_size"],
                scale=spec["crop_scale"],
                ratio=spec["crop_ratio"],
                interpolation=interpolation,
                antialias=spec["antialias"],
            ),
            transforms.RandomHorizontalFlip(spec["horizontal_flip"]),
            transforms.ColorJitter(**spec["color_jitter"]),
        ]
    else:
        operations = [
            transforms.Resize(
                spec["eval_resize_shorter_edge"],
                interpolation=interpolation,
                antialias=spec["antialias"],
            ),
            transforms.CenterCrop(spec["input_size"]),
        ]
    return transforms.Compose(
        [*operations, transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    )


class ImageRows(Dataset):
    def __init__(self, frame, data_root, config, *, training):
        self.paths = [image_file(data_root, path) for path in frame["image_path"]]
        self.labels = labels_for(frame)
        self.transform = build_transform(config, training=training)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, int(self.labels[index])


def load_pretrained_model(config, *, offline=False):
    """Explicit enum; verify cached files too, and record the full downloaded hash."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    if weights.url != config["weights_url"]:
        raise ValueError("Installed torchvision weight URL differs from preregistration")
    path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if offline and not path.is_file():
        raise FileNotFoundError(f"Pinned pretrained weights unavailable offline: {path}")
    if path.exists() and not file_hash(path).startswith(config["weights_sha256_prefix"]):
        raise ValueError("Cached pretrained weights checksum mismatch")
    model = resnet50(weights=weights)
    digest = file_hash(path)
    if not digest.startswith(config["weights_sha256_prefix"]):
        raise ValueError("Downloaded pretrained weights checksum mismatch")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model, {"enum": config["weights"], "url": weights.url, "sha256": digest}


def set_training_phase(model, epoch, config):
    """Keep frozen layers and all BatchNorm running statistics in eval mode."""
    model.eval().requires_grad_(False)
    model.fc.train().requires_grad_(True)
    if epoch >= config["head_only_epochs"]:
        model.layer4.train().requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def make_optimizer(model, config):
    # Explicit rates for BOTH groups avoid inheriting the historical head LR bug.
    # Layer4 is registered from epoch 0; it has no gradients/updates while frozen.
    return torch.optim.AdamW(
        [
            {"params": list(model.fc.parameters()), "lr": config["head_lr"], "name": "fc"},
            {
                "params": list(model.layer4.parameters()),
                "lr": config["backbone_lr"],
                "name": "layer4",
            },
        ],
        weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
    )


def train_epoch(model, loader, optimizer, scaler, device, config):
    """Sample-weighted accumulation, including non-full microbatches/windows."""
    optimizer.zero_grad(set_to_none=True)
    total_loss, correct, seen, window_images, updates = 0.0, 0, 0, 0, 0
    skipped_updates = 0
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = nn.functional.cross_entropy(
                logits, labels, label_smoothing=config["label_smoothing"], reduction="sum"
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; fixed run aborted")
        # Predivide to keep AMP's scaled microbatch gradients in the usual range.
        # The window correction below still yields sum(grad) / actual_images.
        scaler.scale(loss / config["effective_batch_size"]).backward()
        count = len(labels)
        total_loss += float(loss.detach())
        correct += int((logits.detach().argmax(1) == labels).sum())
        seen += count
        window_images += count
        if (batch_index + 1) % config["accumulation_steps"] == 0 or batch_index + 1 == len(loader):
            scaler.unscale_(optimizer)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(config["effective_batch_size"] / window_images)
            nn.utils.clip_grad_norm_(
                parameters, config["gradient_clip_norm"], error_if_nonfinite=not scaler.is_enabled()
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)
            window_images = 0
            updates += int(not skipped)
            skipped_updates += int(skipped)
    if not seen:
        raise ValueError("Empty training loader")
    return {
        "train_loss": total_loss / seen,
        "train_accuracy": correct / seen,
        "images": seen,
        "optimizer_updates": updates,
        "amp_skipped_updates": skipped_updates,
        "amp_scale": scaler.get_scale(),
    }


def fit_fixed(model, frame, data_root, config, device, *, seed, output):
    """No validation input exists in this API; exactly ten epochs, final state only."""
    dataset = ImageRows(frame, data_root, config, training=True)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=0.0
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(config["epochs"]):
        epoch_seed = seed + epoch
        seed_epoch(epoch_seed)
        set_training_phase(model, epoch, config)
        loader = DataLoader(
            dataset,
            batch_size=config["physical_batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(epoch_seed),
            num_workers=0,
            drop_last=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        rates = {group["name"] + "_lr": group["lr"] for group in optimizer.param_groups}
        metrics = train_epoch(model, loader, optimizer, scaler, device, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "epoch": epoch + 1,
                "seed": epoch_seed,
                "phase": "head" if epoch < config["head_only_epochs"] else "layer4_and_fc",
                **rates,
                **metrics,
                "seconds": time.perf_counter() - start,
                **memory_usage(device),
            }
        )
        pd.DataFrame(history).to_csv(Path(output) / "history.csv", index=False)
        scheduler.step()
    model.eval()
    return history


def memory_usage(device):
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    }


@torch.inference_mode()
def predict_rows(model, frame, data_root, config, device, *, batch_size=None):
    model.eval()
    dataset = ImageRows(frame, data_root, config, training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or config["physical_batch_size"],
        shuffle=False,
        num_workers=0,
    )
    probabilities = []
    for images, _ in loader:
        # Float32 deployment/outer inference; AMP is a training-only choice.
        logits = model(images.to(device))
        probabilities.append(logits.double().softmax(1).cpu().numpy())
    result = np.concatenate(probabilities)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1.0):
        raise FloatingPointError("Invalid model probabilities")
    return result


def outer_plan(frame, protocol, *, assignments_path=None):
    """Recreate primary splits, then validate supplied assignments without reading scores."""
    paths = frame["image_path"].tolist()
    if paths != sorted(paths):
        raise ValueError("Use the same audited, image_path-sorted manifest as primary v3")
    y, groups = labels_for(frame), frame["source_group"].to_numpy(dtype=str)
    folds = grouped_folds(y, groups, protocol["outer_folds"], protocol["seed"])
    assignments = np.full(len(frame), -1, dtype=np.int64)
    for index, (_, test) in enumerate(folds):
        assignments[test] = index
    if assignments_path is not None:
        existing = pd.read_csv(assignments_path, dtype=str, keep_default_na=False)
        columns = ["image_path", "class", "material_id", "source_group"]
        if (
            not set(columns + ["outer_fold"]).issubset(existing.columns)
            or not existing[columns].equals(frame[columns])
            or existing["outer_fold"].tolist() != assignments.astype(str).tolist()
        ):
            raise ValueError("Primary fold assignments disagree with manifest/protocol")
    return folds, assignments


def decoded_hash(path):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode())
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def training_identity(manifest, frame, data_root):
    identity = source_identity(manifest, frame, data_root)
    identity["source_groups"] = sorted(set(frame["source_group"]))
    identity["material_ids"] = sorted(set(frame["material_id"]))
    identity["decoded_rgb_sha256"] = sorted(
        {decoded_hash(image_file(data_root, path)) for path in frame["image_path"]}
    )
    identity["row_identity_sha256"] = json_hash(frame.to_dict("records"))
    return identity


def export_model(model, path, metadata):
    """Deployment state only: no optimizer state and no resumable checkpoint claim."""
    path = Path(path)
    torch.save(
        {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, path
    )
    sidecar = {**metadata, "model_sha256": file_hash(path)}
    sidecar["metadata_sha256"] = json_hash(sidecar)
    write_json(path.with_suffix(".json"), sidecar)
    return sidecar


def load_model(path, device, *, model_factory=None):
    path = Path(path)
    metadata = read_json(path.with_suffix(".json"))
    if metadata.get("metadata_sha256") != json_hash(
        {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    ):
        raise ValueError("Deployment metadata checksum mismatch")
    if file_hash(path) != metadata["model_sha256"]:
        raise ValueError("Deployment state_dict checksum mismatch")
    if (
        metadata["class_order"] != list(CLASS_NAMES)
        or json_hash(metadata["config"]) != metadata["config_sha256"]
        or metadata["preprocessing"] != metadata["config"]["preprocessing"]
    ):
        raise ValueError("Deployment class/config/preprocessing identity mismatch")
    validate_config(metadata["config"])
    model = (
        model_factory()
        if model_factory is not None
        else resnet50(weights=None, num_classes=len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), metadata


def train_run(
    manifest,
    data_root,
    output,
    config,
    protocol,
    *,
    stage,
    device="cuda",
    cpu_threads=4,
    offline=False,
    assignments_path=None,
    config_path=CONFIG_PATH,
    model_factory=None,
):
    """Run either five matched OOF fits OR an independent all-development fit."""
    validate_config(config, protocol)
    if stage not in ("oof", "final"):
        raise ValueError("Choose stage oof or final explicitly")
    registration = preregistration(config_path)
    frame = read_manifest(manifest)
    folds, assignments = outer_plan(frame, protocol, assignments_path=assignments_path)
    source = training_identity(manifest, frame, data_root)
    device = configure_torch(device, cpu_threads)
    hardware = {"device": str(device), "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        hardware.update(
            name=torch.cuda.get_device_name(device),
            total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        **registration,
        "config": config,
        "config_sha256": json_hash(config),
        "primary_protocol_sha256": json_hash(protocol),
        "training_source": source,
        "environment": environment(),
        "hardware": hardware,
        "implementation": implementation_identity(),
        "stage": stage,
        "primary_assignments_sha256": file_hash(assignments_path) if assignments_path else None,
    }
    record = {
        "status": "running",
        "run_identity": identity,
        "run_identity_sha256": json_hash(identity),
    }
    write_json(output / "config.json", config)
    write_json(output / "run.json", record)
    write_json(
        output / "folds.json",
        fold_record(
            folds,
            np.arange(len(frame)),
            frame["source_group"].to_numpy(dtype=str),
            labels_for(frame),
        ),
    )
    assignment_frame = frame[["image_path", "class", "material_id", "source_group"]].copy()
    assignment_frame["outer_fold"] = assignments
    assignment_frame.to_csv(output / "fold_assignments.csv", index=False)
    start = time.perf_counter()
    factory = model_factory or load_pretrained_model
    jobs = (
        [
            (f"fold_{index}", train, test, config["seed"] + index * 1000)
            for index, (train, test) in enumerate(folds)
        ]
        if stage == "oof"
        else [("final", np.arange(len(frame)), None, config["seed"] + 100000)]
    )
    oof = np.full((len(frame), len(CLASS_NAMES)), np.nan)
    expected_weights = None
    try:
        for name, train, test, seed in jobs:
            folder = output / name
            folder.mkdir()
            job_start = time.perf_counter()
            seed_epoch(seed)
            model, weights = factory(config, offline=offline)
            if expected_weights is not None and weights != expected_weights:
                raise ValueError("Pretrained weight identity changed between folds")
            expected_weights = weights
            model = model.to(device)
            fit_frame = frame.iloc[train].reset_index(drop=True)
            fit_source = copy.deepcopy(source)
            fit_source.update(
                rows=len(train),
                image_paths=fit_frame["image_path"].tolist(),
                source_groups=sorted(set(fit_frame["source_group"])),
                material_ids=sorted(set(fit_frame["material_id"])),
                sources=[source["sources"][index] for index in train],
                decoded_rgb_sha256=sorted(
                    {decoded_hash(image_file(data_root, path)) for path in fit_frame["image_path"]}
                ),
                row_identity_sha256=json_hash(fit_frame.to_dict("records")),
            )
            fit_source["source_content_sha256"] = json_hash(fit_source["sources"])
            metadata = {
                "schema_version": 1,
                "architecture": config["architecture"],
                "class_order": list(CLASS_NAMES),
                "preprocessing": config["preprocessing"],
                "config": config,
                "config_sha256": json_hash(config),
                "training_source": fit_source,
                "training_stage": name,
                "initial_weights": weights,
                "run_identity_sha256": record["run_identity_sha256"],
                "primary_protocol_sha256": json_hash(protocol),
                "implementation": identity["implementation"],
                "environment": identity["environment"],
                **registration,
            }
            write_json(folder / "fit.json", {"status": "running", **metadata})
            history = fit_fixed(
                model, fit_frame, data_root, config, device, seed=seed, output=folder
            )
            saved = export_model(model, folder / "model.pt", metadata)
            evaluation_seconds = 0.0
            if test is not None:
                evaluation_start = time.perf_counter()
                probabilities = predict_rows(model, frame.iloc[test], data_root, config, device)
                evaluation_seconds = time.perf_counter() - evaluation_start
                oof[test] = probabilities
                prediction_frame(frame.iloc[test], probabilities, assignments[test]).to_csv(
                    folder / "outer_predictions.csv", index=False
                )
                write_json(
                    folder / "outer_metrics.json",
                    probability_metrics(labels_for(frame.iloc[test]), probabilities),
                )
            write_json(
                folder / "fit.json",
                {
                    "status": "complete",
                    **metadata,
                    "seconds": time.perf_counter() - job_start,
                    "outer_evaluation_seconds": evaluation_seconds,
                    "epochs_completed": len(history),
                    "history_sha256": file_hash(folder / "history.csv"),
                    "model_sha256": saved["model_sha256"],
                    "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in history),
                    "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in history),
                },
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if stage == "oof":
            if not np.isfinite(oof).all():
                raise AssertionError("OOF does not cover every manifest row")
            prediction_frame(frame, oof, assignments).to_csv(output / "resnet_oof.csv", index=False)
            write_json(output / "metrics.json", probability_metrics(labels_for(frame), oof))
        record.update(
            status="complete",
            seconds=time.perf_counter() - start,
            initial_weights=expected_weights,
            outputs={
                path.relative_to(output).as_posix(): file_hash(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "run.json"
            },
        )
        write_json(output / "run.json", record)
        return record
    except Exception as exc:
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            seconds=time.perf_counter() - start,
        )
        write_json(output / "run.json", record)
        raise


def check_fresh(frame, source, training_source):
    required = ("source_groups", "material_ids", "sources", "decoded_rgb_sha256")
    if any(not training_source.get(key) for key in required) or not source.get(
        "decoded_rgb_sha256"
    ):
        raise ValueError("Missing final training identity for fresh evaluation")
    if set(frame["source_group"]) & set(training_source["source_groups"]):
        raise ValueError("Fresh evaluation source families overlap development data")
    if set(frame["material_id"]) & set(training_source["material_ids"]):
        raise ValueError("Fresh evaluation material IDs overlap development data")
    trained_hashes = {row["sha256"] for row in training_source["sources"]}
    if any(row["sha256"] in trained_hashes for row in source["sources"]):
        raise ValueError("Fresh evaluation image content overlaps development data")
    if set(source["decoded_rgb_sha256"]) & set(training_source["decoded_rgb_sha256"]):
        raise ValueError("Fresh evaluation decoded image content overlaps development data")


def predict_manifest(
    artifact,
    manifest,
    data_root,
    output,
    *,
    evaluate=False,
    device="cpu",
    cpu_threads=4,
    batch_size=16,
    model_factory=None,
):
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    frame = read_manifest(manifest)
    source = training_identity(manifest, frame, data_root)
    device = configure_torch(device, cpu_threads)
    model, metadata = load_model(artifact, device, model_factory=model_factory)
    if evaluate:
        if metadata["training_stage"] != "final":
            raise ValueError("Fresh evaluation requires the all-development final model")
        check_fresh(frame, source, metadata["training_source"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "evaluation_requested": evaluate,
        "model_sha256": metadata["model_sha256"],
        "source": source,
        "config_sha256": metadata["config_sha256"],
        "environment": environment(),
        "fitting": "none; saved final state_dict only",
    }
    write_json(output / "prediction_run.json", record)
    start = time.perf_counter()
    try:
        probabilities = predict_rows(
            model, frame, data_root, metadata["config"], device, batch_size=batch_size
        )
        predictions = prediction_frame(
            frame, probabilities, "holdout" if evaluate else "prediction_only"
        )
        predictions.to_csv(output / "predictions.csv", index=False)
        if evaluate:
            write_json(
                output / "metrics.json", probability_metrics(labels_for(frame), probabilities)
            )
        record.update(
            status="complete",
            rows=len(frame),
            seconds=time.perf_counter() - start,
            predictions_sha256=file_hash(output / "predictions.csv"),
        )
        write_json(output / "prediction_run.json", record)
        return predictions
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "prediction_run.json", record)
        raise
