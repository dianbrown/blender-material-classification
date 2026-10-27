"""Bounded exploratory DINO development selection, separate from frozen nested CV.

No outer predictions enter this module. The saved final-selector fold map is
mandatory. All weights are local/hash verified; inference needs no pretrained
download. Resume files are epoch transactions, separate from deployment weights.
"""

from __future__ import annotations

import gc
import hashlib
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import CLASS_NAMES
from .features import (
    configure_torch,
    directory_lock,
    environment,
    file_hash,
    image_file,
    json_hash,
    load_frozen_model,
    pool_dino_tokens,
    read_json,
    read_manifest,
    source_identity,
    transform_image,
    write_json,
)
from .learning import fold_record, grouped_folds, labels_for, prediction_frame, probability_metrics


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = [
        Path(__file__),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("learning.py"),
        Path(__file__).with_name("__init__.py"),
        root / "scripts/v3/finetune.py",
        root / "scripts/v3/predict_finetuned.py",
    ]
    return {
        path.relative_to(root).as_posix(): json_hash(path.read_text(encoding="utf-8"))
        for path in paths
    }


def validate_config(config, protocol):
    """A bounded follow-up is not a general hyperparameter search entry point."""
    fixed = {
        "schema_version": 1,
        "input_size": 336,
        "dropout": 0.1,
        "unfreeze_blocks": 2,
        "head_only_epochs": 2,
        "max_epochs": 10,
        "patience": 3,
        "backbone_lrs": [3e-6, 1e-5],
        "head_lr": 3e-4,
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
        "clip_grad_norm": 1.0,
        "batch_size": 8,
        "accumulation_steps": 4,
        "folds": 3,
        "fold_seed_offset": 1000,
        "frozen_selector_mean_macro_f1": 0.8035245993,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"Bounded protocol violation: {key} must equal {value}")
    spec = protocol["models"]["dino"]
    if (
        config["model"] != "vit_small_patch14_reg4_dinov2.lvd142m"
        or spec["name"] != config["model"]
        or spec["num_prefix_tokens"] != 5
        or spec["width"] != 384
        or spec["patch_size"] != 14
        or spec["weights_sha256"] != config["weights_sha256"]
        or config["weights_sha256"]
        != "dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d"
    ):
        raise ValueError("Backbone must use the frozen protocol's pinned DINOv2 small reg4")
    aug = config["augmentation"]
    if (
        aug["rrc_scale"] != [0.88, 1.0]
        or aug["rrc_ratio"] != [0.95, 1.05]
        or aug["horizontal_flip"] != 0.5
        or aug["color_jitter"] != [0.1, 0.1, 0.05, 0.02]
    ):
        raise ValueError("Augmentation differs from the finite protocol")


def verify_folds(frame, saved, protocol):
    if frame["image_path"].tolist() != sorted(frame["image_path"]):
        raise ValueError("Manifest must retain the frozen sorted image order; do not reorder rows")
    y, groups = labels_for(frame), frame["source_group"].to_numpy(dtype=str)
    folds = grouped_folds(y, groups, 3, protocol["seed"] + 1000)
    expected = fold_record(folds, np.arange(len(frame)), groups, y)
    if saved != expected:
        raise ValueError("Saved final_folds.json differs from exact frozen all-development folds")
    return folds


class PartialDino(nn.Module):
    def __init__(self, backbone, spec, config):
        super().__init__()
        if backbone.num_prefix_tokens != 5 or len(backbone.blocks) < 2:
            raise ValueError("Expected a reg4 transformer with five prefix tokens")
        self.backbone, self.spec, self.config = backbone, spec, config
        self.dropout = nn.Dropout(config["dropout"])
        self.head = nn.Linear(2 * spec["width"], len(CLASS_NAMES))
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.backbone.requires_grad_(False)
        if epoch > self.config["head_only_epochs"]:
            self.backbone.blocks[-2:].requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # Frozen modules stay in eval, including during the head-only warmup.
        self.backbone.eval()
        if mode and any(p.requires_grad for p in self.backbone.norm.parameters()):
            self.backbone.blocks[-2:].train()
            self.backbone.norm.train()
        return self

    def forward(self, batch):
        tokens = self.backbone.forward_features(batch)
        pooled = pool_dino_tokens(
            tokens,
            5,
            (self.config["input_size"] // self.spec["patch_size"]) ** 2,
            self.spec["width"],
        )
        return self.head(self.dropout(torch.cat([pooled["cls"], pooled["mean"]], dim=1)))


def make_optimizer(model, config, backbone_lr):
    # Include the eventual trainable parameters from the outset so warmup does
    # not require rebuilding the optimizer or losing head Adam moments.
    model.set_epoch(config["head_only_epochs"] + 1)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = config["head_lr"] if name.startswith("head.") else backbone_lr
        decay = 0.0 if param.ndim <= 1 or name.endswith(".bias") else config["weight_decay"]
        groups.setdefault((lr, decay), []).append(param)
    model.set_epoch(1)
    return torch.optim.AdamW(
        [
            {"params": params, "lr": lr, "weight_decay": decay}
            for (lr, decay), params in groups.items()
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transform(config, spec):
    from torchvision import transforms as T

    aug = config["augmentation"]
    return T.Compose(
        [
            T.RandomResizedCrop(
                config["input_size"],
                scale=tuple(aug["rrc_scale"]),
                ratio=tuple(aug["rrc_ratio"]),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.RandomHorizontalFlip(aug["horizontal_flip"]),
            T.ColorJitter(*aug["color_jitter"]),
            T.ToTensor(),
            T.Normalize(spec["mean"], spec["std"]),
        ]
    )


class Images(Dataset):
    def __init__(self, frame, data_root, spec, config, training=False):
        self.frame, self.root, self.spec, self.config = frame, data_root, spec, config
        self.labels = labels_for(frame)
        self.transform = train_transform(config, spec) if training else None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(image_file(self.root, self.frame.iloc[index]["image_path"])) as image:
            tensor = (
                self.transform(ImageOps.exif_transpose(image).convert("RGB"))
                if self.transform
                else transform_image(image, self.spec, self.config["input_size"])
            )
        return tensor, int(self.labels[index])


def epoch_loader(dataset, config, seed, training):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=training,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )


def metrics_from_predictions(loss, y, pred):
    return {
        "loss": loss / len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, average="macro", labels=np.arange(len(CLASS_NAMES)), zero_division=0)
        ),
    }


def run_epoch(model, loader, device, config, optimizer=None, scaler=None):
    """Sum microbatch losses / actual window samples, including the final partial window."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"], reduction="sum")
    total_loss, labels, predictions = 0.0, [], []
    n = len(loader.dataset)
    effective = config["batch_size"] * config["accumulation_steps"]
    seen, steps, skipped_steps = 0, 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for index, (batch, target) in enumerate(loader):
        batch, target = batch.to(device), target.to(device)
        window_start = (seen // effective) * effective
        denominator = min(effective, n - window_start)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ),
        ):
            logits = model(batch)
            loss = criterion(logits, target)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite loss/logits; experiment failed without adaptation")
        if training:
            scaler.scale(loss / denominator).backward()
        seen += len(target)
        if training and ((index + 1) % config["accumulation_steps"] == 0 or seen == n):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["clip_grad_norm"],
                error_if_nonfinite=not scaler.is_enabled(),
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)
            steps += int(not skipped)
            skipped_steps += int(skipped)
        total_loss += float(loss.detach())
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    return {
        **metrics_from_predictions(total_loss, labels, predictions),
        "optimizer_steps": steps,
        "attempted_optimizer_steps": steps + skipped_steps,
        "amp_skipped_steps": skipped_steps,
        "amp_scale": scaler.get_scale() if training else None,
    }


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checked_load(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def cpu_state(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def fit_one(
    frame,
    train,
    valid,
    data_root,
    spec,
    config,
    lr,
    seed,
    directory,
    identity,
    device,
    *,
    epochs=None,
    model_factory=None,
):
    """Epoch commits replay deterministically; completed fits never train again.

    The progress pointer commits an immutable epoch checkpoint and its hash. A
    process death before pointer replacement leaves an uncommitted epoch to replay.
    Every attempt (including inferred hard interruptions) remains in progress.json.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    fit_identity = json_hash(
        {
            "run": identity,
            "lr": lr,
            "seed": seed,
            "train": train.tolist(),
            "valid": valid.tolist(),
            "epochs": epochs,
        }
    )
    progress = (
        read_json(progress_path)
        if progress_path.exists()
        else {"identity_sha256": fit_identity, "status": "pending", "attempts": []}
    )
    if progress["identity_sha256"] != fit_identity:
        raise ValueError("Fit resume identity mismatch")
    if progress["status"] == "complete":
        checked_load(directory / progress["checkpoint"], progress["checkpoint_sha256"])
        return progress
    if progress["status"] == "running":
        progress["attempts"][-1].update(status="interrupted", error="Process stopped before commit")
    progress["attempts"].append({"status": "running", "started_unix": time.time()})
    progress["status"] = "running"
    write_json(progress_path, progress)
    model = optimizer = scaler = None
    try:
        seed_all(seed)
        factory = model_factory or load_frozen_model
        model = PartialDino(
            factory(spec, config["input_size"], device=device, offline=True), spec, config
        ).to(device)
        optimizer = make_optimizer(model, config, lr)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        history, best_state, best_score, best_epoch, stale = [], None, -1.0, 0, 0
        if "checkpoint" in progress:
            saved = checked_load(directory / progress["checkpoint"], progress["checkpoint_sha256"])
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            history, best_state = saved["history"], saved["best_state"]
            best_score, best_epoch, stale = saved["best_score"], saved["best_epoch"], saved["stale"]
            del saved
        train_data = Images(frame.iloc[train], data_root, spec, config, training=True)
        val_data = Images(frame.iloc[valid], data_root, spec, config) if len(valid) else None
        limit = epochs if epochs is not None else config["max_epochs"]
        for epoch in range(len(history) + 1, limit + 1):
            if val_data is not None and stale >= config["patience"]:
                break
            seed_all(seed + epoch)
            model.set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            training = run_epoch(
                model,
                epoch_loader(train_data, config, seed + epoch, True),
                device,
                config,
                optimizer,
                scaler,
            )
            validation = (
                run_epoch(
                    model, epoch_loader(val_data, config, seed + epoch, False), device, config
                )
                if val_data is not None
                else None
            )
            if validation is not None:
                if validation["macro_f1"] > best_score:
                    best_score, best_epoch, stale = validation["macro_f1"], epoch, 0
                    best_state = cpu_state(model)
                else:
                    stale += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            history.append(
                {
                    "epoch": epoch,
                    "head_only": epoch <= config["head_only_epochs"],
                    "train": training,
                    "validation_development_only": validation,
                    "seconds": time.perf_counter() - start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0,
                }
            )
            checkpoint = directory / f"epoch_{epoch:02d}.pt"
            atomic_torch_save(
                checkpoint,
                {
                    "model": cpu_state(model),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "best_state": best_state,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "stale": stale,
                },
            )
            previous = progress.get("checkpoint")
            progress.update(
                checkpoint=checkpoint.name,
                checkpoint_sha256=file_hash(checkpoint),
                history=history,
                best_score=best_score,
                best_epoch=best_epoch,
            )
            write_json(progress_path, progress)
            # Only the previous committed epoch is removed, after the new pointer commits.
            if previous and previous != checkpoint.name:
                (directory / previous).unlink(missing_ok=True)
        progress["status"] = "complete"
        progress["attempts"][-1].update(status="complete", ended_unix=time.time())
        write_json(progress_path, progress)
        return progress
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        progress["status"] = status
        progress["attempts"][-1].update(
            status=status, ended_unix=time.time(), error=f"{type(exc).__name__}: {exc}"
        )
        write_json(progress_path, progress)
        raise
    finally:
        del model, optimizer, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def choose_selection(results, config):
    if len(results) != 6 or {(r["lr"], r["fold"]) for r in results} != {
        (lr, fold) for lr in config["backbone_lrs"] for fold in range(3)
    }:
        raise ValueError("Selection requires all six predeclared completed fits")
    scores = []
    for lr in config["backbone_lrs"]:
        rows = [row for row in results if row["lr"] == lr]
        if any(
            row["status"] != "complete"
            or not np.isfinite(row["best_score"])
            or not 1 <= row["best_epoch"] <= config["max_epochs"]
            for row in rows
        ):
            raise ValueError("Failed or invalid fits cannot enter selection")
        scores.append(
            {
                "lr": lr,
                "mean_best_validation_macro_f1": float(
                    np.mean([row["best_score"] for row in rows])
                ),
                "best_epochs": [row["best_epoch"] for row in rows],
            }
        )
    chosen = min(scores, key=lambda row: (-row["mean_best_validation_macro_f1"], row["lr"]))
    return {
        "scores": scores,
        "backbone_lr": chosen["lr"],
        "final_epochs": max(2, int(np.floor(np.median(chosen["best_epochs"]) + 0.5))),
        "mean_best_validation_macro_f1": chosen["mean_best_validation_macro_f1"],
        "frozen_selector_mean_macro_f1": config["frozen_selector_mean_macro_f1"],
        "difference_from_frozen_selector": chosen["mean_best_validation_macro_f1"]
        - config["frozen_selector_mean_macro_f1"],
        "candidate_choice": "pending explicit parent decision; scores are development only",
    }


def decoded_hash(path):
    """Hash EXIF-normalized RGB pixels and dimensions, independent of file encoding."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode())
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def neural_source_identity(manifest, frame, data_root):
    return {
        **source_identity(manifest, frame, data_root),
        "source_groups": sorted(set(frame["source_group"])),
        "material_ids": sorted(set(frame["material_id"])),
        "decoded_rgb_sha256": sorted(
            {decoded_hash(image_file(data_root, path)) for path in frame["image_path"]}
        ),
    }


def training_identity(manifest, data_root, frozen_run, protocol, config, device):
    validate_config(config, protocol)
    frame = read_manifest(manifest)
    frozen_run = Path(frozen_run)
    saved = read_json(frozen_run / "final_folds.json")
    folds = verify_folds(frame, saved, protocol)
    source = source_identity(manifest, frame, data_root)
    metadata = read_json(frozen_run / "model_metadata.json")
    reference = metadata["training_source"]
    if any(source[key] != reference[key] for key in source):
        raise ValueError("Development manifest/content differs from frozen final-selector input")
    if (
        metadata["protocol_sha256"] != json_hash(protocol)
        or metadata["class_order"] != list(CLASS_NAMES)
        or abs(
            metadata["final_selection"]["inner_mean_macro_f1"]
            - config["frozen_selector_mean_macro_f1"]
        )
        > 1e-9
    ):
        raise ValueError("Frozen baseline selector/protocol differs from the declared reference")
    frozen_hash = file_hash(frozen_run / "model.joblib")
    if frozen_hash != metadata["model_sha256"]:
        raise ValueError("Frozen artifact checksum mismatch")
    identity = {
        "config": config,
        "config_sha256": json_hash(config),
        "protocol": protocol,
        "protocol_sha256": json_hash(protocol),
        "source_hashes": source_hashes(),
        "source": neural_source_identity(manifest, frame, data_root),
        "folds_sha256": file_hash(frozen_run / "final_folds.json"),
        "frozen_model_sha256": frozen_hash,
        "frozen_metadata_sha256": file_hash(frozen_run / "model_metadata.json"),
        "environment": environment(),
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "threads": torch.get_num_threads(),
    }
    return frame, folds, identity


def run_finetune(
    manifest,
    data_root,
    frozen_run,
    output,
    protocol,
    config,
    *,
    stage="select",
    resume=False,
    selection_sha256=None,
    device="cuda",
    cpu_threads=4,
):
    if stage not in ("select", "final"):
        raise ValueError("Stage must be select or final")
    device = configure_torch(device, cpu_threads)
    frame, folds, identity = training_identity(
        manifest, data_root, frozen_run, protocol, config, device
    )
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Output exists; --resume requires an identical run identity")
    output.mkdir(parents=True, exist_ok=True)
    with directory_lock(output):
        record_path = output / "run.json"
        if record_path.exists():
            record = read_json(record_path)
            if record["identity"] != identity or record["identity_sha256"] != json_hash(identity):
                raise ValueError("Run resume identity mismatch; no settings may change")
        else:
            if stage != "select" or any(p.name != ".writer.lock" for p in output.iterdir()):
                raise ValueError("Start selection in an empty output directory")
            record = {
                "schema_version": 1,
                "identity": identity,
                "identity_sha256": json_hash(identity),
                "status": "created",
                "interpretation": config["interpretation"],
            }
            write_json(record_path, record)
            write_json(
                output / "final_folds.json", read_json(Path(frozen_run) / "final_folds.json")
            )
        spec = protocol["models"]["dino"]
        try:
            if stage == "select":
                if (output / "selection.json").exists() and "selection_sha256" in record:
                    selection = read_json(output / "selection.json")
                    if file_hash(output / "selection.json") != record["selection_sha256"]:
                        raise ValueError("Selection checksum mismatch")
                    return {**selection, "selection_sha256": record["selection_sha256"]}
                record["status"] = "selecting"
                write_json(record_path, record)
                results = []
                for lr in config["backbone_lrs"]:
                    for fold, (train, valid) in enumerate(folds):
                        progress = fit_one(
                            frame,
                            train,
                            valid,
                            data_root,
                            spec,
                            config,
                            lr,
                            protocol["seed"] + fold * 10000,
                            output / "resume" / f"lr_{lr:g}_fold_{fold}",
                            record["identity_sha256"],
                            device,
                        )
                        results.append(
                            {
                                "lr": lr,
                                "fold": fold,
                                "status": progress["status"],
                                "best_score": progress["best_score"],
                                "best_epoch": progress["best_epoch"],
                                "checkpoint_sha256": progress["checkpoint_sha256"],
                            }
                        )
                selection = {
                    **choose_selection(results, config),
                    "fits": results,
                    "run_identity_sha256": record["identity_sha256"],
                    "interpretation": config["interpretation"],
                }
                if neural_source_identity(manifest, frame, data_root) != identity["source"]:
                    raise ValueError("Development source changed during selection")
                write_json(output / "selection.json", selection)
                record.update(
                    status="selection_frozen", selection_sha256=file_hash(output / "selection.json")
                )
                write_json(record_path, record)
                return {**selection, "selection_sha256": record["selection_sha256"]}
            if not selection_sha256 or file_hash(output / "selection.json") != selection_sha256:
                raise ValueError("Final fit requires the exact --selection-sha256 from selection")
            selection = read_json(output / "selection.json")
            if (
                selection_sha256 != record["selection_sha256"]
                or selection["run_identity_sha256"] != record["identity_sha256"]
            ):
                raise ValueError("Final fit selection/run identity mismatch")
            derived = choose_selection(selection["fits"], config)
            if any(selection[key] != value for key, value in derived.items()):
                raise ValueError("Saved selection differs from the bounded selection rule")
            if record["status"] == "complete":
                for exported in record["exports"].values():
                    if file_hash(output / exported["path"]) != exported["sha256"]:
                        raise ValueError("Completed export checksum mismatch")
                if (
                    file_hash(output / "candidates/finetuned/model.json")
                    != (record["exports"]["finetuned"]["metadata_sha256"])
                ):
                    raise ValueError("Completed export metadata checksum mismatch")
                if (
                    file_hash(output / "candidates/frozen/model_metadata.json")
                    != (identity["frozen_metadata_sha256"])
                ):
                    raise ValueError("Completed frozen metadata checksum mismatch")
                return record["exports"]
            record["status"] = "fitting_all_development"
            write_json(record_path, record)
            progress = fit_one(
                frame,
                np.arange(len(frame)),
                np.array([], dtype=int),
                data_root,
                spec,
                config,
                selection["backbone_lr"],
                protocol["seed"] + 900000,
                output / "resume" / "all_development",
                record["identity_sha256"],
                device,
                epochs=selection["final_epochs"],
            )
            # Rehash the development corpus before releasing either final artifact.
            if neural_source_identity(manifest, frame, data_root) != identity["source"]:
                raise ValueError("Development source content changed during training")
            saved = checked_load(
                output / "resume/all_development" / progress["checkpoint"],
                progress["checkpoint_sha256"],
            )
            destination = output / "candidates/finetuned"
            destination.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(destination / "model.pt", saved["model"])
            metadata = {
                "schema_version": 1,
                "kind": "exploratory-partial-dinov2",
                "architecture": spec,
                "config": config,
                "config_sha256": json_hash(config),
                "class_order": list(CLASS_NAMES),
                "selection": selection,
                "selection_sha256": selection_sha256,
                "training_source": identity["source"],
                "source_hashes": identity["source_hashes"],
                "environment": identity["environment"],
                "run_identity_sha256": record["identity_sha256"],
                "model_sha256": file_hash(destination / "model.pt"),
                "pretrained_weights_sha256": spec["weights_sha256"],
                "inference_precision": "float32; autocast and TF32 disabled",
                "interpretation": config["interpretation"],
            }
            write_json(destination / "model.json", metadata)
            frozen_destination = output / "candidates/frozen"
            frozen_destination.mkdir(parents=True, exist_ok=True)
            if (
                file_hash(Path(frozen_run) / "model.joblib") != identity["frozen_model_sha256"]
                or file_hash(Path(frozen_run) / "model_metadata.json")
                != identity["frozen_metadata_sha256"]
            ):
                raise ValueError("Frozen candidate changed during training")
            for name in ("model.joblib", "model_metadata.json"):
                shutil.copyfile(Path(frozen_run) / name, frozen_destination / name)
            record.update(
                status="complete",
                exports={
                    "finetuned": {
                        "path": "candidates/finetuned/model.pt",
                        "sha256": metadata["model_sha256"],
                        "metadata_sha256": file_hash(destination / "model.json"),
                    },
                    "frozen": {
                        "path": "candidates/frozen/model.joblib",
                        "sha256": file_hash(frozen_destination / "model.joblib"),
                    },
                },
                candidate_choice="parent must freeze explicitly before fresh evaluation",
            )
            write_json(record_path, record)
            return record["exports"]
        except BaseException as exc:
            record.update(
                status="interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            write_json(record_path, record)
            raise


def load_deployed(
    artifact, device, *, expected_sha256=None, metadata_sha256=None, model_factory=None
):
    artifact = Path(artifact)
    sidecar = artifact.with_suffix(".json")
    if metadata_sha256 and file_hash(sidecar) != metadata_sha256:
        raise ValueError("Deployment metadata checksum mismatch")
    metadata = read_json(sidecar)
    if (
        metadata["class_order"] != list(CLASS_NAMES)
        or json_hash(metadata["config"]) != metadata["config_sha256"]
        or (expected_sha256 and expected_sha256 != metadata["model_sha256"])
    ):
        raise ValueError("Deployment class order/config/checksum mismatch")
    if "source_hashes" in metadata and metadata["source_hashes"] != source_hashes():
        raise ValueError("Deployment implementation changed; use its recorded source revision")
    current = environment()["packages"]
    if any(
        current[name] != metadata["environment"]["packages"][name]
        for name in ("torch", "torchvision", "timm", "numpy", "Pillow")
    ):
        raise ValueError("Inference runtime differs from exported environment")
    spec, config = metadata["architecture"], metadata["config"]
    if model_factory:
        backbone = model_factory(spec, config["input_size"], device="cpu", offline=True)
    else:
        import timm

        backbone = timm.create_model(
            spec["name"], pretrained=False, num_classes=0, img_size=config["input_size"]
        )
    model = PartialDino(backbone, spec, config)
    model.load_state_dict(checked_load(artifact, metadata["model_sha256"]), strict=True)
    return model.to(device).eval().requires_grad_(False), metadata


def verify_fresh(frame, source, training):
    required = ("source_groups", "material_ids", "image_paths", "sources", "decoded_rgb_sha256")
    if any(not training.get(key) for key in required) or not source.get("decoded_rgb_sha256"):
        raise ValueError("Missing training/evaluation identity for fresh evaluation")
    if set(frame["source_group"]) & set(training["source_groups"]) or set(
        frame["material_id"]
    ) & set(training["material_ids"]):
        raise ValueError("Fresh evaluation source/material overlap with development")
    if set(frame["image_path"]) & set(training["image_paths"]) or {
        row["sha256"] for row in source["sources"]
    } & {row["sha256"] for row in training["sources"]}:
        raise ValueError("Fresh evaluation image overlap with development")
    if set(source["decoded_rgb_sha256"]) & set(training["decoded_rgb_sha256"]):
        raise ValueError("Fresh evaluation decoded image overlap with development")


@torch.inference_mode()
def predict_finetuned(
    artifact,
    data_root,
    output,
    *,
    manifest=None,
    image=None,
    evaluate=False,
    device="cpu",
    cpu_threads=4,
    expected_sha256=None,
    metadata_sha256=None,
    model_factory=None,
):
    if (manifest is None) == (image is None) or (evaluate and manifest is None):
        raise ValueError("Choose image or manifest; --evaluate requires a labeled manifest")
    device = configure_torch(device, cpu_threads)
    model, metadata = load_deployed(
        artifact,
        device,
        expected_sha256=expected_sha256,
        metadata_sha256=metadata_sha256,
        model_factory=model_factory,
    )
    spec, config = metadata["architecture"], metadata["config"]
    output = Path(output)
    if image is not None:
        path = image_file(data_root, image)
        with Image.open(path) as picture:
            tensor = transform_image(picture, spec, config["input_size"])
        logits = model(tensor[None].to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite prediction logits")
        probabilities = logits.softmax(dim=1).double().cpu().numpy()[0]
        probabilities /= probabilities.sum()
        result = {
            "image_path": str(path),
            "image_sha256": file_hash(path),
            "y_pred": int(probabilities.argmax()),
            "predicted_class": CLASS_NAMES[int(probabilities.argmax())],
            "probabilities": dict(zip(CLASS_NAMES, probabilities.tolist(), strict=True)),
            "model_sha256": metadata["model_sha256"],
        }
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "prediction.json", result)
        return result
    frame = read_manifest(manifest)
    source = neural_source_identity(manifest, frame, data_root)
    if evaluate:
        verify_fresh(frame, source, metadata["training_source"])
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "model_sha256": metadata["model_sha256"],
        "metadata_sha256": file_hash(Path(artifact).with_suffix(".json")),
        "source": source,
        "evaluation_requested": evaluate,
        "fitting": "none; immutable all-development fine-tuned candidate",
    }
    write_json(output / "prediction_run.json", record)
    try:
        data = Images(frame, data_root, spec, config)
        values = []
        for batch, _ in epoch_loader(data, config, 0, False):
            # Deployment is deterministic float32; no stochastic training AMP path.
            logits = model(batch.to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite prediction logits")
            values.append(logits.softmax(dim=1).double().cpu().numpy())
        probabilities = np.concatenate(values)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if neural_source_identity(manifest, frame, data_root) != source:
            raise ValueError("Prediction source changed during inference")
        prediction_frame(frame, probabilities, "holdout" if evaluate else "prediction_only").to_csv(
            output / "predictions.csv", index=False
        )
        if evaluate:
            write_json(
                output / "metrics.json",
                {
                    **probability_metrics(labels_for(frame), probabilities),
                    "interpretation": "Fresh evaluation of fixed exploratory development artifact; "
                    "independence also requires source provenance review by the parent.",
                },
            )
        record.update(
            status="complete",
            rows=len(frame),
            predictions_sha256=file_hash(output / "predictions.csv"),
        )
        write_json(output / "prediction_run.json", record)
        return {"rows": len(frame), "output": str(output)}
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "prediction_run.json", record)
        raise
