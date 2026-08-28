"""Training engine for fixed-split headline experiments.

The engine accepts DataLoaders so data policy lives in :mod:`data`; in
particular, this module never creates a test loader.  Checkpoints are complete
and atomically replaced after every epoch, making interruption recoverable.
"""

from __future__ import annotations

import copy
import csv
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .metrics import classification_metrics
from .models import classifier_parameter_names
from .utils import (
    atomic_json_dump,
    atomic_torch_save,
    capture_environment,
    count_parameters,
    load_torch_checkpoint,
    now_utc,
    seed_everything,
    stable_hash,
)

LOGGER = logging.getLogger(__name__)


def _without_batch_size(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.get("training", {}).pop("batch_size", None)
    return value


def _safe_oom_config_change(old: dict[str, Any], new: dict[str, Any]) -> bool:
    return _without_batch_size(old) == _without_batch_size(new)


def _load_adjustments(path: Path) -> list[dict[str, int]]:
    if not path.exists():
        return []
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    chain = payload.get("adjustments")
    if not isinstance(chain, list):
        raise RuntimeError("Malformed OOM adjustment chain")
    previous = None
    for item in chain:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("from_batch_size"), int)
            or not isinstance(item.get("to_batch_size"), int)
        ):
            raise RuntimeError("Malformed OOM adjustment transition")
        if item["to_batch_size"] >= item["from_batch_size"] or (
            previous is not None and item["from_batch_size"] != previous
        ):
            raise RuntimeError("Non-contiguous OOM adjustment chain")
        previous = item["to_batch_size"]
    if chain and payload.get("final_batch_size") != chain[-1]["to_batch_size"]:
        raise RuntimeError("OOM adjustment final batch does not match transition chain")
    return chain


def _recorded_batch_transition(path: Path, old: dict[str, Any], new: dict[str, Any]) -> bool:
    chain = _load_adjustments(path)
    old_batch = int(old.get("training", {}).get("batch_size", 0))
    new_batch = int(new.get("training", {}).get("batch_size", 0))
    return (
        bool(chain)
        and chain[0]["from_batch_size"] == old_batch
        and chain[-1]["to_batch_size"] == new_batch
    )


def _batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"], batch["label"] if "label" in batch else batch["target"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("A batch must be (images, labels) or a dict with image/label keys")


def _amp_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _trainable(model: nn.Module, names: set[str] | None, head_only: bool) -> None:
    if not head_only:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    names = names or set()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in names


def _set_training_mode(model: nn.Module, training: bool, head_only: bool) -> None:
    """Keep frozen-backbone BatchNorm running statistics unchanged."""
    model.train(training)
    if training and head_only:
        model.eval()
        prefixes = ("fc", "head", "heads", "classifier")
        for name, module in model.named_modules():
            if name and any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                module.train(True)


def _optimizer(model: nn.Module, cfg: dict[str, Any]) -> AdamW:
    opt = cfg.get("optimizer", {})
    names = classifier_parameter_names(model)
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        (head if name in names else backbone).append(parameter)
    if not head:
        head, backbone = backbone, []
    head_lr = float(opt.get("head_learning_rate", opt.get("learning_rate", 3e-4)))
    back_lr = float(opt.get("backbone_learning_rate", head_lr / 10))
    groups = [{"params": head, "lr": head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": back_lr})
    return AdamW(groups, weight_decay=float(opt.get("weight_decay", 1e-4)))


def _scheduler(optimizer: AdamW, cfg: dict[str, Any]) -> LambdaLR:
    train = cfg.get("training", {})
    sched = cfg.get("scheduler", {})
    total = max(1, int(train.get("max_epochs", 40)))
    warmup = max(0, int(sched.get("warmup_epochs", 3)))

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / max(1, warmup))
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def _epoch(
    model: nn.Module,
    loader: Iterable[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: Any,
    amp: bool,
    clip: float,
    head_only: bool = False,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    _set_training_mode(model, training, head_only)
    total_loss, total = 0.0, 0
    true: list[int] = []
    pred: list[int] = []
    probs: list[torch.Tensor] = []
    for raw in loader:
        images, labels = _batch(raw)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with _autocast(amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total += batch_size
        true.extend(labels.detach().cpu().tolist())
        pred.extend(logits.detach().argmax(1).cpu().tolist())
        probs.append(torch.softmax(logits.detach(), dim=1).cpu())
    probability = torch.cat(probs).numpy() if probs else None
    return total_loss / max(1, total), classification_metrics(true, pred, probability)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    config: dict[str, Any],
    checkpoint_dir: Path,
    device: torch.device | None = None,
    force: bool = False,
    class_to_idx: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Train a model on train/val only and return the best validation record."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "run_state.json"
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    adjustments_path = checkpoint_dir / "oom_adjustments.json"
    config_hash = stable_hash(config)
    state = {}
    if state_path.exists() and not force:
        import json

        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_hash") and state["config_hash"] != config_hash:
            if not (
                state.get("status") == "oom"
                and state.get("config")
                and _safe_oom_config_change(state["config"], config)
                and _recorded_batch_transition(adjustments_path, state["config"], config)
            ):
                raise RuntimeError(
                    "Completed/resumable run config differs; use a new namespace or --force"
                )
        if state.get("status") == "completed" and best_path.exists():
            return state["best"]
    seed_everything(int(config.get("seed", 42)))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    resolved = dict(config)
    resolved["config_hash"] = config_hash
    resolved["model"] = getattr(model, "_material_model_name", type(model).__name__)
    resolved["class_to_idx"] = class_to_idx or {}
    resolved["input_preprocessing"] = config.get(
        "input_preprocessing",
        {
            "resize": 256,
            "center_crop": 224,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
    )
    resolved["effective_batch_size"] = config.get("training", {}).get("batch_size")
    resolved["oom_adjustments"] = _load_adjustments(checkpoint_dir / "oom_adjustments.json")
    atomic_json_dump(resolved, checkpoint_dir / "resolved_config.json")
    train_cfg = config.get("training", {})
    max_epochs = int(train_cfg.get("max_epochs", 40))
    patience = int(train_cfg.get("early_stopping_patience", 8))
    freeze_epochs = (
        int(config.get("fine_tuning", {}).get("freeze_head_epochs", 3))
        if getattr(model, "_material_model_name", "") != "custom_cnn"
        else 0
    )
    head_names = classifier_parameter_names(model)
    _trainable(model, head_names, freeze_epochs > 0)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = _amp_scaler(amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("optimizer", {}).get("label_smoothing", 0.1))
    )
    start_epoch, best_f1, best_loss, stale = 0, -float("inf"), float("inf"), 0
    history: list[dict[str, Any]] = []
    resume_payload = None
    if last_path.exists() and not force:
        payload = load_torch_checkpoint(last_path, device)
        if payload.get("config_hash") and payload["config_hash"] != config_hash:
            if not (
                state.get("status") == "oom"
                and payload.get("config")
                and _safe_oom_config_change(payload["config"], config)
                and _recorded_batch_transition(adjustments_path, payload["config"], config)
            ):
                raise RuntimeError(
                    "Checkpoint config differs from requested config; use a new "
                    "namespace or --force"
                )
        resume_payload = payload
    if resume_payload is not None:
        payload = resume_payload
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload.get("epoch", 0))
        best_f1, best_loss, stale = (
            float(payload.get("best_f1", best_f1)),
            float(payload.get("best_loss", best_loss)),
            int(payload.get("stale", 0)),
        )
        history = list(payload.get("history", []))
        if start_epoch >= freeze_epochs:
            _trainable(model, head_names, False)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(checkpoint_dir / "tensorboard"))
    except (ImportError, ModuleNotFoundError):
        LOGGER.warning("TensorBoard unavailable; continuing with CSV history")
    resume_metadata = resume_payload if resume_payload is not None else state
    started_at = (
        str(resume_metadata.get("started_at")) if resume_metadata.get("started_at") else now_utc()
    )
    previous_time = (
        float(resume_metadata.get("total_training_time_seconds", 0.0)) if resume_metadata else 0.0
    )
    wall_start = time.perf_counter()
    atomic_json_dump(
        {
            "status": "running",
            "started_at": started_at,
            "config_hash": config_hash,
            "seed": config.get("seed", 42),
            "model": getattr(model, "_material_model_name", type(model).__name__),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "effective_batch_size": train_cfg.get("batch_size"),
            "environment": capture_environment(),
            "resolved_config": resolved,
        },
        state_path,
    )
    for epoch in range(start_epoch, max_epochs):
        started = time.perf_counter()
        if epoch == freeze_epochs:
            _trainable(model, head_names, False)
        try:
            train_loss, train_m = _epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                amp,
                float(train_cfg.get("gradient_clip_norm", 1.0)),
                head_only=epoch < freeze_epochs,
            )
            with torch.no_grad():
                val_loss, val_m = _epoch(
                    model, val_loader, criterion, device, None, scaler, amp, 0.0
                )
        except Exception as exc:
            if writer:
                writer.close()
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                atomic_json_dump(
                    {
                        "status": "oom",
                        "config": config,
                        "config_hash": config_hash,
                        "started_at": started_at,
                        "model": getattr(model, "_material_model_name", type(model).__name__),
                        "oom_recoverable": True,
                        "epoch": epoch + 1,
                        "message": str(exc),
                        "batch_size": train_cfg.get("batch_size"),
                        "total_training_time_seconds": previous_time
                        + time.perf_counter()
                        - wall_start,
                    },
                    state_path,
                )
                raise RuntimeError(
                    "CUDA OOM: reduce batch_size or use the CLI retry; adjustment was recorded"
                ) from exc
            raise
        epoch_time = time.perf_counter() - started
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_m["accuracy"],
            "train_macro_f1": train_m["f1_macro"],
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["f1_macro"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": epoch_time,
        }
        history.append(row)
        _write_history(checkpoint_dir / "history.csv", history)
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch + 1)
        scheduler.step()
        improved = val_m["f1_macro"] > best_f1 or (
            val_m["f1_macro"] == best_f1 and val_loss < best_loss
        )
        if improved:
            best_f1, best_loss, stale = val_m["f1_macro"], val_loss, 0
            best_payload = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": config,
                "config_hash": config_hash,
                "class_to_idx": class_to_idx or {},
                "input_preprocessing": resolved["input_preprocessing"],
                "best_f1": best_f1,
                "best_loss": best_loss,
                "history": history,
                "environment": capture_environment(),
                "seed": config.get("seed", 42),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "effective_batch_size": train_cfg.get("batch_size"),
                "started_at": started_at,
                "total_training_time_seconds": previous_time + time.perf_counter() - wall_start,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            }
            atomic_torch_save(best_payload, best_path)
        else:
            stale += 1
        last_payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": config,
            "config_hash": config_hash,
            "class_to_idx": class_to_idx or {},
            "input_preprocessing": resolved["input_preprocessing"],
            "best_f1": best_f1,
            "best_loss": best_loss,
            "stale": stale,
            "history": history,
            "seed": config.get("seed", 42),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "effective_batch_size": train_cfg.get("batch_size"),
            "started_at": started_at,
            "last_epoch_time_seconds": epoch_time,
            "total_training_time_seconds": previous_time + time.perf_counter() - wall_start,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        atomic_torch_save(last_payload, last_path)
        if stale >= patience:
            LOGGER.info("Early stopping at epoch %d", epoch + 1)
            break
    if writer:
        writer.close()
    best = load_torch_checkpoint(best_path, "cpu") if best_path.exists() else last_payload
    completed_at = now_utc()
    total_time = previous_time + time.perf_counter() - wall_start
    best.update(
        {
            "config": config,
            "config_hash": config_hash,
            "resolved_config": resolved,
            "history": history,
            "started_at": started_at,
            "completed_at": completed_at,
            "total_training_time_seconds": total_time,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "environment": capture_environment(),
            "effective_batch_size": train_cfg.get("batch_size"),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        }
    )
    best["oom_adjustments"] = []
    if adjustments_path.exists():
        import json

        best["oom_adjustments"] = _load_adjustments(adjustments_path)
        resolved["oom_adjustments"] = best["oom_adjustments"]
    resolved["effective_batch_size"] = train_cfg.get("batch_size")
    best["resolved_config"] = resolved
    atomic_json_dump(resolved, checkpoint_dir / "resolved_config.json")
    atomic_torch_save(best, best_path)
    if last_path.exists():
        last = load_torch_checkpoint(last_path, "cpu")
        last.update(
            {
                "config": config,
                "config_hash": config_hash,
                "resolved_config": resolved,
                "oom_adjustments": best.get("oom_adjustments", []),
                "started_at": started_at,
                "completed_at": completed_at,
                "total_training_time_seconds": total_time,
                "environment": best.get("environment"),
                "effective_batch_size": train_cfg.get("batch_size"),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            }
        )
        atomic_torch_save(last, last_path)
    result = {
        "best_epoch": best["epoch"],
        "val_macro_f1": best["best_f1"],
        "val_loss": best["best_loss"],
        "history": history,
    }
    atomic_json_dump(
        {
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "config_hash": config_hash,
            "seed": config.get("seed", 42),
            "best": result,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "effective_batch_size": train_cfg.get("batch_size"),
            "total_training_time_seconds": total_time,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "environment": best.get("environment"),
            "resolved_config": resolved,
            "oom_adjustments": best.get("oom_adjustments", []),
            "model": getattr(model, "_material_model_name", type(model).__name__),
        },
        state_path,
    )
    return result


def train_with_oom_retry(
    model_factory: Any,
    loader_factory: Any,
    config: dict[str, Any],
    checkpoint_dir: Path,
    device: torch.device | None = None,
    class_to_idx: dict[str, int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Retry CUDA OOM runs with halved batches and persist every adjustment."""
    working = copy.deepcopy(config)
    adjustment_path = Path(checkpoint_dir) / "oom_adjustments.json"
    if force:
        # Explicit force starts a clean experiment chain.
        if adjustment_path.exists():
            adjustment_path.unlink()
        recorded = []
    else:
        recorded = _load_adjustments(adjustment_path)
        if recorded:
            requested_batch = int(working.get("training", {}).get("batch_size", 32))
            first_batch = recorded[0]["from_batch_size"]
            final_batch = recorded[-1]["to_batch_size"]
            if requested_batch == first_batch:
                working.setdefault("training", {})["batch_size"] = final_batch
            elif requested_batch != final_batch:
                raise RuntimeError("Requested batch size is not the recorded OOM retry transition")
    run_state_path = Path(checkpoint_dir) / "run_state.json"
    unfinished = False
    if not force and run_state_path.exists():
        import json

        unfinished = json.loads(run_state_path.read_text(encoding="utf-8")).get("status") == "oom"
    # A restarted unfinished run must append to, not replace, its validated chain.
    adjustments: list[dict[str, int]] = list(recorded) if unfinished else []
    while True:
        batch_size = int(working.get("training", {}).get("batch_size", 32))
        try:
            # Factories may initialize classifier weights or shuffle generators;
            # seed before constructing either so OOM retries are reproducible.
            seed_everything(int(working.get("seed", 42)))
            train_loader, val_loader = loader_factory(batch_size)
            result = train_model(
                model_factory(),
                train_loader,
                val_loader,
                working,
                checkpoint_dir,
                device,
                force,
                class_to_idx,
            )
            if adjustments:
                atomic_json_dump(
                    {"adjustments": adjustments, "final_batch_size": batch_size},
                    adjustment_path,
                )
            return result
        except RuntimeError as exc:
            if (
                "CUDA OOM" not in str(exc)
                or batch_size <= 1
                or (device is not None and device.type != "cuda")
            ):
                raise
            next_batch = max(1, batch_size // 2)
            adjustments.append({"from_batch_size": batch_size, "to_batch_size": next_batch})
            working.setdefault("training", {})["batch_size"] = next_batch
            # Persist before retrying: the next train_model call may finalize
            # an existing last/best checkpoint after a prior saved epoch.
            atomic_json_dump(
                {"adjustments": adjustments, "final_batch_size": next_batch},
                adjustment_path,
            )
            force = False
