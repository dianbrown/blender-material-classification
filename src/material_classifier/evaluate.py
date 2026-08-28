"""Locked validation/test evaluation and the immutable methodology gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .metrics import classification_metrics
from .models import build_model, forward_with_embeddings
from .train import _autocast, _batch
from .utils import atomic_json_dump, load_torch_checkpoint, now_utc, sha256_file, stable_hash


def methodology_lock_path(root: Path) -> Path:
    return Path(root) / "reports" / "results" / "methodology_lock.json"


def require_methodology_lock(root: Path) -> dict[str, Any]:
    path = methodology_lock_path(root)
    if not path.exists():
        raise RuntimeError(
            "Held-out TEST evaluation is blocked: create the immutable methodology lock first"
        )
    lock = json.loads(path.read_text(encoding="utf-8"))
    if not lock.get("locked") or lock.get("test_evaluation_allowed") is not True:
        raise RuntimeError("Methodology lock exists but does not authorize test evaluation")
    lock_hash = lock.get("lock_payload_hash")
    unsigned = {key: value for key, value in lock.items() if key != "lock_payload_hash"}
    if not lock_hash or stable_hash(unsigned) != lock_hash:
        raise RuntimeError("Methodology lock integrity check failed")
    for name, expected in lock.get("config_sha256", {}).items():
        config = Path(root) / "configs" / name
        if not config.exists() or sha256_file(config) != expected:
            raise RuntimeError(f"Methodology lock invalidated by changed config: {name}")
    for relative, expected in lock.get("artifacts_sha256", {}).items():
        artifact = Path(root) / relative
        if not artifact.exists() or sha256_file(artifact) != expected:
            raise RuntimeError(f"Methodology lock invalidated by changed artifact: {relative}")
    return lock


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    class_names: list[str] | None = None,
    amp: bool = True,
    output_dir: Path | None = None,
    declared_split: str | None = None,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    true: list[int] = []
    pred: list[int] = []
    probabilities: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    total_loss, total = 0.0, 0
    use_amp = amp and device.type == "cuda"
    for raw in loader:
        images, labels = _batch(raw)
        if isinstance(raw, dict):
            batch_meta = raw
            if declared_split is not None:
                actual = batch_meta.get("split")
                values = actual if isinstance(actual, (list, tuple)) else [actual] * len(labels)
                if any(str(value) != declared_split for value in values):
                    raise AssertionError(
                        f"Loader record split disagrees with declared split {declared_split!r}"
                    )
            for i in range(len(labels)):
                metadata.append(
                    {
                        key: (value[i] if isinstance(value, (list, tuple)) else value)
                        for key, value in batch_meta.items()
                        if key not in {"image", "label", "target"}
                    }
                )
        elif declared_split is not None:
            raise AssertionError("Metadata batches are required when asserting a declared split")
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with _autocast(use_amp):
            logits, embedding = forward_with_embeddings(model, images)
            loss = criterion(logits, labels)
        n = labels.shape[0]
        total_loss += float(loss) * n
        total += n
        true.extend(labels.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
        probabilities.append(torch.softmax(logits, 1).cpu())
        embeddings.append(embedding.cpu())
    probs = torch.cat(probabilities).numpy() if probabilities else None
    result = classification_metrics(true, pred, probs, class_names, total_loss / max(1, total))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import csv

        names = class_names or [str(i) for i in range(probs.shape[1] if probs is not None else 0)]
        pred_path = output_dir / "predictions.csv"
        fields = [
            "image_path",
            "material_id",
            "split",
            "true_label",
            "pred_label",
            "confidence",
        ] + [f"prob_{name}" for name in names]
        with pred_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i, (truth, guess, probability) in enumerate(zip(true, pred, probs)):
                meta = metadata[i] if i < len(metadata) else {}
                row = {
                    "image_path": meta.get("image_path", ""),
                    "material_id": meta.get("material_id", ""),
                    "split": meta.get("split", declared_split or ""),
                    "true_label": names[truth],
                    "pred_label": names[guess],
                    "confidence": float(probability[guess]),
                }
                row.update({f"prob_{name}": float(probability[j]) for j, name in enumerate(names)})
                writer.writerow(row)
        npz_path = output_dir / "embeddings.npz"
        import numpy as np

        np.savez_compressed(
            npz_path,
            embeddings=torch.cat(embeddings).numpy(),
            image_path=np.asarray([m.get("image_path", "") for m in metadata]),
            material_id=np.asarray([m.get("material_id", "") for m in metadata]),
            true_label=np.asarray(true),
            pred_label=np.asarray(pred),
        )
        result["predictions_path"], result["embeddings_path"] = str(pred_path), str(npz_path)
    return result


def evaluate_checkpoint(
    checkpoint: Path,
    loader: Iterable[Any],
    root: Path,
    split: str,
    class_names: list[str] | None = None,
    device: torch.device | None = None,
    model: nn.Module | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint; test requires the lock and can only run once."""
    root = Path(root)
    split = split.lower()
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    checkpoint_path = Path(checkpoint)
    payload = load_torch_checkpoint(checkpoint_path, "cpu")
    config = payload.get("config", {})
    model_name = config.get("name", config.get("architecture", checkpoint_path.parent.name))
    checkpoint_hash = sha256_file(checkpoint_path)
    if split == "test":
        require_methodology_lock(root)
        marker = (
            root
            / "reports"
            / "results"
            / f"test_evaluated_{model_name}_{checkpoint_hash[:16]}.json"
        )
        if marker.exists():
            cached = json.loads(marker.read_text(encoding="utf-8"))
            cached_path = Path(cached.get("metrics_file", ""))
            prediction_path = Path(cached.get("predictions_file", ""))
            embedding_path = Path(cached.get("embeddings_file", ""))
            if (
                cached.get("status") == "completed"
                and cached.get("checkpoint_sha256") == checkpoint_hash
                and cached_path.exists()
                and prediction_path.exists()
                and embedding_path.exists()
                and cached.get("metrics_sha256") == sha256_file(cached_path)
                and cached.get("predictions_sha256") == sha256_file(prediction_path)
                and cached.get("embeddings_sha256") == sha256_file(embedding_path)
            ):
                return json.loads(cached_path.read_text(encoding="utf-8"))
            if cached.get("status") == "completed":
                raise RuntimeError(
                    "Completed TEST cache failed integrity checks; refusing duplicate inference"
                )
    if model is None:
        model = build_model(model_name, int(config.get("num_classes", 7)), pretrained=False)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model_key = getattr(model, "_material_model_name", str(model_name))
    target = Path(output_dir or (root / "reports" / "results")) / model_key / split
    result = evaluate_loader(
        model, loader, device, class_names, amp=True, output_dir=target, declared_split=split
    )
    result.update(
        {
            "model": model_key,
            "checkpoint": str(checkpoint),
            "split": split,
            "best_epoch": payload.get("epoch"),
        }
    )
    target.mkdir(parents=True, exist_ok=True)
    metrics_path = target / "metrics.json"
    atomic_json_dump(result, metrics_path)
    if split == "test":
        prediction_path = Path(result["predictions_path"])
        embedding_path = Path(result["embeddings_path"])
        atomic_json_dump(
            {
                "status": "completed",
                "evaluated_at": now_utc(),
                "checkpoint_sha256": checkpoint_hash,
                "metrics_file": str(metrics_path.resolve()),
                "predictions_file": str(prediction_path.resolve()),
                "embeddings_file": str(embedding_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(prediction_path),
                "embeddings_sha256": sha256_file(embedding_path),
            },
            marker,
        )
    return result
