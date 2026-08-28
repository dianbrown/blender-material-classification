"""Checkpoint loading, single-image prediction, and latency benchmarking."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from .models import build_model, extract_embeddings
from .utils import load_torch_checkpoint


def evaluation_transform(size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint: Path, device: torch.device | None = None):
    """Reconstruct a checkpoint without downloading pretrained weights."""
    payload = load_torch_checkpoint(Path(checkpoint), "cpu")
    cfg = payload.get("config", {})
    name = cfg.get("name", cfg.get("architecture"))
    model = build_model(name, int(cfg.get("num_classes", 7)), pretrained=False)
    model.load_state_dict(payload["model_state"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, payload, device


@torch.no_grad()
def predict_image(
    checkpoint: Path,
    image: Path,
    class_names: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    model, payload, device = load_model(checkpoint, device)
    transform = evaluation_transform(int(payload.get("config", {}).get("input_size", 224)))
    with Image.open(image) as handle:
        tensor = transform(handle.convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    mapping = payload.get("class_to_idx", {})
    checkpoint_names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    names = class_names or checkpoint_names or [str(i) for i in range(len(probabilities))]
    return {
        "class_index": index,
        "class": names[index],
        "confidence": float(probabilities[index]),
        "probabilities": probabilities.cpu().tolist(),
    }


def benchmark_inference(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    device: torch.device,
    warmup: int = 5,
    repeats: int = 30,
) -> dict[str, float]:
    """Benchmark synchronized CUDA timing for the supplied batch iterable."""
    model.eval()
    iterator = iter(batches)
    sample = next(iterator).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    batch_size = int(sample.shape[0])
    return {
        "batch_size": batch_size,
        "latency_ms_batch": elapsed * 1000 / repeats,
        "latency_ms_per_image": elapsed * 1000 / (repeats * batch_size),
        "throughput_images_per_second": repeats * batch_size / max(elapsed, 1e-12),
    }


@torch.no_grad()
def embedding_batch(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return extract_embeddings(model, images.to(device)).cpu()
