"""Conservative explainability helpers for the three transfer models.

Grad-CAM is optional.  The module supplies architecture-correct target-layer
selection and a gradient saliency fallback, while the curated generator is
deliberately gated on completed, locked final evaluation artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn

LOGGER = logging.getLogger(__name__)


def target_layer_for_model(model: nn.Module, model_name: str | None = None) -> nn.Module:
    """Return the final spatial feature layer suitable for Grad-CAM.

    ResNet's final bottleneck convolution and ConvNeXt's final feature block
    retain spatial maps.  ViT uses its final encoder layer; ``vit_reshape_transform``
    converts its token sequence into a patch grid for CAM implementations.
    """
    name = (model_name or getattr(model, "_material_model_name", "")).lower().replace("-", "_")
    if name in {"resnet50", "resnet_50"} or hasattr(model, "layer4"):
        # The final Bottleneck output includes the residual addition and is
        # the conventional Grad-CAM target for torchvision ResNet.
        return model.layer4[-1]
    if name in {"convnext_tiny", "convnext"} or (
        hasattr(model, "features") and hasattr(model, "classifier")
    ):
        return model.features[-1]
    if name in {"vit_b16", "vit_b_16", "vit"} or hasattr(model, "class_token"):
        encoder = model.encoder
        layers = getattr(encoder, "layers", encoder)
        block = layers[-1]
        return getattr(block, "ln_1", block)
    raise ValueError("Unsupported model; expected ResNet50, ConvNeXt-Tiny or ViT-B/16")


def vit_reshape_transform(
    tensor: torch.Tensor, height: int | None = None, width: int | None = None
) -> torch.Tensor:
    """Transform ViT ``[batch, tokens, channels]`` into CAM ``[batch,C,H,W]``.

    Torchvision's ViT sequence includes one CLS token.  Patch dimensions are
    inferred as a square when not supplied (ViT-B/16 at 224px is 14 x 14).
    """
    if tensor.ndim != 3:
        raise ValueError(f"ViT activations must be 3-D, got shape {tuple(tensor.shape)}")
    patches = tensor[:, 1:, :]
    count = patches.shape[1]
    if height is None or width is None:
        side = int(round(count**0.5))
        if side * side != count:
            raise ValueError("Cannot infer non-square ViT patch grid; pass height and width")
        height, width = side, side
    if int(height) * int(width) != count:
        raise ValueError("ViT patch grid does not match token count")
    return patches.reshape(tensor.shape[0], int(height), int(width), tensor.shape[2]).permute(
        0, 3, 1, 2
    )


def assert_explainability_ready(root: str | Path, model_name: str, split: str = "test") -> Path:
    """Require methodology lock and completed final metrics before CAM output."""
    if split.lower() != "test":
        raise ValueError("Curated explainability is only enabled for final TEST artifacts")
    root_path = Path(root)
    from .evaluate import require_methodology_lock

    require_methodology_lock(root_path)
    model_key = model_name.lower().replace("-", "_").replace("/", "_")
    model_key = {"vit_b_16": "vit_b16", "resnet_50": "resnet50"}.get(model_key, model_key)
    metrics = root_path / "reports" / "results" / model_key / "test" / "metrics.json"
    predictions = root_path / "reports" / "results" / model_key / "test" / "predictions.csv"
    if not metrics.exists() or not predictions.exists():
        raise FileNotFoundError(
            f"Completed final evaluation artifacts required for {model_key}: "
            f"{metrics} and {predictions}"
        )
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if payload.get("split") not in {None, "test"}:
        raise ValueError(f"Metrics artifact is not a TEST result: {metrics}")
    checkpoint = root_path / "artifacts" / "checkpoints" / model_key / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint required to verify TEST marker: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    valid_marker = False
    for marker in (root_path / "reports" / "results").glob(f"test_evaluated_{model_key}_*.json"):
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            marker_metrics = Path(marker_payload.get("metrics_file", "")).resolve()
            valid_marker = (
                marker_payload.get("status") == "completed"
                and marker_payload.get("checkpoint_sha256") == digest
                and marker_metrics == metrics.resolve()
                and marker_metrics.exists()
                and marker.name.endswith(f"{digest[:16]}.json")
            )
        except (OSError, json.JSONDecodeError):
            valid_marker = False
        if valid_marker:
            break
    if not valid_marker:
        raise RuntimeError(
            f"No valid completed TEST marker with a matching checkpoint hash for {model_key}"
        )
    return metrics


def gradient_saliency(
    model: nn.Module, image: torch.Tensor, target: int | None = None
) -> tuple[torch.Tensor, int]:
    """Compute normalized input-gradient saliency as a dependency-free fallback."""
    was_training = model.training
    model.eval()
    tensor = image.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    index = int(logits.argmax(1).item()) if target is None else int(target)
    logits[:, index].sum().backward()
    saliency = tensor.grad.detach().abs().amax(dim=1)
    saliency = saliency - saliency.amin(dim=(1, 2), keepdim=True)
    saliency = saliency / saliency.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    if was_training:
        model.train()
    return saliency.cpu(), index


def _overlay(image: torch.Tensor, heatmap: torch.Tensor) -> np.ndarray:
    from matplotlib import colormaps

    raw = image.detach().cpu()
    if raw.ndim == 4:
        raw = raw[0]
    raw = raw.permute(1, 2, 0).numpy()
    raw = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-8)
    heat = heatmap.detach().cpu()
    if heat.ndim == 3:
        heat = heat[0]
    heat_array = heat.numpy()
    heat_array = (
        np.asarray(
            Image.fromarray(np.uint8(heat_array * 255)).resize(
                (raw.shape[1], raw.shape[0]), Image.BILINEAR
            )
        )
        / 255.0
    )
    colour = colormaps.get_cmap("magma")(heat_array)[..., :3]
    return np.clip(0.58 * raw + 0.42 * colour, 0, 1)


def cam_or_saliency(
    model: nn.Module,
    image: torch.Tensor,
    model_name: str | None = None,
    target: int | None = None,
    display_image: torch.Tensor | None = None,
) -> tuple[np.ndarray, str, int]:
    """Return a CAM overlay, falling back to input saliency if Grad-CAM is absent."""
    image = image if image.ndim == 4 else image.unsqueeze(0)
    name = (model_name or getattr(model, "_material_model_name", "")).lower()
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        layer = target_layer_for_model(model, name)
        kwargs: dict[str, Any] = {"model": model, "target_layers": [layer]}
        if "vit" in name:
            kwargs["reshape_transform"] = vit_reshape_transform
        with GradCAM(**kwargs) as cam:
            logits = model(image)
            index = int(logits.argmax(1).item()) if target is None else int(target)
            heatmap = torch.from_numpy(
                cam(input_tensor=image, targets=[ClassifierOutputTarget(index)])
            )
        return (
            _overlay(display_image if display_image is not None else image, heatmap),
            "grad-cam",
            index,
        )
    except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
        LOGGER.warning("Grad-CAM unavailable for %s (%s); using input saliency", name, exc)
        heatmap, index = gradient_saliency(model, image, target)
        return (
            _overlay(display_image if display_image is not None else image, heatmap),
            "saliency",
            index,
        )


def generate_curated_gallery(
    root: str | Path,
    model_name: str,
    model: nn.Module,
    records: Iterable[dict[str, Any]],
    output: str | Path,
    device: torch.device | None = None,
    max_examples: int = 12,
) -> Path:
    """Generate a small, curated explanation gallery after the methodology gate.

    ``records`` must contain cached ``image`` tensors or image paths and labels;
    no dataset is instantiated and no prediction pass is performed here.
    """
    assert_explainability_ready(root, model_name, "test")
    import matplotlib.pyplot as plt

    selected = list(records)[:max_examples]
    if not selected:
        raise ValueError("No cached records supplied for explainability gallery")
    device = device or next(model.parameters()).device
    model.to(device).eval()
    columns = 4
    fig, axes = plt.subplots(
        int(np.ceil(len(selected) / columns)),
        columns,
        figsize=(12, 3.3 * int(np.ceil(len(selected) / columns))),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, selected):
        image = record.get("image")
        display_image = record.get("display_image", image)
        if image is None:
            image_path = Path(str(record.get("image_path", "")))
            if not image_path.is_absolute():
                image_path = Path(root) / image_path
            with Image.open(image_path) as handle:
                image = torch.from_numpy(
                    np.asarray(handle.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
        tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
        if tensor.ndim == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
        display_tensor = display_image if isinstance(display_image, torch.Tensor) else None
        overlay, method, predicted = cam_or_saliency(
            model,
            tensor.unsqueeze(0).to(device),
            model_name,
            record.get("target"),
            display_image=display_tensor,
        )
        axis.imshow(overlay)
        axis.set_title(
            f"True: {record.get('true_label', '?')}\n"
            f"Pred: {record.get('pred_label', predicted)} · {method}",
            fontsize=8,
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination
