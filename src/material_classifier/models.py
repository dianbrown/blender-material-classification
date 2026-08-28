"""The four model families used by the material-classification study.

The module deliberately keeps model construction in one place.  This makes it
hard for a training script to accidentally introduce an unplanned architecture
or to evaluate a differently configured classifier.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

MODEL_NAMES = ("custom_cnn", "resnet50", "convnext_tiny", "vit_b16")


class CustomCNN(nn.Module):
    """Small from-scratch baseline with five convolutional stages."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.30) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 384)
        stages: list[nn.Module] = []
        in_channels = 3
        for width in widths:
            stages.extend(
                [
                    nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = width
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], num_classes)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_embeddings(x)[0]

    def forward_with_embeddings(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.forward_features(x)
        return self.classifier(embedding), embedding


def _torchvision_model(
    name: str, num_classes: int, pretrained: bool, weights: Any = None
) -> nn.Module:
    """Build a torchvision model, handling the current weights API."""
    from torchvision import models

    builders = {
        "resnet50": (models.resnet50, "ResNet50_Weights"),
        "convnext_tiny": (models.convnext_tiny, "ConvNeXt_Tiny_Weights"),
        "vit_b16": (models.vit_b_16, "ViT_B_16_Weights"),
    }
    builder, weight_enum_name = builders[name]
    if weights is None and pretrained:
        weights = getattr(models, weight_enum_name).DEFAULT
    # Passing weights=None is supported by modern torchvision and avoids an
    # accidental network/download during tests.
    model = builder(weights=weights if pretrained else None)
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "convnext_tiny":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def build_model(
    name: str, num_classes: int = 7, pretrained: bool = True, weights: Any = None
) -> nn.Module:
    """Construct exactly one of the four approved architectures."""
    key = name.lower().replace("-", "_").replace("/", "_")
    aliases = {"customcnn": "custom_cnn", "vit_b_16": "vit_b16", "resnet_50": "resnet50"}
    key = aliases.get(key, key)
    if key not in MODEL_NAMES:
        raise ValueError(f"Unknown architecture {name!r}; expected one of {MODEL_NAMES}")
    model = (
        CustomCNN(num_classes)
        if key == "custom_cnn"
        else _torchvision_model(key, num_classes, pretrained, weights)
    )
    model._material_model_name = key  # type: ignore[attr-defined]
    model._material_num_classes = num_classes  # type: ignore[attr-defined]
    return model


def _vit_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # torchvision's ViT has no public feature method.  This mirrors its
    # forward path and returns the CLS embedding before the classification head.
    x = model._process_input(x)  # type: ignore[attr-defined]
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)  # type: ignore[attr-defined]
    x = torch.cat([batch_class_token, x], dim=1)
    x = model.encoder(x)  # type: ignore[attr-defined]
    return x[:, 0]


def _resnet_embeddings(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model.conv1(x)
    y = model.bn1(y)
    y = model.relu(y)
    y = model.maxpool(y)
    y = model.layer1(y)
    y = model.layer2(y)
    y = model.layer3(y)
    y = model.layer4(y)
    return model.avgpool(y).flatten(1)


def _convnext_embeddings(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ConvNeXt's 4-D LayerNorm2d before flattening.

    ``classifier[0]`` is LayerNorm2d in current torchvision. Applying it to a
    flattened tensor is a subtle shape/semantics bug and breaks newer builds.
    """
    pooled = model.avgpool(model.features(x))
    normalized = model.classifier[0](pooled)
    embedding = model.classifier[1](normalized)
    logits = model.classifier[2](embedding)
    return logits, embedding


def forward_with_embeddings(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logits and aligned penultimate embeddings in one backbone pass."""
    name = getattr(model, "_material_model_name", "")
    if name == "custom_cnn" or isinstance(model, CustomCNN):
        return model.forward_with_embeddings(x)  # type: ignore[attr-defined]
    if name == "resnet50" or hasattr(model, "layer4"):
        embedding = _resnet_embeddings(model, x)
        return model.fc(embedding), embedding
    if name == "convnext_tiny" or hasattr(model, "features") and hasattr(model, "classifier"):
        return _convnext_embeddings(model, x)
    if name == "vit_b16" or hasattr(model, "class_token"):
        embedding = _vit_features(model, x)
        return model.heads(embedding), embedding
    raise ValueError("Cannot identify model architecture for embedding extraction")


def extract_embeddings(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return penultimate features for all approved architectures."""
    return forward_with_embeddings(model, x)[1]


def classifier_parameter_names(model: nn.Module) -> set[str]:
    """Return parameter names belonging to the classification head."""
    prefixes = ("fc", "head", "heads", "classifier")
    return {name for name, _ in model.named_parameters() if name.split(".", 1)[0] in prefixes}
