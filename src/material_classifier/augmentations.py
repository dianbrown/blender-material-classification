"""Conservative, train-only image preprocessing for material renders."""

from __future__ import annotations

from typing import Any

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _transforms() -> Any:
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("torchvision is required to construct image transforms") from exc
    return transforms


def build_train_transform(input_size: int = 224) -> Any:
    """Build conservative augmentation applied only to training images."""

    transforms = _transforms()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(input_size: int = 224, resize_size: int = 256) -> Any:
    """Build deterministic validation/test preprocessing with no augmentation."""

    transforms = _transforms()
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
