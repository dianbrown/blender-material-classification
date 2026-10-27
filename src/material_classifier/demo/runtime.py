"""Lazy trusted-model loading and exact reviewed deployment inference paths."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from . import CLASS_NAMES, LOCK_PATH, MODEL_NAMES, DemoError

# Fixed locations, not request-controlled serialization paths.
ARTIFACTS = {
    "frozen_selected": (
        "artifacts/v3/run_final/model.joblib",
        "artifacts/v3/run_final/model_metadata.json",
    ),
    "frozen_control": (
        "artifacts/v3/run_final/control_model.joblib",
        "artifacts/v3/run_final/control_model_metadata.json",
    ),
    "resnet_fixed": (
        "artifacts/v3/resnet-final/final/model.pt",
        "artifacts/v3/resnet-final/final/model.json",
    ),
    "dinov2_finetuned": (
        "artifacts/v3/finetune/candidates/finetuned/model.pt",
        "artifacts/v3/finetune/candidates/finetuned/model.json",
    ),
}
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_EDGE = 8192
MAX_IMAGE_ASPECT = 32
LATENCY_SCOPE = (
    "Live single-image decode, exact preprocessing, device transfer, resident model forward, "
    "saved head and host probabilities, including CUDA synchronization. Excludes runtime/model "
    "loading, integrity checks, request queueing and JSON serialization. Not a study benchmark."
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_paths(root, model_id):
    """Check the final lock BEFORE any executable joblib deserialization."""
    root = Path(root).resolve()
    lock_path = root / LOCK_PATH
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DemoError(
            "missing_artifact", "The final model lock is missing from resources."
        ) from exc
    except (OSError, ValueError) as exc:
        raise DemoError("artifact_integrity", "The final model lock cannot be read.") from exc
    if lock["class_order"] != list(CLASS_NAMES) or lock["primary_model"] != "frozen_selected":
        raise DemoError(
            "artifact_integrity", "The model lock has an unexpected class/model identity."
        )
    record = lock["models"][model_id]
    paths = tuple(root / relative for relative in ARTIFACTS[model_id])
    for path, field, digest in zip(
        paths,
        ("artifact_path", "metadata_path"),
        ("artifact_sha256", "metadata_sha256"),
        strict=True,
    ):
        resolved = path.resolve()
        if (
            not resolved.is_relative_to(root)
            or (lock_path.parent / record[field]).resolve() != resolved
        ):
            raise DemoError(
                "artifact_integrity", "The locked model path is not a trusted resource."
            )
        if not resolved.is_file():
            raise DemoError("missing_artifact", f"Required model resource is missing: {path.name}.")
        if sha256(resolved) != record[digest]:
            raise DemoError(
                "artifact_integrity", f"Model resource failed SHA-256 check: {path.name}."
            )
    return paths, record


def check_image_path(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DemoError("invalid_image", "image_path must be an absolute local image filename.")
    if value.startswith(("\\\\", "//")):
        raise DemoError(
            "invalid_image", "Choose an absolute local file, not a network/device path."
        )
    path = Path(value)
    if not path.is_absolute():
        raise DemoError("invalid_image", "image_path must be absolute.")
    try:
        if not path.is_file():
            raise DemoError("invalid_image", "The selected image does not exist or is not a file.")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise DemoError("image_too_large", "Images must be at most 32 MiB.")
    except OSError as exc:
        raise DemoError("invalid_image", "The selected image cannot be read.") from exc
    return path


def decode_image(path):
    import warnings

    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if (
                    image.width * image.height > MAX_IMAGE_PIXELS
                    or max(image.size) > MAX_IMAGE_EDGE
                ):
                    raise DemoError(
                        "image_too_large", "Images must be at most 16 MP and 8192 px/edge."
                    )
                # Reviewed transforms resize the short edge before cropping. Bound
                # extreme aspect ratios so a tiny input cannot expand to gigabytes.
                if max(image.size) / min(image.size) > MAX_IMAGE_ASPECT:
                    raise DemoError("image_too_large", "Image aspect ratio must be at most 32:1.")
                if image.format not in ("PNG", "JPEG", "WEBP", "BMP", "TIFF"):
                    raise DemoError("invalid_image", "Choose a PNG, JPEG, WebP, BMP or TIFF image.")
                if getattr(image, "n_frames", 1) != 1:
                    raise DemoError("invalid_image", "Choose a single-frame image.")
                image.load()
                # Preserve EXIF until the exact reviewed family transform handles orientation.
                return image.copy()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise DemoError("image_too_large", "The image exceeds safe decoded dimensions.") from exc
    except (OSError, ValueError) as exc:
        raise DemoError(
            "invalid_image", "The selected file is not a readable supported image."
        ) from exc


class ModelRuntime:
    """Fixed bounded cache: four heads/models, with shared frozen feature backbones.

    No image features or predictions are cached: every request runs real inference.
    Construction imports no optional dependencies and does not inspect CUDA.
    """

    def __init__(self, root, device="auto"):
        self.root = Path(root).resolve()
        self.requested_device = device
        self.device = None
        self.models = {}
        self.backbones = {}

    def configure(self):
        if self.device is not None:
            return
        # These are set before huggingface_hub (or a reviewed loader) is imported.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        bundled_cache = self.root / "runtime/huggingface"
        if bundled_cache.is_dir():
            os.environ.setdefault("HF_HOME", str(bundled_cache))
        try:
            import torch

            from material_classifier.v3.features import configure_torch
        except (ImportError, OSError) as exc:
            raise DemoError(
                "missing_runtime", "The local ML runtime or its native libraries are unavailable."
            ) from exc

        requested = self.requested_device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested == "cuda" and not torch.cuda.is_available():
            raise DemoError("device_unavailable", "CUDA is unavailable; choose auto or cpu.")
        self.device = configure_torch(requested, threads=4)

    def _backbone(self, spec, size):
        from material_classifier.v3.features import load_frozen_model

        key = (json.dumps(spec, sort_keys=True), size)
        if key not in self.backbones:
            self.backbones[key] = load_frozen_model(spec, size, device=self.device, offline=True)
        return self.backbones[key]

    def load(self, model_id):
        if model_id in self.models:
            return self.models[model_id]
        (artifact, _), record = verified_paths(self.root, model_id)
        if model_id.startswith("frozen_"):
            from material_classifier.v3.features import jobs_for_protocol
            from material_classifier.v3.inference import _check_feature_runtime, load_artifact

            bundle = load_artifact(artifact)
            metadata = bundle["metadata"]
            _check_feature_runtime(metadata)
            jobs = [
                (prefix, spec, size, self._backbone(spec, size))
                for prefix, spec, size in jobs_for_protocol(
                    metadata["protocol"], metadata["recipe"]["parts"]
                )
            ]
            loaded = (bundle, jobs)
        elif model_id == "resnet_fixed":
            from material_classifier.v3.resnet_baseline import build_transform, load_model

            model, metadata = load_model(artifact, self.device)
            loaded = (model, build_transform(metadata["config"], training=False))
        else:
            from material_classifier.v3.finetune import load_deployed

            loaded = load_deployed(
                artifact,
                self.device,
                expected_sha256=record["artifact_sha256"],
                metadata_sha256=record["metadata_sha256"],
            )
        self.models[model_id] = loaded
        return loaded

    def _probabilities(self, model_id, loaded, image):
        import torch
        from PIL import ImageOps
        from threadpoolctl import threadpool_limits

        from material_classifier.v3.features import (
            assemble_features,
            extract_batch,
            transform_image,
        )
        from material_classifier.v3.learning import class_probabilities

        with torch.inference_mode(), threadpool_limits(limits=4):
            if model_id.startswith("frozen_"):
                bundle, jobs = loaded
                base = {}
                for prefix, spec, size, model in jobs:
                    batch = transform_image(image, spec, size)[None].to(self.device)
                    base.update(
                        {
                            prefix + "_" + key: value
                            for key, value in extract_batch(model, batch, spec, size).items()
                        }
                    )
                metadata = bundle["metadata"]
                features = assemble_features(base, metadata["recipe"]["parts"])
                if features.shape[1] != metadata["input_features"]:
                    raise ValueError("Extracted image feature width differs from final head")
                return class_probabilities(bundle["pipeline"], features)[0]
            model, transform_or_metadata = loaded
            if model_id == "resnet_fixed":
                tensor = transform_or_metadata(ImageOps.exif_transpose(image).convert("RGB"))
                logits = model(tensor[None].to(self.device))
                return logits.double().softmax(1).cpu().numpy()[0]
            metadata = transform_or_metadata
            tensor = transform_image(
                image, metadata["architecture"], metadata["config"]["input_size"]
            )
            logits = model(tensor[None].to(self.device))
            probabilities = logits.softmax(1).double().cpu().numpy()[0]
            probabilities /= probabilities.sum()
            return probabilities

    def synchronize(self):
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def predict(self, model_id, image_path):
        if model_id not in MODEL_NAMES:
            raise DemoError("unsupported_model", "Choose one of the four bundled study models.")
        path = check_image_path(image_path)
        # Reject corrupt/oversized images before importing torch or loading weights.
        start = time.perf_counter()
        image = decode_image(path)
        decode_ms = (time.perf_counter() - start) * 1000
        try:
            cached = model_id in self.models
            start = time.perf_counter()
            self.configure()
            loaded = self.load(model_id)
            self.synchronize()
            load_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            probabilities = self._probabilities(model_id, loaded, image)
            self.synchronize()
            inference_ms = decode_ms + (time.perf_counter() - start) * 1000
        finally:
            image.close()
        values = [float(p) for p in probabilities]
        import math

        if (
            len(values) != len(CLASS_NAMES)
            or any(not math.isfinite(p) or p < 0 for p in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-6)
        ):
            raise DemoError(
                "invalid_probabilities", "The model returned invalid class probabilities."
            )
        return {
            "model": model_id,
            "class_names": list(CLASS_NAMES),
            "probabilities": values,
            "predicted_class": CLASS_NAMES[max(range(len(values)), key=values.__getitem__)],
            "inference_ms": inference_ms,
            "device": str(self.device),
            "load_ms": load_ms,
            "model_cached": cached,
            "latency_scope": LATENCY_SCOPE,
        }
