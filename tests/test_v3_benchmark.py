"""CPU stubs only: no downloaded models, CUDA execution, accuracy or selection."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from material_classifier.v3 import CLASS_NAMES, features, finetune, inference, resnet_baseline
from material_classifier.v3 import benchmark as bm
from material_classifier.v3.learning import class_probabilities

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_cpu_only(monkeypatch):
    import huggingface_hub
    import timm

    def forbidden(*args, **kwargs):
        raise AssertionError("No downloads, real model construction or GPU execution in tests")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(timm, "create_model", forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    monkeypatch.setattr(resnet_baseline, "resnet50", forbidden)
    for name in (
        "synchronize",
        "empty_cache",
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    torch.set_num_threads(1)


class TinyDino(nn.Module):
    num_prefix_tokens = 5

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.scale = nn.Parameter(torch.ones(3))
        # PartialDino checks the block inventory; these have no trainable state.
        self.blocks = nn.Sequential(nn.Identity(), nn.Identity())
        self.norm = nn.Identity()

    def forward_features(self, batch):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        assert batch.shape[-2:] == (self.size, self.size)
        signal = batch.mean((2, 3)) * self.scale
        count = (self.size // 2) ** 2
        return torch.cat(
            [
                2 * signal[:, None],
                torch.full((len(batch), 4, 3), 10000.0),
                signal[:, None].expand(-1, count, -1),
            ],
            dim=1,
        )


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.full((3,), 2.0))

    def forward_features(self, batch):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        return batch * self.scale[None, :, None, None]

    def forward_head(self, features, pre_logits):
        assert pre_logits
        return features.mean((2, 3))


class TinyResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.fc = nn.Linear(4, len(CLASS_NAMES))

    def forward(self, batch):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        return self.fc(self.stem(batch.mean((2, 3))))


@pytest.fixture
def corpus(tmp_path):
    rows = []
    for index in range(4):
        path = f"image_{index}.png"
        pixels = np.arange(13 * 17 * 3, dtype=np.uint8).reshape(13, 17, 3)
        Image.fromarray(np.roll(pixels, index, axis=1)).save(tmp_path / path)
        rows.append(
            {
                "image_path": path,
                "class": CLASS_NAMES[index],
                "material_id": f"m{index}",
                "source_group": f"g{index}",
                # Historical 'test' is development according to the immutable provenance.
                "original_split": "test",
            }
        )
    frame = pd.DataFrame(rows)
    manifest = tmp_path / "development.csv"
    frame.to_csv(manifest, index=False)
    return manifest, frame, features.source_identity(manifest, frame, tmp_path)


def fixed_pipeline(width):
    """Install deterministic fitted arrays directly; never invoke fit or select."""
    classifier = LogisticRegression()
    classifier.classes_ = np.arange(len(CLASS_NAMES))
    classifier.coef_ = np.arange(len(CLASS_NAMES) * width).reshape(7, width) / 100.0
    classifier.intercept_ = np.arange(7) / 10.0
    classifier.n_features_in_ = width
    scaler = StandardScaler()
    scaler.mean_, scaler.var_, scaler.scale_ = np.zeros(width), np.ones(width), np.ones(width)
    scaler.n_features_in_, scaler.n_samples_seen_ = width, 7
    return Pipeline([("scaler", scaler), ("classifier", classifier)])


@pytest.fixture
def frozen(tmp_path, corpus, monkeypatch):
    protocol = features.read_json(ROOT / "configs/v3/protocol.json")
    protocol["models"]["dino"].update(width=3, patch_size=2, resolutions=[8, 12])
    protocol["models"]["convnext"].update(width=3, resolutions=[8])
    recipe = {
        "name": "offline-test",
        "parts": [
            "dino_8_cls",
            "dino_8_mean",
            "dino_8_std",
            "dino_12_mean",
            "convnext_8_pooled",
        ],
    }
    protocol["candidates"] = [recipe]
    models = {"dino_8": TinyDino(8), "dino_12": TinyDino(12), "convnext_8": TinyConv()}
    paths = {}
    for name, spec in protocol["models"].items():
        path = tmp_path / f"{name}.safetensors"
        save_file(models[f"{name}_8"].state_dict(), path)
        spec["weights_sha256"] = features.file_hash(path)
        paths[spec["repository"]] = path
    metadata = {
        "class_order": list(CLASS_NAMES),
        "protocol": protocol,
        "protocol_sha256": features.json_hash(protocol),
        "recipe": recipe,
        "feature_implementation_sha256": features.implementation_hash(),
        "feature_environment": features.environment(),
        "learning_environment": features.environment(),
        "training_source": corpus[2],
        "input_features": 15,
    }
    bundle = {"metadata": metadata, "pipeline": fixed_pipeline(15)}
    artifact = tmp_path / "selected.joblib"
    joblib.dump(bundle, artifact)
    features.write_json(
        artifact.with_name("selected_metadata.json"),
        {
            **metadata,
            "model_sha256": features.file_hash(artifact),
        },
    )
    loads = []

    def factory(spec, size, device, offline):
        assert offline and torch.device(device).type == "cpu"
        loads.append((spec["name"], size))
        return TinyDino(size) if "patch_size" in spec else TinyConv()

    def cached(repository, filename, *, revision, local_files_only):
        assert local_files_only and revision and filename == "model.safetensors"
        return str(paths[repository])

    import huggingface_hub

    monkeypatch.setattr(features, "load_frozen_model", factory)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached)
    return artifact, bundle, models, loads


def test_frozen_warm_parity_and_prefix_exclusion(frozen, corpus, tmp_path):
    artifact, bundle, models, loads = frozen
    backend = bm.load_backend("frozen", artifact, torch.device("cpu"))
    images, _ = bm.development_images(corpus[0], tmp_path, bundle["metadata"], 2, bm.DEFAULT_SEED)
    actual = backend.predict(images)
    # Independent recipe assembly through the public feature functions.
    base = {}
    for prefix, spec, size in backend.jobs:
        model = models[prefix].eval().requires_grad_(False)
        tensors = torch.stack([features.transform_image(image, spec, size) for image in images])
        values = features.extract_batch(model, tensors, spec, size)
        if "patch_size" in spec:
            assert np.max(np.abs(values["mean"])) < 10  # registers are 10,000
        base.update({prefix + "_" + key: value for key, value in values.items()})
    expected = class_probabilities(
        bundle["pipeline"], features.assemble_features(base, bundle["metadata"]["recipe"]["parts"])
    )
    np.testing.assert_array_equal(actual, expected)
    for image, row in zip(images, actual, strict=True):
        path = tmp_path / "parity.png"
        image.save(path)
        reference = inference.predict_image(
            artifact,
            path,
            device="cpu",
            cpu_threads=1,
            offline=True,
            model_factory=features.load_frozen_model,
        )
        np.testing.assert_allclose(row, list(reference["probabilities"].values()), atol=1e-14)
    before = len(loads)
    result = bm.measure_warm(backend, images, warmup=1, repetitions=2)
    assert len(loads) == before
    assert result["measurement_kind"] == "backend_only_unverified"
    assert result["peak_allocated_bytes"] is None


@pytest.mark.parametrize("family", ["resnet", "dinov2"])
def test_neural_warm_parity(family, corpus, tmp_path, monkeypatch):
    artifact = tmp_path / f"{family}.pt"
    if family == "resnet":
        config = resnet_baseline.load_config()
        model = TinyResnet()
        metadata = {
            "config": config,
            "config_sha256": features.json_hash(config),
            "preprocessing": config["preprocessing"],
            "class_order": list(CLASS_NAMES),
            "training_stage": "final",
            "training_source": corpus[2],
            "implementation": resnet_baseline.implementation_identity(),
            "environment": features.environment(),
        }
        resnet_baseline.export_model(model, artifact, metadata)

        def construct(*, weights, num_classes):
            assert weights is None and num_classes == 7
            return TinyResnet()

        monkeypatch.setattr(resnet_baseline, "resnet50", construct)
    else:
        import timm

        config = features.read_json(ROOT / "configs/v3/finetune.json")
        config.update(input_size=8, batch_size=2)
        spec = features.read_json(ROOT / "configs/v3/protocol.json")["models"]["dino"]
        spec.update(width=3, patch_size=2)
        model = finetune.PartialDino(TinyDino(8), spec, config)
        torch.save(model.state_dict(), artifact)
        metadata = {
            "kind": "exploratory-partial-dinov2",
            "config": config,
            "config_sha256": features.json_hash(config),
            "architecture": spec,
            "class_order": list(CLASS_NAMES),
            "training_source": corpus[2],
            "source_hashes": finetune.source_hashes(),
            "environment": features.environment(),
            "model_sha256": features.file_hash(artifact),
        }
        features.write_json(artifact.with_suffix(".json"), metadata)

        def construct(name, *, pretrained, num_classes, img_size):
            assert name == spec["name"] and not pretrained and num_classes == 0 and img_size == 8
            return TinyDino(img_size)

        monkeypatch.setattr(timm, "create_model", construct)
    backend = bm.load_backend(family, artifact, "cpu")
    images = [Image.open(tmp_path / relative).convert("RGB") for relative in corpus[1].image_path]
    actual = backend.predict(images)
    if family == "resnet":
        expected = resnet_baseline.predict_rows(
            model, corpus[1], tmp_path, config, torch.device("cpu"), batch_size=4
        )
    else:
        finetune.predict_finetuned(
            artifact, tmp_path, tmp_path / "reference", manifest=corpus[0], cpu_threads=1
        )
        predictions = pd.read_csv(tmp_path / "reference/predictions.csv")
        expected = predictions[["p_" + name for name in CLASS_NAMES]].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    assert not backend.model.training
    assert all(not p.requires_grad for p in backend.model.parameters())
    inventory = bm.deployment_inventory(family, artifact, backend)
    assert inventory["pretrained_weight_files"] == []
    assert inventory["head_parameters"] == sum(
        p.numel() for p in (model.fc if family == "resnet" else model.head).parameters()
    )
    assert inventory["runtime_unique_parameters"] == sum(p.numel() for p in model.parameters())
    artifact.write_bytes(b"corrupt neural state")
    with pytest.raises(ValueError, match="checksum"):
        bm.load_backend(family, artifact, "cpu")


def test_statistics_seconds_and_throughput():
    result = bm.statistics([0.001, 0.002, 0.003], 16)
    assert result["median_seconds"] == pytest.approx(0.002)
    assert result["mean_seconds"] == pytest.approx(0.002)
    assert result["p95_seconds"] == pytest.approx(0.0029)
    assert result["std_seconds"] == pytest.approx(np.std([0.001, 0.002, 0.003]))
    assert result["images_per_second"] == pytest.approx(8000)
    for samples in ([], [0], [-1], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="seconds"):
            bm.statistics(samples, 1)


def test_cuda_synchronization_and_peak_scope_are_mocked(monkeypatch):
    events = []
    device = torch.device("cuda:2")
    for name, value in (
        ("synchronize", None),
        ("reset_peak_memory_stats", None),
        ("memory_allocated", 10),
        ("memory_reserved", 20),
        ("max_memory_allocated", 30),
        ("max_memory_reserved", 40),
    ):

        def mocked(actual_device, name=name, value=value):
            assert actual_device == device
            events.append(name)
            return value

        monkeypatch.setattr(torch.cuda, name, mocked)
    times = iter([100.0, 100.25, 200.0, 200.75])

    def clock():
        events.append("clock")
        return next(times)

    def predict(images):
        events.append("predict")
        return np.full((len(images), 7), 1 / 7)

    monkeypatch.setattr(bm.time, "perf_counter", clock)
    result = bm.measure_warm(
        SimpleNamespace(device=device, predict=predict), [object()], warmup=1, repetitions=2
    )
    assert events == [
        "predict",
        "synchronize",
        "memory_allocated",
        "memory_reserved",
        "reset_peak_memory_stats",
        "synchronize",
        "clock",
        "predict",
        "synchronize",
        "clock",
        "synchronize",
        "clock",
        "predict",
        "synchronize",
        "clock",
        "max_memory_allocated",
        "max_memory_reserved",
    ]
    assert result["samples_seconds"] == [0.25, 0.75]
    assert result["peak_allocated_bytes"] == 30 and result["peak_reserved_bytes"] == 40
    assert result["memory_before_trials"] == {"allocated_bytes": 10, "reserved_bytes": 20}


def test_frozen_deployment_bytes_and_parameter_deduplication(frozen):
    artifact, bundle, models, _ = frozen
    inventory = bm.deployment_inventory("frozen", artifact, bm.FrozenBackend(bundle, models, "cpu"))
    assert inventory["artifact_role"] == "classifier_and_scaler_only"
    assert len(inventory["pretrained_weight_files"]) == 2
    dino = next(r for r in inventory["pretrained_weight_files"] if len(r["runtime_instances"]) == 2)
    assert dino["runtime_instances"] == ["dino_8", "dino_12"]
    assert inventory["unique_pretrained_weight_bytes"] == sum(
        Path(r["path"]).stat().st_size for r in inventory["pretrained_weight_files"]
    )
    assert inventory["total_deployment_bytes"] == (
        artifact.stat().st_size
        + artifact.with_name("selected_metadata.json").stat().st_size
        + inventory["unique_pretrained_weight_bytes"]
    )
    assert inventory["runtime_backbone_instances"] == 3
    assert inventory["head_parameters"] == 7 * (15 + 1)
    assert inventory["runtime_unique_parameters"] == 9 + 7 * 16
    assert inventory["unique_source_parameters_including_head"] == 6 + 7 * 16


def test_exact_development_only_and_deterministic_selection(corpus, tmp_path):
    metadata = {"training_source": corpus[2]}
    images, selection = bm.development_images(corpus[0], tmp_path, metadata, 2, 123)
    repeated, other = bm.development_images(corpus[0], tmp_path, metadata, 2, 123)
    assert selection == other and all(image.mode == "RGB" for image in images)
    assert [i.tobytes() for i in images] == [i.tobytes() for i in repeated]
    wrong = copy.deepcopy(metadata)
    wrong["training_source"]["manifest_sha256"] = "holdout"
    with pytest.raises(ValueError, match="never holdout"):
        bm.development_images(corpus[0], tmp_path, wrong, 2, 123)
    with pytest.raises(ValueError, match="smaller"):
        bm.development_images(corpus[0], tmp_path, metadata, 16, 123)
    Image.new("RGB", (17, 13)).save(tmp_path / corpus[1].iloc[0].image_path)
    with pytest.raises(ValueError, match="content"):
        bm.development_images(corpus[0], tmp_path, metadata, 2, 123)


@pytest.mark.parametrize("cold", [0, 3])
def test_small_json_csv_and_resident_models(frozen, corpus, tmp_path, cold):
    artifact, _, _, loads = frozen
    output = tmp_path / f"benchmark_{cold}"
    result = bm.benchmark(
        "frozen",
        artifact,
        corpus[0],
        tmp_path,
        output,
        device="cpu",
        cpu_threads=1,
        batches=(1, 2),
        warmup=1,
        repetitions=2,
        cold_repetitions=cold,
    )
    assert result["status"] == "complete"
    assert len(loads) == 3 * (5 if cold else 1)
    assert len(result["results"]) == (3 if cold else 2)
    assert features.read_json(output / "benchmark.json") == result
    table = pd.read_csv(output / "benchmark.csv")
    assert len(table) == len(result["results"])
    assert table["artifact_bytes"].eq(artifact.stat().st_size).all()
    assert result["csv_sha256"] == features.file_hash(output / "benchmark.csv")
    assert result["hardware"]["device"] == "cpu"
    assert "GPU" not in result["hardware"]
    assert "Not full CLI startup" in result["scopes"]["process_warm_reload"]
    assert len(result["development"]["selected"]) == 2
    assert result["provenance"]["files"]["src/material_classifier/v3/benchmark.py"]
    with pytest.raises(FileExistsError, match="new directory"):
        bm.benchmark("frozen", artifact, corpus[0], tmp_path, output, device="cpu")


def test_no_disk_io_in_warm(frozen, corpus, tmp_path, monkeypatch):
    artifact, bundle, _, _ = frozen
    backend = bm.load_backend("frozen", artifact, "cpu")
    images, _ = bm.development_images(corpus[0], tmp_path, bundle["metadata"], 2, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm path tried to read disk or load a model")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(features, "file_hash", forbidden)
    monkeypatch.setattr(features, "load_frozen_model", forbidden)
    bm.measure_warm(backend, images, warmup=1, repetitions=2)


def test_validation_failures(frozen, corpus, tmp_path, monkeypatch):
    artifact, bundle, models, _ = frozen
    with pytest.raises(ValueError, match="exactly"):
        bm.FrozenBackend(bundle, {}, "cpu")
    with pytest.raises(ValueError, match="explicitly"):
        bm.load_backend("auto", artifact, "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        bm.benchmark("frozen", artifact, corpus[0], tmp_path, tmp_path / "gpu", device="cuda")
    assert not (tmp_path / "gpu").exists()
    for kwargs in (
        {"batches": (0,)},
        {"batches": (1, 1)},
        {"warmup": 0},
        {"repetitions": 0},
        {"cpu_threads": 0},
        {"cold_repetitions": 1},
        {"seed": -1},
    ):
        with pytest.raises(ValueError, match="Invalid"):
            bm.benchmark(
                "frozen", artifact, corpus[0], tmp_path, tmp_path / "bad", device="cpu", **kwargs
            )
    wrong = copy.deepcopy(bundle)
    wrong["metadata"]["input_features"] = 100
    with pytest.raises(ValueError, match="width"):
        bm.FrozenBackend(wrong, models, "cpu").predict([Image.new("RGB", (16, 16))])
    wrong = copy.deepcopy(bundle)
    wrong["pipeline"].named_steps["classifier"].classes_ = np.arange(7)[::-1]
    with pytest.raises(ValueError, match="class order"):
        bm.FrozenBackend(wrong, models, "cpu").predict([Image.new("RGB", (16, 16))])
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        bm.benchmark("frozen", artifact, corpus[0], tmp_path, tmp_path / "corrupt", device="cpu")
    failed = features.read_json(tmp_path / "corrupt/benchmark.json")
    assert failed["status"] == "failed" and failed["results"] == []
    assert not (tmp_path / "corrupt/benchmark.csv").exists()


def test_feature_runtime_and_weight_hash_failures(frozen, monkeypatch):
    artifact, bundle, models, _ = frozen
    changed = features.environment()
    changed["packages"]["timm"] = "different"
    monkeypatch.setattr(inference, "environment", lambda: changed)
    with pytest.raises(ValueError, match="feature runtime"):
        bm.load_backend("frozen", artifact, "cpu")
    spec = bundle["metadata"]["protocol"]["models"]["dino"]
    spec["weights_sha256"] = "bad"
    with pytest.raises(ValueError, match="weights checksum"):
        bm.deployment_inventory("frozen", artifact, bm.FrozenBackend(bundle, models, "cpu"))


def test_missing_cache_no_retry_or_fallback(frozen, corpus, tmp_path, monkeypatch):
    attempts = []

    def missing(*args, **kwargs):
        attempts.append(kwargs)
        raise FileNotFoundError("Pinned cache unavailable offline")

    monkeypatch.setattr(features, "load_frozen_model", missing)
    with pytest.raises(FileNotFoundError, match="offline"):
        bm.benchmark(
            "frozen",
            frozen[0],
            corpus[0],
            tmp_path,
            tmp_path / "missing",
            device="cpu",
            batches=(1,),
            warmup=1,
            repetitions=1,
        )
    assert len(attempts) == 1 and attempts[0]["offline"] is True
    assert features.read_json(tmp_path / "missing/benchmark.json")["status"] == "failed"


def test_oom_and_bad_probabilities_abort_without_retry():
    attempts = []

    def oom(images):
        attempts.append(len(images))
        raise torch.cuda.OutOfMemoryError("simulated, no GPU execution")

    backend = SimpleNamespace(device=torch.device("cpu"), predict=oom)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated"):
        bm.measure_warm(backend, [object(), object()], warmup=1, repetitions=2)
    assert attempts == [2]
    backend.predict = lambda images: np.zeros((len(images), 7))
    with pytest.raises(ValueError, match="probabilities"):
        bm.measure_warm(backend, [object()], warmup=1, repetitions=2)


def test_content_mutation_fails_without_summary_csv(frozen, corpus, tmp_path, monkeypatch):
    original = bm.measure_warm

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        Image.new("RGB", (13, 17), "red").save(tmp_path / corpus[1].iloc[0].image_path)
        return result

    monkeypatch.setattr(bm, "measure_warm", mutate)
    with pytest.raises(ValueError, match="content changed"):
        bm.benchmark(
            "frozen",
            frozen[0],
            corpus[0],
            tmp_path,
            tmp_path / "changed",
            device="cpu",
            batches=(1,),
            warmup=1,
            repetitions=1,
        )
    assert features.read_json(tmp_path / "changed/benchmark.json")["status"] == "failed"
    assert not (tmp_path / "changed/benchmark.csv").exists()


def test_cli_defaults_and_explicit_choices(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "v3_benchmark_cli", ROOT / "scripts/v3/benchmark.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "benchmark", run)
    args = [
        "--family",
        "frozen",
        "--artifact",
        "model.joblib",
        "--development-manifest",
        "dev.csv",
        "--data-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "output"),
        "--device",
        "cpu",
    ]
    cli.main(args)
    assert captured["batches"] == [1, 16] and captured["cpu_threads"] == 4
    assert captured["warmup"] == 5 and captured["repetitions"] == 30
    assert captured["seed"] == bm.DEFAULT_SEED and captured["cold_repetitions"] == 0
    with pytest.raises(SystemExit):
        cli.main(args[:-2])
