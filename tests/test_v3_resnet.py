"""Offline CPU checks: real training loop, tiny injected backbone, no downloads."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, PngImagePlugin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3 import resnet_baseline as baseline
from material_classifier.v3.evaluation import evaluate_predictions, validate_predictions
from material_classifier.v3.features import json_hash, load_protocol, read_json, write_json
from material_classifier.v3.learning import grouped_folds, labels_for

ROOT = Path(__file__).resolve().parents[1]


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.fc = nn.Linear(8, len(CLASS_NAMES))

    def forward(self, images):
        return self.fc(self.layer4(self.stem(images.mean((2, 3)))))


def tiny_factory(config, *, offline):
    assert offline
    return TinyResNet(), {"enum": config["weights"], "url": "offline-test-only", "sha256": "a" * 64}


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not download weights")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    monkeypatch.setattr(baseline.ResNet50_Weights, "get_state_dict", forbidden)
    torch.set_num_threads(1)


@pytest.fixture
def config():
    return baseline.load_config()


@pytest.fixture
def protocol():
    return load_protocol(ROOT / "configs/v3/protocol.json")


@pytest.fixture
def development(tmp_path):
    records = []
    for label, name in enumerate(CLASS_NAMES):
        for group in range(5):
            for view in range(2):
                path = f"{name}_{group}_{view}.png"
                Image.new("RGB", (12, 14), (label * 20 + 1, group * 21 + 2, view * 31 + 3)).save(
                    tmp_path / path
                )
                records.append(
                    {
                        "image_path": path,
                        "class": name,
                        "material_id": f"{name}_{group}",
                        "source_group": f"family_{name}_{group}",
                        "original_split": "development",
                    }
                )
    frame = pd.DataFrame(records).sort_values("image_path").reset_index(drop=True)
    manifest = tmp_path / "development.csv"
    frame.to_csv(manifest, index=False)
    return manifest, frame


def test_fixed_config_and_preregistration(config, protocol):
    baseline.validate_config(config, protocol)
    assert config["epochs"] == 10 and config["head_only_epochs"] == 2
    assert config["weights"] == "ResNet50_Weights.IMAGENET1K_V2"
    smaller = baseline.load_config(physical_batch_size=8)
    assert smaller["accumulation_steps"] == 4
    assert json_hash(smaller) != json_hash(config)
    assert smaller["effective_batch_size"] == config["effective_batch_size"] == 32
    assert len(baseline.preregistration()["config_commit"]) == 40
    changed = copy.deepcopy(config)
    changed["epochs"] = 9
    with pytest.raises(ValueError, match="fixed recipe"):
        baseline.validate_config(changed, protocol)
    protocol["seed"] += 1
    with pytest.raises(ValueError, match="primary protocol"):
        baseline.validate_config(config, protocol)
    with pytest.raises(ValueError, match="16 or 8"):
        baseline.load_config(physical_batch_size=4)


def test_pretrained_loader_refuses_missing_or_corrupt_cache(config, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="unavailable offline"):
        baseline.load_pretrained_model(config, offline=True)
    checkpoint = tmp_path / "checkpoints/resnet50-11ad3fa6.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"invalid cached weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        baseline.load_pretrained_model(config, offline=True)


def test_group_isolation_and_primary_assignment_validation(development, protocol, tmp_path):
    _, frame = development
    folds, assignments = baseline.outer_plan(frame, protocol)
    reference = grouped_folds(labels_for(frame), frame["source_group"], 5, protocol["seed"])
    for (train, test), (expected_train, expected_test) in zip(folds, reference, strict=True):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
        assert not set(frame.iloc[train]["source_group"]) & set(frame.iloc[test]["source_group"])
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(len(frame)))
    stored = frame[["image_path", "class", "material_id", "source_group"]].copy()
    stored["outer_fold"] = assignments
    path = tmp_path / "fold_assignments.csv"
    stored.to_csv(path, index=False)
    baseline.outer_plan(frame, protocol, assignments_path=path)
    stored.loc[0, "outer_fold"] = (assignments[0] + 1) % 5
    stored.to_csv(path, index=False)
    with pytest.raises(ValueError, match="assignments disagree"):
        baseline.outer_plan(frame, protocol, assignments_path=path)
    with pytest.raises(ValueError, match="sorted manifest"):
        baseline.outer_plan(frame.iloc[::-1], protocol)


def test_imagenet_normalization_and_epoch_seed(config):
    image = Image.new("RGB", (260, 240), (64, 128, 192))
    transform = baseline.build_transform(config, training=False)
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
    expected = torch.tensor([64, 128, 192]) / 255 - torch.tensor([0.485, 0.456, 0.406])
    expected /= torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(tensor[:, 20, 20], expected)
    torch.testing.assert_close(tensor, baseline.ResNet50_Weights.IMAGENET1K_V2.transforms()(image))
    transform = baseline.build_transform(config, training=True)
    baseline.seed_epoch(81)
    first = transform(image)
    baseline.seed_epoch(81)
    torch.testing.assert_close(transform(image), first, rtol=0, atol=0)


def test_freeze_unfreeze_bn_and_explicit_learning_rates(config):
    model = TinyResNet()
    bn_buffers = {key: value.clone() for key, value in model.named_buffers()}
    optimizer = baseline.make_optimizer(model, config)
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 1e-5]
    baseline.set_training_phase(model, 0, config)
    assert all(p.requires_grad == name.startswith("fc.") for name, p in model.named_parameters())
    x = torch.randn(3, 3, 8, 8)
    assert torch.isfinite(model(x)).all()
    baseline.set_training_phase(model, 2, config)
    assert all(
        p.requires_grad == name.startswith(("fc.", "layer4."))
        for name, p in model.named_parameters()
    )
    before = {key: value.clone() for key, value in model.named_parameters()}
    nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2])).backward()
    optimizer.step()
    assert any(
        not torch.equal(before[key], value)
        for key, value in model.named_parameters()
        if key.startswith("layer4.")
    )
    assert all(
        torch.equal(before[key], value)
        for key, value in model.named_parameters()
        if key.startswith("stem.")
    )
    assert all(torch.equal(bn_buffers[key], value) for key, value in model.named_buffers())


@pytest.mark.parametrize("physical_batch_size", [16, 8])
def test_accumulation_matches_actual_batch_including_partial(physical_batch_size):
    config = baseline.load_config(physical_batch_size=physical_batch_size)
    torch.manual_seed(12)
    # 83 = 32+32+19. Last window contains differently sized microbatches.
    x, y = torch.randn(83, 3), torch.randint(0, 7, (83,))
    accumulated = nn.Linear(3, 7)
    reference = copy.deepcopy(accumulated)
    optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.03)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    loader = DataLoader(TensorDataset(x, y), batch_size=physical_batch_size, shuffle=False)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = baseline.train_epoch(
        accumulated, loader, optimizer, scaler, torch.device("cpu"), config
    )
    for start in range(0, len(x), 32):
        ref_optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            reference(x[start : start + 32]), y[start : start + 32], label_smoothing=0.05
        )
        loss.backward()
        nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        ref_optimizer.step()
    for actual, expected in zip(accumulated.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert metrics["optimizer_updates"] == 3 and metrics["images"] == 83
    assert np.isfinite(metrics["train_loss"])
    assert 0 <= metrics["train_accuracy"] <= 1


def test_nonfinite_training_fails(config):
    model = nn.Linear(3, 7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loader = DataLoader(TensorDataset(torch.full((2, 3), float("nan")), torch.tensor([0, 1])))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        baseline.train_epoch(
            model,
            loader,
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            torch.device("cpu"),
            config,
        )


@pytest.fixture
def small_images(monkeypatch):
    # Actual recipe/normalization is tested above. Reduce CPU pixel work only in
    # integration tests; the exact production epoch/optimizer/export code runs.
    def transform(config, *, training):
        spec = config["preprocessing"]
        return transforms.Compose(
            [
                transforms.Resize((8, 8)),
                transforms.ToTensor(),
                transforms.Normalize(spec["mean"], spec["std"]),
            ]
        )

    monkeypatch.setattr(baseline, "build_transform", transform)


def test_full_oof_boundary_fixed_budget_and_final_parity(
    config, protocol, development, tmp_path, monkeypatch, small_images
):
    manifest, frame = development
    folds, _ = baseline.outer_plan(frame, protocol)
    original_fit, original_predict = baseline.fit_fixed, baseline.predict_rows
    original_export = baseline.export_model
    seen_fits, seen_predictions = [], []

    def export(model, path, metadata):
        if len(seen_fits) <= 5:
            train, test = folds[len(seen_fits) - 1]
            assert metadata["training_source"]["material_ids"] == sorted(
                set(frame.iloc[train]["material_id"])
            )
            assert not set(metadata["training_source"]["material_ids"]) & set(
                frame.iloc[test]["material_id"]
            )
        sample = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
        expected = model(sample).detach()
        result = original_export(model, path, metadata)
        reloaded, _ = baseline.load_model(path, torch.device("cpu"), model_factory=TinyResNet)
        torch.testing.assert_close(reloaded(sample), expected, rtol=0, atol=0)
        return result

    def fit(model, train_frame, *args, **kwargs):
        if len(seen_fits) < 5:
            expected, excluded = folds[len(seen_fits)]
            assert train_frame["image_path"].tolist() == frame.iloc[expected]["image_path"].tolist()
            assert not set(train_frame["source_group"]) & set(frame.iloc[excluded]["source_group"])
        assert (Path(kwargs["output"]) / "fit.json").exists()
        history = original_fit(model, train_frame, *args, **kwargs)
        assert len(history) == 10
        assert [row["phase"] for row in history] == ["head"] * 2 + ["layer4_and_fc"] * 8
        assert history[0]["fc_lr"] == 3e-4
        assert history[0]["layer4_lr"] == 1e-5
        assert history[2]["layer4_lr"] == pytest.approx(1e-5 * (1 + np.cos(2 * np.pi / 10)) / 2)
        assert all(np.isfinite(row["train_loss"]) for row in history)
        seen_fits.append(train_frame["image_path"].tolist())
        return history

    def predict(model, test_frame, *args, **kwargs):
        assert len(seen_fits) == len(seen_predictions) + 1
        expected = folds[len(seen_predictions)][1]
        assert test_frame["image_path"].tolist() == frame.iloc[expected]["image_path"].tolist()
        seen_predictions.append(test_frame["image_path"].tolist())
        return original_predict(model, test_frame, *args, **kwargs)

    monkeypatch.setattr(baseline, "fit_fixed", fit)
    monkeypatch.setattr(baseline, "predict_rows", predict)
    monkeypatch.setattr(baseline, "export_model", export)
    output = tmp_path / "oof"
    result = baseline.train_run(
        manifest,
        tmp_path,
        output,
        config,
        protocol,
        stage="oof",
        device="cpu",
        cpu_threads=1,
        offline=True,
        model_factory=tiny_factory,
    )
    assert result["status"] == "complete"
    assert result["run_identity"]["training_source"]["material_ids"] == sorted(
        set(frame["material_id"])
    )
    assert len(seen_fits) == len(seen_predictions) == 5
    oof = pd.read_csv(output / "resnet_oof.csv")
    assert oof["image_path"].tolist() == frame["image_path"].tolist()
    assert all("p_" + name in oof for name in CLASS_NAMES)
    assert all((output / f"fold_{fold}/model.pt").exists() for fold in range(5))
    monkeypatch.setattr(baseline, "predict_rows", original_predict)

    final = tmp_path / "final-run"
    baseline.train_run(
        manifest,
        tmp_path,
        final,
        config,
        protocol,
        stage="final",
        device="cpu",
        cpu_threads=1,
        offline=True,
        model_factory=tiny_factory,
    )
    assert seen_fits[-1] == frame["image_path"].tolist()
    assert len(seen_fits) == 6 and len(seen_predictions) == 5
    assert not (final / "metrics.json").exists()
    artifact = final / "final/model.pt"
    model, metadata = baseline.load_model(artifact, torch.device("cpu"), model_factory=TinyResNet)
    assert metadata["training_source"]["source_groups"] == sorted(set(frame["source_group"]))
    assert metadata["training_source"]["material_ids"] == sorted(set(frame["material_id"]))
    assert metadata["training_source"]["rows"] == len(frame)
    probabilities = original_predict(model, frame, tmp_path, config, torch.device("cpu"))
    predicted = baseline.predict_manifest(
        artifact,
        manifest,
        tmp_path,
        tmp_path / "prediction",
        model_factory=TinyResNet,
        cpu_threads=1,
    )
    np.testing.assert_allclose(predicted[["p_" + name for name in CLASS_NAMES]], probabilities)
    assert set(
        validate_predictions(tmp_path / "prediction/predictions.csv", CLASS_NAMES).outer_fold
    ) == {"prediction_only"}

    # Save/reload state_dict has identical logits and carries no optimizer payload.
    reloaded, _ = baseline.load_model(artifact, torch.device("cpu"), model_factory=TinyResNet)
    sample = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(model(sample), reloaded(sample), rtol=0, atol=0)
    assert all(
        isinstance(value, torch.Tensor)
        for value in torch.load(artifact, weights_only=True).values()
    )
    with pytest.raises(ValueError, match="source families overlap"):
        baseline.predict_manifest(
            artifact,
            manifest,
            tmp_path,
            tmp_path / "bad-evaluation",
            evaluate=True,
            model_factory=TinyResNet,
        )
    with pytest.raises(ValueError, match="all-development final"):
        baseline.predict_manifest(
            output / "fold_0/model.pt",
            manifest,
            tmp_path,
            tmp_path / "bad-fold-evaluation",
            evaluate=True,
            model_factory=TinyResNet,
        )
    with pytest.raises(FileExistsError):
        baseline.train_run(
            manifest,
            tmp_path,
            final,
            config,
            protocol,
            stage="final",
            device="cpu",
            offline=True,
            model_factory=tiny_factory,
        )

    fresh = frame.iloc[:1].copy()
    fresh["source_group"] = "independent-fresh-family"
    fresh["image_path"] = "fresh.png"
    Image.new("RGB", (12, 14), (251, 252, 253)).save(tmp_path / "fresh.png")
    fresh_manifest = tmp_path / "fresh.csv"
    fresh.to_csv(fresh_manifest, index=False)
    with pytest.raises(ValueError, match="material IDs overlap"):
        baseline.predict_manifest(
            artifact,
            fresh_manifest,
            tmp_path,
            tmp_path / "bad-material",
            evaluate=True,
            model_factory=TinyResNet,
        )
    fresh["material_id"] = "independent-fresh-material"
    fresh.to_csv(fresh_manifest, index=False)
    fresh_output = tmp_path / "fresh-output"
    baseline.predict_manifest(
        artifact, fresh_manifest, tmp_path, fresh_output, evaluate=True, model_factory=TinyResNet
    )
    assert read_json(fresh_output / "prediction_run.json")["status"] == "complete"
    assert (fresh_output / "metrics.json").exists()
    csv = fresh_output / "predictions.csv"
    assert set(validate_predictions(csv, CLASS_NAMES).outer_fold) == {"holdout"}
    assert (
        evaluate_predictions(csv, CLASS_NAMES, estimate_type="fresh_holdout", n_bootstrap=0)[
            "estimate_kind"
        ]
        == "fresh_holdout"
    )

    # Renaming a training image or changing PNG metadata cannot pass as fresh.
    original_path = tmp_path / frame.iloc[0]["image_path"]
    (tmp_path / "fresh.png").write_bytes(original_path.read_bytes())
    with pytest.raises(ValueError, match="image content overlaps"):
        baseline.predict_manifest(
            artifact,
            fresh_manifest,
            tmp_path,
            tmp_path / "bad-renamed",
            evaluate=True,
            model_factory=TinyResNet,
        )
    with Image.open(original_path) as image:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("comment", "same pixels in a different file")
        image.save(tmp_path / "fresh.png", pnginfo=pnginfo)
    with pytest.raises(ValueError, match="decoded image content overlaps"):
        baseline.predict_manifest(
            artifact,
            fresh_manifest,
            tmp_path,
            tmp_path / "bad-reencoded",
            evaluate=True,
            model_factory=TinyResNet,
        )

    sidecar = artifact.with_suffix(".json")
    saved = read_json(sidecar)
    changed = copy.deepcopy(saved)
    changed["preprocessing"]["mean"][0] = 0.0
    write_json(sidecar, changed)
    with pytest.raises(ValueError, match="metadata checksum"):
        baseline.load_model(artifact, torch.device("cpu"), model_factory=TinyResNet)
    write_json(sidecar, saved)
    with artifact.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="state_dict checksum"):
        baseline.load_model(artifact, torch.device("cpu"), model_factory=TinyResNet)


def test_oom_is_failure_without_retry(config, protocol, development, tmp_path, monkeypatch):
    manifest, _ = development
    calls = []

    def oom(*args, **kwargs):
        calls.append(1)
        raise torch.OutOfMemoryError("simulated OOM")

    monkeypatch.setattr(baseline, "fit_fixed", oom)
    output = tmp_path / "failed"
    with pytest.raises(torch.OutOfMemoryError):
        baseline.train_run(
            manifest,
            tmp_path,
            output,
            config,
            protocol,
            stage="oof",
            device="cpu",
            offline=True,
            model_factory=tiny_factory,
        )
    assert len(calls) == 1
    assert read_json(output / "run.json")["status"] == "failed"
    assert read_json(output / "config.json")["physical_batch_size"] == 16
    assert not (output / "resnet_oof.csv").exists()


@pytest.mark.parametrize("key", ["source_groups", "material_ids", "sources", "decoded_rgb_sha256"])
@pytest.mark.parametrize("missing", [True, False])
def test_fresh_requires_complete_training_identity(tmp_path, development, key, missing):
    manifest, frame = development
    source = baseline.training_identity(manifest, frame, tmp_path)
    incomplete = copy.deepcopy(source)
    if missing:
        del incomplete[key]
    else:
        incomplete[key] = []
    with pytest.raises(ValueError, match="Missing.*identity"):
        baseline.check_fresh(frame, source, incomplete)
