"""CPU-only tests with an injected tiny backbone; forbid all pretrained downloads."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageOps, PngImagePlugin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3 import finetune as ft
from material_classifier.v3.evaluation import evaluate_predictions, validate_predictions
from material_classifier.v3.features import (
    environment,
    file_hash,
    json_hash,
    read_json,
    read_manifest,
    source_identity,
    write_json,
)
from material_classifier.v3.learning import fold_record, grouped_folds, labels_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import huggingface_hub
    import timm

    def forbidden(*args, **kwargs):
        raise AssertionError("Test attempted pretrained download/model construction")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(timm, "create_model", forbidden)
    torch.set_num_threads(1)


@pytest.fixture
def config():
    return read_json(ROOT / "configs/v3/finetune.json")


@pytest.fixture
def protocol():
    return read_json(ROOT / "configs/v3/protocol.json")


class DummyBackbone(nn.Module):
    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(3, 3)
        self.blocks = nn.Sequential(*[nn.Linear(3, 3) for _ in range(4)])
        self.norm = nn.LayerNorm(3)

    def forward_features(self, batch):
        signal = self.norm(self.blocks(self.patch_embed(batch.mean(dim=(2, 3)))))
        registers = torch.full((len(batch), 4, 3), 10000.0, device=batch.device)
        return torch.cat([signal[:, None] * 2, registers, signal[:, None].expand(-1, 4, -1)], dim=1)


def factory(spec, size, device="cpu", offline=True):
    assert offline
    return DummyBackbone().to(device)


@pytest.fixture
def tiny(config, protocol):
    config = copy.deepcopy(config)
    config.update(input_size=8, max_epochs=3)
    spec = copy.deepcopy(protocol["models"]["dino"])
    spec.update(width=3, patch_size=4)
    return spec, config


@pytest.fixture
def corpus(tmp_path):
    rows = []
    for label, name in enumerate(CLASS_NAMES):
        for group in range(3):
            filename = f"{name}_{group}.png"
            Image.new("RGB", (10, 12), (20 + label * 10, 30 + group * 30, 120)).save(
                tmp_path / filename
            )
            rows.append(
                {
                    "image_path": filename,
                    "class": name,
                    "source_group": f"{name}_{group}",
                    "material_id": f"{name}_{group}",
                    "original_split": "development",
                }
            )
    frame = pd.DataFrame(rows).sort_values("image_path").reset_index(drop=True)
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_bound_config(config, protocol):
    ft.validate_config(config, protocol)
    config["batch_size"] = 4
    with pytest.raises(ValueError, match="batch_size"):
        ft.validate_config(config, protocol)


def test_prefix_exclusion_and_trainable_policy(tiny):
    spec, config = tiny
    model = ft.PartialDino(factory(spec, 8), spec, config)
    captured = []
    model.head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    model.eval()
    batch = torch.ones(2, 3, 8, 8)
    tokens = model.backbone.forward_features(batch)
    model(batch)
    assert torch.allclose(captured[0], torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=1))
    for epoch in (1, 2):
        model.set_epoch(epoch)
        assert {n for n, p in model.named_parameters() if p.requires_grad} == {
            "head.weight",
            "head.bias",
        }
    model.set_epoch(3)
    allowed = ("head.", "backbone.blocks.2.", "backbone.blocks.3.", "backbone.norm.")
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(allowed)
    model.train()
    assert not model.backbone.blocks[0].training
    assert model.backbone.blocks[3].training
    optimizer = ft.make_optimizer(model, config, 3e-6)
    by_id = {id(p): name for name, p in model.named_parameters()}
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = by_id[id(param)]
            assert name.startswith(allowed)
            assert group["weight_decay"] == (0 if param.ndim <= 1 else 0.01)
            assert group["lr"] == (3e-4 if name.startswith("head.") else 3e-6)


@pytest.mark.parametrize("count", [1, 7, 8, 9, 31, 32, 33, 45, 65])
def test_accumulation_matches_actual_effective_batches(config, count):
    torch.manual_seed(3)
    model = nn.Linear(3, 7)
    reference = copy.deepcopy(model)
    x, y = torch.randn(count, 3), torch.arange(count) % 7
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
    metrics = ft.run_epoch(
        model,
        DataLoader(TensorDataset(x, y), batch_size=8),
        torch.device("cpu"),
        config,
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
    )
    for start in range(0, count, 32):
        expected_optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            reference(x[start : start + 32]), y[start : start + 32], label_smoothing=0.05
        )
        loss.backward()
        nn.utils.clip_grad_norm_(reference.parameters(), 1)
        expected_optimizer.step()
    assert metrics["optimizer_steps"] == (count + 31) // 32
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_fold_identity_and_sorted_rows(corpus, protocol):
    _, frame = corpus
    y, groups = labels_for(frame), frame["source_group"].to_numpy()
    folds = grouped_folds(y, groups, 3, protocol["seed"] + 1000)
    saved = fold_record(folds, np.arange(len(y)), groups, y)
    actual = ft.verify_folds(frame, saved, protocol)
    for expected, received in zip(folds, actual, strict=True):
        np.testing.assert_array_equal(expected[1], received[1])
    bad = copy.deepcopy(saved)
    bad[0]["validation_rows"][0] += 1
    with pytest.raises(ValueError, match="exact frozen"):
        ft.verify_folds(frame, bad, protocol)
    with pytest.raises(ValueError, match="sorted"):
        ft.verify_folds(frame.iloc[::-1], saved, protocol)


def test_amp_overflow_skips_update_reduces_scale_and_recovers(config):
    class HalfLinear(nn.Linear):
        def forward(self, batch):
            with torch.autocast("cpu", dtype=torch.float16):
                logits = super().forward(batch)
            return logits.float()

    model = HalfLinear(3, len(CLASS_NAMES))
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    before = ft.cpu_state(model)
    loader = DataLoader(TensorDataset(torch.ones(1, 3), torch.zeros(1, dtype=torch.long)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=131072)

    metrics = ft.run_epoch(model, loader, torch.device("cpu"), config, optimizer, scaler)
    assert np.isfinite(metrics["loss"])
    assert metrics["attempted_optimizer_steps"] == metrics["amp_skipped_steps"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["amp_scale"] == scaler.get_scale() == 65536
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    # Resume must retain the reduced scale, allowing a subsequent finite update.
    resumed_scaler = torch.amp.GradScaler("cpu")
    resumed_scaler.load_state_dict(scaler.state_dict())
    metrics = ft.run_epoch(model, loader, torch.device("cpu"), config, optimizer, resumed_scaler)
    assert metrics["attempted_optimizer_steps"] == metrics["optimizer_steps"] == 1
    assert metrics["amp_skipped_steps"] == 0
    assert metrics["amp_scale"] == 65536
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(torch.isfinite(value).all() for value in model.parameters())


def test_disabled_scaler_nonfinite_gradients_still_fail(config):
    model = nn.Linear(3, len(CLASS_NAMES))
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    loader = DataLoader(TensorDataset(torch.ones(1, 3), torch.zeros(1, dtype=torch.long)))
    with pytest.raises(RuntimeError, match="non-finite"):
        ft.run_epoch(
            model,
            loader,
            torch.device("cpu"),
            config,
            torch.optim.SGD(model.parameters(), lr=0.1),
            torch.amp.GradScaler("cpu", enabled=False),
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_nonfinite_logits_still_fail(config, enabled):
    model = nn.Linear(3, len(CLASS_NAMES))
    loader = DataLoader(
        TensorDataset(torch.full((1, 3), float("nan")), torch.zeros(1, dtype=torch.long))
    )
    with pytest.raises(FloatingPointError, match="Nonfinite loss/logits"):
        ft.run_epoch(
            model,
            loader,
            torch.device("cpu"),
            config,
            torch.optim.SGD(model.parameters(), lr=0.1),
            torch.amp.GradScaler("cpu", enabled=enabled),
        )


def test_selection_tie_epoch_floor_and_failures(config):
    results = [
        {"lr": lr, "fold": fold, "status": "complete", "best_score": 0.8, "best_epoch": epoch}
        for lr in config["backbone_lrs"]
        for fold, epoch in enumerate((1, 1, 3))
    ]
    chosen = ft.choose_selection(results, config)
    assert chosen["backbone_lr"] == 3e-6
    assert chosen["final_epochs"] == 2
    results[0]["status"] = "failed"
    with pytest.raises(ValueError, match="Failed"):
        ft.choose_selection(results, config)
    with pytest.raises(ValueError, match="six"):
        ft.choose_selection(results[1:], config)


def test_interrupted_epoch_resume_exact_and_completed_skip(tmp_path, corpus, tiny, monkeypatch):
    _, frame = corpus
    spec, config = tiny
    train, valid = np.arange(14), np.arange(14, 21)

    def fit(path):
        return ft.fit_one(
            frame,
            train,
            valid,
            tmp_path,
            spec,
            config,
            3e-6,
            17,
            path,
            "identity",
            torch.device("cpu"),
            model_factory=factory,
        )

    clean = fit(tmp_path / "clean")
    original = ft.run_epoch
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("test epoch two interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(ft, "run_epoch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit(tmp_path / "resumed")
    broken = read_json(tmp_path / "resumed/progress.json")
    assert broken["status"] == "interrupted"
    assert len(broken["history"]) == 1
    monkeypatch.setattr(ft, "run_epoch", original)
    resumed = fit(tmp_path / "resumed")
    assert [a["status"] for a in resumed["attempts"]] == ["interrupted", "complete"]
    clean_state = ft.checked_load(
        tmp_path / "clean" / clean["checkpoint"], clean["checkpoint_sha256"]
    )
    resumed_state = ft.checked_load(
        tmp_path / "resumed" / resumed["checkpoint"], resumed["checkpoint_sha256"]
    )
    for key in clean_state["model"]:
        torch.testing.assert_close(
            clean_state["model"][key], resumed_state["model"][key], rtol=0, atol=0
        )
    for actual, expected in zip(resumed["history"], clean["history"], strict=True):
        assert actual["train"] == expected["train"]
        assert actual["validation_development_only"] == expected["validation_development_only"]
    monkeypatch.setattr(ft, "run_epoch", lambda *a, **k: pytest.fail("Completed fold retrained"))
    assert fit(tmp_path / "resumed") == resumed
    with (tmp_path / "resumed" / resumed["checkpoint"]).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        fit(tmp_path / "resumed")


@pytest.mark.parametrize(
    "error", [FloatingPointError("nonfinite"), torch.cuda.OutOfMemoryError("simulated OOM")]
)
def test_failed_attempts_are_recorded(tmp_path, corpus, tiny, monkeypatch, error):
    _, frame = corpus
    spec, config = tiny

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(ft, "run_epoch", fail)
    with pytest.raises(type(error)):
        ft.fit_one(
            frame,
            np.arange(14),
            np.arange(14, 21),
            tmp_path,
            spec,
            config,
            3e-6,
            1,
            tmp_path / "failure",
            "identity",
            torch.device("cpu"),
            model_factory=factory,
        )
    progress = read_json(tmp_path / "failure/progress.json")
    assert progress["status"] == "failed"
    assert type(error).__name__ in progress["attempts"][0]["error"]


def test_deployment_prediction_schema_checksums_and_fresh_overlap(tmp_path, tiny, corpus):
    manifest, frame = corpus
    spec, config = tiny
    model = ft.PartialDino(factory(spec, 8), spec, config)
    artifact = tmp_path / "model.pt"
    ft.atomic_torch_save(artifact, ft.cpu_state(model))
    training = ft.neural_source_identity(manifest, frame, tmp_path)
    metadata = {
        "class_order": list(CLASS_NAMES),
        "config": config,
        "config_sha256": json_hash(config),
        "environment": environment(),
        "model_sha256": file_hash(artifact),
        "architecture": spec,
        "training_source": training,
    }
    write_json(artifact.with_suffix(".json"), metadata)
    result = ft.predict_finetuned(
        artifact, tmp_path, tmp_path / "pred", manifest=manifest, model_factory=factory
    )
    assert result["rows"] == 21
    predictions = pd.read_csv(tmp_path / "pred/predictions.csv")
    assert predictions.columns.tolist() == [
        "image_path",
        "class",
        "material_id",
        "source_group",
        "outer_fold",
        "y_true",
        "y_pred",
        *["p_" + c for c in CLASS_NAMES],
    ]
    np.testing.assert_allclose(predictions[["p_" + c for c in CLASS_NAMES]].sum(axis=1), 1)
    assert set(validate_predictions(tmp_path / "pred/predictions.csv", CLASS_NAMES).outer_fold) == {
        "prediction_only"
    }
    assert not (tmp_path / "pred/metrics.json").exists()
    ft.predict_finetuned(
        artifact,
        tmp_path,
        tmp_path / "image",
        image=frame.iloc[0]["image_path"],
        model_factory=factory,
    )
    with pytest.raises(ValueError, match="source/material overlap"):
        ft.predict_finetuned(
            artifact,
            tmp_path,
            tmp_path / "eval",
            manifest=manifest,
            evaluate=True,
            model_factory=factory,
        )
    altered = frame.copy()
    altered["source_group"] += "fresh"
    altered["material_id"] += "fresh"
    altered["image_path"] += "fresh"
    with pytest.raises(ValueError, match="image overlap"):
        ft.verify_fresh(altered, training, training)

    fresh = frame.iloc[:1].copy()
    fresh["source_group"] = "fresh-family"
    fresh["material_id"] = "fresh-material"
    fresh["image_path"] = "fresh.png"
    Image.new("RGB", (10, 12), (251, 252, 253)).save(tmp_path / "fresh.png")
    fresh_manifest = tmp_path / "fresh.csv"
    fresh.to_csv(fresh_manifest, index=False)
    ft.predict_finetuned(
        artifact,
        tmp_path,
        tmp_path / "fresh-output",
        manifest=fresh_manifest,
        evaluate=True,
        model_factory=factory,
    )
    csv = tmp_path / "fresh-output/predictions.csv"
    assert set(validate_predictions(csv, CLASS_NAMES).outer_fold) == {"holdout"}
    assert (
        evaluate_predictions(csv, CLASS_NAMES, estimate_type="fresh_holdout", n_bootstrap=0)[
            "estimate_kind"
        ]
        == "fresh_holdout"
    )

    with Image.open(tmp_path / frame.iloc[0]["image_path"]) as image:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("comment", "same pixels, new metadata")
        image.save(tmp_path / "fresh.png", pnginfo=pnginfo)
    assert file_hash(tmp_path / "fresh.png") != training["sources"][0]["sha256"]
    with pytest.raises(ValueError, match="decoded image overlap"):
        ft.predict_finetuned(
            artifact,
            tmp_path,
            tmp_path / "bad-reencoded",
            manifest=fresh_manifest,
            evaluate=True,
            model_factory=factory,
        )
    with pytest.raises(ValueError, match="checksum"):
        ft.load_deployed(
            artifact, torch.device("cpu"), expected_sha256="bad", model_factory=factory
        )
    with pytest.raises(ValueError, match="metadata checksum"):
        ft.load_deployed(
            artifact, torch.device("cpu"), metadata_sha256="bad", model_factory=factory
        )


@pytest.mark.parametrize(
    "key", ["source_groups", "material_ids", "image_paths", "sources", "decoded_rgb_sha256"]
)
def test_fresh_requires_complete_training_identity(tmp_path, corpus, key):
    manifest, frame = corpus
    source = ft.neural_source_identity(manifest, frame, tmp_path)
    incomplete = {name: value for name, value in source.items() if name != key}
    with pytest.raises(ValueError, match="Missing.*identity"):
        ft.verify_fresh(frame, source, incomplete)


def test_decoded_hash_includes_dimensions_and_normalizes_exif(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (2, 3), (1, 2, 3)).save(first)
    Image.new("RGB", (3, 2), (1, 2, 3)).save(second)
    assert ft.decoded_hash(first) != ft.decoded_hash(second)
    image = Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    exif = Image.Exif()
    exif[274] = 6
    image.save(first, exif=exif)
    with Image.open(first) as original:
        normalized = ImageOps.exif_transpose(original).convert("RGB")
        normalized.save(second)
    assert file_hash(first) != file_hash(second)
    assert ft.decoded_hash(first) == ft.decoded_hash(second)


def test_real_parent_fold_contract_without_images(protocol):
    # Only repository development metadata is needed; no parent/holdout access.
    frame = read_manifest(ROOT / "metadata/v3/manifest.csv")
    y, groups = labels_for(frame), frame["source_group"].to_numpy()
    folds = grouped_folds(y, groups, 3, protocol["seed"] + 1000)
    ft.verify_folds(frame, fold_record(folds, np.arange(len(frame)), groups, y), protocol)


def test_full_selection_final_export_contract(tmp_path, corpus, tiny, protocol, monkeypatch):
    manifest, frame = corpus
    spec, config = tiny
    protocol["models"]["dino"] = spec
    monkeypatch.setattr(ft, "validate_config", lambda *args: None)
    monkeypatch.setattr(ft, "load_frozen_model", factory)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    y, groups = labels_for(frame), frame["source_group"].to_numpy()
    folds = grouped_folds(y, groups, 3, protocol["seed"] + 1000)
    write_json(frozen / "final_folds.json", fold_record(folds, np.arange(len(y)), groups, y))
    # The fine-tune pipeline copies the frozen artifact without deserializing it.
    (frozen / "model.joblib").write_bytes(b"original-frozen-artifact-test-fixture")
    metadata = {
        "training_source": source_identity(manifest, frame, tmp_path),
        "protocol_sha256": json_hash(protocol),
        "class_order": list(CLASS_NAMES),
        "final_selection": {"inner_mean_macro_f1": config["frozen_selector_mean_macro_f1"]},
        "model_sha256": file_hash(frozen / "model.joblib"),
    }
    write_json(frozen / "model_metadata.json", metadata)
    output = tmp_path / "run"

    def run(**kwargs):
        return ft.run_finetune(
            manifest,
            tmp_path,
            frozen,
            output,
            protocol,
            config,
            device="cpu",
            cpu_threads=1,
            **kwargs,
        )

    selection = run()
    assert len(selection["fits"]) == 6
    assert selection["selection_sha256"] == file_hash(output / "selection.json")
    assert not (output / "candidates").exists()
    assert run(resume=True) == selection
    with pytest.raises(ValueError, match="exact --selection-sha256"):
        run(resume=True, stage="final", selection_sha256="wrong")
    exports = run(resume=True, stage="final", selection_sha256=selection["selection_sha256"])
    assert file_hash(output / exports["frozen"]["path"]) == file_hash(frozen / "model.joblib")
    artifact = output / exports["finetuned"]["path"]
    assert read_json(artifact.with_suffix(".json"))["training_source"] == (
        ft.neural_source_identity(manifest, frame, tmp_path)
    )
    state = torch.load(artifact, weights_only=True)
    assert "head.weight" in state and "optimizer" not in state
    full = read_json(output / "resume/all_development/progress.json")
    assert len(full["history"]) == selection["final_epochs"]
    assert all(row["validation_development_only"] is None for row in full["history"])
    ft.load_deployed(
        artifact,
        torch.device("cpu"),
        model_factory=factory,
        expected_sha256=exports["finetuned"]["sha256"],
        metadata_sha256=exports["finetuned"]["metadata_sha256"],
    )
    assert read_json(output / "run.json")["status"] == "complete"
    assert (
        run(resume=True, stage="final", selection_sha256=selection["selection_sha256"]) == exports
    )
    # A changed input cannot be resumed even if its class/group map stays the same.
    Image.new("RGB", (10, 12), (255, 0, 0)).save(tmp_path / frame.iloc[0]["image_path"])
    with pytest.raises(ValueError, match="manifest/content"):
        run(resume=True)
