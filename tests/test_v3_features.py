"""Offline extraction tests: no pretrained downloads and no CUDA execution."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from material_classifier.v3 import CLASS_NAMES
from material_classifier.v3 import features as features_module
from material_classifier.v3.features import (
    assemble_features,
    extract_batch,
    extract_features,
    load_feature_cache,
    load_protocol,
    pool_dino_tokens,
    transform_image,
)


@pytest.fixture(autouse=True)
def forbid_weight_downloads(monkeypatch):
    import huggingface_hub

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline unit tests must not request pretrained weights")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)


class DummyDino(torch.nn.Module):
    num_prefix_tokens = 5

    def __init__(self, size, width=3, patch_size=2):
        super().__init__()
        self.size, self.width, self.patch_size = size, width, patch_size

    def forward_features(self, batch):
        assert not torch.is_grad_enabled()
        assert batch.shape[-2:] == (self.size, self.size)
        signal = batch.mean(dim=(2, 3))[:, : self.width]
        count = (self.size // self.patch_size) ** 2
        patches = signal[:, None, :].expand(-1, count, -1)
        cls = signal[:, None, :] * 2
        registers = torch.full((len(batch), 4, self.width), 10000.0, device=batch.device)
        return torch.cat([cls, registers, patches], dim=1)


class DummyConv(torch.nn.Module):
    def forward_features(self, batch):
        assert not torch.is_grad_enabled()
        return batch

    def forward_head(self, features, pre_logits):
        assert pre_logits
        return features.mean(dim=(2, 3))


def dummy_factory(spec, size, device="cpu", offline=False):
    del offline
    model = (
        DummyDino(size, spec["width"], spec["patch_size"]) if "patch_size" in spec else DummyConv()
    )
    return model.eval().requires_grad_(False).to(device)


@pytest.fixture
def small_protocol():
    protocol = copy.deepcopy(
        load_protocol(Path(__file__).resolve().parents[1] / "configs/v3/protocol.json")
    )
    protocol["models"]["dino"].update(width=3, patch_size=2, resolutions=[8, 12])
    protocol["models"]["convnext"].update(width=3, resolutions=[8])
    for candidate in protocol["candidates"]:
        candidate["parts"] = [
            part.replace("224", "8").replace("336", "12") for part in candidate["parts"]
        ]
    return protocol


@pytest.fixture
def image_manifest(tmp_path):
    records = []
    for index in range(3):
        path = f"image_{index}.png"
        Image.new("RGB", (13, 17), color=(20 + index * 30, 60, 100)).save(tmp_path / path)
        records.append(
            {
                "image_path": path,
                "class": CLASS_NAMES[index],
                "material_id": f"m{index}",
                "source_group": f"g{index}",
                "original_split": "development",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest, tmp_path


def test_register_exclusion_fp32_population_std_and_shapes():
    tokens = torch.tensor(
        [[[4, 8], [1000, 1000], [2000, 2000], [1, 3], [3, 7]]], dtype=torch.float16
    )
    pooled = pool_dino_tokens(tokens, 3, expected_patches=2, width=2)
    np.testing.assert_array_equal(pooled["cls"], [[4, 8]])
    np.testing.assert_array_equal(pooled["mean"], [[2, 5]])
    np.testing.assert_array_equal(pooled["std"], [[1, 2]])
    assert all(tensor.dtype == torch.float32 for tensor in pooled.values())
    with pytest.raises(ValueError, match="patch count"):
        pool_dino_tokens(tokens, 1, expected_patches=2)
    with pytest.raises(ValueError, match="width"):
        pool_dino_tokens(tokens, 3, width=3)


def test_transform_is_deterministic_and_model_specific(small_protocol):
    spec = small_protocol["models"]["dino"]
    image = Image.new("RGB", (15, 10), (255, 0, 128))
    first = transform_image(image, spec, 8)
    assert first.shape == (3, 8, 8)
    assert torch.equal(first, transform_image(image, spec, 8))
    expected = (torch.tensor([1, 0, 128 / 255]) - torch.tensor(spec["mean"])) / torch.tensor(
        spec["std"]
    )
    torch.testing.assert_close(first[:, 0, 0], expected)
    model = DummyDino(8)
    with pytest.raises(ValueError, match="frozen"):
        extract_batch(model, first[None], spec, 8)


def test_cache_alignment_content_and_identity(small_protocol, image_manifest):
    manifest, root = image_manifest
    output = root / "cache"
    record = extract_features(
        manifest,
        root,
        output,
        small_protocol,
        device="cpu",
        batch_size=2,
        model_factory=dummy_factory,
    )
    base, loaded = load_feature_cache(
        output, manifest=manifest, data_root=root, protocol=small_protocol
    )
    assert record == loaded
    assert record["complete"]
    assert len(base) == 7
    assert assemble_features(base, small_protocol["candidates"][-1]["parts"]).shape == (3, 21)
    assert np.allclose(base["dino_8_std"], 0)
    assert base["dino_8_cls"][0, 0] != base["dino_8_cls"][1, 0]
    changed = copy.deepcopy(small_protocol)
    changed["models"]["dino"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_feature_cache(output, manifest=manifest, protocol=changed)
    with pytest.raises(FileExistsError):
        extract_features(manifest, root, output, small_protocol, device="cpu")
    with pytest.raises(ValueError, match="identity mismatch"):
        extract_features(
            manifest,
            root,
            output,
            small_protocol,
            device="cpu",
            batch_size=1,
            resume=True,
            model_factory=dummy_factory,
        )
    frame = pd.read_csv(manifest)
    frame.iloc[::-1].to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="alignment mismatch"):
        load_feature_cache(output, manifest=manifest)
    frame.to_csv(manifest, index=False)
    frame.loc[0, "source_group"] = "corrected-family"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="alignment mismatch"):
        load_feature_cache(output, manifest=manifest)
    frame.loc[0, "source_group"] = "g0"
    frame.to_csv(manifest, index=False)
    Image.new("RGB", (13, 17), "red").save(root / "image_0.png")
    with pytest.raises(ValueError, match="source image content"):
        load_feature_cache(output, manifest=manifest, data_root=root)


def test_resume_reuses_only_hash_verified_batches(small_protocol, image_manifest):
    manifest, root = image_manifest
    output = root / "cache"
    calls = []

    def interrupt(spec, size, **kwargs):
        calls.append(size)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return dummy_factory(spec, size, **kwargs)

    with pytest.raises(RuntimeError, match="simulated"):
        extract_features(
            manifest,
            root,
            output,
            small_protocol,
            device="cpu",
            batch_size=2,
            model_factory=interrupt,
        )
    calls.clear()

    def resume_factory(spec, size, **kwargs):
        calls.append(size)
        return dummy_factory(spec, size, **kwargs)

    extract_features(
        manifest,
        root,
        output,
        small_protocol,
        device="cpu",
        batch_size=2,
        resume=True,
        model_factory=resume_factory,
    )
    assert calls == [12, 8]  # completed first resolution is never recomputed
    base, _ = load_feature_cache(output, manifest=manifest)
    assert base["dino_12_cls"].shape == (3, 3)
    with (output / "dino_8_cls.npy").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_feature_cache(output, manifest=manifest)


def test_resume_recovers_at_committed_batch_boundary(small_protocol, image_manifest, monkeypatch):
    manifest, root = image_manifest
    output = root / "partial-cache"
    original = features_module.extract_batch
    batch_sizes = []

    def fail_second_batch(model, batch, spec, size):
        batch_sizes.append(len(batch))
        if len(batch_sizes) == 2:
            raise RuntimeError("interrupted mid-resolution")
        return original(model, batch, spec, size)

    monkeypatch.setattr(features_module, "extract_batch", fail_second_batch)
    with pytest.raises(RuntimeError, match="mid-resolution"):
        extract_features(
            manifest,
            root,
            output,
            small_protocol,
            device="cpu",
            batch_size=2,
            model_factory=dummy_factory,
        )
    batch_sizes.clear()

    def track_resume(model, batch, spec, size):
        batch_sizes.append(len(batch))
        return original(model, batch, spec, size)

    monkeypatch.setattr(features_module, "extract_batch", track_resume)
    extract_features(
        manifest,
        root,
        output,
        small_protocol,
        device="cpu",
        batch_size=2,
        resume=True,
        model_factory=dummy_factory,
    )
    assert batch_sizes == [1, 2, 1, 2, 1]
    base, _ = load_feature_cache(output, manifest=manifest)
    assert base["dino_8_cls"].shape == (3, 3)
