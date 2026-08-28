from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from scripts import benchmark_models as benchmark
from torch import nn


def _tiny_model(input_size: int = 8) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * input_size * input_size, 7))


def _completed_runs(root: Path) -> None:
    for model_name in benchmark.MODEL_NAMES:
        directory = root / "artifacts" / "checkpoints" / model_name
        directory.mkdir(parents=True)
        config_hash = f"hash-{model_name}"
        torch.save(
            {
                "epoch": 2,
                "model_state": _tiny_model().state_dict(),
                "config": {"name": model_name, "num_classes": 7, "input_size": 8},
                "config_hash": config_hash,
                "peak_vram_bytes": 4096,
                "effective_batch_size": 4,
                "total_training_time_seconds": 12.5,
                "oom_adjustments": {"attempts": []},
            },
            directory / "best.pt",
        )
        (directory / "resolved_config.json").write_text(
            json.dumps({"config_hash": config_hash}), encoding="utf-8"
        )
        (directory / "run_state.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "model": model_name,
                    "config_hash": config_hash,
                    "best": {"best_epoch": 2},
                    "total_training_time_seconds": 12.5,
                    "peak_vram_bytes": 4096,
                    "effective_batch_size": 4,
                    "oom_adjustments": {"attempts": []},
                }
            ),
            encoding="utf-8",
        )


def test_run_benchmarks_four_models_and_reuses_cache(tmp_path, monkeypatch):
    _completed_runs(tmp_path)
    loads: list[Path] = []

    def fake_load_model(checkpoint, device):
        loads.append(Path(checkpoint))
        return _tiny_model().to(device).eval(), {}, device

    monkeypatch.setattr(benchmark, "load_model", fake_load_model)
    json_path, csv_path, reused = benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=2)
    assert reused == []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == benchmark.SCHEMA_VERSION
    assert [item["model"] for item in payload["models"]] == list(benchmark.MODEL_NAMES)
    assert all(item["benchmark"]["1"]["available"] for item in payload["models"])
    assert all(item["benchmark"]["32"]["available"] for item in payload["models"])
    assert all(len(item["checkpoint_sha256"]) == 64 for item in payload["models"])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["model"] for row in rows} == set(benchmark.MODEL_NAMES)
    before_json, before_csv = json_path.read_bytes(), csv_path.read_bytes()

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("valid cache should avoid model loading")

    monkeypatch.setattr(benchmark, "load_model", forbidden_load)
    _, _, reused = benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=2)
    assert reused == list(benchmark.MODEL_NAMES)
    assert json_path.read_bytes() == before_json
    assert csv_path.read_bytes() == before_csv
    assert len(loads) == 4


def test_batch_32_oom_preserves_batch_1(tmp_path, monkeypatch):
    _completed_runs(tmp_path)
    prerequisites = benchmark.validate_prerequisites(tmp_path, "custom_cnn")
    monkeypatch.setattr(
        benchmark,
        "load_model",
        lambda _checkpoint, device: (_tiny_model().to(device).eval(), {}, device),
    )

    def fake_batch(_model, _device, _size, batch_size, _warmup, _repeats):
        if batch_size == 32:
            raise RuntimeError("CUDA out of memory")
        return {
            "available": True,
            "reason": None,
            "batch_size": 1,
            "latency_ms_batch": 1.0,
            "latency_ms_per_image": 1.0,
            "throughput_images_per_second": 1000.0,
        }

    monkeypatch.setattr(benchmark, "_benchmark_batch", fake_batch)
    settings = benchmark.benchmark_settings(torch.device("cpu"), 1, 2)
    result = benchmark.benchmark_one(
        tmp_path,
        "custom_cnn",
        torch.device("cpu"),
        1,
        2,
        settings,
        prerequisites,
    )
    assert result["benchmark"]["1"]["available"] is True
    assert result["benchmark"]["32"]["available"] is False
    assert result["benchmark"]["32"]["batch_size"] == 32
    assert "OOM" in result["benchmark"]["32"]["reason"]


def test_partial_results_resume_after_interruption(tmp_path, monkeypatch):
    _completed_runs(tmp_path)
    calls = 0

    def interrupted_load(_checkpoint, device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return _tiny_model().to(device).eval(), {}, device

    monkeypatch.setattr(benchmark, "load_model", interrupted_load)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=1)
    json_path = tmp_path / "reports/results/efficiency_metrics.json"
    partial = json.loads(json_path.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert [item["model"] for item in partial["models"]] == ["custom_cnn"]

    monkeypatch.setattr(
        benchmark,
        "load_model",
        lambda _checkpoint, device: (_tiny_model().to(device).eval(), {}, device),
    )
    _, _, reused = benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=1)
    completed = json.loads(json_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert reused == ["custom_cnn"]
    assert len(completed["models"]) == 4


def test_prerequisites_reject_incomplete_run(tmp_path):
    _completed_runs(tmp_path)
    state_path = tmp_path / "artifacts/checkpoints/custom_cnn/run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not completed"):
        benchmark.validate_prerequisites(tmp_path, "custom_cnn")


def test_corrupt_completed_json_forces_full_regeneration(tmp_path, monkeypatch):
    _completed_runs(tmp_path)
    loads = 0

    def fake_load(_checkpoint, device):
        nonlocal loads
        loads += 1
        return _tiny_model().to(device).eval(), {}, device

    monkeypatch.setattr(benchmark, "load_model", fake_load)
    json_path, csv_path, _ = benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=1)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["models"][0]["benchmark"]["1"]["latency_ms_batch"] = None
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark._atomic_write_csv(payload["models"], csv_path)

    _, _, reused = benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=1)
    repaired = json.loads(json_path.read_text(encoding="utf-8"))
    assert reused == []
    assert loads == 8
    assert repaired["models"][0]["benchmark"]["1"]["latency_ms_batch"] > 0


def test_corrupt_completed_csv_forces_full_regeneration(tmp_path, monkeypatch):
    _completed_runs(tmp_path)
    loads = 0

    def fake_load(_checkpoint, device):
        nonlocal loads
        loads += 1
        return _tiny_model().to(device).eval(), {}, device

    monkeypatch.setattr(benchmark, "load_model", fake_load)
    _, csv_path, _ = benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=1)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["batch_1_latency_ms_batch"] = "corrupt"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _, _, reused = benchmark.run(tmp_path, device_arg="cpu", warmup=1, repeats=1)
    assert reused == []
    assert loads == 8
    with csv_path.open(newline="", encoding="utf-8") as handle:
        repaired = list(csv.DictReader(handle))
    assert repaired[0]["batch_1_latency_ms_batch"] != "corrupt"


def test_prerequisites_reject_conflicting_checkpoint_metadata(tmp_path):
    _completed_runs(tmp_path)
    checkpoint = tmp_path / "artifacts/checkpoints/custom_cnn/best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["effective_batch_size"] = 8
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="effective batch size mismatch"):
        benchmark.validate_prerequisites(tmp_path, "custom_cnn")
