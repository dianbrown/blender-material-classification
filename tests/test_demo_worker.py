"""Protocol/caching tests with dummy models; never load real weights or use a GPU."""

import io
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from material_classifier.demo import CLASS_NAMES, LOCK_PATH, DemoError
from material_classifier.demo.runtime import (
    ARTIFACTS,
    MAX_IMAGE_BYTES,
    ModelRuntime,
    check_image_path,
    decode_image,
    sha256,
    verified_paths,
)
from material_classifier.demo.service import MAX_LINE_BYTES, Service, run_service

ROOT = Path(__file__).resolve().parents[1]


class DummyRuntime:
    models = {}
    requested_device = "auto"
    device = None

    def predict(self, model, path):
        print("third-party diagnostic")
        return {
            "model": model,
            "class_names": list(CLASS_NAMES),
            "probabilities": [1.0, 0, 0, 0, 0, 0, 0],
            "predicted_class": "wood",
            "inference_ms": 1.25,
            "device": "cpu",
        }


def test_protocol_recovers_from_bad_lines_and_stops_on_shutdown(capsys):
    stream = io.BytesIO(
        b"not json\n\xff\n"
        + b"x" * (MAX_LINE_BYTES + 1)
        + b"\n"
        + b'{"id":"p","op":"predict","model":"frozen_selected",'
        b'"image_path":"dummy"}\n'
        + b'{"id":"s","op":"shutdown"}\n'
        + b'{"id":"never","op":"health"}\n'
    )
    output = io.StringIO()
    run_service(DummyRuntime(), stream, output)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[0] == {"event": "ready", "protocol_version": 1}
    assert [r["error"]["code"] for r in rows[1:4]] == [
        "invalid_json",
        "invalid_json",
        "request_too_large",
    ]
    assert rows[4]["id"] == "p" and rows[4]["ok"]
    assert rows[5] == {"id": "s", "ok": True}
    assert len(rows) == 6
    assert "third-party diagnostic" in capsys.readouterr().err


@pytest.mark.parametrize(
    "payload,code",
    [
        ([], "invalid_request"),
        ({"id": 1}, "invalid_request"),
        ({"id": ""}, "invalid_request"),
        ({"id": "x" * 257}, "invalid_request"),
        ({"id": "x", "op": "load_joblib"}, "invalid_request"),
        ({"id": "x", "op": "predict", "model": []}, "unsupported_model"),
        ({"id": "x", "op": "predict", "model": "arbitrary.joblib"}, "unsupported_model"),
    ],
)
def test_request_validation(payload, code):
    response = Service(DummyRuntime()).handle(payload)
    assert response["ok"] is False
    assert response["error"]["code"] == code


@pytest.mark.parametrize(
    "exception,code",
    [
        (ModuleNotFoundError("torch"), "missing_runtime"),
        (
            type("LocalEntryNotFoundError", (OSError,), {})("private-cache-path"),
            "offline_weights_missing",
        ),
        (ValueError("Weights SHA-256 mismatch"), "artifact_integrity"),
        (ValueError("Frozen feature runtime differs"), "runtime_mismatch"),
        (RuntimeError("broken model"), "inference_failed"),
    ],
)
def test_errors_are_structured_without_private_details(exception, code):
    class Failed(DummyRuntime):
        def predict(self, *args):
            raise exception

    response = Service(Failed()).handle(
        {"id": "retained", "op": "predict", "model": "resnet_fixed"}
    )
    assert response["id"] == "retained"
    assert response["error"]["code"] == code
    assert str(exception) not in response["error"]["message"]


def test_real_service_process_ready_health_without_ml_or_cwd_dependency(tmp_path):
    # -S disables site-packages: startup and health must work with no ML runtime at all.
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/demo/worker.py"),
        "--root",
        str(tmp_path),
        "--device",
        "auto",
    ]
    image = tmp_path / "valid.png"
    Image.new("RGB", (2, 2)).save(image)
    with subprocess.Popen(
        command,
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as worker:
        try:
            # Reading ready BEFORE writing anything verifies a flushed unsolicited handshake.
            lines = queue.Queue()
            threading.Thread(
                target=lambda: lines.put(worker.stdout.readline()), daemon=True
            ).start()
            ready = lines.get(timeout=5)
            assert json.loads(ready) == {"event": "ready", "protocol_version": 1}
            requests = [
                {"id": "h", "op": "health"},
                {"id": "p", "op": "predict", "model": "resnet_fixed", "image_path": str(image)},
                {"id": "s", "op": "shutdown"},
                {"id": "ignored", "op": "health"},
            ]
            stdout, stderr = worker.communicate(
                "".join(json.dumps(r) + "\n" for r in requests), timeout=5
            )
            rows = [json.loads(line) for line in stdout.splitlines()]
            assert worker.returncode == 0
            assert len(rows) == 3
            assert rows[0]["device"] is None and rows[0]["loaded_models"] == []
            assert rows[1]["error"]["code"] == "missing_runtime"
            assert rows[2] == {"id": "s", "ok": True}
            assert "ModuleNotFoundError" in stderr
        finally:
            if worker.poll() is None:
                worker.kill()


@pytest.mark.parametrize("value", [None, "relative.png", "", "\x00"])
def test_invalid_image_paths(value):
    with pytest.raises(DemoError, match="absolute"):
        check_image_path(value)


def test_image_validation_before_model_loading(tmp_path):
    runtime = ModelRuntime(tmp_path)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(DemoError) as error:
        runtime.predict("resnet_fixed", str(bad))
    assert error.value.code == "invalid_image"
    assert runtime.device is None and not runtime.models
    huge = tmp_path / "huge.png"
    with huge.open("wb") as stream:
        stream.truncate(MAX_IMAGE_BYTES + 1)
    with pytest.raises(DemoError) as error:
        check_image_path(str(huge))
    assert error.value.code == "image_too_large"
    thin = tmp_path / "thin.png"
    Image.new("RGB", (1024, 1)).save(thin)
    with pytest.raises(DemoError) as error:
        decode_image(thin)
    assert error.value.code == "image_too_large"
    wide = tmp_path / "wide.png"
    Image.new("RGB", (8193, 1)).save(wide)
    with pytest.raises(DemoError) as error:
        decode_image(wide)
    assert error.value.code == "image_too_large"


def make_lock(root):
    paths = [root / p for p in ARTIFACTS["frozen_selected"]]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy trusted bytes")
    lock = root / LOCK_PATH
    lock.parent.mkdir(parents=True)
    record = dict(
        zip(
            ("artifact_path", "metadata_path"),
            (os.path.relpath(p, lock.parent) for p in paths),
            strict=True,
        )
    )
    record.update(artifact_sha256=sha256(paths[0]), metadata_sha256=sha256(paths[1]))
    content = {
        "class_order": list(CLASS_NAMES),
        "primary_model": "frozen_selected",
        "models": {"frozen_selected": record},
    }
    lock.write_text(json.dumps(content))
    return paths, lock, content


def test_hash_and_path_checks_precede_deserialization(tmp_path):
    paths, lock, content = make_lock(tmp_path)
    assert verified_paths(tmp_path, "frozen_selected")[0] == tuple(paths)
    paths[0].write_bytes(b"tampered")
    with pytest.raises(DemoError) as error:
        ModelRuntime(tmp_path).load("frozen_selected")
    assert error.value.code == "artifact_integrity"
    paths[0].write_bytes(b"dummy trusted bytes")
    content["models"]["frozen_selected"]["artifact_path"] = "../../../../outside.joblib"
    lock.write_text(json.dumps(content))
    with pytest.raises(DemoError, match="trusted"):
        verified_paths(tmp_path, "frozen_selected")


def test_missing_or_corrupt_lock(tmp_path):
    with pytest.raises(DemoError) as error:
        verified_paths(tmp_path, "frozen_selected")
    assert error.value.code == "missing_artifact"
    lock = tmp_path / LOCK_PATH
    lock.parent.mkdir(parents=True)
    lock.write_text("not json")
    with pytest.raises(DemoError) as error:
        verified_paths(tmp_path, "frozen_selected")
    assert error.value.code == "artifact_integrity"


def test_shared_backbone_and_model_cache(monkeypatch, tmp_path):
    from material_classifier.v3 import features, inference

    spec = {"name": "dummy", "resolutions": [224]}
    metadata = {
        "protocol": {"models": {"convnext": spec}},
        "recipe": {"parts": ["convnext_224_pooled"]},
    }
    loads, backbone_loads = [], []
    monkeypatch.setattr(
        "material_classifier.demo.runtime.verified_paths",
        lambda root, model: ((Path(model), Path("meta")), {}),
    )
    monkeypatch.setattr(inference, "_check_feature_runtime", lambda metadata: None)

    def load_artifact(path):
        loads.append(str(path))
        return {"metadata": metadata}

    def load_backbone(*args, **kwargs):
        backbone_loads.append(kwargs)
        return object()

    monkeypatch.setattr(inference, "load_artifact", load_artifact)
    monkeypatch.setattr(features, "load_frozen_model", load_backbone)
    runtime = ModelRuntime(tmp_path)
    runtime.device = "cpu"
    selected = runtime.load("frozen_selected")
    assert runtime.load("frozen_selected") is selected
    control = runtime.load("frozen_control")
    assert control[1][0][3] is selected[1][0][3]
    assert len(loads) == 2 and len(backbone_loads) == 1
    assert backbone_loads[0]["offline"] is True


def test_auto_cpu_only_when_cuda_absent(monkeypatch, tmp_path):
    import torch

    from material_classifier.v3 import features

    configured = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        features,
        "configure_torch",
        lambda device, threads: configured.append((device, threads)) or device,
    )
    runtime = ModelRuntime(tmp_path)
    runtime.configure()
    runtime.configure()
    assert configured == [("cpu", 4)]
    with pytest.raises(DemoError) as error:
        ModelRuntime(tmp_path, "cuda").configure()
    assert error.value.code == "device_unavailable"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ModelRuntime(tmp_path).configure()
    assert configured[-1] == ("cuda", 4)


def test_neural_probability_precision_matches_reviewed_paths(monkeypatch, tmp_path):
    import numpy as np
    import torch

    from material_classifier.v3 import features

    logits = torch.tensor([[0.234, 1.1, -0.5, 0.8, -2.2, 3.7, 0.01]], dtype=torch.float32)
    model = lambda batch: logits  # noqa: E731
    tensor = torch.zeros(3, 2, 2)
    monkeypatch.setattr(features, "transform_image", lambda *args: tensor)
    runtime = ModelRuntime(tmp_path)
    runtime.device = torch.device("cpu")
    with Image.new("RGB", (2, 2)) as picture:
        resnet = runtime._probabilities("resnet_fixed", (model, lambda image: tensor), picture)
        dino = runtime._probabilities(
            "dinov2_finetuned", (model, {"architecture": {}, "config": {"input_size": 2}}), picture
        )
    assert np.array_equal(resnet, logits.double().softmax(1).numpy()[0])
    expected = logits.softmax(1).double().numpy()[0]
    expected /= expected.sum()
    assert np.array_equal(dino, expected)


def test_every_prediction_runs_forward_and_no_bug_cpu_fallback(monkeypatch, tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(image)
    runtime = ModelRuntime(tmp_path)
    runtime.device = SimpleNamespace(type="cpu")
    calls = []
    monkeypatch.setattr(runtime, "load", lambda model: runtime.models.setdefault(model, object()))
    monkeypatch.setattr(
        runtime, "_probabilities", lambda *args: calls.append(1) or [1, 0, 0, 0, 0, 0, 0]
    )
    first = runtime.predict("resnet_fixed", str(image))
    second = runtime.predict("resnet_fixed", str(image))
    assert not first["model_cached"] and second["model_cached"] and len(calls) == 2
    assert first["inference_ms"] > 0 and "Excludes" in first["latency_scope"]

    def broken(*args):
        raise RuntimeError("model bug")

    runtime.device = SimpleNamespace(type="cuda")
    monkeypatch.setattr(runtime, "synchronize", lambda: None)
    monkeypatch.setattr(runtime, "_probabilities", broken)
    response = Service(runtime).handle(
        {"id": "bad", "op": "predict", "model": "resnet_fixed", "image_path": str(image)}
    )
    assert response["error"]["code"] == "inference_failed"
    assert runtime.device.type == "cuda"  # No reconfigure/retry on a model failure.
