"""Version 1 JSONL service; stdlib only until the first prediction.

Request: {id: string (1..256 characters), op: health|predict|shutdown}.
Predict adds model (fixed ID) and image_path (absolute local file).
Ready: {event: ready, protocol_version: 1}. Every request gets one id/ok response.
Failures: {id: original string or null, ok: false, error: {code, message}}.
Successful predictions contain class_names and a same-order probability array.
Health reports requested/active device and loaded model IDs without loading torch.
EOF exits cleanly. Shutdown acknowledges before exit. Requests are processed serially.
"""

from __future__ import annotations

import contextlib
import json
import sys
import traceback

from . import MODEL_NAMES, DemoError

MAX_LINE_BYTES = 65536
MAX_ID_LENGTH = 256


def error_response(request_id, code, message):
    return {"id": request_id, "ok": False, "error": {"code": code, "message": message}}


def exception_response(request_id, exc):
    if isinstance(exc, DemoError):
        return error_response(request_id, exc.code, str(exc))
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return error_response(
            request_id,
            "missing_runtime",
            "The local ML runtime is incomplete; use the packaged Python runtime.",
        )
    # Offline hub misses are an OSError subclass; do not silently download weights.
    if type(exc).__name__ == "LocalEntryNotFoundError":
        return error_response(
            request_id,
            "offline_weights_missing",
            "Pinned pretrained weights are absent from the local offline cache.",
        )
    if isinstance(exc, ValueError) and any(
        word in str(exc).lower()
        for word in ("checksum", "sha-256", "sha256", "identity", "corrupt")
    ):
        return error_response(
            request_id,
            "artifact_integrity",
            "A model resource failed the reviewed loader's integrity check.",
        )
    if isinstance(exc, ValueError) and "runtime differs" in str(exc).lower():
        return error_response(
            request_id,
            "runtime_mismatch",
            "Use the exact runtime versions recorded with the saved models.",
        )
    return error_response(
        request_id,
        "inference_failed",
        "Local inference failed. See worker diagnostics; the device was not changed.",
    )


class Service:
    def __init__(self, runtime):
        self.runtime = runtime

    def handle(self, request):
        request_id = request.get("id") if isinstance(request, dict) else None
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= MAX_ID_LENGTH:
            return error_response(
                None, "invalid_request", "id must be a string of 1 to 256 characters."
            )
        try:
            op = request.get("op")
            if op == "health":
                return {
                    "id": request_id,
                    "ok": True,
                    "protocol_version": 1,
                    "models": list(MODEL_NAMES),
                    "loaded_models": list(self.runtime.models),
                    "requested_device": self.runtime.requested_device,
                    "device": str(self.runtime.device) if self.runtime.device else None,
                }
            if op == "shutdown":
                return {"id": request_id, "ok": True}
            if op != "predict":
                raise DemoError("invalid_request", "op must be health, predict or shutdown.")
            model = request.get("model")
            if not isinstance(model, str) or model not in MODEL_NAMES:
                raise DemoError("unsupported_model", "Choose one of the four bundled study models.")
            return {
                "id": request_id,
                "ok": True,
                **self.runtime.predict(model, request.get("image_path")),
            }
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return exception_response(request_id, exc)


def run_service(runtime, incoming, outgoing):
    """incoming is binary so the line size bound applies to UTF-8 bytes, not characters."""

    def emit(value):
        outgoing.write(
            json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
        )
        outgoing.flush()

    emit({"event": "ready", "protocol_version": 1})
    service = Service(runtime)
    while line := incoming.readline(MAX_LINE_BYTES + 1):
        if len(line) > MAX_LINE_BYTES:
            while not line.endswith(b"\n"):
                line = incoming.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
            emit(error_response(None, "request_too_large", "Request exceeds 64 KiB."))
            continue
        try:
            request = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            emit(
                error_response(
                    None, "invalid_json", "Request must be one UTF-8 JSON object per line."
                )
            )
            continue
        # Third-party Python prints also belong on stderr when used in-process.
        with contextlib.redirect_stdout(sys.stderr):
            response = service.handle(request)
        emit(response)
        if isinstance(request, dict) and request.get("op") == "shutdown" and response["ok"]:
            return
