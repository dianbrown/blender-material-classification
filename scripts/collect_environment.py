"""Record the runtime environment used for a reproducible experiment.

PyTorch is optional at repository-bootstrap time. Missing ML dependencies are
recorded as ``null`` so this command remains useful before installation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError):
        return None
    return getattr(module, "__version__", None)


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def collect(project_root: Path, seed: int = 42) -> dict[str, object]:
    details: dict[str, object] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": _version("torch"),
        "torchvision_version": _version("torchvision"),
        "cuda_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_bytes": None,
        "seed": seed,
        "git_commit": _git_commit(project_root),
        "configs": [
            "configs/data.yaml",
            "configs/custom_cnn.yaml",
            "configs/resnet50.yaml",
            "configs/convnext_tiny.yaml",
            "configs/vit_b16.yaml",
        ],
    }
    try:
        import torch

        details["cuda_version"] = torch.version.cuda
        details["cuda_available"] = bool(torch.cuda.is_available())
        if details["cuda_available"]:
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            details["gpu_name"] = properties.name
            details["gpu_vram_bytes"] = properties.total_memory
    except (ImportError, OSError, RuntimeError):
        pass
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/results/environment.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(collect(args.project_root.resolve(), args.seed), indent=2) + "\n"
    output.write_text(payload, encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
