"""Small shared reproducibility helpers; no global monkey patches."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "30def79976ca3b1cdcb5fcbfa0165524e44001de"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_fingerprint(model):
    """Hash weights independently of changing optimizer/progress metadata."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextlib.contextmanager
def isolated_rng(seed):
    state = capture_rng()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state)


def autocast(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError("Supported precision: bf16 or fp32; no implicit fp16 fallback")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16; explicitly select fp32")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def code_manifest():
    files = [*ROOT.glob("mdlm_rl/*.py"), *ROOT.glob("experiments/*/config.json"),
             *ROOT.glob("experiments/*/run.sh"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("requirements*.txt"), *ROOT.glob("sources.json")]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(files)}


def environment_record():
    packages = {}
    for name in ("torch", "numpy", "transformers", "tokenizers", "huggingface-hub",
                 "einops", "omegaconf", "pyarrow", "pytest", "safetensors", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "upstream_commit": UPSTREAM_COMMIT, "source_sha256": code_manifest(),
    }


def validate_environment():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("RL uses Python 3.12; use the pinned isolated environment")
    expected = {"torch": "2.6.0", "numpy": "1.26.4", "transformers": "4.45.0",
                "tokenizers": "0.20.3", "huggingface-hub": "0.30.2", "einops": "0.8.1",
                "omegaconf": "2.3.0", "pyarrow": "19.0.1", "safetensors": "0.5.2"}
    for name, version in expected.items():
        installed = importlib.metadata.version(name)
        if installed.split("+")[0] != version:
            raise RuntimeError(f"{name}=={version} required, found {installed}")


def cost_record(wall_seconds, world_size, rate):
    # Includes idle GPU time during rank-zero generation validation.
    gpu_hours = wall_seconds * world_size / 3600
    return {"wall_seconds": wall_seconds, "allocated_gpu_hours": gpu_hours,
            "hourly_cost_rmb_per_gpu": rate,
            "estimated_cost_rmb": None if rate is None else gpu_hours * rate,
            "cost_basis": "allocated wall time, including validation and idle ranks"}
