"""Atomic tensor storage and trusted initialization from the selected SFT state."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch


SCHEMA = "mdlm-sft-v1"
# These are the SHA-256 fingerprints of the two earlier on-disk schema labels.
# Allow only those formats when loading historical training states.
LEGACY_SCHEMA_HASHES = {
    "42180d37beef52b5494497f4c03386b5022a9d0dd6863fcb973ccef82a62c0d9",
    "07f2c8ec2f2f4105b8359e9141f87440cacf9782e6b189a35c77b7bbc7c8aab1",
}


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_sft(path):
    # Only load trusted, locally created training files. They contain RNG tuples.
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    schema = state.get("schema")
    if not isinstance(schema, str) or (schema != SCHEMA and
            hashlib.sha256(schema.encode()).hexdigest() not in LEGACY_SCHEMA_HASHES):
        raise ValueError("RL initialization requires an SFT training checkpoint")
    for key in ("model", "optimizer", "scheduler", "progress", "rng_by_rank", "config", "identity"):
        if key not in state:
            raise ValueError(f"Incomplete SFT checkpoint: missing {key}")
    return state

