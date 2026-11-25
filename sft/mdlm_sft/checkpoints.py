"""Atomic SFT state checkpoints, distinct from pretrained initialization."""
from __future__ import annotations

import json
import copy
import hashlib
import os
import shutil
from pathlib import Path

import torch

from .runtime import atomic_json

SCHEMA = "mdlm-sft-v1"
# These immutable identifiers allow checkpoints produced by the two previous
# storage schemas to remain readable after the package rename.
LEGACY_SCHEMA_DIGESTS = ('42180d37beef52b5494497f4c03386b5022a9d0dd6863fcb973ccef82a62c0d9', '07f2c8ec2f2f4105b8359e9141f87440cacf9782e6b189a35c77b7bbc7c8aab1')
LEGACY_FORMAT_DIGESTS = ('c5d70e41555c1f485289e9734d3e3266269e419d51a3a83127842cc40c25c992', '15c0d4da8a6d9545595b6a1107389744aeaa5fcd3469123cdd406814dfafc129')


def matches_legacy(value, digests):
    return isinstance(value, str) and hashlib.sha256(value.encode("utf-8")).hexdigest() in digests


def is_sft_schema(value):
    return value == SCHEMA or matches_legacy(value, LEGACY_SCHEMA_DIGESTS)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_copy(source, destination):
    destination = Path(destination)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, destination)


def load_sft(path):
    # Only load trusted, locally created training files. They contain RNG tuples.
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not is_sft_schema(state.get("schema")):
        raise ValueError("Resume/evaluate requires an MDLM SFT checkpoint; use --init for pretrained")
    for key in ("model", "optimizer", "scheduler", "progress", "rng_by_rank", "config", "identity"):
        if key not in state:
            raise ValueError(f"Incomplete SFT checkpoint: missing {key}")
    return state


def resume_config_signature(config):
    # Bounds may be extended after a stopped run, without changing the LR plan.
    mutable = {"output_dir", "data_dir", "tokenizer_path", "pretrained_checkpoint",
               "max_steps", "max_wall_seconds", "checkpoint_every_steps",
               "hourly_cost_rmb_per_gpu"}
    return {k: v for k, v in config.items() if k not in mutable}


def check_resume(state, config, identity):
    if resume_config_signature(state["config"]) != resume_config_signature(config):
        raise ValueError("Resume cannot change training/decoding configuration; start a new experiment")
    if state["identity"] != identity:
        raise ValueError("Resume requires identical data, code, architecture, tokenizer and world size")
    if len(state["rng_by_rank"]) != identity["world_size"]:
        raise ValueError("Missing per-rank random states")


class CheckpointManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, step):
        return self.directory / f"step-{step:08d}.pt"

    def save(self, payload):
        path = self.path(payload["progress"]["global_step"])
        atomic_torch_save(payload, path)
        self._publish_last(path)
        return path

    def _publish_last(self, path):
        # last is an alias of the newest of the two ordinary checkpoints.
        temporary = self.directory / "last.pt.tmp"
        temporary.unlink(missing_ok=True)
        try:
            os.link(path, temporary)
        except OSError:
            shutil.copyfile(path, temporary)
        os.replace(temporary, self.directory / "last.pt")

    def finish(self, payload, accuracy=None):
        progress = copy.deepcopy(payload["progress"])
        old_best = progress.get("best_accuracy")
        improved = accuracy is not None and (old_best is None or accuracy > old_best)
        if improved:
            progress["best_accuracy"] = accuracy
            progress["best_step"] = progress["global_step"]
        progress["pending_validation"] = None
        finalized = {**payload, "progress": progress}
        path = self.path(progress["global_step"])
        # Keep last pointing to the recoverable pending state until best has
        # been published. Replacing path leaves last's hardlink inode intact.
        atomic_torch_save(finalized, path)
        if improved:
            atomic_copy(path, self.directory / "best.pt")
            atomic_json(self.directory / "best.json", {
                "accuracy": accuracy, "step": progress["global_step"],
                "tie_policy": "strictly greater; retain earlier checkpoint"})
        self._publish_last(path)
        payload["progress"].clear()
        payload["progress"].update(progress)
        # This is called only AFTER generation and best-copy have succeeded.
        for old in sorted(self.directory.glob("step-*.pt"))[:-2]:
            old.unlink()
        return improved
