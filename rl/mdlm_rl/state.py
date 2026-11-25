"""Deterministic question order and explicit RL checkpoint boundaries.

Full RL checkpoints are trusted local pickle files and contain optimizer and
random states. Model-only ``best.pt`` files deliberately cannot resume training.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
import hashlib
from pathlib import Path
import random

import torch


SCHEMA = "mdlm-diffu-grpo-v1"
LEGACY_TRAINING_SCHEMA_SHA256 = "065d5f1a01235992e68047bb4b3ae631ceb962bf71541942d827366f3d078e00"
MODEL_SCHEMA = "mdlm-rl-model-v1"
SAMPLER_SCHEMA = "shuffled-questions-v1"
REQUIRED_STATE_KEYS = (
    "model", "optimizer", "scheduler", "progress", "rng_by_rank",
    "sampler", "config", "identity",
)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _as_tuple(value):
    """Allow a sampler state to round-trip through JSON as well as torch.save."""
    if isinstance(value, (list, tuple)):
        return tuple(_as_tuple(item) for item in value)
    return value


class QuestionSampler:
    """Uniform shuffled passes, with no repeated question within a rollout.

    Each pass visits every index exactly once. If a batch crosses a pass
    boundary, the next permutation is shuffled and indices already selected
    for that batch are stably postponed to its end. Thus the cross-boundary
    order is conditioned on the unfinished batch, but no question is dropped,
    duplicated, or padded. A private RNG leaves training/generation RNG alone.
    """

    def __init__(self, size, seed, state=None):
        self.size = _integer(size, "size", 1)
        self.seed = _integer(seed, "seed")
        self._rng = random.Random(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = list(range(size))
        if state is None:
            self._rng.shuffle(self.order)
        else:
            self._restore(state)

    def _restore(self, state):
        if not isinstance(state, Mapping) or state.get("schema") != SAMPLER_SCHEMA:
            raise ValueError("Invalid question sampler schema")
        saved_size = _integer(state.get("size"), "sampler size", 1)
        saved_seed = _integer(state.get("seed"), "sampler seed")
        if saved_size != self.size or saved_seed != self.seed:
            raise ValueError("Question sampler size/seed changed")
        epoch = _integer(state.get("epoch"), "sampler epoch")
        cursor = _integer(state.get("cursor"), "sampler cursor")
        order = state.get("order")
        if cursor > self.size:
            raise ValueError("Sampler cursor is outside its permutation")
        if (not isinstance(order, (list, tuple)) or len(order) != self.size
                or any(type(index) is not int for index in order)
                or sorted(order) != list(range(self.size))):
            raise ValueError("Sampler order must contain every question exactly once")
        try:
            self._rng.setstate(_as_tuple(state["rng_state"]))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid question sampler random state") from error
        self.epoch, self.cursor, self.order = epoch, cursor, list(order)

    @property
    def questions_seen(self):
        return self.epoch * self.size + self.cursor

    def next_batch(self, n):
        n = _integer(n, "questions per batch", 1)
        if n > self.size:
            raise ValueError("A rollout cannot contain more unique questions than the dataset")
        selected = []
        while len(selected) < n:
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.order = list(range(self.size))
                self._rng.shuffle(self.order)
                if selected:
                    used = set(selected)
                    self.order = ([index for index in self.order if index not in used]
                                  + [index for index in self.order if index in used])
            take = min(n - len(selected), self.size - self.cursor)
            selected.extend(self.order[self.cursor:self.cursor + take])
            self.cursor += take
        if len(set(selected)) != n:
            raise AssertionError("Question sampler produced duplicate rollout questions")
        return selected

    def state_dict(self):
        return {
            "schema": SAMPLER_SCHEMA,
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.copy(),
            "rng_state": self._rng.getstate(),
        }


def _validate_checkpoint(state):
    schema = state.get("schema") if isinstance(state, Mapping) else None
    if not isinstance(schema, str) or (schema != SCHEMA and
            hashlib.sha256(schema.encode()).hexdigest() != LEGACY_TRAINING_SCHEMA_SHA256):
        raise ValueError("Resume requires a full diffu-GRPO checkpoint, not SFT or model-only weights")
    for key in REQUIRED_STATE_KEYS:
        if key not in state:
            raise ValueError(f"Incomplete diffu-GRPO checkpoint: missing {key}")
    for key in ("model", "optimizer", "scheduler", "progress", "sampler", "config", "identity"):
        if not isinstance(state[key], Mapping):
            raise ValueError(f"Invalid diffu-GRPO checkpoint mapping: {key}")
    _integer(state["progress"].get("global_step"), "global_step")
    world_size = _integer(state["identity"].get("world_size"), "world_size", 1)
    if (not isinstance(state["rng_by_rank"], (list, tuple))
            or len(state["rng_by_rank"]) != world_size
            or any(not isinstance(rng, Mapping) for rng in state["rng_by_rank"])):
        raise ValueError("Missing per-rank random states")
    # Validate saved ordering immediately, before the caller changes any model.
    sampler = state["sampler"]
    QuestionSampler(sampler.get("size"), sampler.get("seed"), sampler)
    return state


def load_rl(path):
    """Load only trusted locally generated checkpoints (contains Python RNG)."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    return _validate_checkpoint(state)


def resume_signature(config):
    """Only execution bounds and the output destination may change on resume."""
    if not isinstance(config, Mapping):
        raise ValueError("RL configuration must be a mapping")
    mutable = {"max_steps", "max_wall_seconds", "output_dir"}
    return {key: copy.deepcopy(value) for key, value in config.items() if key not in mutable}


def check_resume(state, config, identity):
    """Fail closed on changed data, code, models, decoding, or RNG world size."""
    _validate_checkpoint(state)
    if resume_signature(state["config"]) != resume_signature(config):
        raise ValueError("Resume cannot change RL training/decoding configuration")
    if state["identity"] != identity:
        raise ValueError("Resume requires identical data, code, architecture, tokenizer and world size")
    world_size = _integer(identity.get("world_size"), "world_size", 1)
    if len(state["rng_by_rank"]) != world_size:
        raise ValueError("Missing per-rank random states")
    limit = config.get("max_steps")
    if limit is not None and _integer(limit, "max_steps") < state["progress"]["global_step"]:
        raise ValueError("max_steps is earlier than the saved training step")


def make_best_payload(payload):
    """Build a model-only artifact; tensor storage is shared until serialization.

    Callers must serialize atomically before performing another optimizer step.
    Metadata is copied so subsequent progress updates do not alter this record.
    The initial SFT checkpoint remains the authority for the frozen reference.
    """
    _validate_checkpoint(payload)
    result = {
        "schema": MODEL_SCHEMA,
        "model": payload["model"],
        "progress": copy.deepcopy(payload["progress"]),
        "identity": copy.deepcopy(payload["identity"]),
        "config": copy.deepcopy(payload["config"]),
    }
    if "model_config" in payload:
        result["model_config"] = copy.deepcopy(payload["model_config"])
    elif "model" in payload["identity"]:
        result["model_config"] = copy.deepcopy(payload["identity"]["model"])
    return result
