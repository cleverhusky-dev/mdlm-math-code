"""Strict, portable configuration for the MDLM diffu-GRPO experiment.

Validation performs no I/O. Loading resolves the four input/output paths from
the project root, so launching from another working directory changes nothing.
The runner separately verifies input identities and the execution environment.
"""
from __future__ import annotations

import json
import copy
import math
from pathlib import Path
import re

from .runtime import ROOT


PATH_KEYS = ("data_dir", "tokenizer_path", "initial_checkpoint", "output_dir")
CONFIG_KEYS = frozenset({
    "name", *PATH_KEYS, "initial_checkpoint_sha256", "initial_global_step",
    "seed", "world_size", "precision", "questions_per_batch",
    "samples_per_question", "inner_updates", "micro_batch_size",
    "gradient_accumulation_steps", "generation_batch_size", "learning_rate",
    "betas", "adam_eps", "weight_decay", "grad_clip", "warmup_steps",
    "max_steps", "max_wall_seconds", "validation_every_steps",
    "checkpoint_every_steps", "prompt_mask_probability", "clip_epsilon",
    "kl_beta", "estimator_temperature", "expected_train_questions",
    "expected_val_questions", "expected_data_manifest_sha256", "evaluation",
    "rollout",
})
ROLLOUT_KEYS = frozenset({"length", "steps", "temperature", "use_float64"})
EVALUATION_KEYS = ROLLOUT_KEYS | {
    "batch_size", "seed", "timeout_s", "memory_mb",
}


def _keys(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing, extra = expected - value.keys(), value.keys() - expected
    if missing or extra:
        raise ValueError(f"{label} keys: missing={sorted(missing)}, unknown={sorted(extra)}")


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _number(value, label, *, minimum=None, strict=False):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        operator = ">" if strict else ">="
        raise ValueError(f"{label} must be {operator} {minimum}")


def _seed(value, label):
    _integer(value, label, 0)
    if value >= 2**32:
        raise ValueError(f"{label} must be below 2**32")


def _decoding(settings, label):
    _integer(settings["length"], f"{label}.length")
    if settings["length"] != 512:
        raise ValueError(f"{label}.length must remain 512 total tokens (prompt + completion)")
    _integer(settings["steps"], f"{label}.steps")
    _number(settings["temperature"], f"{label}.temperature", minimum=0, strict=True)
    if settings["use_float64"] is not True:
        raise ValueError(f"{label}.use_float64 must remain true")


def validate_config(config):
    """Reject misspellings and inconsistent global/accumulation semantics.

    ``inner_updates`` counts optimizer updates over the entire rollout, not
    microbatches or denoising steps. World size one is available for CPU test
    fixtures; the published experiment is configured for two-GPU DDP.
    """
    _keys(config, CONFIG_KEYS, "config")
    if not isinstance(config["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", config["name"]):
        raise ValueError("name must be a simple experiment identifier")
    for key in PATH_KEYS:
        if not isinstance(config[key], str) or not config[key].strip() or "\x00" in config[key]:
            raise ValueError(f"{key} must be a nonempty path string")
    for key in ("initial_checkpoint_sha256", "expected_data_manifest_sha256"):
        if not isinstance(config[key], str) or not re.fullmatch(r"[0-9a-f]{64}", config[key]):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    _integer(config["initial_global_step"], "initial_global_step", 0)
    _seed(config["seed"], "seed")
    for key in ("world_size", "questions_per_batch", "samples_per_question", "inner_updates",
                "micro_batch_size", "gradient_accumulation_steps", "generation_batch_size",
                "max_steps", "validation_every_steps", "checkpoint_every_steps",
                "expected_train_questions", "expected_val_questions"):
        _integer(config[key], key)
    if config["world_size"] not in (1, 2):
        raise ValueError("Supported world_size is 1 (fixtures) or 2 (formal DDP)")
    if config["precision"] not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    if config["samples_per_question"] < 2:
        raise ValueError("samples_per_question must be >= 2 for group advantages")
    if config["questions_per_batch"] > config["expected_train_questions"]:
        raise ValueError("questions_per_batch exceeds the unique training question count")
    if config["questions_per_batch"] % config["world_size"]:
        raise ValueError("questions_per_batch must divide evenly across world_size")
    rollout_size = config["questions_per_batch"] * config["samples_per_question"]
    update_size = config["world_size"] * config["micro_batch_size"] * config["gradient_accumulation_steps"]
    if rollout_size != update_size:
        raise ValueError("n * G must equal world_size * micro_batch_size * gradient_accumulation_steps")
    for key in ("max_steps", "validation_every_steps", "checkpoint_every_steps"):
        if config[key] % config["inner_updates"]:
            raise ValueError(f"{key} must be a multiple of inner_updates (complete rollout boundary)")
    _integer(config["warmup_steps"], "warmup_steps", 0)
    if config["warmup_steps"] > config["max_steps"]:
        raise ValueError("warmup_steps must not exceed max_steps")
    for key in ("learning_rate", "adam_eps", "grad_clip", "max_wall_seconds"):
        _number(config[key], key, minimum=0, strict=True)
    for key in ("weight_decay", "kl_beta", "prompt_mask_probability"):
        _number(config[key], key, minimum=0)
    if config["prompt_mask_probability"] > 1:
        raise ValueError("prompt_mask_probability must be in [0, 1]")
    _number(config["clip_epsilon"], "clip_epsilon", minimum=0, strict=True)
    if config["clip_epsilon"] >= 1:
        raise ValueError("clip_epsilon must be below 1")
    if not isinstance(config["betas"], (list, tuple)) or len(config["betas"]) != 2:
        raise ValueError("betas must contain two Adam coefficients")
    for index, value in enumerate(config["betas"]):
        _number(value, f"betas[{index}]", minimum=0)
        if value >= 1:
            raise ValueError("Adam betas must be below 1")
    _number(config["estimator_temperature"], "estimator_temperature", minimum=0, strict=True)
    if config["estimator_temperature"] != 1.0:
        raise ValueError("estimator_temperature must remain 1.0 for the documented d1 surrogate")
    _keys(config["rollout"], ROLLOUT_KEYS, "rollout")
    _keys(config["evaluation"], EVALUATION_KEYS, "evaluation")
    _decoding(config["rollout"], "rollout")
    _decoding(config["evaluation"], "evaluation")
    if any(config["rollout"][key] != config["evaluation"][key] for key in ROLLOUT_KEYS):
        raise ValueError("rollout and evaluation decoding settings must match in this experiment")
    for key in ("batch_size", "memory_mb"):
        _integer(config["evaluation"][key], f"evaluation.{key}")
    _seed(config["evaluation"]["seed"], "evaluation.seed")
    _number(config["evaluation"]["timeout_s"], "evaluation.timeout_s", minimum=0, strict=True)
    return config


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON configuration key: {key}")
        result[key] = value
    return result


def load_config(path):
    """Load a JSON configuration and resolve paths relative to the project."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    validate_config(config)
    for key in PATH_KEYS:
        value = Path(config[key])
        config[key] = str((value if value.is_absolute() else ROOT / value).resolve())
    return config


def apply_run_overrides(config, *, output_dir=None, max_steps=None,
                        max_wall_seconds=None, smoke=False, device="cuda"):
    """Apply CLI overrides consistently in the launcher and distributed workers."""
    config = copy.deepcopy(config)
    training_dir = Path(config["output_dir"]).resolve()
    if output_dir is not None:
        config["output_dir"] = str(Path(output_dir).resolve())
    if max_steps is not None:
        config["max_steps"] = max_steps
    if max_wall_seconds is not None:
        config["max_wall_seconds"] = max_wall_seconds
    if smoke:
        smoke_dir = Path(output_dir).resolve() if output_dir else training_dir.with_name(training_dir.name + "-smoke")
        if smoke_dir.is_relative_to(training_dir) or training_dir.is_relative_to(smoke_dir):
            raise ValueError("Smoke output must be separate from the training directory")
        config["output_dir"] = str(smoke_dir)
        config.update(max_steps=8, warmup_steps=2,
                      validation_every_steps=4, checkpoint_every_steps=4)
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if device == "cpu" and not smoke:
        raise ValueError("CPU execution is reserved for explicitly requested smoke checks")
    if device == "cuda" and config["world_size"] != 2:
        raise ValueError("CUDA RL requires world_size=2")
    return validate_config(config)
