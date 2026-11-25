"""Explicit pretrained EMA-to-ordinary-weight initialization.

This entry point never restores optimizers, loops, counters, RNG or data
progress. SFT resume belongs to train.py and checkpoints.py.
"""
from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path
from types import ModuleType

import torch

from .checkpoints import LEGACY_FORMAT_DIGESTS, is_sft_schema, matches_legacy

UPSTREAM_COMMIT = "30def79976ca3b1cdcb5fcbfa0165524e44001de"
# Reference revisions for the mapping below; no upstream checkout is needed
# at runtime. Actual checkpoint keys, shapes, order and copied tensors are
# validated by initialize_pretrained.
UPSTREAM_MAPPING_SOURCE_SHA256_LF = {
    "models/dit.py": "7b13ae13ac2c8f3c776238c231b803f5dc6399d97c501b428e512d885a69bc81",
    "models/ema.py": "f1417e66a46b2c2144e3fd47bb5c91c819f8c55f6bd18c75b72a75dc36bafb60",
    "trainer_base.py": "d2c41e92bca8b78790f44e275a262e3cb4a33bb0ce0f2353755a9d1a6a2c03b0",
}


class _TokenizerMetadata:
    """Unpickling-only stand-in for upstream's tokenizer wrapper.

    The checkpoint's serialized tokenizer is never used for encoding. Avoid
    importing upstream dataloader (and the complete unrelated training stack)
    just to deserialize its wrapper. Its state remains available for auditing.
    """
    pass


class _CheckpointUnpickler(pickle.Unpickler):
    # Observed pickle globals in the supplied published checkpoint. Unknown
    # types fail closed; no import of checkpoint-provided Python source.
    ALLOWED = {
        ("__builtin__", "dict"), ("__builtin__", "list"), ("__builtin__", "long"),
        ("builtins", "dict"), ("builtins", "list"), ("builtins", "int"),
        ("_codecs", "encode"), ("collections", "OrderedDict"),
        ("collections", "defaultdict"), ("typing", "Any"),
        ("omegaconf.base", "ContainerMetadata"), ("omegaconf.base", "Metadata"),
        ("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
        ("omegaconf.nodes", "AnyNode"), ("tokenizers", "AddedToken"),
        ("tokenizers", "Tokenizer"), ("tokenizers.models", "Model"),
        ("torch", "DoubleStorage"), ("torch", "FloatStorage"),
        ("torch", "HalfStorage"), ("torch", "BFloat16Storage"),
        ("torch", "LongStorage"), ("torch", "ByteStorage"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("transformers.models.gpt2.tokenization_gpt2_fast", "GPT2TokenizerFast"),
    }

    def find_class(self, module, name):
        if (module, name) == ("dataloader", "VocabSizeTokenizerWrapper"):
            return _TokenizerMetadata
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Unexpected checkpoint global {module}.{name}")
        return super().find_class(module, name)


_PICKLE = ModuleType("mdlm_checkpoint_pickle")
_PICKLE.Unpickler = _CheckpointUnpickler
_PICKLE.load = lambda stream, **kwargs: _CheckpointUnpickler(stream, **kwargs).load()
_PICKLE.loads = lambda value, **kwargs: _CheckpointUnpickler(io.BytesIO(value), **kwargs).load()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pretrained_checkpoint(path):
    """Read user-supplied published checkpoint on CPU without adopting state.

    mmap avoids allocating pretrained Adam states. PyTorch's weights-only
    loader cannot read the serialized OmegaConf/tokenizer hyperparameters;
    the explicit restricted unpickler permits only observed library classes.
    This is still intended for the published checkpoint, not arbitrary files.
    """
    return torch.load(Path(path), map_location="cpu", mmap=True,
                      weights_only=False, pickle_module=_PICKLE)


def _clean_state_dict(state):
    cleaned = {}
    for name, value in state.items():
        name = name.replace("._orig_mod.", ".")
        if name in cleaned:
            raise ValueError(f"Duplicate state key after compile-prefix removal: {name}")
        cleaned[name] = value
    return cleaned


def _check_architecture_metadata(model, checkpoint):
    config = checkpoint.get("hyper_parameters", {}).get("config")
    if config is None:
        return {"available": False}
    from omegaconf import OmegaConf
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=False)
    expected = {
        "algo.name": "mdlm", "algo.time_conditioning": False,
        "algo.causal_attention": False, "algo.adaLN": True,
        "algo.parameterization": "subs", "algo.T": 0, "algo.loss_type": "elbo",
        "noise.type": "log-linear", "noise.eps": model.config.noise_eps,
        "model.hidden_size": model.config.hidden_size,
        "model.cond_dim": model.config.cond_dim,
        "model.n_blocks": model.config.n_blocks,
        "model.n_heads": model.config.n_heads,
        "model.dropout": model.config.dropout,
        "model.length": model.config.length,
    }
    checked = {}
    for dotted, value in expected.items():
        observed = config
        for key in dotted.split("."):
            if not isinstance(observed, dict) or key not in observed:
                raise ValueError(f"Missing checkpoint architecture metadata: {dotted}")
            observed = observed[key]
        if observed != value:
            raise ValueError(f"Checkpoint {dotted}={observed!r}, expected {value!r}")
        checked[dotted] = observed
    # Older published configurations may predate these fields. If present,
    # none may silently alter the default schedule or training objective.
    for section, key, expected_value in [("noise", "adaptive", False),
                                          ("noise", "alpha_min", None),
                                          ("noise", "alpha_max", None),
                                          ("algo", "post_process_mode", "log_probs")]:
        if key in config.get(section, {}) and config[section][key] != expected_value:
            raise ValueError(f"Unsupported checkpoint configuration: {section}.{key}")
    return {"available": True, "checked": checked}


def _check_embedded_tokenizer(model, checkpoint):
    embedded = checkpoint.get("hyper_parameters", {}).get("tokenizer")
    if embedded is None:
        return {"available": False}
    if isinstance(embedded, _TokenizerMetadata):
        embedded = vars(embedded).get("_tokenizer")
    if embedded is None:
        raise ValueError("Malformed embedded tokenizer metadata")
    # The published tokenizer pickle uses a newer Transformers attribute
    # layout (_special_tokens_map); invoking 4.45 properties on that object
    # would fail on missing _bos_token. Inspect its serialized metadata and
    # Rust backend directly; never use the embedded object for encoding.
    state = vars(embedded)
    backend = state.get("_tokenizer")
    if backend is None:
        raise ValueError("Missing embedded tokenizer backend metadata")
    special = state.get("_special_tokens_map", {})
    def special_id(name):
        value = special.get(name, state.get("_" + name))
        if value is None:
            return None
        identity = backend.token_to_id(str(value))
        if identity is None:
            raise ValueError(f"Embedded special token is absent from vocabulary: {name}")
        return identity
    observed = {"vocab_without_mask": backend.get_vocab_size(with_added_tokens=True),
                "bos_id": special_id("bos_token"), "eos_id": special_id("eos_token"),
                "pad_id": special_id("pad_token"), "mask_id": special_id("mask_token")}
    expected = {"vocab_without_mask": model.vocab_size - 1, "bos_id": 0,
                "eos_id": 0, "pad_id": model.mask_index - 1, "mask_id": None}
    if observed != expected:
        raise ValueError(f"Embedded checkpoint tokenizer disagrees: {observed}")
    return {"available": True, "checked": observed,
            "metadata_layout": "_special_tokens_map" if special else "legacy_individual_attributes"}


def initialize_pretrained(model, path=None, *, checkpoint=None):
    """Copy validated EMA parameters into the ordinary model; return audit.

    `checkpoint=` is useful for independent fixtures. Real runs use `path=`
    so the identity/hash is recorded. Missing EMA alone permits ordinary
    weights; malformed/mismatched EMA never silently falls back.
    """
    if (path is None) == (checkpoint is None):
        raise ValueError("Supply exactly one pretrained path or checkpoint")
    if checkpoint is None:
        checkpoint = load_pretrained_checkpoint(path)
    if (is_sft_schema(checkpoint.get("schema"))
            or checkpoint.get("format") == "mdlm_sft_v1"
            or matches_legacy(checkpoint.get("format"), LEGACY_FORMAT_DIGESTS)
            or "model_state" in checkpoint):
        raise ValueError("SFT checkpoint must use the explicit resume/evaluation path")
    if "state_dict" not in checkpoint:
        raise ValueError("Not an upstream pretrained checkpoint")
    architecture = _check_architecture_metadata(model, checkpoint)
    embedded_tokenizer = _check_embedded_tokenizer(model, checkpoint)
    state = _clean_state_dict(checkpoint["state_dict"])
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"State keys differ: missing={set(expected)-set(state)}, "
                         f"unexpected={set(state)-set(expected)}")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or tensor.shape != expected[name].shape:
            raise ValueError(f"State shape mismatch: {name}")
    # Exact upstream EMA constructor order: backbone params, then noise params;
    # requires_grad filter happens inside ExponentialMovingAverage.__init__.
    ordered = [("backbone." + name, parameter)
               for name, parameter in model.backbone.named_parameters()
               if parameter.requires_grad]
    ordered += [("noise." + name, parameter)
                for name, parameter in model.noise.named_parameters()
                if parameter.requires_grad]
    expected_names = [name for name, _ in ordered]
    parameter_names = set(expected_names)
    if [name for name in state if name in parameter_names] != expected_names:
        raise ValueError("Checkpoint parameter insertion order disagrees with upstream EMA order")
    selected = dict(state)
    mapping = []
    source = "ordinary_no_ema"
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        if not isinstance(ema, dict) or not isinstance(ema.get("shadow_params"), (list, tuple)):
            raise ValueError("EMA exists but is malformed")
        shadows = ema["shadow_params"]
        if len(shadows) != len(ordered):
            raise ValueError(f"EMA count {len(shadows)} != expected {len(ordered)}")
        for index, ((name, parameter), shadow) in enumerate(zip(ordered, shadows)):
            if not torch.is_tensor(shadow) or shadow.shape != parameter.shape:
                raise ValueError(f"EMA mapping shape mismatch at {index}: {name}")
            if shadow.dtype != state[name].dtype or not shadow.is_floating_point():
                raise ValueError(f"EMA mapping dtype mismatch at {index}: {name}")
            selected[name] = shadow
            mapping.append({"index": index, "name": name,
                            "shape": list(shadow.shape), "dtype": str(shadow.dtype)})
        source = "pretrained_ema"
    for name, tensor in selected.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite selected checkpoint tensor: {name}")
    model.load_state_dict(selected, strict=True)
    # Verify the actual post-copy tensors, not just successful load_state_dict.
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), selected[name].cpu()):
            raise ValueError(f"Selected weight did not copy exactly: {name}")
    return {
        "source": source, "upstream_commit": UPSTREAM_COMMIT,
        "upstream_reference_source_sha256_lf": dict(UPSTREAM_MAPPING_SOURCE_SHA256_LF),
        "adapted_dit_sha256": sha256_file(Path(__file__).with_name("transformer.py")),
        "checkpoint_path": str(Path(path).resolve()) if path is not None else None,
        "checkpoint_sha256": sha256_file(path) if path is not None else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "ema_mapping": mapping, "architecture": architecture,
        "embedded_tokenizer": embedded_tokenizer,
        "ema_order_basis": "backbone.named_parameters then noise.named_parameters; checked against state_dict insertion order",
        "pretrained_progress_ignored": {"epoch": checkpoint.get("epoch"),
                                         "global_step": checkpoint.get("global_step")},
        "copied_exactly": True, "sft_ema_enabled": False,
        "vocab_size": model.vocab_size, "mask_index": model.mask_index,
        "optimizer_scheduler_rng_data_progress_restored": False,
    }
