"""Audited TinyGSM-style data encoding and grouped GSM8K-PoT splits.

``target_mask`` is a corruption/loss mask, NOT a Transformer attention mask.
Every sequence is independent, with bidirectional attention over all 512 slots.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
from typing import Any

TOKENIZER_ID = "HuggingFaceTB/SmolLM-135M"
TOKENIZER_REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
SEPARATOR = "\\n"  # Two characters: backslash, lower-case n; never a newline.
SEQUENCE_LENGTH = 512
TOKENIZER_FILES_SHA256 = {
    "config.json": "a1fe6f43e20f7a6c6dbc6380222af9526b5cef262446391a281c038249e3e3b7",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "238ad6b60d48e471624ea70bc79e92f2611844d5016471fee8c167854bcb98e8",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}
TOKENIZER_DEPENDENCIES = {
    "transformers": "4.45.0", "tokenizers": "0.20.3", "huggingface-hub": "0.30.2",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def question_sort_key(gsm8k_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"(train|test):(0|[1-9][0-9]*)", gsm8k_id)
    if match is None:
        raise ValueError(f"Invalid official question ID: {gsm8k_id!r}")
    return match.group(1), int(match.group(2))


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Check identity consistency before doing any split or tokenization."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        identity = row["gsm8k_id"]
        split, official_index = question_sort_key(identity)
        if split != "train":
            raise ValueError("SFT preparation only accepts official training IDs")
        if row.get("gsm8k_train_row_0based", official_index) != official_index:
            raise ValueError(f"Official row index disagrees with {identity}")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"Missing question for {identity}")
        if not isinstance(row["code"], str) or not row["code"].strip():
            raise ValueError(f"Missing code for {identity}")
        gold = row["gold_answer"]
        if isinstance(gold, bool) or not isinstance(gold, (int, float)) or not math.isfinite(gold):
            raise ValueError(f"Non-numeric gold answer for {identity}")
        members = groups.setdefault(identity, [])
        if members and (members[0]["question"] != row["question"] or members[0]["gold_answer"] != gold):
            raise ValueError(f"Inconsistent question or gold answer for {identity}")
        if any(previous["code"] == row["code"] for previous in members):
            raise ValueError(f"Duplicate code for {identity}; do not silently change the clean pool")
        members.append(row)
    if not groups:
        raise ValueError("Empty clean dataset")
    return groups


def split_grouped(
    rows: list[dict[str, Any]], *, seed: int = 42, val_questions: int = 639,
    expected_questions: int | None = 6386, expected_samples: int | None = 14409,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """Numerically sort IDs, shuffle with a private Python Random(seed), take val first.

    Persisted ID lists, rather than assumptions about future PRNG versions, are
    the authority for reproducing an experiment. Input row order does not affect
    assignment or output ordering. No global RNG state is touched.
    """
    groups = group_rows(rows)
    if expected_questions is not None and len(groups) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, got {len(groups)}")
    if expected_samples is not None and len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {len(rows)}")
    if not 0 < val_questions < len(groups):
        raise ValueError("Validation size must leave nonempty train and validation splits")
    shuffled = sorted(groups, key=question_sort_key)
    random.Random(seed).shuffle(shuffled)
    ids = {
        "val": sorted(shuffled[:val_questions], key=question_sort_key),
        "train": sorted(shuffled[val_questions:], key=question_sort_key),
    }
    partitions = {
        split: [row for key in keys for row in sorted(groups[key], key=lambda row: row["code"])]
        for split, keys in ids.items()
    }
    if set(ids["train"]) & set(ids["val"]) or sum(map(len, partitions.values())) != len(rows):
        raise AssertionError("Grouped split lost records or leaked question IDs")
    return partitions, ids


def load_tokenizer(path: str | Path | None = None, *, local_files_only: bool = False):
    """Load only the audited revision and fail closed on file/version differences.

    PAD is appended exactly as upstream. MASK remains an extra model vocabulary
    entry at ``len(tokenizer)``; it is deliberately NOT added to the tokenizer.
    Consequently use ``len(tokenizer) + 1`` for model vocab size, not the native
    GPT2 tokenizer's ``vocab_size`` property (which excludes the appended PAD).
    """
    for package, expected in TOKENIZER_DEPENDENCIES.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package}=={expected} required, found {actual}")
    from huggingface_hub import snapshot_download
    from tokenizers import processors
    from transformers import AutoTokenizer, GPT2TokenizerFast

    if path is None or str(path) == TOKENIZER_ID:
        path = snapshot_download(
            TOKENIZER_ID, revision=TOKENIZER_REVISION,
            allow_patterns=list(TOKENIZER_FILES_SHA256), local_files_only=local_files_only,
        )
    path = Path(path)
    for name, expected in TOKENIZER_FILES_SHA256.items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"Tokenizer file does not match audited revision: {name}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if not isinstance(tokenizer, GPT2TokenizerFast):
        raise ValueError("Expected GPT2TokenizerFast from Transformers 4.45.0")
    tokenizer._tokenizer.post_processor = processors.BertProcessing(
        (tokenizer.bos_token, tokenizer.bos_token_id),
        (tokenizer.eos_token, tokenizer.eos_token_id),
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, len(tokenizer)) != (0, 0, 49152, 49153):
        raise ValueError("Tokenizer special-token IDs or vocabulary size changed")
    if tokenizer.mask_token_id is not None:
        raise ValueError("Audited SmolLM tokenizer must not already have a MASK token")
    pretokenizer = json.loads(tokenizer.backend_tokenizer.pre_tokenizer.__getstate__())
    expected_pretokenizer = {
        "type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        ],
    }
    if pretokenizer != expected_pretokenizer:
        raise ValueError("Audited digit-wise pretokenizer changed")
    separator_ids = tokenizer.encode(SEPARATOR, add_special_tokens=False)
    # Independently read from the hash-verified serialized tokenizer backend.
    if separator_ids != [76, 94]:
        raise ValueError("Literal backslash-n separator differs from audited token IDs")
    tokenizer.mdlm_source_path = str(path.resolve())
    return tokenizer


def tokenizer_identity(tokenizer) -> dict[str, Any]:
    return {
        "repo": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
        "files_sha256": TOKENIZER_FILES_SHA256,
        "dependencies": TOKENIZER_DEPENDENCIES,
        "bos_id": tokenizer.bos_token_id, "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id, "mask_id": len(tokenizer),
        "vocab_without_mask": len(tokenizer), "model_vocab_size": len(tokenizer) + 1,
        "separator": SEPARATOR, "separator_ids": tokenizer.encode(SEPARATOR, add_special_tokens=False),
    }


def _prefix(question: str, tokenizer) -> list[int]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question is required")
    return [tokenizer.bos_token_id] + tokenizer.encode(question.strip(), add_special_tokens=False) + tokenizer.encode(SEPARATOR, add_special_tokens=False)


def encode_example(row: dict[str, Any], tokenizer, length: int = SEQUENCE_LENGTH) -> dict[str, Any]:
    prefix = _prefix(row["question"], tokenizer)
    code_ids = tokenizer.encode(row["code"].strip(), add_special_tokens=False)
    sequence = prefix + code_ids + [tokenizer.eos_token_id]
    if len(sequence) > length:
        raise ValueError(f"{row.get('gsm8k_id', '<fixture>')} has {len(sequence)} tokens > {length}; no truncation allowed")
    for field, actual in (("question_tokens", len(prefix) - 1 - len(tokenizer.encode(SEPARATOR, add_special_tokens=False))), ("code_tokens", len(code_ids)), ("total_tokens", len(sequence))):
        if field in row and row[field] != actual:
            raise ValueError(f"Audited {field} changed for {row.get('gsm8k_id')}: {row[field]} != {actual}")
    return {
        "input_ids": sequence + [tokenizer.pad_token_id] * (length - len(sequence)),
        "target_mask": [False] * len(prefix) + [True] * (length - len(prefix)),
        "prefix_length": len(prefix), "sequence_length": len(sequence),
        "gsm8k_id": row.get("gsm8k_id", ""),
    }


def encode_prompt(question: str, tokenizer, length: int = SEQUENCE_LENGTH, *, mask_id: int | None = None) -> dict[str, Any]:
    """Generation API accepts ONLY the question; all target slots start as MASK."""
    prefix = _prefix(question, tokenizer)
    if len(prefix) >= length:
        raise ValueError(f"Question prefix occupies {len(prefix)} slots; no generation space")
    mask_id = len(tokenizer) if mask_id is None else mask_id
    return {
        "input_ids": prefix + [mask_id] * (length - len(prefix)),
        "target_mask": [False] * len(prefix) + [True] * (length - len(prefix)),
        "prefix_length": len(prefix),
    }


def verify_prepared(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("Unsupported prepared-data schema or sequence length")
    for name, expected in manifest["artifacts_sha256"].items():
        if Path(name).name != name or sha256_file(directory / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    return manifest


class PreparedDataset:
    """Small in-memory arrays; no stochastic preprocessing or worker RNG state.

    ``__getitem__`` imports torch lazily, so data preparation itself does not
    require a CUDA/PyTorch installation. DataLoader's default collate works.
    """
    def __init__(self, directory: str | Path, split: str = "train"):
        import numpy as np
        if split not in {"train", "val"}:
            raise ValueError("Only prepared train/val code samples may be loaded")
        self.manifest = verify_prepared(directory)
        with np.load(Path(directory) / f"{split}.npz", allow_pickle=False) as cache:
            self.input_ids = cache["input_ids"].copy()
            self.target_mask = cache["target_mask"].copy()
            self.prefix_lengths = cache["prefix_lengths"].copy()
            self.gsm8k_ids = cache["gsm8k_ids"].tolist()
        self.target_counts = self.target_mask.sum(axis=1).tolist()
        if self.input_ids.shape != self.target_mask.shape or self.input_ids.shape != (self.manifest["counts"][split]["samples"], SEQUENCE_LENGTH):
            raise ValueError("Prepared tensor shapes disagree with manifest")
        expected_mask = np.arange(SEQUENCE_LENGTH)[None, :] >= self.prefix_lengths[:, None]
        if not np.array_equal(self.target_mask, expected_mask):
            raise ValueError("Prepared target mask disagrees with prefix boundaries")
        if not ((self.input_ids >= 0) & (self.input_ids < self.manifest["tokenizer"]["vocab_without_mask"])).all():
            raise ValueError("Clean target cache contains MASK or invalid token IDs")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        import torch
        return {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "target_mask": torch.tensor(self.target_mask[index], dtype=torch.bool),
            "prefix_length": int(self.prefix_lengths[index]),
            "gsm8k_id": self.gsm8k_ids[index],
        }


def load_questions(directory: str | Path, split: str = "val") -> list[dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError("Prepared question splits are train/val only")
    manifest = verify_prepared(directory)
    rows = read_jsonl(Path(directory) / f"{split}_questions.jsonl")
    ids = [row["gsm8k_id"] for row in rows]
    if len(ids) != len(set(ids)) or ids != sorted(ids, key=question_sort_key):
        raise ValueError("Questions are not unique and numerically ordered")
    if len(rows) != manifest["counts"][split]["questions"]:
        raise ValueError("Question count disagrees with manifest")
    if any("code" in row for row in rows):
        raise ValueError("Generation question file must not contain reference code")
    return rows
