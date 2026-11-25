"""Check prepared training data against its manifest without fixed sample counts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import unicodedata

from .data import load_questions, question_sort_key, read_jsonl, verify_prepared


PREPARED_ARTIFACTS = {
    f"{split}{suffix}"
    for split in ("train", "val")
    for suffix in (".jsonl", ".npz", "_ids.json", "_questions.jsonl")
}


def training_question_sort_key(identity):
    """Order official training questions before explicitly namespaced replay."""
    match = re.fullmatch(r"tinygsm:(0|[1-9][0-9]*)", identity)
    if match:
        return 1, int(match.group(1))
    split, index = question_sort_key(identity)
    if split != "train":
        raise ValueError("SFT training must use official training questions only, plus declared TinyGSM replay")
    return 0, index


def _check_mixed_question(row):
    source, _ = training_question_sort_key(row["gsm8k_id"])
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("Mixed training question text must be nonempty")
    if source == 1:
        if (row.get("source") != "TinyGSM"
                or row.get("answer_provenance") != "not_provided_by_source"
                or "gold_answer" not in row or row["gold_answer"] is not None):
            raise ValueError("TinyGSM replay requires source markers and an explicitly null gold_answer")
    else:
        if row.get("source") == "TinyGSM":
            raise ValueError("TinyGSM replay must not impersonate an official question ID")
        gold = row.get("gold_answer")
        if isinstance(gold, bool) or not isinstance(gold, (int, float)) or not math.isfinite(gold):
            raise ValueError("Official training questions require a finite numeric gold_answer")


def load_training_questions(directory):
    """Load replay identities without changing the frozen evaluation data module.

    TinyGSM has no independent gold answers. Its execution results must never
    become evaluation labels; only a declared training mixture accepts nulls.
    """
    directory = Path(directory)
    manifest = verify_prepared(directory)
    if manifest.get("derivation", {}).get("kind") != "tinygsm_replay_mix":
        return load_questions(directory, "train")
    questions = read_jsonl(directory / "train_questions.jsonl")
    for row in questions:
        _check_mixed_question(row)
        if "code" in row:
            raise ValueError("Training question file must not contain reference code")
    ids = [row["gsm8k_id"] for row in questions]
    if len(ids) != len(set(ids)) or ids != sorted(ids, key=training_question_sort_key):
        raise ValueError("Mixed training questions must be unique and ordered official-first")
    if len(questions) != manifest["counts"]["train"]["questions"]:
        raise ValueError("Training question count disagrees with manifest")
    return questions


def labeled_training_questions(directory, limit=8):
    """Select only real official labels for smoke generation scoring."""
    questions = [
        row for row in load_training_questions(directory)
        if training_question_sort_key(row["gsm8k_id"])[0] == 0]
    if not questions:
        raise ValueError("Smoke generation requires labeled official training questions")
    return questions[:limit]


def _question_text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_training_data(directory, train_data, val_data, val_questions):
    """Validate counts, cache order and question isolation before any optimizer step.

    PreparedDataset already checks tensor shapes, masks and token ranges. This
    cross-check also requires all eight checksummed files, so a shortened
    manifest cannot silently leave an input unverified. No RNG is consumed.
    """
    directory = Path(directory)
    manifest = verify_prepared(directory)
    if set(manifest["artifacts_sha256"]) != PREPARED_ARTIFACTS:
        raise ValueError("Prepared manifest must checksum all eight train/validation artifacts")
    is_mix = manifest.get("derivation", {}).get("kind") == "tinygsm_replay_mix"
    split_ids = {}
    replay_rows = []
    official_texts = {_question_text_key(row["question"]) for row in val_questions}
    for split, dataset in (("train", train_data), ("val", val_data)):
        if dataset.manifest != manifest:
            raise ValueError("Prepared manifest changed after loading data")
        counts = manifest["counts"][split]
        if any(type(counts[key]) is not int or counts[key] <= 0
               for key in ("samples", "questions", "target_tokens")):
            raise ValueError(f"Invalid {split} counts in prepared manifest")
        if len(dataset) != counts["samples"] or sum(dataset.target_counts) != counts["target_tokens"]:
            raise ValueError(f"{split} sample/target-token count disagrees with manifest")
        questions = (load_training_questions(directory) if split == "train"
                     else load_questions(directory, split))
        if split == "val" and questions != val_questions:
            raise ValueError("Loaded validation questions disagree with prepared data")
        ids = [row["gsm8k_id"] for row in questions]
        saved_ids = json.loads((directory / f"{split}_ids.json").read_text(encoding="utf-8"))
        if saved_ids != ids or set(dataset.gsm8k_ids) != set(ids):
            raise ValueError(f"{split} question IDs disagree across cache, questions and split list")
        if not (is_mix and split == "train") and any(
                question_sort_key(identity)[0] != "train" for identity in ids):
            raise ValueError("SFT training and validation must use official training questions only")
        rows = read_jsonl(directory / f"{split}.jsonl")
        if len(rows) != len(dataset) or [row["gsm8k_id"] for row in rows] != dataset.gsm8k_ids:
            raise ValueError(f"{split} reference rows disagree with cache count/order")
        by_id = {row["gsm8k_id"]: row for row in questions}
        if any(any(row[key] != by_id[row["gsm8k_id"]][key] for key in ("question", "gold_answer"))
               for row in rows):
            raise ValueError(f"{split} reference question/answer differs from generation questions")
        if is_mix and split == "train":
            for row in rows:
                _check_mixed_question(row)
                if training_question_sort_key(row["gsm8k_id"])[0] == 1:
                    replay_rows.append(row)
                else:
                    official_texts.add(_question_text_key(row["question"]))
        split_ids[split] = set(ids)
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Training and validation question IDs overlap")
    if is_mix:
        mixing = manifest.get("mixing", {})
        parent_count, replay_count = mixing.get("parent_samples"), mixing.get("replay_samples")
        if (type(parent_count) is not int or type(replay_count) is not int
                or min(parent_count, replay_count) <= 0
                or len(replay_rows) != replay_count
                or parent_count + replay_count != len(train_data)):
            raise ValueError("Replay and parent sample counts disagree with mixture manifest")
        if any(_question_text_key(row["question"]) in official_texts for row in replay_rows):
            raise ValueError("TinyGSM replay overlaps official training or validation question text")
        replay_texts = [_question_text_key(row["question"]) for row in replay_rows]
        if len(set(replay_texts)) != len(replay_texts):
            raise ValueError("TinyGSM replay question text is duplicated")
    return manifest["counts"]
