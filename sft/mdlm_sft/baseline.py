"""Validate a completed experiment0 before reusing its baseline in SFT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL_KEYS = ("tokenizer_path", "pretrained_checkpoint", "seed", "noise_eps",
                 "precision", "validation_loss_batch_size", "validation_loss_seed", "evaluation")
VALIDATION_FILES = ("val.jsonl", "val.npz", "val_ids.json", "val_questions.jsonl")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def experiment1_checkpoint(directory):
    """Use the three-epoch weights for the step-609 diagnostics."""
    final = directory / "step-00000609.pt"
    if final.is_file():
        return final
    # Older archives retained only the selected checkpoint and its metadata.
    best = directory / "best.pt"
    if best.is_file() and read_json(directory / "best.json").get("step") == 609:
        return best
    return final


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def check_validation_identity(root, original, current, loss, protocol, entries):
    """Permit changed training data only when held-out artifacts are identical.

    The original manifest is anchored in both the input lock and experiment0's
    loss record. Comparing manifests alone would not detect stale/tampered files.
    """
    original_path, current_path = original / "manifest.json", current / "manifest.json"
    original_sha = sha256(original_path)
    locked_sha = next((row["sha256"] for row in entries
                       if resolve_path(root, row["path"]) == original_path), None)
    if original_sha != loss.get("data_manifest_sha256") or original_sha != locked_sha:
        raise ValueError("Baseline data manifest is not the recorded, locked dataset")
    previous, present = read_json(original_path), read_json(current_path)
    counts = previous.get("counts", {}).get("val", {})
    if (counts.get("questions") != 639 or counts.get("samples") != 1456
            or counts.get("target_tokens") != loss.get("target_tokens")
            or present.get("counts", {}).get("val") != counts):
        raise ValueError("Validation data counts differ from the baseline")
    for key in ("sequence_length", "tokenizer"):
        if key not in previous or present.get(key) != previous[key]:
            raise ValueError(f"Validation data encoding differs: {key}")
    if previous["tokenizer"] != protocol.get("tokenizer"):
        raise ValueError("Validation tokenizer differs from the baseline protocol")
    identities = {}
    for name in VALIDATION_FILES:
        expected = previous.get("artifacts_sha256", {}).get(name)
        if (not expected or present.get("artifacts_sha256", {}).get(name) != expected
                or sha256(original / name) != expected or sha256(current / name) != expected):
            raise ValueError(f"Validation artifact differs from the baseline: {name}")
        identities[name] = expected
    questions = [json.loads(line) for line in (current / "val_questions.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    question_sha = hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (len(questions) != 639 or protocol.get("questions_sha256") != question_sha
            or protocol.get("question_order") != [row["gsm8k_id"] for row in questions]):
        raise ValueError("Validation questions differ from the baseline protocol")
    return {"baseline_manifest_sha256": original_sha, "data_manifest_sha256": sha256(current_path),
            "validation_artifacts_sha256": identities, "questions_sha256": question_sha}


def check_baseline(root, baseline, config):
    """Require experiment0's complete, matching held-out evaluation before SFT."""
    root = Path(root).resolve()
    baseline = resolve_path(root, baseline)
    result = read_json(baseline / "result.json")
    metadata = read_json(baseline / "experiment.json")
    if result.get("experiment") != "experiment0" or result.get("status") != "complete":
        raise ValueError("Run experiment0 to completion before experiment1")
    if metadata.get("experiment") != "experiment0":
        raise ValueError("Baseline metadata does not identify experiment0")
    if metadata.get("inputs_lock_sha256") != sha256(root / "inputs.lock.json"):
        raise ValueError("Baseline input identity differs from the current input lock")
    for key in PROTOCOL_KEYS:
        expected, actual = metadata.get("config", {}).get(key), config.get(key)
        if key in ("tokenizer_path", "pretrained_checkpoint"):
            if expected is None or actual is None:
                raise ValueError(f"Baseline configuration missing: {key}")
            expected, actual = resolve_path(root, expected), resolve_path(root, actual)
        if expected != actual:
            raise ValueError(f"Baseline configuration differs: {key}")
    loss = read_json(baseline / "loss/summary.json")
    score = read_json(baseline / "validation/summary.json")
    if loss.get("complete") is not True or loss.get("code_samples") != 1456:
        raise ValueError("Baseline loss must cover all 1456 validation reference programs")
    if score.get("complete") is not True or score.get("num_questions") != 639 or score.get("question_count") != 639:
        raise ValueError("Baseline generation must cover all 639 validation questions")
    correct = score.get("counts", {}).get("correct")
    if type(correct) is not int or not 0 <= correct <= 639 or score.get("accuracy") != correct / 639:
        raise ValueError("Baseline accuracy must match the correct count out of 639 questions")
    entries = read_json(root / "inputs.lock.json")["files"]
    checkpoint = resolve_path(root, config["pretrained_checkpoint"])
    checkpoint_sha = next(row["sha256"] for row in entries if resolve_path(root, row["path"]) == checkpoint)
    for summary in (loss, score):
        source = summary.get("checkpoint_source", {})
        if source.get("source") != "pretrained_ema" or source.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Baseline checkpoint differs from the original pretrained EMA")
    protocol = read_json(baseline / "validation/protocol.json")
    digest = protocol.pop("protocol_sha256")
    if hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != digest or score.get("protocol_sha256") != digest:
        raise ValueError("Baseline protocol checksum mismatch")
    if protocol.get("generation") != config["evaluation"] or protocol.get("precision") != config["precision"]:
        raise ValueError("Baseline generation settings differ")
    source_files = protocol.get("evaluation_source", {})
    expected = {"evaluation.py", "sampling.py", "scoring.py", "executor.py", "data.py", "diffusion.py", "transformer.py", "runtime.py"}
    if set(source_files) != expected or any(sha256(root / "mdlm_sft" / name) != digest for name, digest in source_files.items()):
        raise ValueError("Evaluation code changed since experiment0; establish a new baseline")
    if score.get("predictions_sha256") != sha256(baseline / "validation/predictions.jsonl"):
        raise ValueError("Baseline predictions checksum mismatch")
    previous_data, current_data = metadata.get("config", {}).get("data_dir"), config.get("data_dir")
    if previous_data is None or current_data is None:
        raise ValueError("Baseline configuration missing: data_dir")
    previous_data, current_data = resolve_path(root, previous_data), resolve_path(root, current_data)
    validation_identity = (check_validation_identity(root, previous_data, current_data, loss, protocol, entries)
                           if previous_data != current_data else None)
    reference = {"directory": str(baseline), "result_sha256": sha256(baseline / "result.json"),
            "summary_sha256": sha256(baseline / "validation/summary.json"),
            "protocol_sha256": score["protocol_sha256"], "predictions_sha256": score["predictions_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "accuracy": score["accuracy"], "validation_loss": loss["loss"], "git": metadata["git"]}
    if validation_identity is not None:
        reference["validation_identity"] = validation_identity
    return reference
