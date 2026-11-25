"""Execute original and style-aligned references in the unchanged Linux sandbox.

No extraction, syntax repair, sample filtering, model loading or GPU use occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mdlm_sft.data import read_jsonl, sha256_file, verify_prepared
from mdlm_sft.scoring import (
    ExecutionConfig, InfrastructureError, SCORER_VERSION, numbers_equal,
    preflight_executor, run_program,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def code_digest(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_pairs(parent, data, splits=("train", "val")):
    original_manifest, current_manifest = verify_prepared(parent), verify_prepared(data)
    if original_manifest["tokenizer"] != current_manifest["tokenizer"]:
        raise ValueError("Style transformation changed the tokenizer")
    pairs = []
    for split in splits:
        before, after = read_jsonl(parent / f"{split}.jsonl"), read_jsonl(data / f"{split}.jsonl")
        if len(before) != len(after) or len(before) != current_manifest["counts"][split]["samples"]:
            raise ValueError(f"Sample count changed: {split}")
        for suffix in ("_ids.json", "_questions.jsonl"):
            if sha256_file(parent / f"{split}{suffix}") != sha256_file(data / f"{split}{suffix}"):
                raise ValueError(f"Question identities, order or gold answers changed: {split}{suffix}")
        if split == "val" and before != after:
            raise ValueError("Validation references must remain unchanged")
        for index, (old, new) in enumerate(zip(before, after)):
            for key in ("gsm8k_id", "question", "gold_answer", "source_rows_0based"):
                if old.get(key) != new.get(key):
                    raise ValueError(f"Reference identity changed: {split}:{index}:{key}")
            for side, row in (("original", old), ("transformed", new)):
                if row.get("code_sha256") != code_digest(row["code"]):
                    raise ValueError(f"Code checksum mismatch: {split}:{index}:{side}")
            identity = {"split": split, "row_index": index, "gsm8k_id": old["gsm8k_id"],
                        "question_sha256": code_digest(old["question"]),
                        "gold_answer": old["gold_answer"],
                        "original_code_sha256": code_digest(old["code"]),
                        "transformed_code_sha256": code_digest(new["code"])}
            pairs.append({**identity, "row_sha256": digest(identity),
                          "original_code": old["code"], "transformed_code": new["code"]})
    return pairs


def score_result(result, gold):
    correct = result["status"] == "ok" and numbers_equal(result["predicted_answer"], gold)
    return {**result, "correct": correct,
            "score_status": ("correct" if correct else "wrong_answer")
            if result["status"] == "ok" else result["status"]}


def check_pair(pair, config):
    evidence = {key: value for key, value in pair.items()
                if key not in ("original_code", "transformed_code")}
    evidence["shared_execution"] = pair["original_code"] == pair["transformed_code"]
    started = time.monotonic()
    try:
        before = run_program(pair["original_code"], config)
        after = before if evidence["shared_execution"] else run_program(pair["transformed_code"], config)
        evidence["original"] = score_result(before, pair["gold_answer"])
        evidence["transformed"] = score_result(after, pair["gold_answer"])
        # Exact observed worker protocol equality, not a tolerance-based gold comparison.
        fields = ("status", "predicted_answer", "raw_type", "used_stdout_fallback", "stdout", "error")
        evidence["behavior_equal"] = all(before.get(key) == after.get(key) for key in fields)
        evidence["changed_fields"] = [key for key in fields if before.get(key) != after.get(key)]
        evidence["infrastructure_error"] = None
    except InfrastructureError as error:
        evidence["infrastructure_error"] = str(error)
        evidence["behavior_equal"] = None
    evidence["wall_seconds"] = time.monotonic() - started
    return evidence


def load_completed(path, pairs):
    if not path.exists():
        return []
    completed = read_jsonl(path)
    if len(completed) > len(pairs):
        raise ValueError("Saved checks contain more records than the selected dataset")
    for index, record in enumerate(completed):
        if record.get("row_sha256") != pairs[index]["row_sha256"]:
            raise ValueError(f"Saved check belongs to a different row at index {index}")
        if record.get("infrastructure_error") is not None:
            raise ValueError("Saved check contains an infrastructure error; use a new output directory")
        if "original" not in record or "transformed" not in record or type(record.get("behavior_equal")) is not bool:
            raise ValueError(f"Incomplete saved check at index {index}")
    return completed


def summarize(records, expected, full_count):
    splits = {}
    for split in ("train", "val"):
        selected = [r for r in records if r["split"] == split]
        valid = [r for r in selected if r.get("infrastructure_error") is None]
        sides = {}
        for side in ("original", "transformed"):
            counts = Counter(row[side]["score_status"] for row in valid)
            sides[side] = {"counts": dict(sorted(counts.items())),
                           "runnable": sum(row[side]["status"] == "ok" for row in valid),
                           "correct": counts["correct"],
                           "incorrect_or_failed": len(valid) - counts["correct"]}
        splits[split] = {"checked": len(selected), **sides,
                         "behavior_changes": sum(r.get("behavior_equal") is False for r in selected),
                         "infrastructure_errors": len(selected) - len(valid),
                         "shared_executions": sum(r["shared_execution"] for r in selected)}
    errors = sum(v["infrastructure_errors"] for v in splits.values())
    changes = sum(v["behavior_changes"] for v in splits.values())
    complete = len(records) == expected and errors == 0
    return {"status": "complete" if complete else "incomplete", "complete": complete,
            "scope": "full_train_and_validation" if expected == full_count else "limited_self_check",
            "selected_samples": expected, "full_dataset_samples": full_count,
            "completed_samples": len(records), "splits": splits,
            "all_observed_behaviors_preserved": complete and changes == 0,
            "all_transformed_answers_correct": complete and all(
                s["transformed"]["incorrect_or_failed"] == 0 for s in splits.values()),
            "behavior_comparison": "Exact worker status, numeric answer, raw return type, stdout, fallback flag and error",
            "gold_comparison": "Unchanged scoring.numbers_equal; tolerance 1e-3 for floats",
            "execution_policy": "Unchanged Linux RLIMIT + seccomp + AST capability policy; raw code without extraction or repair",
            "scorer_version": SCORER_VERSION}


def validate(args):
    started = time.monotonic()
    pairs = load_pairs(args.parent_dir, args.data_dir)
    full_count = len(pairs)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        pairs = pairs[:args.limit]
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be between 1 and 32")
    config = ExecutionConfig(args.timeout_s, args.memory_mb)
    config.validate()
    sources = {"script": Path(__file__), "scoring": ROOT / "mdlm_sft/scoring.py",
               "executor": ROOT / "mdlm_sft/executor.py"}
    identity = {"parent_manifest_sha256": sha256_file(args.parent_dir / "manifest.json"),
                "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
                "code_sha256": {key: sha256_file(value) for key, value in sources.items()},
                "execution_config": asdict(config), "limit": args.limit,
                "python_version": platform.python_version(), "scorer_version": SCORER_VERSION}
    out = args.output_dir
    if out.resolve() in (args.parent_dir.resolve(), args.data_dir.resolve()):
        raise ValueError("Execution evidence must not be written into a dataset directory")
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        if not args.resume:
            raise ValueError("Output already exists; pass --resume or use a new output directory")
        if json.loads(protocol_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume protocol differs in data, code or runtime")
    elif any(out.iterdir()):
        raise ValueError("Output directory is nonempty without an audit protocol")
    else:
        write_json(protocol_path, identity)
    records = load_completed(out / "checks.jsonl", pairs)
    try:
        preflight = preflight_executor(config)
        write_json(out / "preflight.json", preflight)
        # Keep at most workers tasks in flight and preserve input order. A broken
        # executor stops the audit without queuing thousands of more processes.
        with (out / "checks.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                remaining = iter(pairs[len(records):])
                pending = [pool.submit(check_pair, pair, config)
                           for pair in [next(remaining, None) for _ in range(args.workers)]
                           if pair is not None]
                while pending:
                    record = pending.pop(0).result()
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    if record.get("infrastructure_error") is not None:
                        raise InfrastructureError(record["infrastructure_error"])
                    if len(records) % 128 == 0 or len(records) == len(pairs):
                        print(json.dumps({"execution_checked": len(records), "total": len(pairs)}), flush=True)
                    pair = next(remaining, None)
                    if pair is not None:
                        pending.append(pool.submit(check_pair, pair, config))
    except BaseException as error:
        report = summarize(records, len(pairs), full_count)
        report.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       "complete": False, "error": f"{type(error).__name__}: {error}",
                       "wall_seconds_this_invocation": time.monotonic() - started})
        write_json(out / "report.json", report)
        raise
    report = summarize(records, len(pairs), full_count)
    report.update({"wall_seconds_this_invocation": time.monotonic() - started,
                   "checks_sha256": sha256_file(out / "checks.jsonl"),
                   "protocol_sha256": sha256_file(protocol_path)})
    write_json(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["all_observed_behaviors_preserved"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    raise SystemExit(validate(parser.parse_args()))


if __name__ == "__main__":
    main()
