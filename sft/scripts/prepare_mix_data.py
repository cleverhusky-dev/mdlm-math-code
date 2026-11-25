"""Prepare TinyGSM replay data: select locally, check on Linux, then assemble.

TinyGSM supplies question/code pairs, not independently verified numeric labels.
Successful execution is recorded separately; it is never promoted to gold.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdlm_sft.data import (
    PreparedDataset, encode_example, load_tokenizer, read_jsonl,
    sha256_file, tokenizer_identity, verify_prepared,
)
from mdlm_sft.executor import validate_ast
from mdlm_sft.prepare_data import write_json, write_jsonl
from mdlm_sft.scoring import ExecutionConfig, SCORER_VERSION, preflight_executor, run_program


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_question(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"^\s*(?:student|question)\s*:\s*", "", value)
    return " ".join(re.findall(r"\w+", value))


def shingles(value):
    words = normalized_question(value).split()
    return {tuple(words[i:i + 4]) for i in range(len(words) - 3)}


class QuestionExclusions:
    """Exact normalized and conservative lexical near-duplicate exclusion."""
    def __init__(self, questions):
        self.exact = {normalized_question(q) for q in questions}
        self.grams = [shingles(q) for q in questions]
        self.index = defaultdict(set)
        for i, grams in enumerate(self.grams):
            for gram in grams:
                self.index[gram].add(i)

    def matches(self, question):
        if normalized_question(question) in self.exact:
            return "exact"
        query = shingles(question)
        if len(query) < 7:
            return None
        candidates = set().union(*(self.index.get(g, set()) for g in query))
        for i in candidates:
            other = self.grams[i]
            small, large = min(len(query), len(other)), max(len(query), len(other))
            if small >= 7 and small / large >= 0.8 and len(query & other) / small >= 0.8:
                return "lexical_near"
        return None


def select(args):
    import pyarrow.parquet as pq

    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing selection: {args.output_dir}")
    parent = verify_prepared(args.parent_dir)
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=True)
    if tokenizer_identity(tokenizer) != parent["tokenizer"]:
        raise ValueError("Tokenizer differs from parent")
    exclusions = []
    official_sources = {}
    for path in (args.official_train, args.official_test):
        # Test QUESTIONS are read only to exclude contamination. No answers or scoring.
        table = pq.read_table(path, columns=["question"])
        exclusions.extend(table["question"].to_pylist())
        official_sources[path.name] = {"sha256": sha256_file(path), "questions": len(table)}
    exclusions.extend(r["question"] for r in read_jsonl(args.parent_dir / "train_questions.jsonl"))
    exclusions.extend(r["question"] for r in read_jsonl(args.parent_dir / "val_questions.jsonl"))
    blocked = QuestionExclusions(exclusions)
    files = sorted(args.tinygsm_dir.glob("train-*-of-*.parquet"))
    if not files:
        raise ValueError("No TinyGSM parquet shards found")
    sources, total = [], 0
    for path in files:
        parquet = pq.ParquetFile(path)
        if set(parquet.schema_arrow.names) != {"question", "code"}:
            raise ValueError(f"Unexpected TinyGSM schema: {path}")
        digest = sha256_file(path)
        metadata = args.tinygsm_dir.parent / ".cache/huggingface/download/data" / (path.name + ".metadata")
        revision = None
        if metadata.exists():
            lines = metadata.read_text().splitlines()
            revision = lines[0]
            if lines[1].strip('"') != digest:
                raise ValueError(f"TinyGSM file differs from download checksum: {path}")
        sources.append({"name": path.name, "sha256": digest, "rows": parquet.metadata.num_rows,
                        "global_start": total, "revision": revision})
        total += parquet.metadata.num_rows
    # Sample across the full source, not just the first shard or consecutive rows.
    chosen = random.Random(args.seed).sample(range(total), min(args.draw_count, total))
    ranks = {index: rank for rank, index in enumerate(chosen)}
    sampled = {}
    for path, source in zip(files, sources):
        start = source["global_start"]
        indices = sorted(i - start for i in chosen if start <= i < start + source["rows"])
        if not indices:
            continue
        parquet, offset, pointer = pq.ParquetFile(path), 0, 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            group_indices = []
            while pointer < len(indices) and indices[pointer] < offset + size:
                group_indices.append(indices[pointer])
                pointer += 1
            if group_indices:
                records = parquet.read_row_group(group, columns=["question", "code"])
                for index in group_indices:
                    record = {key: records[key][index - offset].as_py() for key in ("question", "code")}
                    sampled[start + index] = (record, source["name"], index)
            offset += size
        print(json.dumps({"source_read": path.name, "sampled": len(sampled)}), flush=True)
    accepted, rejected, seen = [], [], set()
    for index in chosen:
        record, filename, row_index = sampled[index]
        reason = None
        if not all(isinstance(record[k], str) and record[k].strip() for k in ("question", "code")):
            reason = "empty_or_nontext"
        elif overlap := blocked.matches(record["question"]):
            reason = "gsm8k_" + overlap
        elif normalized_question(record["question"]) in seen:
            reason = "duplicate_question"
        else:
            try:
                tree = validate_ast(record["code"])
                compile(record["code"], "<TinyGSM-reference>", "exec")
                entry = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "simple_math_problem"]
                if len(entry) != 1 or entry[0].args.args or entry[0].args.posonlyargs or entry[0].args.kwonlyargs:
                    raise ValueError("Missing zero-argument entrypoint")
            except (SyntaxError, ValueError, RuntimeError, TypeError) as error:
                reason = "syntax_or_entrypoint:" + type(error).__name__
            except Exception as error:
                from mdlm_sft.executor import PolicyError
                if not isinstance(error, PolicyError):
                    raise
                reason = "ast_policy"
        if reason is None:
            row = {**record, "gsm8k_id": f"tinygsm:{index}", "gold_answer": None,
                   "source": "TinyGSM", "answer_provenance": "not_provided_by_source",
                   "code_sha256": text_sha(record["code"]), "source_file": filename,
                   "source_row_0based": row_index, "selection_rank": ranks[index]}
            try:
                encoded = encode_example(row, tokenizer)
            except ValueError:
                reason = "over_512_tokens"
            else:
                row.update(question_tokens=encoded["prefix_length"] - 3,
                           code_tokens=len(tokenizer.encode(row["code"].strip(), add_special_tokens=False)),
                           total_tokens=encoded["sequence_length"])
                accepted.append(row)
                seen.add(normalized_question(row["question"]))
        if reason:
            rejected.append({"gsm8k_id": f"tinygsm:{index}", "reason": reason})
    if len(accepted) < args.replay_count:
        raise ValueError(f"Only {len(accepted)} candidates; increase --draw-count")
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "candidates.jsonl", accepted)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    report = {"stage": "selected", "seed": args.seed, "source_rows": total,
              "draw_count": len(chosen), "eligible_candidates": len(accepted),
              "requested_replay_samples": args.replay_count,
              "parent_manifest_sha256": sha256_file(args.parent_dir / "manifest.json"),
              "tinygsm_sources": sources, "official_exclusion_sources": official_sources,
              "exclusion_policy": "Normalized exact + 4-word shingle containment >=0.8 with size ratio >=0.8; not semantic deduplication",
              "test_access": "Question column for exclusion only; no answers read or evaluation",
              "rejection_counts": dict(Counter(r["reason"] for r in rejected)),
              "candidates_sha256": sha256_file(args.output_dir / "candidates.jsonl"),
              "code_policy": "Source question/code preserved verbatim, including docstrings/comments",
              "script_sha256": sha256_file(Path(__file__)), "python": platform.python_version()}
    write_json(args.output_dir / "selection.json", report)
    print(json.dumps({k: report[k] for k in ("eligible_candidates", "rejection_counts")}), flush=True)


def execute_candidates(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing execution results: {args.output_dir}")
    rows = read_jsonl(args.candidates)
    config = ExecutionConfig(args.timeout_s, 512)
    preflight = preflight_executor(config)
    args.output_dir.mkdir(parents=True)
    checked, accepted = [], []
    # Bounded groups avoid queuing all programs or consuming GPU resources.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with (args.output_dir / "checks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for start in range(0, len(rows), args.workers):
                group = rows[start:start + args.workers]
                results = list(pool.map(lambda row: run_program(row["code"], config), group))
                for row, result in zip(group, results):
                    record = {"gsm8k_id": row["gsm8k_id"], "code_sha256": row["code_sha256"],
                              "result": result, "accepted": result["status"] == "ok"}
                    checked.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    if record["accepted"] and len(accepted) < args.replay_count:
                        accepted.append({**row, "execution_answer": result["predicted_answer"]})
                stream.flush()
                if len(checked) % 128 == 0 or len(accepted) == args.replay_count:
                    print(json.dumps({"checked": len(checked), "accepted": len(accepted), "target": args.replay_count}), flush=True)
                if len(accepted) == args.replay_count:
                    break
    if len(accepted) != args.replay_count:
        raise ValueError(f"Insufficient runnable candidates: {len(accepted)}")
    write_jsonl(args.output_dir / "accepted.jsonl", accepted)
    report = {"status": "passed", "checked": len(checked), "accepted": len(accepted),
              "result_counts": dict(Counter(r["result"]["status"] for r in checked)),
              "candidates_sha256": sha256_file(args.candidates),
              "accepted_sha256": sha256_file(args.output_dir / "accepted.jsonl"),
              "checks_sha256": sha256_file(args.output_dir / "checks.jsonl"),
              "scorer_version": SCORER_VERSION, "preflight": preflight,
              "source_sha256": {name: sha256_file(ROOT / "mdlm_sft" / name)
                                for name in ("scoring.py", "executor.py")},
              "script_sha256": sha256_file(Path(__file__)), "python": platform.python_version(),
              "correctness": "Executable finite numeric result only; no independent gold in TinyGSM; mathematical correctness unverified"}
    write_json(args.output_dir / "execution.json", report)
    print(json.dumps(report), flush=True)


def assemble(args):
    import numpy as np
    from mdlm_sft.dataset_checks import validate_training_data
    from mdlm_sft.data import load_questions

    paths = [p.resolve() for p in (args.parent_dir, args.output_dir, args.selection_dir, args.execution_dir)]
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Parent, output, selection and execution directories must be disjoint")
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing dataset: {args.output_dir}")
    parent = verify_prepared(args.parent_dir)
    parent_sha = sha256_file(args.parent_dir / "manifest.json")
    selection = json.loads((args.selection_dir / "selection.json").read_text(encoding="utf-8"))
    execution = json.loads((args.execution_dir / "execution.json").read_text(encoding="utf-8"))
    if selection["parent_manifest_sha256"] != parent_sha:
        raise ValueError("Parent changed after selection")
    if selection["candidates_sha256"] != execution["candidates_sha256"] or execution["status"] != "passed":
        raise ValueError("Execution evidence belongs to different selection")
    if sha256_file(args.selection_dir / "candidates.jsonl") != selection["candidates_sha256"]:
        raise ValueError("Selected candidate file changed")
    if sha256_file(args.execution_dir / "accepted.jsonl") != execution["accepted_sha256"]:
        raise ValueError("Accepted replay data changed")
    if sha256_file(args.execution_dir / "checks.jsonl") != execution["checks_sha256"]:
        raise ValueError("Execution evidence changed")
    candidates = {r["gsm8k_id"]: r for r in read_jsonl(args.selection_dir / "candidates.jsonl")}
    checks = {r["gsm8k_id"]: r for r in read_jsonl(args.execution_dir / "checks.jsonl")}
    replay = read_jsonl(args.execution_dir / "accepted.jsonl")
    if len(replay) != selection["requested_replay_samples"] or len({r["gsm8k_id"] for r in replay}) != len(replay):
        raise ValueError("Replay count/identity mismatch")
    for row in replay:
        if {k: v for k, v in row.items() if k != "execution_answer"} != candidates[row["gsm8k_id"]]:
            raise ValueError("Replay text differs from selected source")
        check = checks[row["gsm8k_id"]]
        answer = row["execution_answer"]
        if (not check["accepted"] or check["result"]["status"] != "ok"
                or isinstance(answer, bool) or not isinstance(answer, (int, float))
                or not math.isfinite(answer) or check["code_sha256"] != text_sha(row["code"])
                or answer != check["result"]["predicted_answer"]):
            raise ValueError("Replay execution evidence mismatch")
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=True)
    if tokenizer_identity(tokenizer) != parent["tokenizer"]:
        raise ValueError("Tokenizer differs from parent")
    original = PreparedDataset(args.parent_dir, "train")
    parent_rows = read_jsonl(args.parent_dir / "train.jsonl")
    replay.sort(key=lambda r: int(r["gsm8k_id"].split(":")[1]))
    rows = parent_rows + replay
    values = [encode_example(row, tokenizer) for row in rows]
    for i in range(len(parent_rows)):
        if (values[i]["input_ids"] != original.input_ids[i].tolist()
                or values[i]["target_mask"] != original.target_mask[i].tolist()):
            raise ValueError("Parent text/cache mismatch")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="mix.building-", dir=args.output_dir.parent))
    for name in parent["artifacts_sha256"]:
        if name.startswith("val"):
            shutil.copyfile(args.parent_dir / name, staging / name)
    write_jsonl(staging / "train.jsonl", rows)
    questions = read_jsonl(args.parent_dir / "train_questions.jsonl") + [
        {k: row[k] for k in ("gsm8k_id", "question", "gold_answer", "source", "answer_provenance")}
        for row in replay]
    write_jsonl(staging / "train_questions.jsonl", questions)
    write_json(staging / "train_ids.json", [r["gsm8k_id"] for r in questions])
    np.savez_compressed(staging / "train.npz",
                        input_ids=np.asarray([r["input_ids"] for r in values], dtype=np.int32),
                        target_mask=np.asarray([r["target_mask"] for r in values], dtype=np.bool_),
                        prefix_lengths=np.asarray([r["prefix_length"] for r in values], dtype=np.int32),
                        gsm8k_ids=np.asarray([r["gsm8k_id"] for r in values]))
    digest = hashlib.sha256()
    for split in ("train", "val"):
        with np.load(staging / f"{split}.npz", allow_pickle=False) as cache:
            for ids in cache["input_ids"]:
                digest.update(np.asarray(ids, dtype="<u4").tobytes())
    manifest = copy.deepcopy(parent)
    manifest.update(
        source_sha256={"parent_manifest": parent_sha, "selection": sha256_file(args.selection_dir / "selection.json"),
                       "execution": sha256_file(args.execution_dir / "execution.json")},
        source_filenames={"parent_manifest": "manifest.json", "selection": "selection.json", "execution": "execution.json"},
        source_locations_at_preparation={"parent_dir": str(args.parent_dir.resolve()),
                                         "selection_dir": str(args.selection_dir.resolve()), "execution_dir": str(args.execution_dir.resolve())},
        split={"seed": selection["seed"], "algorithm": "Preserve parent train/val; append unique executable TinyGSM replay, numeric source ID order", "python_version": platform.python_version()},
        derivation={"kind": "tinygsm_replay_mix", "parent_manifest_sha256": parent_sha,
                    "validation_references": "byte-identical to parent", "parent_rows": "retained unchanged in original order",
                    "tinygsm_code": "verbatim including docstrings/comments", "tinygsm_gold": "not available; execution_answer is not independent ground truth"},
        mixing={"parent_samples": len(parent_rows), "replay_samples": len(replay),
                "replay_fraction": len(replay) / len(rows), "sampling": "uniform shuffled training rows",
                "steps_per_epoch_at_batch64": (len(rows) + 63) // 64},
        official_check={"scope": "parent GSM8K rows only", "parent": parent["official_check"],
                        "tinygsm_test_overlap_check": selection["test_access"]},
        checks={"parent_rows_and_encoded_arrays_unchanged": True, "validation_files_byte_identical": True,
                "tinygsm_unique_normalized_questions": True, "tinygsm_executable": True,
                "tinygsm_independent_answer_correctness_verified": False, "all_total_tokens_within_512": True},
        encoded_token_sha256=digest.hexdigest(),
        preparation_code_sha256={"scripts/prepare_mix_data.py": sha256_file(Path(__file__)),
                                 "mdlm_sft/data.py": sha256_file(ROOT / "mdlm_sft/data.py")},
        artifacts_sha256={name: sha256_file(staging / name) for name in parent["artifacts_sha256"]},
        invocation=[sys.executable, *sys.argv],
    )
    manifest["counts"]["train"] = {"samples": len(rows), "questions": len(questions),
                                      "target_tokens": sum(sum(r["target_mask"]) for r in values)}
    write_json(staging / "manifest.json", manifest)
    train_data, val_data = PreparedDataset(staging, "train"), PreparedDataset(staging, "val")
    validate_training_data(staging, train_data, val_data, load_questions(staging, "val"))
    if not np.array_equal(train_data.input_ids[:len(original)], original.input_ids):
        raise AssertionError("Original tokens changed")
    for name, expected in parent["artifacts_sha256"].items():
        if name.startswith("val") and sha256_file(staging / name) != expected:
            raise AssertionError("Validation changed")
    if sha256_file(args.parent_dir / "manifest.json") != parent_sha or verify_prepared(args.parent_dir) != parent:
        raise AssertionError("Parent changed while assembling")
    destination = args.output_dir.resolve()
    if (staging.resolve().parent != paths[1].parent
            or destination != paths[1] or destination.exists()):
        raise ValueError("Resolved output/staging paths changed; preserving prepared files")
    staging.rename(destination)
    print(json.dumps({"status": "prepared", "output_dir": str(args.output_dir), "counts": manifest["counts"],
                      "mixing": manifest["mixing"], "manifest_sha256": sha256_file(args.output_dir / "manifest.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--tinygsm-dir", type=Path, required=True)
    select_parser.add_argument("--official-train", type=Path, required=True)
    select_parser.add_argument("--official-test", type=Path, required=True)
    select_parser.add_argument("--seed", type=int, default=8)
    select_parser.add_argument("--draw-count", type=int, default=6144)
    select_parser.add_argument("--replay-count", type=int, default=3072)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--candidates", type=Path, required=True)
    check_parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    check_parser.add_argument("--timeout-s", type=float, default=15.0)
    check_parser.add_argument("--replay-count", type=int, default=3072)
    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument("--selection-dir", type=Path, required=True)
    assemble_parser.add_argument("--execution-dir", type=Path, required=True)
    for sub in (select_parser, assemble_parser):
        sub.add_argument("--parent-dir", type=Path, default=Path("data/style"))
        sub.add_argument("--tokenizer", type=Path, default=Path("inputs/tokenizer"))
    for sub in (select_parser, check_parser, assemble_parser):
        sub.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    {"select": select, "check": execute_candidates, "assemble": assemble}[args.action](args)


if __name__ == "__main__":
    main()
