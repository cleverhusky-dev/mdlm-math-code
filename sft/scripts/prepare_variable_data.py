"""Normalize training variable names, retaining all rows and frozen validation.

This is static data preparation only: code is parsed/compiled, never executed.
All preparation reports and staging remain outside the source directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdlm_sft.data import (
    PreparedDataset, SEPARATOR, encode_example, load_questions, load_tokenizer,
    read_jsonl, sha256_file, tokenizer_identity, verify_prepared,
)
from mdlm_sft.dataset_checks import validate_training_data
from mdlm_sft.prepare_data import write_json, write_jsonl
from mdlm_sft.variable_transform import transform_code, verify_transform

ARTIFACTS = tuple(f"{split}{suffix}" for split in ("train", "val")
                  for suffix in (".jsonl", ".npz", "_ids.json", "_questions.jsonl"))
CHANGED_ROW_FIELDS = {"code", "code_sha256", "code_tokens", "total_tokens"}


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(parent_dir, output_dir, tokenizer_path, report_dir, *,
            expected_samples=12288, expected_questions=5747):
    import numpy as np
    from tokenizers import Tokenizer

    parent_dir, output_dir, tokenizer_path, report_dir = (
        Path(path).resolve() for path in (parent_dir, output_dir, tokenizer_path, report_dir))
    paths = (parent_dir, output_dir, report_dir)
    if report_dir.is_relative_to(ROOT.resolve()) and not report_dir.is_relative_to(ROOT / "data"):
        raise ValueError("Reports and staging must be outside the source directory")
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Parent, output and report directories must be disjoint")
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("Preserving existing output/report; choose unused directories")
    parent = verify_prepared(parent_dir)
    parent_sha = sha256_file(parent_dir / "manifest.json")
    if set(parent["artifacts_sha256"]) != set(ARTIFACTS):
        raise ValueError("Parent manifest must checksum all eight prepared artifacts")
    original, validation = PreparedDataset(parent_dir, "train"), PreparedDataset(parent_dir, "val")
    val_questions = load_questions(parent_dir, "val")
    validate_training_data(parent_dir, original, validation, val_questions)
    if (parent["counts"]["train"]["samples"] != expected_samples
            or parent["counts"]["train"]["questions"] != expected_questions):
        raise ValueError(f"Expected {expected_samples} samples and {expected_questions} training questions")
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=True)
    if tokenizer_identity(tokenizer) != parent["tokenizer"]:
        raise ValueError("Tokenizer does not match parent dataset")
    backend = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    parent_rows = read_jsonl(parent_dir / "train.jsonl")
    report_dir.mkdir(parents=True, exist_ok=False)
    rewritten, encoded_rows, changes = [], [], []
    for index, row in enumerate(parent_rows):
        stage = "parent_text_cache"
        try:
            if text_sha(row["code"]) != row["code_sha256"]:
                raise ValueError(f"Parent code hash mismatch at row {index}")
            previous = encode_example(row, tokenizer)
            if (not np.array_equal(previous["input_ids"], original.input_ids[index])
                    or not np.array_equal(previous["target_mask"], original.target_mask[index])
                    or previous["prefix_length"] != int(original.prefix_lengths[index])
                    or row["gsm8k_id"] != original.gsm8k_ids[index]):
                raise ValueError(f"Parent text/cache mismatch at row {index}")
            stage = "transform"
            code, metadata = transform_code(row["code"])
            verify_transform(row["code"], code)
            compile(code, "<variable-normalized-reference>", "exec")
            stage = "encode"
            code_ids = tokenizer.encode(code.strip(), add_special_tokens=False)
            if code_ids != backend.encode(code.strip(), add_special_tokens=False).ids:
                raise ValueError(f"Serialized tokenizer mismatch at row {index}")
            updated = copy.deepcopy(row)
            updated.update(code=code, code_sha256=text_sha(code), code_tokens=len(code_ids),
                           total_tokens=previous["prefix_length"] + len(code_ids) + 1)
            value = encode_example(updated, tokenizer)  # Reject >512; never truncate/drop/revert.
            if (value["prefix_length"] != previous["prefix_length"]
                    or value["target_mask"] != previous["target_mask"]
                    or value["input_ids"][:value["prefix_length"]]
                    != previous["input_ids"][:previous["prefix_length"]]):
                raise ValueError("Variable normalization changed question conditioning")
            if ({k: v for k, v in updated.items() if k not in CHANGED_ROW_FIELDS}
                    != {k: v for k, v in row.items() if k not in CHANGED_ROW_FIELDS}):
                raise AssertionError("Variable normalization changed unrelated row metadata")
        except Exception as error:
            write_json(report_dir / "failure.json", {
                "status": "failed", "stage": stage, "row_index": index,
                "gsm8k_id": row["gsm8k_id"], "error_type": type(error).__name__,
                "error": str(error), "transform_reason": getattr(error, "reason", None),
            })
            raise
        rewritten.append(updated)
        encoded_rows.append(value)
        changes.append({
            "row_index": index, "gsm8k_id": row["gsm8k_id"],
            "before_sha256": row["code_sha256"], "after_sha256": updated["code_sha256"],
            "before_code_tokens": len(tokenizer.encode(row["code"].strip(), add_special_tokens=False)),
            "after_code_tokens": updated["code_tokens"],
            "before_total_tokens": previous["sequence_length"], "after_total_tokens": value["sequence_length"],
            "transform": metadata,
        })
        if (index + 1) % 1000 == 0:
            print(json.dumps({"prepared": index + 1, "total": len(parent_rows)}), flush=True)
    if Counter(r["gsm8k_id"] for r in rewritten) != Counter(r["gsm8k_id"] for r in parent_rows):
        raise AssertionError("Question sampling multiplicity changed")
    parent_duplicates = Counter((r["gsm8k_id"], r["code"]) for r in parent_rows)
    duplicates = Counter((r["gsm8k_id"], r["code"]) for r in rewritten)
    duplicate_extra_rows = sum(count - 1 for count in duplicates.values())
    staging = Path(tempfile.mkdtemp(prefix="data.building-", dir=report_dir))
    for name in ARTIFACTS:
        if name not in {"train.jsonl", "train.npz"}:
            shutil.copyfile(parent_dir / name, staging / name)
    write_jsonl(staging / "train.jsonl", rewritten)
    np.savez_compressed(
        staging / "train.npz",
        input_ids=np.asarray([row["input_ids"] for row in encoded_rows], dtype=np.int32),
        target_mask=np.asarray([row["target_mask"] for row in encoded_rows], dtype=np.bool_),
        prefix_lengths=np.asarray([row["prefix_length"] for row in encoded_rows], dtype=np.int32),
        gsm8k_ids=np.asarray([row["gsm8k_id"] for row in encoded_rows]),
    )
    digest = hashlib.sha256()
    for split in ("train", "val"):
        with np.load(staging / f"{split}.npz", allow_pickle=False) as cache:
            for ids in cache["input_ids"]:
                digest.update(np.asarray(ids, dtype="<u4").tobytes())
    write_jsonl(report_dir / "changes.jsonl", changes)
    changed_indices = [i for i, (a, b) in enumerate(zip(parent_rows, rewritten)) if a["code"] != b["code"]]
    preview_indices = changed_indices[:8] or list(range(min(8, len(rewritten))))
    write_json(report_dir / "preview.json", [
        {"row_index": i, "gsm8k_id": parent_rows[i]["gsm8k_id"], "question": parent_rows[i]["question"],
         "before": parent_rows[i]["code"], "after": rewritten[i]["code"], "transform": changes[i]["transform"]}
        for i in preview_indices])
    manifest = copy.deepcopy(parent)
    manifest.update(
        source_sha256={"parent_manifest": parent_sha, **parent["artifacts_sha256"]},
        source_filenames={"parent_manifest": "manifest.json"},
        source_locations_at_preparation={"parent_dir": str(parent_dir)},
        split={"seed": parent["split"]["seed"], "python_version": platform.python_version(),
               "algorithm": "Preserve parent question split, all training rows, row order and multiplicity"},
        encoded_token_sha256=digest.hexdigest(), invocation=[sys.executable, *sys.argv],
        preparation_code_sha256={path.relative_to(ROOT).as_posix(): sha256_file(path) for path in (
            Path(__file__).resolve(), ROOT / "mdlm_sft/variable_transform.py",
            ROOT / "mdlm_sft/data.py", ROOT / "mdlm_sft/dataset_checks.py")},
        artifacts_sha256={name: sha256_file(staging / name) for name in ARTIFACTS},
        derivation={"kind": "training_variable_normalization", "parent_manifest_sha256": parent_sha,
                    "transform": "mdlm_sft.variable_transform.transform_code",
                    "verification": "verify_transform plus syntax compilation; no program execution",
                    "changed_row_fields": sorted(CHANGED_ROW_FIELDS), "removed_samples": 0,
                    "changes_sha256": sha256_file(report_dir / "changes.jsonl"),
                    "duplicate_solutions": "retained with original row multiplicity; no new deduplication",
                    "validation_references": "byte-identical to parent",
                    "runtime_validation": "separate report from validate_style_data.py"},
        checks={"no_question_overlap": True, "question_split_unchanged": True,
                "parent_cache_matches_text": True, "retained_text_matches_cache": True,
                "train_row_order_and_weights_unchanged": True, "unrelated_row_metadata_unchanged": True,
                "question_prefixes_and_masks_unchanged": True, "question_files_byte_identical": True,
                "validation_files_byte_identical": True, "all_transform_verifications_passed": True,
                "all_transformed_code_compiles": True, "all_total_tokens_within_512": True,
                "exact_training_duplicates_remaining": duplicate_extra_rows,
                "exact_training_duplicate_groups": sum(n > 1 for n in duplicates.values())},
    )
    manifest["counts"]["train"]["target_tokens"] = sum(sum(row["target_mask"]) for row in encoded_rows)
    if manifest["counts"] != parent["counts"]:
        raise AssertionError("Training sample/question/target counts or validation counts changed")
    for name in ARTIFACTS:
        if name not in {"train.jsonl", "train.npz"} and manifest["artifacts_sha256"][name] != parent["artifacts_sha256"][name]:
            raise AssertionError(f"Copied artifact changed: {name}")
    write_json(staging / "manifest.json", manifest)
    new_cache, new_validation = PreparedDataset(staging, "train"), PreparedDataset(staging, "val")
    validate_training_data(staging, new_cache, new_validation, load_questions(staging, "val"))
    written_rows = read_jsonl(staging / "train.jsonl")
    if written_rows != rewritten:
        raise AssertionError("Serialized training rows changed")
    for index, row in enumerate(written_rows):
        expected = encode_example(row, tokenizer)
        if (not np.array_equal(expected["input_ids"], new_cache.input_ids[index])
                or not np.array_equal(expected["target_mask"], new_cache.target_mask[index])
                or expected["prefix_length"] != int(new_cache.prefix_lengths[index])
                or expected["gsm8k_id"] != new_cache.gsm8k_ids[index]):
            raise AssertionError(f"Written cache/text mismatch at row {index}")
    if not np.array_equal(new_cache.target_mask, original.target_mask):
        raise AssertionError("Written target masks changed")
    if verify_prepared(parent_dir) != parent or sha256_file(parent_dir / "manifest.json") != parent_sha:
        raise ValueError("Parent changed during preparation")
    for name, expected in parent["tokenizer"]["files_sha256"].items():
        if sha256_file(tokenizer_path / name) != expected:
            raise ValueError(f"Tokenizer changed during preparation: {name}")
    if output_dir.exists() or not staging.resolve().is_relative_to(report_dir):
        raise FileExistsError("Output/staging identity changed; preserving existing data")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output_dir)
    report = {
        "status": "prepared", "runtime_validation": "not_run", "training_started": False,
        "output_dir": str(output_dir), "parent_manifest_sha256": parent_sha,
        "derived_manifest_sha256": sha256_file(output_dir / "manifest.json"), "counts": manifest["counts"],
        "train_samples": len(rewritten), "train_questions": expected_questions, "removed_samples": 0,
        "changed_rows": len(changed_indices),
        "parent_duplicate_extra_rows": sum(n - 1 for n in parent_duplicates.values()),
        "duplicate_extra_rows_retained": duplicate_extra_rows,
        "duplicate_groups": sum(n > 1 for n in duplicates.values()),
        "before_code_tokens": sum(c["before_code_tokens"] for c in changes),
        "after_code_tokens": sum(c["after_code_tokens"] for c in changes),
        "before_max_total_tokens": max(c["before_total_tokens"] for c in changes),
        "after_max_total_tokens": max(c["after_total_tokens"] for c in changes),
        "bindings_renamed": sum(c["transform"].get("bindings_renamed", 0) for c in changes),
        "name_occurrences_changed": sum(c["transform"].get("name_occurrences_changed", 0) for c in changes),
        "imports_aliased": sum(c["transform"].get("imports_aliased", 0) for c in changes),
        "checks": manifest["checks"],
        "report_artifacts_sha256": {name: sha256_file(report_dir / name) for name in ("changes.jsonl", "preview.json")},
    }
    write_json(report_dir / "preparation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=ROOT / "data/style")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/vars")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "inputs/tokenizer")
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT.parent / ".mdlm-data-work" / "runs/experiment12-prepare/preparation")
    parser.add_argument("--expected-samples", type=int, default=12288)
    parser.add_argument("--expected-questions", type=int, default=5747)
    args = parser.parse_args()
    result = prepare(args.parent_dir, args.output_dir, args.tokenizer, args.report_dir,
                     expected_samples=args.expected_samples, expected_questions=args.expected_questions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
