"""Derive style-aligned training text and caches without changing the frozen split.

Run with the project's pinned Python environment. Runtime correctness is checked
separately by validate_style_data.py under the existing Linux executor.
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
    PreparedDataset, SEPARATOR, encode_example, load_tokenizer, read_jsonl,
    sha256_file, tokenizer_identity, verify_prepared,
)
from mdlm_sft.prepare_data import write_json, write_jsonl
from mdlm_sft.style_transform import transform_code


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(parent_dir: Path, output_dir: Path, tokenizer_path: Path, report_dir: Path):
    import numpy as np
    from tokenizers import Tokenizer

    parent_dir, output_dir = parent_dir.resolve(), output_dir.resolve()
    report_dir = report_dir.resolve()
    for left, right in ((parent_dir, output_dir), (parent_dir, report_dir),
                        (output_dir, report_dir)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("Parent, derived data and reports must be separate directories")
    if output_dir.exists():
        raise FileExistsError(f"Preserving existing output: {output_dir}")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Preserving existing reports: {report_dir}")
    parent_manifest = verify_prepared(parent_dir)
    parent_manifest_sha = sha256_file(parent_dir / "manifest.json")
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=True)
    if tokenizer_identity(tokenizer) != parent_manifest["tokenizer"]:
        raise ValueError("Tokenizer does not match parent dataset")
    backend = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    rows = read_jsonl(parent_dir / "train.jsonl")
    val_rows = read_jsonl(parent_dir / "val.jsonl")
    ids = {split: json.loads((parent_dir / f"{split}_ids.json").read_text(encoding="utf-8"))
           for split in ("train", "val")}
    if set(ids["train"]) & set(ids["val"]):
        raise ValueError("Parent train and validation IDs overlap")
    if list(dict.fromkeys(r["gsm8k_id"] for r in rows)) != ids["train"]:
        raise ValueError("Parent training row order disagrees with saved IDs")
    original = PreparedDataset(parent_dir, "train")
    if len(rows) != len(original):
        raise ValueError("Parent text and cache row counts differ")
    rewritten, encoded, changes = [], [], []
    duplicate_keys = Counter()
    report_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        if text_sha(row["code"]) != row["code_sha256"]:
            raise ValueError(f"Parent code hash mismatch at row {index}")
        previous = encode_example(row, tokenizer)
        if (not np.array_equal(previous["input_ids"], original.input_ids[index])
                or not np.array_equal(previous["target_mask"], original.target_mask[index])
                or previous["prefix_length"] != int(original.prefix_lengths[index])
                or row["gsm8k_id"] != original.gsm8k_ids[index]):
            raise ValueError(f"Parent text/cache mismatch at row {index}")
        try:
            code, metadata = transform_code(row["code"])
        except Exception as error:
            write_json(report_dir / "failure.json", {
                "status": "failed", "stage": "transform", "row_index": index,
                "gsm8k_id": row["gsm8k_id"], "error": str(error),
            })
            raise
        compile(code, "<style-aligned-reference>", "exec")
        updated = dict(row)
        updated.update(code=code, code_sha256=text_sha(code),
                       parent_code_sha256=row["code_sha256"])
        updated["code_tokens"] = len(tokenizer.encode(code.strip(), add_special_tokens=False))
        updated["total_tokens"] = row["question_tokens"] + updated["code_tokens"] + 2 + len(
            tokenizer.encode(SEPARATOR, add_special_tokens=False))
        try:
            value = encode_example(updated, tokenizer)
        except ValueError as error:
            write_json(report_dir / "failure.json", {
                "status": "failed", "stage": "encode", "row_index": index,
                "gsm8k_id": row["gsm8k_id"], "error": str(error),
            })
            raise  # Do not truncate or silently drop/revert long rows.
        if tokenizer.encode(code.strip(), add_special_tokens=False) != backend.encode(
                code.strip(), add_special_tokens=False).ids:
            raise ValueError(f"Serialized tokenizer mismatch at row {index}")
        if (value["prefix_length"] != previous["prefix_length"]
                or value["target_mask"] != previous["target_mask"]
                or value["input_ids"][:value["prefix_length"]]
                != previous["input_ids"][:previous["prefix_length"]]):
            raise AssertionError("Style conversion changed question conditioning")
        rewritten.append(updated)
        encoded.append(value)
        duplicate_keys[(row["gsm8k_id"], code)] += 1
        changes.append({
            "row_index": index, "gsm8k_id": row["gsm8k_id"],
            "before_sha256": row["code_sha256"], "after_sha256": updated["code_sha256"],
            "before_code_tokens": row["code_tokens"], "after_code_tokens": updated["code_tokens"],
            "before_total_tokens": row["total_tokens"], "after_total_tokens": updated["total_tokens"],
            "transform": metadata,
        })
        if (index + 1) % 1000 == 0:
            print(json.dumps({"prepared": index + 1, "total": len(rows)}), flush=True)
    if Counter(r["gsm8k_id"] for r in rows) != Counter(r["gsm8k_id"] for r in rewritten):
        raise AssertionError("Question sampling weights changed")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".building-", dir=output_dir.parent))
    # On failure retain the staging directory as evidence, never touch parent data.
    copied = []
    for name in parent_manifest["artifacts_sha256"]:
        if name not in {"train.jsonl", "train.npz"}:
            shutil.copyfile(parent_dir / name, staging / name)
            copied.append(name)
    write_jsonl(staging / "train.jsonl", rewritten)
    np.savez_compressed(
        staging / "train.npz",
        input_ids=np.asarray([r["input_ids"] for r in encoded], dtype=np.int32),
        target_mask=np.asarray([r["target_mask"] for r in encoded], dtype=np.bool_),
        prefix_lengths=np.asarray([r["prefix_length"] for r in encoded], dtype=np.int32),
        gsm8k_ids=np.asarray([r["gsm8k_id"] for r in encoded]),
    )
    digest = hashlib.sha256()
    for split in ("train", "val"):
        with np.load(staging / f"{split}.npz", allow_pickle=False) as cache:
            for array in cache["input_ids"]:
                digest.update(np.asarray(array, dtype="<u4").tobytes())
    manifest = copy.deepcopy(parent_manifest)
    manifest.update(
        source_sha256={"parent_manifest": parent_manifest_sha,
                       **{name: sha256_file(parent_dir / name)
                          for name in parent_manifest["artifacts_sha256"]}},
        source_filenames={"parent_manifest": "manifest.json"},
        source_locations_at_preparation={"parent_dir": str(parent_dir)},
        split={"seed": parent_manifest["split"]["seed"],
               "algorithm": "Preserve parent question split and exact sample row order; no deduplication",
               "python_version": platform.python_version()},
        encoded_token_sha256=digest.hexdigest(),
        invocation=[sys.executable, *sys.argv],
        preparation_code_sha256={
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
            for p in (Path(__file__), ROOT / "mdlm_sft/style_transform.py",
                      ROOT / "mdlm_sft/data.py")},
        artifacts_sha256={name: sha256_file(staging / name)
                          for name in parent_manifest["artifacts_sha256"]},
    )
    manifest["derivation"] = {
        "kind": "training_code_style_alignment", "parent_manifest_sha256": parent_manifest_sha,
        "rules": ["bare int return annotation (no numeric coercion)", "remove comments",
                  "assign return expression to result", "no added docstring"],
        "validation_references": "byte-identical to parent for comparable denoising loss",
        "duplicate_solutions": "retained with original sampling multiplicity",
        "runtime_validation": "separate report from validate_style_data.py",
    }
    manifest["checks"].update({
        "parent_cache_matches_text": True, "question_prefixes_and_masks_unchanged": True,
        "train_row_order_and_weights_unchanged": True, "validation_files_byte_identical": True,
        "all_transformed_code_compiles": True, "all_total_tokens_within_512": True,
    })
    manifest["counts"]["train"]["target_tokens"] = sum(sum(r["target_mask"]) for r in encoded)
    for name in copied:
        if sha256_file(staging / name) != parent_manifest["artifacts_sha256"][name]:
            raise AssertionError(f"Copied artifact changed: {name}")
    if verify_prepared(parent_dir) != parent_manifest or sha256_file(parent_dir / "manifest.json") != parent_manifest_sha:
        raise ValueError("Parent changed during preparation")
    write_json(staging / "manifest.json", manifest)
    new_cache = PreparedDataset(staging, "train")
    PreparedDataset(staging, "val")
    for index, row in enumerate(read_jsonl(staging / "train.jsonl")):
        expected = encode_example(row, tokenizer)
        if (not np.array_equal(expected["input_ids"], new_cache.input_ids[index])
                or expected["gsm8k_id"] != new_cache.gsm8k_ids[index]):
            raise AssertionError(f"Written cache/text mismatch at row {index}")
    if not np.array_equal(new_cache.target_mask, original.target_mask):
        raise AssertionError("Written target masks changed")
    staging.rename(output_dir)
    report = {
        "status": "prepared", "runtime_validation": "pending",
        "output_dir": str(output_dir), "parent_manifest_sha256": parent_manifest_sha,
        "derived_manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "train_questions": len(ids["train"]), "train_samples": len(rows),
        "validation_questions": len(ids["val"]), "validation_samples": len(val_rows),
        "changed_rows": sum(a["code"] != b["code"] for a, b in zip(rows, rewritten)),
        "duplicate_extra_rows_retained": sum(n - 1 for n in duplicate_keys.values()),
        "before_code_tokens": sum(r["code_tokens"] for r in rows),
        "after_code_tokens": sum(r["code_tokens"] for r in rewritten),
        "before_max_total_tokens": max(r["total_tokens"] for r in rows),
        "after_max_total_tokens": max(r["total_tokens"] for r in rewritten),
        "checks": manifest["checks"],
    }
    write_jsonl(report_dir / "changes.jsonl", changes)
    write_json(report_dir / "preparation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT.parent / ".mdlm-data-work" / "data/style_aligned",
                        help="Intermediate data before deduplication (outside the source directory)")
    parser.add_argument("--tokenizer", type=Path, default=Path("inputs/tokenizer"))
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT.parent / ".mdlm-data-work" / "runs/style_alignment/preparation",
                        help="Preparation evidence directory (defaults outside the source directory)")
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_dir, args.output_dir, args.tokenizer, args.report_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
