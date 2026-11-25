"""Remove exact per-question training duplicates, preserving validation and row order."""
from __future__ import annotations

import argparse
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
    PreparedDataset, encode_example, load_tokenizer, read_jsonl,
    sha256_file, tokenizer_identity, verify_prepared,
)
from mdlm_sft.prepare_data import write_json, write_jsonl


def deduplicate(rows):
    """Keep the first exact (question ID, code string); do not rewrite any row."""
    kept, indices, duplicates = [], [], []
    seen = {}
    for index, row in enumerate(rows):
        key = (row["gsm8k_id"], row["code"])
        if key not in seen:
            seen[key] = (index, len(kept))
            kept.append(row)
            indices.append(index)
            continue
        parent_index, output_index = seen[key]
        first = kept[output_index]
        if any(row[field] != first[field] for field in ("question", "gold_answer")):
            raise ValueError(f"Conflicting question/answer for duplicate at row {index}")
        duplicates.append({
            "removed_row_index": index,
            "kept_parent_row_index": parent_index,
            "kept_output_row_index": output_index,
            "gsm8k_id": row["gsm8k_id"],
            "code_sha256": hashlib.sha256(row["code"].encode("utf-8")).hexdigest(),
            "removed_source_rows_0based": row.get("source_rows_0based"),
            "kept_source_rows_0based": first.get("source_rows_0based"),
        })
    return kept, indices, duplicates


def prepare(parent_dir, output_dir, report_dir, tokenizer_path):
    import numpy as np

    parent_dir, output_dir, report_dir = (
        Path(path).resolve() for path in (parent_dir, output_dir, report_dir))
    paths = (parent_dir, output_dir, report_dir)
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Parent, output and report directories must be disjoint")
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("Preserving existing output/report; choose unused directories")
    parent = verify_prepared(parent_dir)
    parent_sha = sha256_file(parent_dir / "manifest.json")
    rows = read_jsonl(parent_dir / "train.jsonl")
    kept, indices, duplicates = deduplicate(rows)
    if not kept:
        raise ValueError("Training data is empty")
    train_ids = json.loads((parent_dir / "train_ids.json").read_text(encoding="utf-8"))
    val_ids = json.loads((parent_dir / "val_ids.json").read_text(encoding="utf-8"))
    if list(dict.fromkeys(row["gsm8k_id"] for row in kept)) != train_ids:
        raise ValueError("Question set or first-occurrence order changed")
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation questions overlap")
    original = PreparedDataset(parent_dir, "train")
    PreparedDataset(parent_dir, "val")
    if len(rows) != len(original):
        raise ValueError("Parent training text and cache counts disagree")
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=True)
    if tokenizer_identity(tokenizer) != parent["tokenizer"]:
        raise ValueError("Tokenizer does not match parent data")
    # Reuse unchanged encoded rows, but check each retained row against its text.
    for row, index in zip(kept, indices):
        encoded = encode_example(row, tokenizer)
        if (not np.array_equal(encoded["input_ids"], original.input_ids[index])
                or not np.array_equal(encoded["target_mask"], original.target_mask[index])
                or encoded["prefix_length"] != int(original.prefix_lengths[index])
                or row["gsm8k_id"] != original.gsm8k_ids[index]):
            raise ValueError(f"Parent text/cache mismatch at row {index}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".building-", dir=output_dir.parent))
    # Keep partial output on failure for diagnosis; never overwrite parent data.
    for name in parent["artifacts_sha256"]:
        if name not in {"train.jsonl", "train.npz"}:
            shutil.copyfile(parent_dir / name, staging / name)
    write_jsonl(staging / "train.jsonl", kept)
    with np.load(parent_dir / "train.npz", allow_pickle=False) as cache:
        np.savez_compressed(staging / "train.npz", **{key: cache[key][indices] for key in cache.files})
    token_digest = hashlib.sha256()
    for split in ("train", "val"):
        with np.load(staging / f"{split}.npz", allow_pickle=False) as cache:
            for ids in cache["input_ids"]:
                token_digest.update(np.asarray(ids, dtype="<u4").tobytes())
    manifest = copy.deepcopy(parent)
    manifest.update(
        source_sha256={"parent_manifest": parent_sha, **parent["artifacts_sha256"]},
        source_filenames={"parent_manifest": "manifest.json"},
        source_locations_at_preparation={"parent_dir": str(parent_dir)},
        split={"seed": parent["split"]["seed"], "python_version": platform.python_version(),
               "algorithm": "Preserve question split; retain first exact (gsm8k_id, code) in training"},
        encoded_token_sha256=token_digest.hexdigest(),
        invocation=[sys.executable, *sys.argv],
        preparation_code_sha256={path.relative_to(ROOT).as_posix(): sha256_file(path)
                                 for path in (Path(__file__).resolve(), ROOT / "mdlm_sft/data.py",
                                              ROOT / "mdlm_sft/prepare_data.py")},
        artifacts_sha256={name: sha256_file(staging / name) for name in parent["artifacts_sha256"]},
        derivation={"kind": "exact_training_deduplication", "parent_manifest_sha256": parent_sha,
                    "key": ["gsm8k_id", "code"], "keep": "first", "removed_samples": len(duplicates),
                    "retained_rows": "unchanged", "validation_references": "byte-identical to parent",
                    "sampling": "Uniform over retained samples; per-question multiplicity may change"},
        checks={"no_question_overlap": True, "question_split_unchanged": True,
                "retained_row_order_unchanged": True, "retained_text_matches_cache": True,
                "exact_training_duplicates_remaining": 0, "validation_files_byte_identical": True},
    )
    manifest["counts"]["train"] = {
        "questions": len(train_ids), "samples": len(kept),
        "target_tokens": int(original.target_mask[indices].sum()),
    }
    for name, digest in parent["artifacts_sha256"].items():
        if name not in {"train.jsonl", "train.npz"} and manifest["artifacts_sha256"][name] != digest:
            raise AssertionError(f"Copied artifact changed: {name}")
    write_json(staging / "manifest.json", manifest)
    written = PreparedDataset(staging, "train")
    if read_jsonl(staging / "train.jsonl") != kept:
        raise AssertionError("Serialized retained rows changed")
    for key in ("input_ids", "target_mask", "prefix_lengths"):
        if not np.array_equal(getattr(written, key), getattr(original, key)[indices]):
            raise AssertionError(f"Written cache mismatch: {key}")
    if written.gsm8k_ids != [row["gsm8k_id"] for row in kept]:
        raise AssertionError("Written cache question IDs changed")
    if verify_prepared(parent_dir) != parent or sha256_file(parent_dir / "manifest.json") != parent_sha:
        raise ValueError("Parent changed during preparation")
    report_dir.mkdir(parents=True)
    write_json(report_dir / "kept_parent_indices.json", indices)
    write_jsonl(report_dir / "duplicates.jsonl", duplicates)
    staging.rename(output_dir)
    report = {
        "status": "complete", "parent_manifest_sha256": parent_sha,
        "output_manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "input_train_samples": len(rows), "output_train_samples": len(kept),
        "removed_duplicates": len(duplicates),
        "duplicate_groups": len({item["kept_parent_row_index"] for item in duplicates}),
        "counts": manifest["counts"], "checks": manifest["checks"],
        "output_dir": str(output_dir), "runtime_execution": "Not rerun; retained code is unchanged",
        "report_artifacts_sha256": {name: sha256_file(report_dir / name)
                                    for name in ("kept_parent_indices.json", "duplicates.jsonl")},
    }
    write_json(report_dir / "preparation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path,
                        default=ROOT.parent / ".mdlm-data-work" / "data/style_aligned")
    parser.add_argument("--output-dir", type=Path, default=Path("data/style"))
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT.parent / ".mdlm-data-work" / "runs/style_alignment/deduplication",
                        help="Deduplication evidence directory (defaults outside the source directory)")
    parser.add_argument("--tokenizer", type=Path, default=Path("inputs/tokenizer"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_dir, args.output_dir, args.report_dir, args.tokenizer),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
