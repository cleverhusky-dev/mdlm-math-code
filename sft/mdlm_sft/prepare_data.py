"""Prepare grouped, auditable training/validation artifacts without touching test data."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import platform
import sys

from .data import (
    SEQUENCE_LENGTH, encode_example, group_rows, load_tokenizer, read_jsonl,
    sha256_file, split_grouped, tokenizer_identity, verify_prepared,
)


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def verify_official_train(rows, official_train: Path) -> dict:
    """Check all retained IDs against official TRAIN only; no scoring relaxation."""
    import pyarrow.parquet as pq
    official = pq.read_table(official_train, columns=["question", "answer"]).to_pylist()
    if len(official) != 7473:
        raise ValueError(f"Expected 7473 official training rows, got {len(official)}")
    groups = group_rows(rows)
    for identity, members in groups.items():
        row = members[0]
        index = int(identity.split(":")[1])
        if index >= len(official) or row["question"] != official[index]["question"]:
            raise ValueError(f"Question differs from official training row: {identity}")
        text = official[index]["answer"]
        if text.count("####") != 1:
            raise ValueError(f"Invalid official answer marker for {identity}")
        try:
            gold = Decimal(text.split("####")[1].strip().replace(",", ""))
        except InvalidOperation as error:
            raise ValueError(f"Invalid official answer for {identity}") from error
        if not gold.is_finite() or gold != Decimal(str(row["gold_answer"])):
            raise ValueError(f"Clean gold answer differs from official training answer: {identity}")
    return {"official_train_rows": len(official), "questions_checked": len(groups), "official_test_accessed": False}


def prepare(clean_data: Path, official_train: Path, tokenizer_path: Path | None, output_dir: Path, *, local_files_only=False) -> dict:
    """Generate once; an existing complete cache is verified and reused, never rewritten."""
    import numpy as np
    from tokenizers import Tokenizer
    source_hashes = {"clean_data": sha256_file(clean_data), "official_train": sha256_file(official_train)}
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=local_files_only)
    identity = tokenizer_identity(tokenizer)
    if output_dir.exists() and any(output_dir.iterdir()):
        previous = verify_prepared(output_dir)
        if previous["source_sha256"] != source_hashes or previous["tokenizer"] != identity:
            raise ValueError("Existing cache belongs to different data/tokenizer; use a new output directory")
        return previous
    rows = read_jsonl(clean_data)
    official_check = verify_official_train(rows, official_train)
    partitions, ids = split_grouped(rows)
    if len(ids["train"]) != 5747 or len(ids["val"]) != 639:
        raise AssertionError("Unexpected grouped split size")
    # Independent serialized backend matches the cleaning path. Check ALL question
    # and code segments, not only lengths (equal lengths do not prove equal IDs).
    backend = Tokenizer.from_file(str(Path(tokenizer.mdlm_source_path) / "tokenizer.json"))
    encoded = {}
    token_digest = hashlib.sha256()
    for split, records in partitions.items():
        encoded[split] = []
        for row in records:
            for text in (row["question"].strip(), row["code"].strip()):
                actual = tokenizer.encode(text, add_special_tokens=False)
                if actual != backend.encode(text, add_special_tokens=False).ids:
                    raise ValueError(f"Serialized cleaning backend mismatch: {row['gsm8k_id']}")
            value = encode_example(row, tokenizer)
            encoded[split].append(value)
            token_digest.update(np.asarray(value["input_ids"], dtype="<u4").tobytes())
    if any(sha256_file(path) != source_hashes[key] for key, path in (("clean_data", clean_data), ("official_train", official_train))):
        raise ValueError("Input changed during preparation")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for split, records in partitions.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)
        write_json(output_dir / f"{split}_ids.json", ids[split])
        questions = [{key: member[0][key] for key in ("gsm8k_id", "question", "gold_answer")} for member in group_rows(records).values()]
        write_jsonl(output_dir / f"{split}_questions.jsonl", questions)
        values = encoded[split]
        np.savez_compressed(
            output_dir / f"{split}.npz",
            input_ids=np.asarray([r["input_ids"] for r in values], dtype=np.int32),
            target_mask=np.asarray([r["target_mask"] for r in values], dtype=np.bool_),
            prefix_lengths=np.asarray([r["prefix_length"] for r in values], dtype=np.int32),
            gsm8k_ids=np.asarray([r["gsm8k_id"] for r in values]),
        )
        artifacts.extend(f"{split}{suffix}" for suffix in (".jsonl", "_ids.json", "_questions.jsonl", ".npz"))
    manifest = {
        "schema_version": 1,
        "source_sha256": source_hashes,
        "source_filenames": {"clean_data": clean_data.name, "official_train": official_train.name},
        "source_locations_at_preparation": {"clean_data": str(clean_data.resolve()), "official_train": str(official_train.resolve())},
        "split": {"seed": 42, "algorithm": "numeric-ID sort; random.Random(42).shuffle; first 639 val; numeric-ID output order; lexical-code order within ID", "python_version": platform.python_version()},
        "counts": {split: {"questions": len(ids[split]), "samples": len(records), "target_tokens": sum(sum(row["target_mask"]) for row in encoded[split])} for split, records in partitions.items()},
        "sequence_length": SEQUENCE_LENGTH,
        "tokenizer": identity,
        "official_check": official_check,
        "checks": {"no_question_overlap": True, "all_clean_samples_retained": True, "all_segments_match_cleaning_backend": True, "audited_lengths_match": True},
        "encoded_token_sha256": token_digest.hexdigest(),
        "rl_exclusion_file": "val_ids.json",
        "preparation_code_sha256": {p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("data.py"))},
        "artifacts_sha256": {name: sha256_file(output_dir / name) for name in artifacts},
        "invocation": [sys.executable, *sys.argv],
    }
    write_json(output_dir / "manifest.json", manifest)
    return verify_prepared(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="Audited local tokenizer directory; omitted downloads only six fixed-revision tokenizer/config files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.clean_data, args.official_train, args.tokenizer, args.output_dir, local_files_only=args.local_files_only)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": manifest["counts"], "source_sha256": manifest["source_sha256"], "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    main()
