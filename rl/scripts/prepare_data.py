"""Export the frozen training/validation questions as a minimal RL dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdlm_rl.data import load_questions, question_sort_key, sha256_file


PARENT_MANIFEST_SHA256 = "a155bd95cdbb92184e3c3db5a9c91f98d62f77ed3dfcfec547ea627034f814e8"
PUBLIC_PARENT_SHA256 = "82f7b0219548cb81d2880eab2453d5370416a52ed5d9605cd250672865cfc34b"
QUESTION_FILES = ("train_questions.jsonl", "val_questions.jsonl")


def _resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _validate_rows(rows, split, expected_count):
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("Expected question counts must be positive integers")
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} {split} questions, found {len(rows)}")
    for row in rows:
        identity = row["gsm8k_id"]
        if question_sort_key(identity)[0] != "train":
            raise ValueError("RL preparation accepts official training IDs only")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Missing question for {identity}")
        gold = row.get("gold_answer")
        if (isinstance(gold, bool) or not isinstance(gold, (int, float))
                or not math.isfinite(gold)):
            raise ValueError(f"A finite numeric gold answer is required for {identity}")


def _destination_matches(destination, contents):
    return (destination.is_dir()
            and {entry.name for entry in destination.iterdir()} == set(contents)
            and all((destination / name).is_file()
                    and (destination / name).read_bytes() == data
                    for name, data in contents.items()))


def export_data(source_dir, output_dir, *, expected_parent_sha256=PARENT_MANIFEST_SHA256,
                expected_train_questions=5747, expected_val_questions=639):
    """Copy question bytes and commit one deterministic manifest; never overwrite."""
    source, destination = _resolve(source_dir), _resolve(output_dir)
    source_manifest_path = source / "manifest.json"
    parent_hash = sha256_file(source_manifest_path)
    if parent_hash != expected_parent_sha256:
        raise ValueError("Source manifest SHA256 differs from the selected SFT dataset")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train = load_questions(source, "train")
    val = load_questions(source, "val")
    _validate_rows(train, "train", expected_train_questions)
    _validate_rows(val, "val", expected_val_questions)
    if {row["gsm8k_id"] for row in train} & {row["gsm8k_id"] for row in val}:
        raise ValueError("Training and validation question IDs overlap")
    contents = {name: (source / name).read_bytes() for name in QUESTION_FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    if any(source_manifest["artifacts_sha256"].get(name) != digest
           for name, digest in hashes.items()):
        raise ValueError("Question bytes differ from the verified source manifest")
    if sha256_file(source_manifest_path) != parent_hash:
        raise ValueError("Source manifest changed during export")
    manifest = {
        "schema_version": 1,
        "sequence_length": 512,
        "kind": "rl_questions",
        "parent_manifest_sha256": parent_hash,
        "artifacts_sha256": hashes,
        "source_files_sha256": hashes.copy(),
        "counts": {"train": {"questions": len(train)}, "val": {"questions": len(val)}},
    }
    contents["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if destination.exists():
        if not _destination_matches(destination, contents):
            raise FileExistsError("Destination differs from this export; existing data was not overwritten")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(destination.parent.resolve()):
            raise RuntimeError("Staging path must stay inside the destination parent")
        for name, data in contents.items():
            (staging / name).write_bytes(data)
        # A complete directory becomes visible at once. A concurrent writer
        # must produce exactly the same bytes, otherwise its data is retained.
        try:
            staging.rename(destination)
        except OSError:
            if not _destination_matches(destination, contents):
                raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="../sft/data/vars")
    parser.add_argument("--output-dir", default="runs/data/gsm8k",
                        help="Directory for the exported questions and provenance manifest")
    args = parser.parse_args(argv)
    source = _resolve(args.source_dir)
    parent_digest = sha256_file(source / "manifest.json")
    if parent_digest not in {PARENT_MANIFEST_SHA256, PUBLIC_PARENT_SHA256}:
        raise ValueError("Expected the experiment 12 SFT dataset manifest")
    manifest = export_data(source, args.output_dir, expected_parent_sha256=parent_digest)
    output = _resolve(args.output_dir)
    print(json.dumps({"status": "ready", "data_dir": str(output),
                      "manifest_sha256": sha256_file(output / "manifest.json"),
                      "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
