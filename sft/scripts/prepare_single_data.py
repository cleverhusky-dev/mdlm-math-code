"""Select one unchanged training solution per question; preserve validation bytes."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mdlm_sft.data import (
    PreparedDataset, encode_example, group_rows, load_questions, load_tokenizer,
    question_sort_key, read_jsonl, sha256_file, tokenizer_identity, verify_prepared,
)
from mdlm_sft.prepare_data import write_json, write_jsonl

ARTIFACTS = tuple(f"{split}{suffix}" for split in ("train", "val")
                  for suffix in (".jsonl", ".npz", "_ids.json", "_questions.jsonl"))
SELECTION_RULE = (
    "Numerically sort official training question IDs; lexicographically sort each "
    "question's distinct code strings; use private random.Random(seed).randrange "
    "once per question (including singleton groups); retain selected rows in "
    "their original parent row order without rewriting any field."
)


def select_single(rows, seed=42):
    groups = group_rows(rows)
    positions = {(row["gsm8k_id"], row["code"]): index for index, row in enumerate(rows)}
    rng = random.Random(seed)
    mapping = []
    for order, identity in enumerate(sorted(groups, key=question_sort_key)):
        candidates = sorted(groups[identity], key=lambda row: row["code"])
        choice = rng.randrange(len(candidates))
        row = candidates[choice]
        mapping.append({
            "question_selection_order": order, "gsm8k_id": identity,
            "parent_row_index": positions[(identity, row["code"])],
            "candidate_count": len(candidates), "selected_candidate_index": choice,
            "code_sha256": hashlib.sha256(row["code"].encode("utf-8")).hexdigest(),
            "source_rows_0based": row.get("source_rows_0based"),
        })
    mapping.sort(key=lambda item: item["parent_row_index"])
    for output_index, item in enumerate(mapping):
        item["output_row_index"] = output_index
    indices = [item["parent_row_index"] for item in mapping]
    return [rows[index] for index in indices], indices, mapping


def prepare(parent_dir, output_dir, report_dir, tokenizer_path, *, seed=42, expected_questions=5747):
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
    if set(parent["artifacts_sha256"]) != set(ARTIFACTS):
        raise ValueError("Parent must contain exactly the standard eight prepared artifacts")
    rows = read_jsonl(parent_dir / "train.jsonl")
    kept, indices, mapping = select_single(rows, seed)
    if len(kept) != expected_questions:
        raise ValueError(f"Expected {expected_questions} training questions, got {len(kept)}")
    train_ids = json.loads((parent_dir / "train_ids.json").read_text(encoding="utf-8"))
    val_ids = json.loads((parent_dir / "val_ids.json").read_text(encoding="utf-8"))
    if (len(train_ids) != len(set(train_ids)) or set(train_ids) != {r["gsm8k_id"] for r in kept}
            or len(train_ids) != parent["counts"]["train"]["questions"]):
        raise ValueError("Parent training question IDs/count disagree with rows")
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation questions overlap")
    questions = load_questions(parent_dir, "train")
    if [row["gsm8k_id"] for row in questions] != train_ids:
        raise ValueError("Parent training question file disagrees with IDs")
    question_map = {row["gsm8k_id"]: row for row in questions}
    val_questions = load_questions(parent_dir, "val")
    if [row["gsm8k_id"] for row in val_questions] != val_ids:
        raise ValueError("Parent validation question file disagrees with IDs")
    for row in rows:
        if any(row[key] != question_map[row["gsm8k_id"]][key] for key in ("question", "gold_answer")):
            raise ValueError("Parent training question/gold file disagrees with references")
        if hashlib.sha256(row["code"].encode("utf-8")).hexdigest() != row["code_sha256"]:
            raise ValueError("Parent code hash mismatch")
    original = PreparedDataset(parent_dir, "train")
    PreparedDataset(parent_dir, "val")
    if len(rows) != len(original):
        raise ValueError("Parent training text and cache counts disagree")
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=True)
    if tokenizer_identity(tokenizer) != parent["tokenizer"]:
        raise ValueError("Tokenizer does not match parent data")
    for row, index in zip(kept, indices):
        encoded = encode_example(row, tokenizer)
        if (not np.array_equal(encoded["input_ids"], original.input_ids[index])
                or not np.array_equal(encoded["target_mask"], original.target_mask[index])
                or encoded["prefix_length"] != int(original.prefix_lengths[index])
                or row["gsm8k_id"] != original.gsm8k_ids[index]):
            raise ValueError(f"Parent text/cache mismatch at row {index}")
        compile(row["code"], "<selected-reference>", "exec")
    # Keep all preparation staging outside the source directory. Move only verified data.
    report_dir.mkdir(parents=True)
    staging = Path(tempfile.mkdtemp(prefix="data.building-", dir=report_dir))
    for name in ARTIFACTS:
        if name not in {"train.jsonl", "train.npz"}:
            shutil.copyfile(parent_dir / name, staging / name)
    write_jsonl(staging / "train.jsonl", kept)
    with np.load(parent_dir / "train.npz", allow_pickle=False) as cache:
        np.savez_compressed(staging / "train.npz", **{key: cache[key][indices] for key in cache.files})
    digest = hashlib.sha256()
    for split in ("train", "val"):
        with np.load(staging / f"{split}.npz", allow_pickle=False) as cache:
            for values in cache["input_ids"]:
                digest.update(np.asarray(values, dtype="<u4").tobytes())
    write_jsonl(report_dir / "selection.jsonl", mapping)
    write_json(report_dir / "kept_parent_indices.json", indices)
    manifest = copy.deepcopy(parent)
    manifest.update(
        source_sha256={"parent_manifest": parent_sha, **parent["artifacts_sha256"]},
        source_filenames={"parent_manifest": "manifest.json"},
        source_locations_at_preparation={"parent_dir": str(parent_dir)},
        split={"seed": parent["split"]["seed"], "python_version": platform.python_version(),
               "algorithm": "Preserve parent question train/validation split; one training solution per question"},
        encoded_token_sha256=digest.hexdigest(),
        invocation=[sys.executable, *sys.argv],
        preparation_code_sha256={path.relative_to(ROOT).as_posix(): sha256_file(path)
                                 for path in (Path(__file__).resolve(), ROOT / "mdlm_sft/data.py",
                                              ROOT / "mdlm_sft/prepare_data.py")},
        artifacts_sha256={name: sha256_file(staging / name) for name in ARTIFACTS},
        derivation={"kind": "one_training_solution_per_question", "parent_manifest_sha256": parent_sha,
                    "seed": seed, "algorithm": SELECTION_RULE, "retained_rows": "unchanged",
                    "selection_sha256": sha256_file(report_dir / "selection.jsonl"),
                    "removed_samples": len(rows) - len(kept),
                    "sampling": "One fixed uniformly selected solution per question; uniform question weights",
                    "validation_references": "byte-identical to parent",
                    "runtime_execution": "Not rerun; unchanged rows permit separately audited inherited evidence"},
        checks={"no_question_overlap": True, "question_split_unchanged": True,
                "one_training_solution_per_question": True, "retained_row_order_unchanged": True,
                "retained_text_matches_cache": True, "retained_code_compiles": True,
                "validation_files_byte_identical": True, "question_files_byte_identical": True},
    )
    manifest["counts"]["train"] = {
        "questions": len(train_ids), "samples": len(kept),
        "target_tokens": int(original.target_mask[indices].sum()),
    }
    for name, expected in parent["artifacts_sha256"].items():
        if name not in {"train.jsonl", "train.npz"} and manifest["artifacts_sha256"][name] != expected:
            raise AssertionError(f"Copied artifact changed: {name}")
    write_json(staging / "manifest.json", manifest)
    written = PreparedDataset(staging, "train")
    PreparedDataset(staging, "val")
    if read_jsonl(staging / "train.jsonl") != kept:
        raise AssertionError("Serialized selected rows changed")
    for key in ("input_ids", "target_mask", "prefix_lengths"):
        if not np.array_equal(getattr(written, key), getattr(original, key)[indices]):
            raise AssertionError(f"Written cache mismatch: {key}")
    if written.gsm8k_ids != [row["gsm8k_id"] for row in kept]:
        raise AssertionError("Written cache question IDs changed")
    if verify_prepared(parent_dir) != parent or sha256_file(parent_dir / "manifest.json") != parent_sha:
        raise ValueError("Parent changed during preparation")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output_dir)
    report = {
        "status": "complete", "parent_manifest_sha256": parent_sha,
        "output_manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "input_train_samples": len(rows), "output_train_samples": len(kept),
        "removed_samples": len(rows) - len(kept), "selection_seed": seed,
        "selection_algorithm": SELECTION_RULE, "counts": manifest["counts"],
        "checks": manifest["checks"], "output_dir": str(output_dir),
        "steps_per_epoch_at_batch64": (len(kept) + 63) // 64,
        "total_steps_at_batch64_five_epochs": 5 * ((len(kept) + 63) // 64),
        "runtime_execution": "Not rerun; selected rows are unchanged; inheritance recorded separately",
        "report_artifacts_sha256": {name: sha256_file(report_dir / name)
                                    for name in ("selection.jsonl", "kept_parent_indices.json")},
    }
    write_json(report_dir / "preparation.json", report)
    return report


def package_data(data_dir, package_dir):
    """Write a data-only upload archive and locked file list; no runnable experiment."""
    data_dir, package_dir = Path(data_dir).resolve(), Path(package_dir).resolve()
    verify_prepared(data_dir)
    relative = data_dir.relative_to(ROOT)
    names = ("manifest.json", *ARTIFACTS)
    if any((package_dir / name).exists() for name in ("data.json", "data.tar.gz", "data.tar.gz.sha256")):
        raise FileExistsError("Preserving existing data package")
    package_dir.mkdir(parents=True, exist_ok=True)
    files = [{"path": (relative / name).as_posix(), "size_bytes": (data_dir / name).stat().st_size,
              "sha256": sha256_file(data_dir / name)} for name in names]
    write_json(package_dir / "data.json", {"schema_version": 1, "dataset": data_dir.name, "files": files})
    archive = package_dir / "data.tar.gz"
    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for entry in files:
                path = ROOT / entry["path"]
                info = tar.gettarinfo(str(path), arcname=entry["path"])
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
    with tarfile.open(archive, "r:gz") as tar:
        if tar.getnames() != [entry["path"] for entry in files]:
            raise AssertionError("Package file list changed")
        for entry in files:
            if hashlib.sha256(tar.extractfile(entry["path"]).read()).hexdigest() != entry["sha256"]:
                raise AssertionError(f"Packaged file hash mismatch: {entry['path']}")
    (package_dir / "data.tar.gz.sha256").write_text(
        f"{sha256_file(archive)}  data.tar.gz\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=ROOT / "data/style")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/single")
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT.parent / ".mdlm-data-work" / "runs/experiment11-prepare/preparation")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "inputs/tokenizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-questions", type=int, default=5747)
    parser.add_argument("--package-dir", type=Path, help="Optional external directory for a data transfer archive")
    args = parser.parse_args()
    result = prepare(args.parent_dir, args.output_dir, args.report_dir, args.tokenizer,
                     seed=args.seed, expected_questions=args.expected_questions)
    if args.package_dir is not None:
        package_data(args.output_dir, args.package_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
