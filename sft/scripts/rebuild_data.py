"""Rebuild the five published SFT datasets from immutable public sources."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mdlm_sft.data import read_jsonl, sha256_file, verify_prepared
from mdlm_sft.prepare_data import prepare, write_jsonl
from scripts import deduplicate_data, prepare_mix_data, prepare_single_data, prepare_style_data, prepare_variable_data
from scripts.cleaning.clean_gsm8k_pot import wrap_program
from scripts.manage_inputs import download_missing, verify_file


RECIPES = ROOT / "datasets"
NAMES = ("prepared", "style", "single", "vars", "mix")


def checked_sources(inputs, names, download):
    entries = json.loads((RECIPES / "sources.lock.json").read_bytes())["files"]
    for entry in entries:
        if entry["path"] in names or ("tinygsm" in names and entry["path"].startswith("tinygsm/")):
            if download:
                download_missing(inputs, entry)
            verify_file(inputs / entry["path"], entry)


def restore_clean(inputs, out):
    """Replay the recorded row selection and reapply the original code transform."""
    import pyarrow.parquet as pq

    raw = json.loads((inputs / "source/MathInstruct.json").read_bytes())
    official = pq.read_table(inputs / "source/train.parquet", columns=["question"]).to_pylist()
    with gzip.open(RECIPES / "clean-index.json.gz", "rt", encoding="utf-8") as stream:
        index = json.load(stream)
    with out.open("w", encoding="utf-8", newline="\r\n") as stream:
        for question_id, sources, gold, q_len, c_len, total in index:
            original = raw[sources[0]]["output"]
            code = wrap_program(original)
            row = dict(gsm8k_id=f"train:{question_id}", gsm8k_train_row_0based=question_id,
                       question=official[question_id]["question"], gold_answer=gold, code=code,
                       source_rows_0based=sources, original_code_sha256=hashlib.sha256(original.encode()).hexdigest(),
                       code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                       question_tokens=q_len, code_tokens=c_len, total_tokens=total)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    expected = json.loads((RECIPES / "manifests/prepared.json").read_bytes())["source_sha256"]["clean_data"]
    if sha256_file(out) != expected:
        raise ValueError("Reconstructed clean rows differ from the original verified input")


def build_mix(inputs, style, output, work):
    selection, execution = work / "selection", work / "execution"
    prepare_mix_data.select(argparse.Namespace(
        parent_dir=style, tokenizer=inputs / "tokenizer", official_train=inputs / "source/train.parquet",
        official_test=inputs / "gsm8k/test.parquet", tinygsm_dir=inputs / "tinygsm", output_dir=selection,
        seed=8, draw_count=6144, replay_count=3072))
    execution.mkdir()
    receipt = RECIPES / "replay-execution.json"
    report = json.loads(receipt.read_bytes())
    if sha256_file(selection / "candidates.jsonl") != report["candidates_sha256"]:
        raise ValueError("Public TinyGSM candidates differ from the recorded execution batch")
    with gzip.open(RECIPES / "replay-checks.jsonl.gz", "rb") as archive:
        (execution / "checks.jsonl").write_bytes(archive.read())
    candidates = {r["gsm8k_id"]: r for r in read_jsonl(selection / "candidates.jsonl")}
    accepted = []
    for check in read_jsonl(execution / "checks.jsonl"):
        if check["accepted"] and len(accepted) < report["accepted"]:
            row = candidates[check["gsm8k_id"]]
            if row["code_sha256"] != check["code_sha256"]:
                raise ValueError("Archived execution result refers to different code")
            accepted.append({**row, "execution_answer": check["result"]["predicted_answer"]})
    write_jsonl(execution / "accepted.jsonl", accepted)
    shutil.copyfile(receipt, execution / "execution.json")
    prepare_mix_data.assemble(argparse.Namespace(parent_dir=style, output_dir=output, tokenizer=inputs / "tokenizer",
                                                  selection_dir=selection, execution_dir=execution))


def publish_dataset(name, staging, destination):
    """Validate all data and standardize machine-specific serialization metadata."""
    reference = RECIPES / "manifests" / (name + ".json")
    expected = json.loads(reference.read_bytes())
    observed = verify_prepared(staging)
    for key in ("counts", "encoded_token_sha256", "sequence_length", "tokenizer", "checks", "official_check"):
        if observed[key] != expected[key]:
            raise ValueError(f"Rebuilt {name} differs at {key}")
    for filename, digest in expected["artifacts_sha256"].items():
        path = staging / filename
        if sha256_file(path) == digest:
            continue
        if filename.endswith("_ids.json"):
            path.write_bytes(path.read_text(encoding="utf-8").replace("\n", "\r\n").encode())
        elif filename.endswith(".npz"):
            # The experiment caches used ZIP's DOS host ID. The NPY streams stay untouched.
            with zipfile.ZipFile(path, "a") as archive:
                for member in archive.infolist():
                    member.create_system = 0
                archive.comment = archive.comment  # Mark the central directory for rewrite.
        if sha256_file(path) != digest:
            raise ValueError(f"Rebuilt {name}/{filename} failed its original SHA-256 check")
    shutil.copyfile(reference, staging / "manifest.json")
    verify_prepared(staging)
    staging.rename(destination)


def rebuild(requested, inputs, data_dir, download, include_test):
    names = set(requested)
    if names & {"vars", "single", "mix"}:
        names.add("style")
    if names:
        names.add("prepared")
    source_names = {"gsm8k/test.parquet"} if include_test or "mix" in names else set()
    data_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for name in NAMES:
        if name not in names:
            continue
        path = data_dir / name
        if path.exists():
            verify_prepared(path)
            if (path / "manifest.json").read_bytes() != (RECIPES / "manifests" / (name + ".json")).read_bytes():
                raise ValueError(f"Different dataset already exists: {path}")
        else:
            pending.append(name)
    if "prepared" in pending:
        source_names.update(("source/MathInstruct.json", "source/train.parquet"))
    if "mix" in pending:
        source_names.update(("source/train.parquet", "tinygsm"))
    checked_sources(inputs, source_names, download)
    tokenizer = inputs / "tokenizer"
    for name in pending:
        print(f"Preparing {name} ...", flush=True)
        # Staging alongside the dataset keeps the final rename on the same volume.
        with tempfile.TemporaryDirectory(prefix=".rebuild-", dir=data_dir) as temp:
            staging, work = Path(temp) / name, Path(temp) / "work"
            work.mkdir()
            if name == "prepared":
                clean = work / "clean.jsonl"
                restore_clean(inputs, clean)
                prepare(clean, inputs / "source/train.parquet", tokenizer, staging, local_files_only=True)
            elif name == "style":
                prepare_style_data.prepare(data_dir / "prepared", work / "aligned", tokenizer, work / "style-report")
                deduplicate_data.prepare(work / "aligned", staging, work / "dedup-report", tokenizer)
            elif name == "single":
                prepare_single_data.prepare(data_dir / "style", staging, work / "single-report", tokenizer)
            elif name == "vars":
                prepare_variable_data.prepare(data_dir / "style", staging, tokenizer, work / "vars-report")
            elif name == "mix":
                build_mix(inputs, data_dir / "style", staging, work)
            publish_dataset(name, staging, data_dir / name)
        print(f"Verified {name}: eight original artifacts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=(*NAMES, "all", "test"), help="Required parents are built automatically")
    parser.add_argument("--download", action="store_true", help="Fetch missing public source inputs")
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    names = NAMES if args.dataset == "all" else () if args.dataset == "test" else (args.dataset,)
    rebuild(names, args.inputs.resolve(), args.data_dir.resolve(), args.download, args.dataset in {"all", "test"})


if __name__ == "__main__":
    main()
