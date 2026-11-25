"""Single-GPU unified baseline/SFT evaluation CLI; test is explicitly gated."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from .checkpoints import load_sft
from .data import load_questions, load_tokenizer
from .evaluation import generate_and_score
from .diffusion import build_model
from .runtime import ROOT, atomic_json, environment_record, cost_record, seed_all, sha256, model_fingerprint
from .runtime import validate_environment
from .scoring import extract_gold_answer
from .train import load_config, pretrained_source
from .pretrained import initialize_pretrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments/experiment0/config.json"))
    parser.add_argument("--kind", choices=("pretrained", "sft"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "train-smoke", "test"), default="val")
    parser.add_argument("--final-test-parquet")
    parser.add_argument("--confirm-frozen-protocol", action="store_true")
    parser.add_argument("--tokenizer")
    parser.add_argument("--data-dir")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--max-wall-seconds", type=float, default=43200)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    validate_environment()
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Unified generation evaluation runs on exactly one process/GPU")
    if args.max_wall_seconds <= 0:
        raise ValueError("Positive wall-clock bound required")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evaluation directory; do not overwrite earlier results")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.set)
    if config["noise_eps"] != 0:
        raise ValueError("The approved checkpoint schedule uses noise_eps=0")
    if args.tokenizer:
        config["tokenizer_path"] = args.tokenizer
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.split == "test":
        if not args.confirm_frozen_protocol or not args.final_test_parquet:
            raise ValueError("Final test needs --confirm-frozen-protocol and --final-test-parquet")
        import pyarrow.parquet as pq
        rows = pq.read_table(args.final_test_parquet, columns=["question", "answer"]).to_pylist()
        if len(rows) != 1319:
            raise ValueError("Expected the complete official 1319-question test set")
        questions = [{"gsm8k_id": f"test:{i}", "question": row["question"],
                      "gold_answer": extract_gold_answer(row["answer"])} for i, row in enumerate(rows)]
        atomic_json(output / "test_source.json", {"sha256": sha256(args.final_test_parquet), "frozen_protocol_acknowledged": True})
    else:
        if args.final_test_parquet or args.confirm_frozen_protocol:
            raise ValueError("Test arguments are only accepted with --split test")
        questions = load_questions(config["data_dir"], "val" if args.split == "val" else "train")
        if args.split == "val" and len(questions) != 639:
            raise ValueError("Formal validation must contain all 639 independent questions")
        if args.split == "train-smoke":
            questions = questions[:8]
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    seed_all(config["evaluation"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    tokenizer = load_tokenizer(config["tokenizer_path"], local_files_only=True)
    model = build_model(tokenizer, {"noise_eps": config["noise_eps"]})
    if args.kind == "pretrained":
        audit = initialize_pretrained(model, args.checkpoint)
        atomic_json(output / "pretrained_audit.json", audit)
        source = pretrained_source(audit)
    else:
        state = load_sft(args.checkpoint)
        if state["config"].get("smoke") and args.split != "train-smoke":
            raise ValueError("A smoke checkpoint cannot produce formal validation/test numbers")
        if state["identity"]["model"] != vars(model.config):
            raise ValueError("SFT checkpoint model/noise configuration differs from evaluation configuration")
        model.load_state_dict(state["model"], strict=True)
        source = {"source": "sft_ordinary", "global_step": state["progress"]["global_step"],
                  "model_sha256": model_fingerprint(model)}
        atomic_json(output / "checkpoint_file.json", {"path": str(Path(args.checkpoint).resolve()), "sha256": sha256(args.checkpoint)})
        del state
    model.to(device)
    atomic_json(output / "environment.json", environment_record())
    atomic_json(output / "config.resolved.json", config)
    label = "exploratory_train_smoke" if args.split == "train-smoke" else ("final_test" if args.split == "test" else "validation")
    status = "failed"
    try:
        summary = generate_and_score(model, tokenizer, questions, config["evaluation"], device, config["precision"], output,
            checkpoint_source=source, label=label, deadline=started + args.max_wall_seconds)
        status = "complete"
        print(json.dumps(summary), flush=True)
    finally:
        atomic_json(output / "usage.json", {"status": status,
            **cost_record(time.monotonic() - started, 1 if device.type == "cuda" else 0, config["hourly_cost_rmb_per_gpu"])})


if __name__ == "__main__":
    main()
