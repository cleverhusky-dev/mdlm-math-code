"""Read-only denoising loss on all 1,456 held-out reference programs.

Run from the project root. This uses the existing loss/RNG implementation;
it neither samples programs nor updates model parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mdlm_sft.checkpoints import load_sft
from mdlm_sft.data import PreparedDataset, load_tokenizer, tokenizer_identity
from mdlm_sft.diffusion import build_model
from mdlm_sft.evaluation import validation_loss
from mdlm_sft.pretrained import initialize_pretrained
from mdlm_sft.runtime import (
    atomic_json, cost_record, environment_record, model_fingerprint,
    seed_all, sha256, validate_environment,
)
from mdlm_sft.train import load_config, pretrained_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("pretrained", "sft"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "experiments/experiment0/config.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-wall-seconds", type=float, default=1200)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_OR_TEXT")
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be finite and positive")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        parser.error("Reference loss evaluation uses one process/GPU")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new loss output directory; existing results are not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started, status, rate = time.monotonic(), "failed", None
    try:
        validate_environment()
        config = load_config(args.config, args.set)
        if config["noise_eps"] != 0:
            raise ValueError("The published checkpoint and loss protocol require noise_eps=0")
        batch_size, seed = config["validation_loss_batch_size"], config["validation_loss_seed"]
        if type(batch_size) is not int or batch_size <= 0 or type(seed) is not int:
            raise ValueError("Loss batch size must be a positive integer and seed must be an integer")
        rate = config["hourly_cost_rmb_per_gpu"]
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU evaluation requires explicit --device cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        seed_all(seed)
        tokenizer = load_tokenizer(config["tokenizer_path"], local_files_only=True)
        data = PreparedDataset(config["data_dir"], "val")
        if len(data) != 1456 or len(set(data.gsm8k_ids)) != 639:
            raise ValueError("Loss evaluation requires all 1456 references from 639 validation questions")
        token_identity = tokenizer_identity(tokenizer)
        if data.manifest["tokenizer"] != token_identity:
            raise ValueError("Prepared references and tokenizer identity differ")
        model = build_model(tokenizer, {"noise_eps": config["noise_eps"]})
        if args.kind == "pretrained":
            audit = initialize_pretrained(model, args.checkpoint)
            atomic_json(output / "pretrained_audit.json", audit)
            source = pretrained_source(audit)
            checkpoint_sha256 = audit["checkpoint_sha256"]
        else:
            state = load_sft(args.checkpoint)
            if state["config"].get("smoke"):
                raise ValueError("Smoke checkpoints cannot produce formal validation loss")
            if state["identity"]["model"] != asdict(model.config):
                raise ValueError("SFT checkpoint model/noise configuration differs")
            if state["identity"]["tokenizer"] != token_identity:
                raise ValueError("SFT checkpoint tokenizer identity differs")
            model.load_state_dict(state["model"], strict=True)
            source = {"source": "sft_ordinary", "global_step": state["progress"]["global_step"],
                      "model_sha256": model_fingerprint(model)}
            checkpoint_sha256 = sha256(args.checkpoint)
            del state
        model.to(device)
        data_sha256 = sha256(Path(config["data_dir"]) / "manifest.json")
        protocol = {
            "precision": config["precision"], "batch_size": batch_size, "seed": seed,
            "model": asdict(model.config), "tokenizer": token_identity,
            "data_manifest_sha256": data_sha256,
            "validation_artifacts_sha256": {name: value for name, value in
                data.manifest["artifacts_sha256"].items() if name.startswith("val")},
            "loss_entry_sha256": sha256(__file__),
            "meaning": "Reference denoising loss; generation steps and temperature are unused",
        }
        atomic_json(output / "checkpoint_file.json", {
            "path": str(Path(args.checkpoint).resolve()), "sha256": checkpoint_sha256,
        })
        atomic_json(output / "config.resolved.json", config)
        atomic_json(output / "environment.json", environment_record())
        atomic_json(output / "loss_protocol.json", protocol)
        metrics = validation_loss(model, data, batch_size, seed, device, config["precision"],
                                  deadline=started + args.max_wall_seconds)
        summary = {"complete": True, "label": "validation_denoising_loss", **metrics,
                   "checkpoint_source": source, "checkpoint_sha256": checkpoint_sha256,
                   "data_manifest_sha256": data_sha256, "precision": config["precision"],
                   "batch_size": batch_size, "wall_seconds": time.monotonic() - started}
        atomic_json(output / "summary.json", summary)
        status = "complete"
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as error:
        if isinstance(error, TimeoutError):
            status = "bounded_stop"
        atomic_json(output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        atomic_json(output / "usage.json", {"status": status,
            **cost_record(time.monotonic() - started, 1 if args.device == "cuda" else 0, rate)})


if __name__ == "__main__":
    main()
