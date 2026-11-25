"""Experiment5: read-only, short-question train/validation generalization probe."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_version import capture_version
from generalization_probe import audit_references, check_groups, prediction_diagnostics, select_questions
from manage_inputs import asset_path, load_assets, verify_file
from mdlm_sft.baseline import experiment1_checkpoint, read_json, resolve_path, sha256
from mdlm_sft.data import read_jsonl

LIMITATIONS = ["Exploratory 64-question short-reference subsets, not a full validation benchmark",
              "Length matching does not control question difficulty or reference multiplicity",
              "Training questions were seen during SFT; a train/val gap alone does not establish memorization",
              "One generation seed and one denoising-loss seed; no confidence claim",
              "All four groups are regenerated in subset order; old full-set predictions are not subset scores",
              "Loss is target-token weighted over every reference, including longer alternatives and EOS/PAD; it is not a question mean"]


def verify_scores(root, config, reference, token_identity, val_ids):
    sources, common_protocol = {}, None
    for spec in config["models"]:
        directory = resolve_path(root, spec["score_dir"])
        summary, protocol = read_json(directory / "summary.json"), read_json(directory / "protocol.json")
        digest = protocol.pop("protocol_sha256")
        rows = read_jsonl(directory / "predictions.jsonl")
        counts = {key: 0 for key in ("correct", "format_failure", "execution_failure", "wrong_answer")}
        counts.update(Counter(row["status"] for row in rows))
        if (summary.get("complete") is not True or summary.get("label") != "validation"
                or summary.get("num_questions") != 639 or summary.get("question_count") != 639
                or len(rows) != 639 or counts != summary["counts"]
                or summary["accuracy"] != counts["correct"] / 639
                or sha256(directory / "predictions.jsonl") != summary["predictions_sha256"]):
            raise ValueError(f"Incomplete or inconsistent reference score: {directory}")
        if (set(protocol) != {"generation", "precision", "numerics", "scorer_version", "tokenizer",
                             "question_order", "questions_sha256", "evaluation_source"}
                or hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != digest
                or summary["protocol_sha256"] != digest or protocol["question_order"] != val_ids
                or [row["gsm8k_id"] for row in rows] != val_ids
                or protocol["generation"] != reference["evaluation"]
                or protocol["precision"] != reference["precision"] or protocol["tokenizer"] != token_identity
                or set(protocol["evaluation_source"]) != {"evaluation.py", "sampling.py", "scoring.py", "executor.py",
                                                           "data.py", "diffusion.py", "transformer.py", "runtime.py"}
                or any(sha256(root / "mdlm_sft" / name) != value
                       for name, value in protocol["evaluation_source"].items())):
            raise ValueError("Saved reference evaluation protocol differs")
        if common_protocol is not None and protocol != common_protocol:
            raise ValueError("The two saved reference protocols differ")
        common_protocol = protocol
        experiment = directory.parent if spec["kind"] == "pretrained" else directory.parents[2]
        metadata = read_json(experiment / "experiment.json")
        if metadata["inputs_lock_sha256"] != sha256(root / "inputs.lock.json"):
            raise ValueError("Reference experiment input lock differs")
        if spec["kind"] == "sft":
            training = read_json(experiment / "training/result.json")
            expected_lr = 3e-5
            if (training.get("status") != "complete" or training.get("smoke") is not False
                    or training.get("global_step") != 609 or training.get("epoch") != 3
                    or training.get("pending_validation") is not None
                    or summary["checkpoint_source"].get("global_step") != 609
                    or metadata["config"]["learning_rate"] != expected_lr):
                raise ValueError("SFT references must be completed non-smoke step 609")
        sources[spec["name"]] = {"checkpoint_source": summary["checkpoint_source"],
                               "summary_sha256": sha256(directory / "summary.json"),
                               "protocol_sha256": digest, "protocol": protocol,
                               "experiment_sha256": sha256(experiment / "experiment.json")}
    return sources


def run(args, root=ROOT):
    root = Path(root).resolve()
    config_path = resolve_path(root, args.config)
    config = read_json(config_path)
    reference_path = resolve_path(root, config["reference_config"])
    reference = read_json(reference_path)
    if (reference_path != root / "experiments/experiment1/config.json"
            or config["questions_per_split"] != 64 or config["selection_seed"] != 5
            or config["max_reference_tokens"] != 256 or reference["evaluation"]["steps"] != 32
            or reference["evaluation"]["seed"] != 1 or reference["precision"] != "bf16"
            or reference["noise_eps"] != 0 or reference["epochs"] != 3):
        raise ValueError("Experiment5 requires the frozen 64-question/256-token/32-step protocol")
    if [(m["name"], m["kind"]) for m in config["models"]] != [
            ("pretrained", "pretrained"), ("sft_original", "sft")]:
        raise ValueError("Expected pretrained and original SFT, in that order")
    if any(not math.isfinite(config[key]) or config[key] <= 0
           for key in ("max_wall_seconds", "instance_hourly_rmb")):
        raise ValueError("Positive finite duration and instance rate required")
    formal_output = resolve_path(root, config["output_dir"])
    output = resolve_path(root, args.output_dir) if args.output_dir else (
        formal_output.with_name(formal_output.name + "-prepare") if args.prepare_only else formal_output)
    if args.prepare_only and (output == formal_output or output.is_relative_to(formal_output)):
        raise ValueError("Preparation needs an independent output directory outside the formal run")
    plan = {"experiment": "experiment5", "output": str(output), "prepare_only": args.prepare_only,
            "config": config, "evaluation": reference["evaluation"], "precision": reference["precision"],
            "model_order": [m["name"] for m in config["models"]], "split_order": ["train", "val"],
            "instance_cost_cap_rmb": None if args.prepare_only else config["max_wall_seconds"] / 3600 * config["instance_hourly_rmb"],
            "allocated_gpus": 0 if args.prepare_only else 2,
            "active_model_gpus": 0 if args.prepare_only else 1, "limitations": LIMITATIONS}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Keep failure reporting usable even when GPU/dependency/version preflight fails.
    def write(name, value):
        temp = output / (name + ".tmp")
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(output / name)
    started, deadline, stage, groups = time.monotonic(), None, "preflight", {}
    def progress(value):
        nonlocal stage
        stage = value
        write("status.json", {"status": "running", "stage": stage, "completed_groups": list(groups)})
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Experiment5 wall-clock bound reached")
    try:
        progress("preflight")
        version = None if args.prepare_only else capture_version(root)
        import torch
        from torch.utils.data import Subset
        from mdlm_sft.data import PreparedDataset, load_questions, load_tokenizer, tokenizer_identity
        from mdlm_sft.runtime import environment_record, model_fingerprint, validate_environment
        from mdlm_sft.evaluation import generate_and_score, validation_loss
        from mdlm_sft.diffusion import build_model
        from mdlm_sft.checkpoints import load_sft
        from mdlm_sft.pretrained import initialize_pretrained
        from mdlm_sft.train import pretrained_source
        validate_environment()
        deadline = started + config["max_wall_seconds"]
        if not args.prepare_only and (int(os.environ.get("WORLD_SIZE", 1)) != 1 or not torch.cuda.is_available()):
            raise RuntimeError("Experiment5 requires CUDA and exactly one process; CPU generation is disabled")
        verified = {}
        for entry in load_assets(root):
            if args.prepare_only and entry["kind"] == "checkpoint":
                continue
            verify_file(asset_path(root, entry["path"]), entry)
            verified[entry["path"]] = entry["sha256"]
        progress("selection_and_data_audit")
        directory = resolve_path(root, reference["data_dir"])
        tokenizer = load_tokenizer(resolve_path(root, reference["tokenizer_path"]), local_files_only=True)
        identity = tokenizer_identity(tokenizer)
        datasets = {split: PreparedDataset(directory, split) for split in ("train", "val")}
        questions = {split: load_questions(directory, split) for split in datasets}
        references = {split: read_jsonl(directory / f"{split}.jsonl") for split in datasets}
        for split, data in datasets.items():
            if (data.manifest["tokenizer"] != identity
                    or read_json(directory / f"{split}_ids.json") != [q["gsm8k_id"] for q in questions[split]]):
                raise ValueError("Tokenizer or frozen split ID identity differs")
        selection = select_questions(references["train"], references["val"],
            questions_per_split=config["questions_per_split"], max_reference_tokens=config["max_reference_tokens"],
            seed=config["selection_seed"])
        indices, audit = audit_references(directory, datasets, questions, references, selection, tokenizer)
        write("selection.json", selection)
        write("data_audit.json", audit)
        metadata = {**plan, "git": version, "config_sha256": sha256(config_path),
                    "reference_config_sha256": sha256(reference_path), "inputs_lock_sha256": sha256(root / "inputs.lock.json"),
                    "verified_inputs_sha256": verified, "selection_sha256": sha256(output / "selection.json"),
                    "data_manifest_sha256": sha256(directory / "manifest.json"), "environment": environment_record(),
                    "checkpoint_weights_verified": False}
        write("experiment.json", metadata)
        if args.prepare_only:
            result = {"experiment": "experiment5", "status": "prepared_only", "generation_performed": False,
                      "checkpoint_weights_verified": False, "selection_sha256": metadata["selection_sha256"]}
        else:
            progress("reference_identity")
            sources = verify_scores(root, config, reference, identity, [q["gsm8k_id"] for q in questions["val"]])
            metadata["score_references"] = sources
            write("experiment.json", metadata)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            device = torch.device("cuda")
            for spec in config["models"]:
                name = spec["name"]
                progress(f"{name}/load")
                model = build_model(tokenizer, {"noise_eps": reference["noise_eps"]})
                checkpoint = resolve_path(root, spec["checkpoint"])
                if spec["kind"] == "pretrained":
                    loaded = initialize_pretrained(model, checkpoint)
                    source = pretrained_source(loaded)
                    checkpoint_sha = loaded["checkpoint_sha256"]
                else:
                    if checkpoint.name == "step-00000609.pt":
                        checkpoint = experiment1_checkpoint(checkpoint.parent)
                    state = load_sft(checkpoint)
                    if (state["config"].get("smoke") is not False or state["config"]["epochs"] != 3
                            or state["progress"]["global_step"] != 609
                            or state["identity"]["model"] != vars(model.config)
                            or state["identity"]["tokenizer"] != identity
                            or state["identity"]["data_manifest_sha256"] != metadata["data_manifest_sha256"]):
                        raise ValueError("SFT checkpoint training/data/tokenizer/model identity differs")
                    model.load_state_dict(state["model"], strict=True)
                    source = {"source": "sft_ordinary", "global_step": 609, "model_sha256": model_fingerprint(model)}
                    checkpoint_sha = sha256(checkpoint)
                    del state
                if source != sources[name]["checkpoint_source"]:
                    raise ValueError(f"Loaded weights do not match saved score identity: {name}")
                before = model_fingerprint(model)
                model.requires_grad_(False).eval().to(device)
                write(f"{name}-checkpoint.json", {"path": str(checkpoint), "sha256": checkpoint_sha,
                    "checkpoint_source": source, "model_sha256_before": before})
                for split in ("train", "val"):
                    key, group_dir = f"{name}/{split}", output / name / split
                    progress(key + "/generation")
                    lookup = {q["gsm8k_id"]: q for q in questions[split]}
                    selected_questions = [lookup[key] for key in selection["ids"][split]]
                    score = generate_and_score(model, tokenizer, selected_questions, reference["evaluation"], device,
                        reference["precision"], group_dir, checkpoint_source=source, label=f"exploratory_{split}", deadline=deadline)
                    protocol = read_json(group_dir / "protocol.json")
                    if (score.get("complete") is not True or score["num_questions"] != 64
                            or score["checkpoint_source"] != source
                            or protocol["question_order"] != selection["ids"][split]
                            or any(protocol[field] != sources[name]["protocol"][field]
                                   for field in ("generation", "precision", "numerics", "scorer_version", "tokenizer", "evaluation_source"))):
                        raise ValueError("A subset changed the frozen generation protocol or checkpoint")
                    progress(key + "/reference_loss")
                    loss = validation_loss(model, Subset(datasets[split], indices[split]), reference["validation_loss_batch_size"],
                        reference["validation_loss_seed"], device, reference["precision"], deadline=deadline)
                    loss.update({"complete": True, "label": f"exploratory_{split}", "checkpoint_source": source,
                                 "precision": reference["precision"], "batch_size": reference["validation_loss_batch_size"],
                                 "num_questions": 64, "data_manifest_sha256": metadata["data_manifest_sha256"],
                                 "selection_sha256": metadata["selection_sha256"], "aggregation": "target-token weighted over all references"})
                    diagnostics = prediction_diagnostics(read_jsonl(group_dir / "predictions.jsonl"),
                        {identity[key] for key in ("bos_id", "eos_id", "pad_id", "mask_id")})
                    write(f"{key}/reference_loss.json", loss)
                    write(f"{key}/diagnostics.json", diagnostics)
                    groups[key] = {"generation": score, "reference_loss": loss, "diagnostics": diagnostics}
                    write("result.partial.json", {"status": "running", "groups": groups})
                after = model_fingerprint(model)
                if before != after or model.training:
                    raise RuntimeError("Read-only evaluation changed model weights or training mode")
                write(f"{name}-checkpoint.json", {"path": str(checkpoint), "sha256": checkpoint_sha,
                    "checkpoint_source": source, "model_sha256_before": before, "model_sha256_after": after,
                    "weights_unchanged": True})
                del model
                gc.collect()
                torch.cuda.empty_cache()
            progress("finalize")
            groups = check_groups(output, selection, audit, sources, reference)
            metadata["checkpoint_weights_verified"] = True
            write("experiment.json", metadata)
            result = {"experiment": "experiment5", "status": "complete", "groups": groups,
                      "selection_sha256": metadata["selection_sha256"], "limitations": LIMITATIONS}
    except BaseException as error:
        result = {"experiment": "experiment5", "status": "bounded_stop" if isinstance(error, TimeoutError) else "failed",
                  "stage": stage, "error_type": type(error).__name__, "error": str(error), "completed_groups": list(groups)}
        write("failure.json", result)
        raise
    finally:
        wall = time.monotonic() - started
        result["usage"] = {"wall_seconds": wall, "allocated_gpus": 0 if args.prepare_only else 2,
            "allocated_gpu_hours": 0 if args.prepare_only else 2 * wall / 3600,
            "instance_hourly_rmb": None if args.prepare_only else config["instance_hourly_rmb"],
            "estimated_cost_rmb": None if args.prepare_only else wall / 3600 * config["instance_hourly_rmb"],
            "cost_basis": "CPU-only preparation; actual no-GPU instance rate/cost is unknown" if args.prepare_only
                          else "Entire two-GPU instance wall time, including idle GPU"}
        write("result.json", result)
        write("status.json", {"status": result["status"], "stage": stage, "completed_groups": list(groups)})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/experiment5/config.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(run(parser.parse_args()))
