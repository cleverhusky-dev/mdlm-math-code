"""Shared generation and scoring for pretrained and SFT checkpoints."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import encode_prompt, tokenizer_identity
from .diffusion import validate_temperature
from .runtime import atomic_json, autocast, isolated_rng, sha256, ROOT


def protocol_record(questions, tokenizer, settings, precision):
    from .scoring import SCORER_VERSION
    core = {"generation": settings, "precision": precision,
            "numerics": {"torch": torch.__version__, "noise_eps": 0.0,
                         "time_conditioning": False, "subs_precision": "float32",
                         "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                         "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32},
            "scorer_version": SCORER_VERSION, "tokenizer": tokenizer_identity(tokenizer),
            "question_order": [q["gsm8k_id"] for q in questions],
            "questions_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "evaluation_source": {name: sha256(ROOT / "mdlm_sft" / name) for name in
                                  ("evaluation.py", "sampling.py", "scoring.py", "executor.py", "data.py", "diffusion.py", "transformer.py", "runtime.py")}}
    core["protocol_sha256"] = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    return core


@torch.no_grad()
def generate_and_score(model, tokenizer, questions, settings, device, precision, output_dir,
                       *, checkpoint_source, label, deadline=None, cancel_requested=None):
    from .sampling import ancestral_sample
    from .scoring import score_program, summarize_results, preflight_executor, ExecutionConfig
    if not questions or len({q["gsm8k_id"] for q in questions}) != len(questions):
        raise ValueError("Evaluation needs unique, nonempty questions")
    temperature = validate_temperature(settings["temperature"])
    if settings["length"] != 512 or not settings["use_float64"]:
        raise ValueError("Evaluation uses length=512 and float64 ancestral probabilities")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_record(questions, tokenizer, settings, precision)
    atomic_json(output_dir / "protocol.json", protocol)
    # Incomplete/failed evaluations never leave a usable summary.
    (output_dir / "summary.json").unlink(missing_ok=True)
    start = time.monotonic()
    atomic_json(output_dir / "executor_preflight.json", preflight_executor(
        ExecutionConfig(settings["timeout_s"], settings["memory_mb"])))
    was_training = model.training
    records = []
    model.eval()
    try:
        with isolated_rng(settings["seed"]), (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as out:
            for offset in range(0, len(questions), settings["batch_size"]):
                if ((deadline is not None and time.monotonic() >= deadline)
                        or (cancel_requested is not None and cancel_requested())):
                    raise TimeoutError("Evaluation wall-clock bound reached; no aggregate score committed")
                chunk = questions[offset:offset + settings["batch_size"]]
                # Do not pass gold/reference code to the generation API.
                prompts = [encode_prompt(q["question"], tokenizer, settings["length"]) for q in chunk]
                prefixes = [p["input_ids"][:p["prefix_length"]] for p in prompts]
                with autocast(device, precision):
                    generated = ancestral_sample(model, prefixes, steps=settings["steps"],
                                                 temperature=temperature,
                                                 length=settings["length"], use_float64=settings["use_float64"],
                                                 deadline=deadline, cancel_requested=cancel_requested)
                for row, prefix, tokens in zip(chunk, prefixes, generated.cpu().tolist()):
                    if ((deadline is not None and time.monotonic() >= deadline)
                            or (cancel_requested is not None and cancel_requested())):
                        raise TimeoutError("Evaluation bound reached before program execution")
                    if tokens[:len(prefix)] != prefix:
                        raise RuntimeError("Sampler changed fixed prefix")
                    target = tokens[len(prefix):]
                    # Upstream decodes every completion slot and skips special
                    # tokens. EOS does not stop fixed-length diffusion sampling.
                    text = tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    result = score_program(text, row["gold_answer"], timeout_s=settings["timeout_s"], memory_mb=settings["memory_mb"])
                    record = {"gsm8k_id": row["gsm8k_id"], "question": row["question"],
                              "generated_text": text, "generated_token_ids": tokens[len(prefix):], **result}
                    out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    out.flush()
                    records.append(record)
                print(json.dumps({"evaluation": label, "completed": len(records), "total": len(questions)}), flush=True)
        if ((deadline is not None and time.monotonic() >= deadline)
                or (cancel_requested is not None and cancel_requested())):
            raise TimeoutError("Evaluation bound reached; aggregate score not committed")
        metrics = summarize_results(records)
        summary = {**metrics, "complete": True, "label": label, "question_count": len(questions),
                   "checkpoint_source": checkpoint_source, "protocol_sha256": protocol["protocol_sha256"],
                   "wall_seconds": time.monotonic() - start,
                   "predictions_sha256": sha256(output_dir / "predictions.jsonl")}
        atomic_json(output_dir / "summary.json", summary)
        return summary
    finally:
        model.train(was_training)


@torch.no_grad()
def validation_loss(model, dataset, batch_size, seed, device, precision, deadline=None, cancel_requested=None):
    numerator, denominator, masked = 0.0, 0, 0
    was_training = model.training
    model.eval()
    try:
        with isolated_rng(seed):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
            for batch in loader:
                if ((deadline is not None and time.monotonic() >= deadline)
                        or (cancel_requested is not None and cancel_requested())):
                    raise TimeoutError("Validation loss wall-clock bound reached")
                with autocast(device, precision):
                    result = model(batch["input_ids"].to(device), batch["target_mask"].to(device))
                value = float(result["loss_sum"].double())
                if not torch.isfinite(result["loss_sum"]):
                    raise FloatingPointError("Non-finite validation loss")
                numerator += value
                denominator += int(result["target_tokens"])
                masked += int(result["masked_tokens"])
        if denominator <= 0:
            raise ValueError("Empty validation target mask")
        return {"loss": numerator / denominator, "loss_sum": numerator,
                "target_tokens": denominator, "masked_tokens": masked,
                "code_samples": len(dataset), "seed": seed}
    finally:
        model.train(was_training)


