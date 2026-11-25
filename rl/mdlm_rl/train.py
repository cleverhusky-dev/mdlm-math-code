"""Full-parameter diffu-GRPO with a fixed SFT reference and rollout transactions.

Rollout generation uses the agreed temperature; d1's one-pass surrogate uses
untempered logits. It approximates, rather than evaluates, the probability of
the complete ancestral trajectory. No gradient is taken through generation or
the code executor. The actor stays in eval mode to disable dropout, with
autograd enabled only for policy updates.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoints import atomic_torch_save, load_sft
from .config import apply_run_overrides, load_config
from .data import encode_prompt, load_questions, load_tokenizer, tokenizer_identity
from .diffusion import build_model
from .distributed import DistributedContext
from .evaluation import generate_and_score
from .objective import PolicyScorer, group_advantages, make_prompt_mask, policy_loss
from .runtime import (
    ROOT, atomic_json, autocast, capture_rng, code_manifest, environment_record,
    model_fingerprint, restore_rng, seed_all, sha256, validate_environment,
)
from .sampling import ancestral_sample
from .scoring import ExecutionConfig, SCORER_VERSION, preflight_executor, score_program
from .state import SCHEMA, QuestionSampler, check_resume, load_rl, make_best_payload
from .training_logs import append_durable_jsonl, reconcile_training_log, recorded_wall_seconds


class TrainingStopped(TimeoutError):
    """A requested stop at a complete rollout boundary, safe to checkpoint."""


def inspect_inputs(config):
    """Read-only CPU preflight; never open the official test set."""
    manifest_path = Path(config["data_dir"]) / "manifest.json"
    if sha256(manifest_path) != config["expected_data_manifest_sha256"]:
        raise ValueError("RL data manifest differs from the selected question dataset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest_sha256 = (manifest["parent_manifest_sha256"]
                             if manifest.get("kind") == "rl_questions"
                             else sha256(manifest_path))
    train = load_questions(config["data_dir"], "train")
    val = load_questions(config["data_dir"], "val")
    for rows, key in ((train, "expected_train_questions"), (val, "expected_val_questions")):
        if len(rows) != config[key]:
            raise ValueError(f"{key}: got {len(rows)}, expected {config[key]}")
        for row in rows:
            gold = row.get("gold_answer")
            if (not row["gsm8k_id"].startswith("train:")
                    or isinstance(gold, bool) or not isinstance(gold, (int, float))
                    or not math.isfinite(gold)):
                raise ValueError("RL accepts official training IDs with finite numeric gold only")
    if {x["gsm8k_id"] for x in train} & {x["gsm8k_id"] for x in val}:
        raise ValueError("Training and validation questions overlap")
    tokenizer = load_tokenizer(config["tokenizer_path"], local_files_only=True)
    for row in train + val:
        encode_prompt(row["question"], tokenizer, config["rollout"]["length"])
    checkpoint = Path(config["initial_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SFT initialization file not found: {checkpoint}. Run SFT experiment 12 "
            "and select its trusted best.pt with scripts/configure_run.py."
        )
    digest = sha256(checkpoint)
    if digest != config["initial_checkpoint_sha256"]:
        raise ValueError("Initial SFT checkpoint SHA256 mismatch")
    state = load_sft(checkpoint)
    if (state["progress"]["global_step"] != config["initial_global_step"]
            or state["config"].get("smoke")
            or state["identity"]["data_manifest_sha256"] != source_manifest_sha256):
        raise ValueError("SFT initialization must match the configured step and dataset and come from a full run")
    return {
        "train_questions": len(train), "validation_questions": len(val),
        "data_manifest_sha256": sha256(manifest_path),
        "source_data_manifest_sha256": source_manifest_sha256,
        "initial_checkpoint_sha256": digest,
        "initial_global_step": state["progress"]["global_step"],
        "model": state["identity"]["model"],
        "tokenizer": tokenizer_identity(tokenizer),
        "scorer_version": SCORER_VERSION,
    }


def warmup_multiplier(step, warmup_steps):
    """Constant after warmup; extending max_steps does not restart the schedule."""
    return min(1.0, (step + 1) / max(1, warmup_steps))


def token_mask(prefix_lengths, length, device):
    return torch.arange(length, device=device)[None, :] >= torch.tensor(prefix_lengths, device=device)[:, None]


def cache_logps(scorer, ids, target, masks, config, device):
    """Cache each mask's old/reference probabilities before any actor update."""
    micro = config["micro_batch_size"]
    cached = []
    with torch.no_grad():
        for mask in masks:
            pieces = []
            for start in range(0, len(ids), micro):
                sl = slice(start, start + micro)
                with autocast(device, config["precision"]):
                    logps = scorer(ids[sl], target[sl], mask[sl],
                                   temperature=config["estimator_temperature"])
                pieces.append(logps.detach().cpu())
            cached.append(torch.cat(pieces))
    return cached


def update_policy(wrapped, model, ids, target, mask, old, reference, advantages,
                  optimizer, scheduler, ctx, config):
    """One global token-mean update over the entire n*G rollout.

    DDP averages rank gradients: multiply each local numerator by world size
    and divide by the GLOBAL number of target tokens, once across all micros.
    """
    denominator = target.sum().to(dtype=torch.float64)
    if ctx.world > 1:
        dist.all_reduce(denominator)
    if denominator <= 0:
        raise ValueError("Empty rollout target region")
    micro = config["micro_batch_size"]
    rounds = math.ceil(len(ids) / micro)
    if rounds != config["gradient_accumulation_steps"]:
        raise ValueError("Rollout partition does not match the configured accumulation")
    optimizer.zero_grad(set_to_none=True)
    names = ("loss_sum", "policy_sum", "kl_sum", "clip_count", "outside_clip_count", "ratio_sum")
    totals = torch.zeros(len(names), device=ctx.device, dtype=torch.float64)
    lr_used = optimizer.param_groups[0]["lr"]
    for index, start in enumerate(range(0, len(ids), micro)):
        sl = slice(start, start + micro)
        sync = wrapped.no_sync() if ctx.world > 1 and index < rounds - 1 else contextlib.nullcontext()
        with sync:
            with autocast(ctx.device, config["precision"]):
                logps = wrapped(ids[sl], target[sl], mask[sl],
                                temperature=config["estimator_temperature"])
            result = policy_loss(
                logps, old[sl].to(ctx.device), reference[sl].to(ctx.device),
                advantages[sl], target[sl], config["clip_epsilon"], config["kl_beta"])
            (result["loss_sum"] * (ctx.world / denominator)).backward()
        totals += torch.stack([result[name].detach().double() for name in names])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    if ctx.world > 1:
        dist.all_reduce(totals)
    values = (totals / denominator).cpu().tolist()
    return {
        "loss": values[0], "policy_loss": values[1], "kl": values[2],
        "clip_fraction": values[3], "outside_clip_fraction": values[4],
        "mean_ratio": values[5], "target_tokens": int(denominator),
        "lr_used": lr_used, "lr_next": optimizer.param_groups[0]["lr"],
        "grad_norm_before_clip": float(norm),
    }


class Trainer:
    """Dependencies are explicit so lifecycle and DDP can be tested on tiny fixtures."""

    def __init__(self, config, ctx, model, reference, tokenizer, train_questions,
                 validation_questions, identity, *, resume=None, smoke=False):
        self.config, self.ctx = config, ctx
        self.model, self.reference, self.tokenizer = model, reference, tokenizer
        self.train_questions, self.validation_questions = train_questions, validation_questions
        self.identity, self.smoke = identity, smoke
        self.output = Path(config["output_dir"])
        self.invocation = uuid.uuid4().hex
        self.started = time.monotonic()
        self.deadline = self.started + config["max_wall_seconds"]
        self.cancelled = False
        model.eval()
        reference.eval().requires_grad_(False)
        self.scorer = PolicyScorer(model).eval()
        self.reference_scorer = PolicyScorer(reference).eval()
        self.wrapped = (DDP(self.scorer, device_ids=[ctx.device.index] if ctx.device.type == "cuda" else None,
                            broadcast_buffers=False) if ctx.world > 1 else self.scorer)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]),
            eps=config["adam_eps"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: warmup_multiplier(step, config["warmup_steps"]))
        self.progress = {
            "global_step": 0, "rollout_batches": 0, "best_correct": None, "best_step": None,
            "validation_pending": True, "validated_step": -1,
        }
        self.sampler = QuestionSampler(len(train_questions), config["seed"])
        if resume is not None:
            check_resume(resume, config, identity)
            model.load_state_dict(resume["model"], strict=True)
            self.optimizer.load_state_dict(resume["optimizer"])
            self.scheduler.load_state_dict(resume["scheduler"])
            self.progress = copy.deepcopy(resume["progress"])
            self.sampler = QuestionSampler(len(train_questions), config["seed"], resume["sampler"])
            step = self.progress["global_step"]
            if (step != self.progress["rollout_batches"] * config["inner_updates"]
                    or self.scheduler.last_epoch != step
                    or self.sampler.questions_seen != self.progress["rollout_batches"] * config["questions_per_batch"]):
                raise ValueError("Checkpoint rollout, sampler and optimizer cursors disagree")
            if step > config["max_steps"]:
                raise ValueError("max_steps precedes the resumed checkpoint")
            ctx.rank_zero(lambda: reconcile_training_log(self.output, step, self.invocation))
            restore_rng(resume["rng_by_rank"][ctx.rank])
        self.previous_wall = ctx.rank_zero(lambda: recorded_wall_seconds(self.output))

    def stopped(self):
        return self.cancelled or time.monotonic() >= self.deadline

    def payload(self):
        return {
            "schema": SCHEMA, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "progress": copy.deepcopy(self.progress),
            "rng_by_rank": self.ctx.gather_rng(), "sampler": self.sampler.state_dict(),
            "config": self.config, "identity": self.identity,
        }

    def save(self, *, best=False):
        payload = self.payload()  # All ranks participate in RNG collection.
        def publish():
            directory = self.output / "checkpoints"
            if best:
                atomic_torch_save(make_best_payload(payload), directory / "best.pt")
                atomic_json(directory / "best.json", {
                    "step": self.progress["best_step"],
                    "correct": self.progress["best_correct"],
                    "num_questions": len(self.validation_questions),
                    "accuracy": self.progress["best_correct"] / len(self.validation_questions),
                    "tie_policy": "strictly greater; retain earlier checkpoint",
                    "includes_step_zero": True,
                })
            atomic_torch_save(payload, directory / "last.pt")
        self.ctx.rank_zero(publish)

    def validate(self):
        """A pending checkpoint is durable before generation starts."""
        self.progress["validation_pending"] = True
        self.save()
        step = self.progress["global_step"]
        def evaluate():
            directory = self.output / "validation" / f"step-{step:08d}"
            directory.mkdir(parents=True, exist_ok=True)
            # Failed attempts remain auditable; never combine partial predictions.
            existing = list(directory.glob("attempt-*"))
            attempt = directory / f"attempt-{len(existing) + 1:03d}"
            source = {
                "source": "sft_initial" if step == 0 else "rl_ordinary",
                "rl_global_step": step, "sft_global_step": self.config["initial_global_step"],
                "initial_checkpoint_sha256": self.identity["initial_checkpoint_sha256"],
                "model_sha256": model_fingerprint(self.model),
            }
            result = generate_and_score(
                self.model, self.tokenizer, self.validation_questions,
                self.config["evaluation"], self.ctx.device, self.config["precision"], attempt,
                checkpoint_source=source,
                label="exploratory_train_smoke" if self.smoke else "validation",
                deadline=self.deadline, cancel_requested=lambda: self.cancelled)
            atomic_json(directory / "summary.json", {**result, "attempt": attempt.name})
            return result
        try:
            result = self.ctx.rank_zero(evaluate)
        except TimeoutError as error:
            # Validation runs only after a complete optimizer transaction.
            raise TrainingStopped(str(error)) from error
        if (not result.get("complete") or result["question_count"] != len(self.validation_questions)):
            raise ValueError("Incomplete validation cannot select a checkpoint")
        count = result["counts"]["correct"]
        improved = self.progress["best_correct"] is None or count > self.progress["best_correct"]
        if improved:
            self.progress["best_correct"], self.progress["best_step"] = count, step
        self.progress["validation_pending"] = False
        self.progress["validated_step"] = step
        self.save(best=improved)
        return result

    def rollout(self, selected, rollout_number):
        """Whole question groups live on one rank; generation has no gold input."""
        config, ctx = self.config, self.ctx
        per_rank = config["questions_per_batch"] // ctx.world
        indices = selected[ctx.rank * per_rank:(ctx.rank + 1) * per_rank]
        questions = [self.train_questions[i] for i in indices]
        rows = [q for q in questions for _ in range(config["samples_per_question"])]
        prefixes = []
        for row in rows:
            encoded = encode_prompt(row["question"], self.tokenizer, config["rollout"]["length"])
            prefixes.append(encoded["input_ids"][:encoded["prefix_length"]])
        generation_micro = config["generation_batch_size"]
        tensors, records = [], []
        for start in range(0, len(rows), generation_micro):
            chunk = rows[start:start + generation_micro]
            chunk_prefixes = prefixes[start:start + generation_micro]
            with autocast(ctx.device, config["precision"]):
                generated = ancestral_sample(
                    self.model, chunk_prefixes, **config["rollout"],
                    deadline=self.deadline, cancel_requested=lambda: self.cancelled)
            tensors.append(generated)
            for offset, (row, prefix, tokens) in enumerate(zip(chunk, chunk_prefixes, generated.cpu().tolist())):
                if self.stopped():
                    raise TimeoutError("Rollout interrupted before executing a program")
                target_ids = tokens[len(prefix):]
                text = self.tokenizer.decode(target_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
                result = score_program(text, row["gold_answer"],
                                       timeout_s=config["evaluation"]["timeout_s"],
                                       memory_mb=config["evaluation"]["memory_mb"])
                records.append({
                    "rollout": rollout_number, "rank": ctx.rank,
                    "gsm8k_id": row["gsm8k_id"], "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "sample_index": (start + offset) % config["samples_per_question"],
                    "generated_text": text, "generated_token_ids": target_ids,
                    "reward": int(result["status"] == "correct"), **result,
                })
        ids = torch.cat(tensors)
        target = token_mask(list(map(len, prefixes)), ids.shape[1], ctx.device)
        rewards = torch.tensor([r["reward"] for r in records], dtype=torch.float32, device=ctx.device)
        advantages = group_advantages(rewards, config["samples_per_question"])
        return ids, target, advantages, records

    def run_rollout(self):
        config, ctx = self.config, self.ctx
        sampler_before, rng_before = self.sampler.state_dict(), capture_rng()
        selected = self.sampler.next_batch(config["questions_per_batch"])
        number = self.progress["rollout_batches"] + 1
        started = time.monotonic()
        interrupted = False
        try:
            ids, target, advantages, records = self.rollout(selected, number)
        except TimeoutError:
            interrupted = True
        # A failed rank never starts optimization on a partially generated batch.
        if ctx.any_true(interrupted or self.stopped()):
            self.sampler = QuestionSampler(len(self.train_questions), config["seed"], sampler_before)
            restore_rng(rng_before)
            raise TrainingStopped("Rollout stopped; sampler and RNG restored to its beginning")
        gathered = ctx.gather_objects(records)
        all_records = [record for rank_records in gathered for record in rank_records]
        group_size = config["samples_per_question"]
        groups = [all_records[i:i + group_size] for i in range(0, len(all_records), group_size)]
        if any(len({r["gsm8k_id"] for r in group}) != 1 for group in groups):
            raise ValueError("A reward group contains different questions")
        stats = {
            "mean_reward": sum(r["reward"] for r in all_records) / len(all_records),
            "mixed_reward_group_fraction": sum(0 < sum(r["reward"] for r in g) < group_size for g in groups) / len(groups),
            "all_correct_group_fraction": sum(all(r["reward"] == 1 for r in g) for g in groups) / len(groups),
            "all_wrong_group_fraction": sum(all(r["reward"] == 0 for r in g) for g in groups) / len(groups),
            "distinct_text_fraction": sum(len({r["generated_text"] for r in g}) for g in groups) / len(all_records),
            "executable_fraction": sum(r["status"] in ("correct", "wrong_answer") for r in all_records) / len(all_records),
            "generation_wall_seconds": time.monotonic() - started,
        }
        self.ctx.rank_zero(lambda: self.write_rollout(number, all_records, selected, stats))
        # No live third model is required: the old policy is current BEFORE
        # these caches, and is represented by immutable cached probabilities.
        masks = [make_prompt_mask(target, config["prompt_mask_probability"],
                                  config["seed"] + number * 100000 + inner * ctx.world + ctx.rank)
                 for inner in range(config["inner_updates"])]
        old = cache_logps(self.scorer, ids, target, masks, config, ctx.device)
        ref = cache_logps(self.reference_scorer, ids, target, masks, config, ctx.device)
        for inner in range(config["inner_updates"]):
            metrics = update_policy(
                self.wrapped, self.model, ids, target, masks[inner], old[inner], ref[inner],
                advantages, self.optimizer, self.scheduler, ctx, config)
            self.progress["global_step"] += 1
            record = {
                "step": self.progress["global_step"], "rollout": number, "inner_update": inner + 1,
                "questions_seen": self.sampler.questions_seen,
                "generated_completions": self.sampler.questions_seen * group_size,
                "wall_seconds": self.previous_wall + time.monotonic() - self.started,
                **stats, **metrics,
            }
            ctx.rank_zero(lambda: append_durable_jsonl(self.output / "train.jsonl", record))
            if ctx.rank == 0:
                print(json.dumps(record, allow_nan=False), flush=True)
        self.progress["rollout_batches"] = number
        # Cancellation is observed after all μ updates, at a recoverable boundary.

    def write_rollout(self, number, records, selected, stats):
        directory = self.output / "rollouts"
        directory.mkdir(exist_ok=True)
        # Invocation id preserves outputs from any updates rolled back on resume.
        path = directory / f"batch-{number:06d}-{self.invocation}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(path.with_suffix(".json"), {
            "rollout": number, "invocation_id": self.invocation,
            "question_indices": selected, "predictions_sha256": sha256(path), **stats,
        })

    def run(self):
        status, failure = "failed", None
        old_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: setattr(self, "cancelled", True))
        try:
            if self.progress["validation_pending"]:
                self.validate()
            while self.progress["global_step"] < self.config["max_steps"]:
                if self.ctx.any_true(self.stopped()):
                    raise TrainingStopped("Training wall-clock boundary or cancellation")
                self.run_rollout()
                step = self.progress["global_step"]
                due = step % self.config["validation_every_steps"] == 0 or step == self.config["max_steps"]
                if due:
                    self.validate()
                elif step % self.config["checkpoint_every_steps"] == 0:
                    self.save()
            status = "complete"
        except TrainingStopped:
            # Storage/collective timeouts during an optimizer transaction must
            # fail instead of publishing an inconsistent resumable checkpoint.
            self.save()
            status = "stopped"
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            if self.ctx.rank == 0:
                seconds = self.previous_wall + time.monotonic() - self.started
                record = {
                    "status": status, "failure": failure, "smoke": self.smoke,
                    "progress": self.progress, "invocation_id": self.invocation,
                    "wall_seconds": seconds,
                    "allocated_gpu_hours": seconds * (self.ctx.world if self.ctx.device.type == "cuda" else 0) / 3600,
                    "best_checkpoint": "checkpoints/best.pt",
                    "latest_checkpoint": "checkpoints/last.pt",
                }
                atomic_json(self.output / "status.json", record)
                append_durable_jsonl(self.output / "usage_history.jsonl", record, atomic=True)
        return status


def prepare_output(output, resume):
    output = Path(output).resolve()
    if resume:
        if Path(resume).resolve() != output / "checkpoints" / "last.pt":
            raise ValueError("Resume must use this run's checkpoints/last.pt")
        if not (output / "config.resolved.json").is_file():
            raise ValueError("Resume requires the original run directory and records")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh RL training requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args(argv)
    validate_environment()
    config = apply_run_overrides(
        load_config(args.config), output_dir=args.output_dir, max_steps=args.max_steps,
        max_wall_seconds=args.max_wall_seconds, smoke=args.smoke, device=args.device)
    ctx = DistributedContext(args.device)
    try:
        if ctx.world != config["world_size"]:
            raise ValueError("Runtime world size differs from the configured two-GPU experiment")
        ctx.rank_zero(lambda: prepare_output(config["output_dir"], args.resume))
        audit = ctx.rank_zero(lambda: inspect_inputs(config))
        preflight = ctx.rank_zero(lambda: preflight_executor(ExecutionConfig(
            config["evaluation"]["timeout_s"], config["evaluation"]["memory_mb"])))
        seed_all(config["seed"] + ctx.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        tokenizer = load_tokenizer(config["tokenizer_path"], local_files_only=True)
        initial = load_sft(config["initial_checkpoint"])
        model = build_model(tokenizer, audit["model"])
        model.load_state_dict(initial["model"], strict=True)
        del initial
        reference = copy.deepcopy(model).to(ctx.device)
        model.to(ctx.device)
        train_questions = load_questions(config["data_dir"], "train")
        questions = train_questions[:8] if args.smoke else load_questions(config["data_dir"], "val")
        identity = {
            **audit, "world_size": ctx.world, "source_sha256": code_manifest(),
            "smoke": args.smoke, "device_type": args.device,
        }
        state = load_rl(args.resume) if args.resume else None
        trainer = Trainer(config, ctx, model, reference, tokenizer, train_questions, questions,
                          identity, resume=state, smoke=args.smoke)
        def record():
            output = Path(config["output_dir"])
            atomic_json(output / "config.resolved.json", config)
            atomic_json(output / "identity.json", identity)
            atomic_json(output / "executor_preflight.json", preflight)
            atomic_json(output / f"environment-{trainer.invocation}.json", environment_record())
        ctx.rank_zero(record)
        status = trainer.run()
        return 0 if status == "complete" else 2
    finally:
        ctx.close()


if __name__ == "__main__":
    raise SystemExit(main())
