"""Bounded full-parameter SFT with explicit fresh initialization and resume.

Single process or single-node two-process torchrun. Validation always
uses rank zero's ordinary module, never the DDP wrapper. Control messages use
Gloo, so long single-GPU generation does not run a pending NCCL collective.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
import uuid

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset

from .batching import rank_slice, step_counts, update_loader, lr_multiplier
from .baseline import check_baseline
from .checkpoints import SCHEMA, CheckpointManager, load_sft, check_resume
from .data import PreparedDataset, load_questions, load_tokenizer, tokenizer_identity
from .dataset_checks import labeled_training_questions, validate_training_data
from .evaluation import generate_and_score, validation_loss
from .diffusion import build_model
from .runtime import (ROOT, atomic_json, autocast, capture_rng, restore_rng,
                      seed_all, sha256, environment_record, code_manifest, cost_record)
from .runtime import model_fingerprint
from .runtime import validate_environment
from .pretrained import initialize_pretrained
from .training_logs import append_durable_jsonl, reconcile_training_log, recorded_wall_seconds


def load_config(path, overrides=()):
    defaults = json.loads((ROOT / "experiments/experiment1/config.json").read_text(encoding="utf-8"))
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    defaults.update(supplied)
    for override in overrides:
        key, raw = override.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parent = defaults
        fields = key.split(".")
        for field in fields[:-1]:
            parent = parent[field]
        if fields[-1] not in parent:
            raise ValueError(f"Unknown override: {key}")
        parent[fields[-1]] = value
    return defaults


def pretrained_source(audit):
    return {k: audit[k] for k in ("source", "checkpoint_sha256", "upstream_commit")}


class DistributedContext:
    def __init__(self, device_name="cuda"):
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.world not in (1, 2):
            raise ValueError("SFT supports one process or single-node two-process DDP")
        self.device = torch.device(device_name, local_rank) if device_name == "cuda" else torch.device("cpu")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; CPU is only for explicitly requested local fixtures")
            torch.cuda.set_device(self.device)
        self.control = None
        if self.world > 1:
            # A long explicit timeout bounds even a rank that is killed during
            # evaluation. Normal Python failures are broadcast immediately.
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=timedelta(hours=24))
            self.control = dist.new_group(backend="gloo", timeout=timedelta(hours=24))

    def rank_zero(self, fn):
        result, error = None, None
        if self.rank == 0:
            try:
                result = fn()
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc),
                         "traceback": traceback.format_exc()}
        if self.world > 1:
            message = [(result, error)]
            dist.broadcast_object_list(message, src=0, group=self.control)
            result, error = message[0]
        if error:
            if error["type"] == "TimeoutError":
                raise TimeoutError(error["message"])
            raise RuntimeError(f"Rank-zero stage failed: {error['type']}: {error['message']}\n{error['traceback']}")
        return result

    def gather_rng(self):
        state = capture_rng()
        if self.world == 1:
            return [state]
        result = [None] * self.world
        dist.all_gather_object(result, state, group=self.control)
        return result

    def any_true(self, value):
        flag = torch.tensor(int(value), device=self.device)
        if self.world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def close(self):
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def train_update(wrapped, model, batch, optimizer, scheduler, context, config):
    """One complete optimizer update. No hidden Lightning loss scaling.

    Full groups: each local micro loss = sum / local target tokens, then
    / accumulation; DDP averages ranks once. This matches the discussed
    upstream estimator (not a global token-weighted optimizer objective).
    Tail groups: weight local micro means by actual example count / group
    count, compensated for DDP averaging. No dropped or duplicated data.
    """
    count = batch["input_ids"].shape[0]
    begin, end = rank_slice(count, context.rank, context.world)
    micro = config["micro_batch_size"]
    rounds = math.ceil(math.ceil(count / context.world) / micro)
    optimizer.zero_grad(set_to_none=True)
    local_sum, local_tokens, objective = 0.0, 0, 0.0
    started = time.monotonic()
    lr_used = optimizer.param_groups[0]["lr"]
    for index in range(rounds):
        lo, hi = begin + index * micro, min(end, begin + (index + 1) * micro)
        valid = max(0, hi - lo)
        if valid:
            ids = batch["input_ids"][lo:hi].to(context.device)
            target = batch["target_mask"][lo:hi].to(context.device)
            kwargs = {"global_batch_size": count, "offset": lo}
        else:
            # A zero-weight forward keeps DDP collective counts equal for a
            # one-example tail. It neither duplicates a training observation
            # nor contributes any gradient/metric.
            ids = batch["input_ids"][:1].to(context.device)
            target = torch.zeros_like(ids, dtype=torch.bool)
            kwargs = {"t": torch.ones(1, device=context.device)}
        sync = wrapped.no_sync() if context.world > 1 and index < rounds - 1 else contextlib.nullcontext()
        with sync:
            with autocast(context.device, config["precision"]):
                result = wrapped(ids, target, **kwargs)
                scaled = result["loss"] * (valid * context.world / count)
            if context.any_true(not bool(torch.isfinite(scaled))):
                raise FloatingPointError("Non-finite training loss on a rank; no optimizer step performed")
            scaled.backward()
        local_sum += float(result["loss_sum"].detach())
        local_tokens += int(result["target_tokens"])
        objective += float(scaled.detach()) / context.world
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    totals = torch.tensor([local_sum, local_tokens, objective], dtype=torch.float64, device=context.device)
    if context.world > 1:
        dist.all_reduce(totals)
    return {"token_mean_loss": float(totals[0] / totals[1]), "optimization_loss": float(totals[2]),
            "target_tokens": int(totals[1]), "examples": count, "lr_used": lr_used,
            "lr_next": optimizer.param_groups[0]["lr"], "grad_norm_before_clip": float(norm),
            "update_wall_seconds": time.monotonic() - started}


def run(args):
    started = time.monotonic()
    if args.baseline_dir and (args.smoke or args.resume):
        raise ValueError("--baseline-dir is only for fresh full SFT; resume uses the saved baseline reference")
    validate_environment()
    config = load_config(args.config, args.set)
    if config["noise_eps"] != 0:
        raise ValueError("The approved checkpoint schedule uses noise_eps=0; changing it requires a new protocol")
    for key in ("max_steps", "max_wall_seconds", "output_dir", "data_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.tokenizer:
        config["tokenizer_path"] = args.tokenizer
    if args.init:
        config["pretrained_checkpoint"] = args.init
    config["smoke"] = bool(args.smoke)
    config["device_type"] = args.device
    ctx = DistributedContext(args.device)
    try:
        run_training(args, config, ctx, started)
    except BaseException:
        # Do not enter any new collective after a possibly rank-local OOM or
        # forward failure. Let torchrun observe failure and stop other ranks.
        raise
    else:
        ctx.close()


def run_training(args, config, ctx, started):
    world, rank, device = ctx.world, ctx.rank, ctx.device
    reference = ctx.rank_zero(lambda: check_baseline(ROOT, args.baseline_dir, config)) if args.baseline_dir else None
    micro, global_batch = config["micro_batch_size"], config["global_batch_size"]
    if not all(isinstance(x, int) and x > 0 for x in (micro, global_batch, config["epochs"])):
        raise ValueError("Batch sizes and epochs must be positive integers")
    if global_batch % (micro * world):
        raise ValueError("global_batch_size must be divisible by micro_batch_size * world_size")
    accumulation = global_batch // (micro * world)
    if config["gradient_accumulation_steps"] not in (None, accumulation):
        raise ValueError("Configured accumulation does not preserve effective global batch")
    config["gradient_accumulation_steps"] = accumulation
    if not 0 <= config["warmup_fraction"] < 1:
        raise ValueError("Invalid warmup fraction")
    if config["max_steps"] is not None and config["max_steps"] < 1:
        raise ValueError("max_steps is an absolute optimizer-step bound and must be positive")
    if config["max_wall_seconds"] is None or config["max_wall_seconds"] <= 0:
        raise ValueError("A positive per-invocation wall-clock bound is mandatory")
    deadline = started + config["max_wall_seconds"]
    output = Path(config["output_dir"]).resolve()
    def prepare_output():
        if args.resume:
            if Path(args.resume).name != "last.pt":
                raise ValueError("Resume through checkpoints/last.pt, the transaction commit pointer")
            if Path(args.resume).resolve().parent.parent != output:
                raise ValueError("Resume in the original output directory to preserve best and evaluation records")
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError("Fresh initialization requires an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
    ctx.rank_zero(prepare_output)
    seed_all(config["seed"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    tokenizer = load_tokenizer(config["tokenizer_path"], local_files_only=True)
    train_data = PreparedDataset(config["data_dir"], "train")
    val_data = PreparedDataset(config["data_dir"], "val")
    questions = load_questions(config["data_dir"], "val")
    validate_training_data(config["data_dir"], train_data, val_data, questions)
    if args.smoke:
        # This explicit mode has no held-out/model-selection claims.
        val_data = Subset(train_data, list(range(min(16, len(train_data)))))
        train_data = Subset(train_data, list(range(min(128, len(train_data)))))
        questions = labeled_training_questions(config["data_dir"], limit=8)
    steps_epoch, total_steps, halfway = step_counts(train_data, global_batch, config["epochs"])
    warmup_steps = math.ceil(total_steps * config["warmup_fraction"])
    model = build_model(tokenizer, {"noise_eps": config["noise_eps"]})
    state = load_sft(args.resume) if args.resume else None
    identity = {"world_size": world, "data_manifest_sha256": sha256(Path(config["data_dir"]) / "manifest.json"),
                "tokenizer": tokenizer_identity(tokenizer), "model": asdict(model.config),
                "source_sha256": code_manifest(), "steps_per_epoch": steps_epoch,
                "total_steps": total_steps, "warmup_steps": warmup_steps,
                "accumulation": accumulation}
    if state:
        check_resume(state, config, identity)
        model.load_state_dict(state["model"], strict=True)
        audit = state["pretrained_audit"]
        progress = state["progress"]
        if (not 0 <= progress["epoch"] <= config["epochs"]
                or not 0 <= progress["update_in_epoch"] < steps_epoch
                or (progress["epoch"] == config["epochs"] and progress["update_in_epoch"] != 0)
                or progress["global_step"] != progress["epoch"] * steps_epoch + progress["update_in_epoch"]):
            raise ValueError("Inconsistent epoch/update/global-step data cursor")
        def check_latest():
            latest = load_sft(output / "checkpoints/last.pt")
            if latest["progress"]["global_step"] != progress["global_step"]:
                raise ValueError("Resume the latest checkpoint; older cursors would mix future metrics/best state")
            if progress.get("best_step") is not None and not (output / "checkpoints/best.pt").exists():
                raise ValueError("The best checkpoint is missing from the run directory")
        ctx.rank_zero(check_latest)
    else:
        audit = initialize_pretrained(model, config["pretrained_checkpoint"])
        progress = {"epoch": 0, "update_in_epoch": 0, "global_step": 0,
                    "best_accuracy": None, "best_step": None, "baseline_completed": False,
                    "pending_validation": {"baseline": True, "loss": False, "generate": False},
                    "wall_seconds": 0.0}
        if reference is not None:
            if reference["checkpoint_sha256"] != audit["checkpoint_sha256"]:
                raise ValueError("Reused baseline does not match the loaded pretrained checkpoint")
            progress["baseline_reference"] = reference
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]),
                                 eps=config["adam_eps"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_multiplier(n, total_steps, warmup_steps))
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scheduler.last_epoch != progress["global_step"]:
            raise ValueError("Scheduler cursor disagrees with training cursor")
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False) if world > 1 else model
    manager = CheckpointManager(output / "checkpoints") if rank == 0 else None
    base_wall = progress["wall_seconds"]
    if args.resume:
        # Include abandoned updates, even after a crash between log recovery
        # and usage publication. Do not sum cumulative time observations.
        base_wall = max(base_wall, ctx.rank_zero(lambda: recorded_wall_seconds(output)))
    invocation_id = ctx.rank_zero(lambda: uuid.uuid4().hex)
    stop_requested = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.__setitem__(0, True))
    if state:
        restore_rng(state["rng_by_rank"][rank])
    # Constructing model/optimizer and DDP must precede restoring RNG.
    state = None
    def write_records():
        atomic_json(output / "config.resolved.json", config)
        atomic_json(output / "identity.json", identity)
        atomic_json(output / "pretrained_audit.json", audit)
        append_durable_jsonl(output / "invocations.jsonl", {**environment_record(), "resume": args.resume,
                      "invocation_id": invocation_id, "start_step": progress["global_step"],
                      "smoke": args.smoke, "config": config}, atomic=True)
        if args.resume:
            reconcile_training_log(output, progress["global_step"], invocation_id)
    print(json.dumps({"rank": rank, "steps_per_epoch": steps_epoch, "total_steps": total_steps,
                      "warmup_steps": warmup_steps, "accumulation": accumulation, "smoke": args.smoke}), flush=True)

    def save_snapshot():
        rngs = ctx.gather_rng()
        progress["wall_seconds"] = base_wall + time.monotonic() - started
        payload = {"schema": SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "progress": progress,
                   "rng_by_rank": rngs, "config": config, "identity": identity,
                   "pretrained_audit": audit}
        ctx.rank_zero(lambda: manager.save(payload).name)
        return payload

    def finish_pending(payload):
        nonlocal progress
        event = progress["pending_validation"] or {}
        step = progress["global_step"]
        prefix = "exploratory_train_smoke" if args.smoke else "validation"
        def validate():
            accuracy = None
            if event.get("baseline"):
                baseline = progress.get("baseline_reference")
                if baseline is not None:
                    atomic_json(output / "baseline_reference.json", baseline)
                    print(json.dumps({"baseline": "reused_experiment0", "accuracy": baseline["accuracy"],
                                      "directory": baseline["directory"]}), flush=True)
                else:
                    source = pretrained_source(audit)
                    baseline = generate_and_score(model, tokenizer, questions, config["evaluation"], device, config["precision"],
                        output / "baseline", checkpoint_source=source, label=prefix + "_pretrained", deadline=deadline,
                        cancel_requested=lambda: stop_requested[0])
                progress["baseline_completed"] = True
                progress["baseline_accuracy"] = baseline["accuracy"]
            if event.get("loss"):
                loss_metrics = validation_loss(model, val_data, config["validation_loss_batch_size"],
                    config["validation_loss_seed"], device, config["precision"], deadline,
                    cancel_requested=lambda: stop_requested[0])
                atomic_json(output / "validation" / f"step-{step:08d}-loss.json", {"step": step, "label": prefix, **loss_metrics})
            if event.get("generate"):
                metrics = generate_and_score(model, tokenizer, questions, config["evaluation"], device, config["precision"],
                    output / "validation" / f"step-{step:08d}",
                    checkpoint_source={"source": "sft_ordinary", "global_step": step,
                                       "model_sha256": model_fingerprint(model)}, label=prefix, deadline=deadline,
                    cancel_requested=lambda: stop_requested[0])
                accuracy = metrics["accuracy"]
            # Baseline is never a best SFT candidate. Smoke selection is labeled
            # exploratory in every artifact and cannot enter formal evaluation.
            progress["wall_seconds"] = base_wall + time.monotonic() - started
            manager.finish(payload, accuracy=accuracy)
            return dict(progress)
        progress = ctx.rank_zero(validate)

    status = "complete"
    try:
        ctx.rank_zero(write_records)
        if progress["pending_validation"]:
            finish_pending(save_snapshot())
        model.train()
        while progress["epoch"] < config["epochs"]:
            loader = update_loader(train_data, global_batch, config["seed"], progress["epoch"], progress["update_in_epoch"])
            for batch in loader:
                bounded = (stop_requested[0] or time.monotonic() >= deadline or
                           (config["max_steps"] is not None and progress["global_step"] >= config["max_steps"]))
                if ctx.any_true(bounded):
                    status = "bounded_stop"
                    payload = save_snapshot()
                    ctx.rank_zero(lambda: manager.finish(payload))
                    break
                metrics = train_update(wrapped, model, batch, optimizer, scheduler, ctx, config)
                progress["global_step"] += 1
                progress["update_in_epoch"] += 1
                end_epoch = progress["update_in_epoch"] == steps_epoch
                validation_due = progress["update_in_epoch"] == halfway or end_epoch
                epoch_fraction = progress["epoch"] + progress["update_in_epoch"] / steps_epoch
                if end_epoch:
                    progress["epoch"] += 1
                    progress["update_in_epoch"] = 0
                ctx.rank_zero(lambda: append_durable_jsonl(output / "train.jsonl", {
                    "invocation_id": invocation_id,
                    "step": progress["global_step"], "epoch": epoch_fraction, "label": "exploratory" if args.smoke else "sft",
                    **metrics, **cost_record(base_wall + time.monotonic() - started, world if device.type == "cuda" else 0, config["hourly_cost_rmb_per_gpu"]),
                    "peak_memory_allocated_bytes_rank0": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}))
                if validation_due:
                    progress["pending_validation"] = {"baseline": False, "loss": True, "generate": end_epoch}
                    finish_pending(save_snapshot())
                elif config["checkpoint_every_steps"] and progress["global_step"] % config["checkpoint_every_steps"] == 0:
                    payload = save_snapshot()
                    ctx.rank_zero(lambda: manager.finish(payload))
                if end_epoch:
                    break
            if status != "complete":
                break
    except TimeoutError:
        status = "bounded_stop_pending_validation"
        # The pre-validation checkpoint is recoverable; pending event will be
        # rerun from its fixed seed before any further optimizer update.
        save_snapshot()
    except Exception as exc:
        status = "failed"
        failure = {
            "invocation_id": invocation_id,
            "type": type(exc).__name__, "error": str(exc), "step": progress["global_step"],
            "checkpoint_policy": "resume last atomic state; partial gradients are never serialized"}
        atomic_json(output / "failures" / f"{invocation_id}-rank-{rank}.json", failure)
        atomic_json(output / f"failure-rank-{rank}.json", failure)
        raise
    finally:
        if rank == 0:
            usage = {
                "invocation_id": invocation_id, "status": status, "global_step": progress["global_step"],
                **cost_record(base_wall + time.monotonic() - started, world if device.type == "cuda" else 0, config["hourly_cost_rmb_per_gpu"])}
            atomic_json(output / "usage.json", usage)
            append_durable_jsonl(output / "usage_history.jsonl", {**usage, "invocation_wall_seconds": time.monotonic() - started}, atomic=True)
    if rank == 0:
        atomic_json(output / "result.json", {"invocation_id": invocation_id, "status": status, "smoke": args.smoke, **progress})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments/experiment1/config.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init", metavar="PRETRAINED_CKPT")
    mode.add_argument("--resume", metavar="LAST_SFT_PT")
    parser.add_argument("--baseline-dir", help="Reuse a verified experiment0 for fresh full SFT; no baseline generation")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_OR_TEXT")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--smoke", action="store_true", help="Use only training-side fixtures, label every result exploratory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
