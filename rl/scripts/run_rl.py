"""Validate inputs and launch one synchronized, bounded diffu-GRPO job."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from mdlm_rl.config import apply_run_overrides, load_config
from mdlm_rl.runtime import validate_environment
from mdlm_rl.scoring import ExecutionConfig, preflight_executor
from mdlm_rl.train import inspect_inputs


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/experiment1/config.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true",
                       help="Verify all input identities and counts without GPU use or output files")
    modes.add_argument("--preflight-only", action="store_true",
                       help="Verify inputs, Linux executor and two-GPU readiness without training")
    parser.add_argument("--resume", help="Resume this run's checkpoints/last.pt")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--smoke", action="store_true",
                       help="Two-rollout integration check (default output: a sibling directory with suffix -smoke)")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def readiness(config, device):
    """Exercise the code sandbox and inspect accelerators without training."""
    validate_environment()
    if platform.system() != "Linux":
        raise RuntimeError("RL execution requires the Linux isolated code executor")
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("PyTorch distributed Gloo support is required")
    gpus = []
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("Two visible CUDA GPUs are required for this RL job")
        if not dist.is_nccl_available():
            raise RuntimeError("This PyTorch build lacks CUDA NCCL support")
        for index in range(config["world_size"]):
            with torch.cuda.device(index):
                if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"CUDA device {index} does not support BF16")
                probe = torch.zeros(1, device=f"cuda:{index}")
                torch.cuda.synchronize(index)
                del probe
            info = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": info.name,
                         "total_memory_bytes": info.total_memory})
    executor = preflight_executor(ExecutionConfig(
        config["evaluation"]["timeout_s"], config["evaluation"]["memory_mb"]))
    return {"device": device, "world_size": config["world_size"],
            "gpus": gpus, "executor": executor}


@contextmanager
def output_lock(output_dir):
    """A sibling lock leaves the training directory empty for fresh-run checks.

    Never guess that an existing lock is stale: another host may own it. Its
    PID and host identify the owner for explicit recovery after a hard crash.
    """
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.lock"
    token = uuid.uuid4().hex
    receipt = {"pid": os.getpid(), "host": platform.node(), "token": token,
               "output_dir": str(output)}
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another job owns {lock}; inspect its PID/host before removing a stale lock") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        # Do not remove a replacement created after external intervention.
        try:
            if json.loads(lock.read_text(encoding="utf-8")).get("token") == token:
                lock.unlink()
        except FileNotFoundError:
            pass


def training_command(args, config):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    command = [sys.executable, "-B", "-m", "torch.distributed.run",
               "--standalone", "--nnodes=1", "--local-addr=127.0.0.1",
               f"--nproc-per-node={config['world_size']}",
               "--module", "mdlm_rl.train", "--config", str(config_path.resolve()),
               "--device", args.device]
    for name in ("resume", "max_steps", "max_wall_seconds", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            if name in ("resume", "output_dir"):
                value = str(Path(value).resolve())
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.smoke:
        command.append("--smoke")
    return command


def run_child(command):
    """Keep the lock owner alive until torchrun has reaped its worker group."""
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    # A separate POSIX session lets a signal sent only to the launcher reach
    # torchrun and its workers, rather than orphaning an active training job.
    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                             start_new_session=(os.name == "posix"))
    handlers = {}

    def forward(signum, _frame):
        if child.poll() is None:
            if os.name == "posix":
                os.killpg(child.pid, signum)
            else:
                child.send_signal(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, forward)
        return child.wait()
    finally:
        # Exceptions in the launcher must not release the lock while a job
        # continues writing the same output directory.
        if child.poll() is None:
            forward(signal.SIGTERM, None)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    args = argument_parser().parse_args(argv)
    config = apply_run_overrides(load_config(args.config), output_dir=args.output_dir,
                                max_steps=args.max_steps, max_wall_seconds=args.max_wall_seconds,
                                smoke=args.smoke, device=args.device)
    audit = inspect_inputs(config)
    if args.dry_run:
        print(json.dumps({"status": "inputs_verified", "config": config, "inputs": audit}, indent=2))
        return 0
    ready = readiness(config, args.device)
    if args.preflight_only:
        print(json.dumps({"status": "ready", "inputs": audit, "readiness": ready}, indent=2))
        return 0
    command = training_command(args, config)
    with output_lock(config["output_dir"]):
        print(json.dumps({"status": "launching", "command": command,
                          "output_dir": config["output_dir"]}), flush=True)
        return run_child(command)


if __name__ == "__main__":
    raise SystemExit(main())
