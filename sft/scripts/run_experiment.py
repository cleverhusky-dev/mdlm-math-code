"""Run a versioned baseline, fresh SFT, or fixed-checkpoint sampling comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mdlm_sft.baseline import check_baseline, experiment1_checkpoint, read_json, resolve_path, sha256


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_plan(root, experiment, output, config, python=sys.executable, master_port=29501,
              baseline_dir="runs/experiment0", sft_dir="runs/experiment1", variant=None):
    path = configuration_path(root, experiment, variant)
    checkpoint = str(resolve_path(root, config["pretrained_checkpoint"]))
    common = ["--config", str(path)]
    limit = config["max_wall_seconds"]
    if experiment == "experiment3":
        sft_checkpoint = experiment1_checkpoint(resolve_path(root, sft_dir) / "training/checkpoints")
        return [{"name": kind, "command": [python, "-m", "mdlm_sft.evaluate", *common,
            "--kind", kind, "--checkpoint", source, "--split", "val",
            "--output-dir", str(output / kind), "--max-wall-seconds", str(limit)],
            "timeout_seconds": limit + 120}
            for kind, source in (("pretrained", checkpoint), ("sft", str(sft_checkpoint)))]
    if experiment == "experiment0":
        return [
            {"name": "loss", "command": [python, str(root / "scripts/evaluate_loss.py"), *common,
                "--kind", "pretrained", "--checkpoint", checkpoint, "--output-dir", str(output / "loss"),
                "--max-wall-seconds", "1200"], "timeout_seconds": 1260},
            {"name": "validation", "command": [python, "-m", "mdlm_sft.evaluate", *common,
                "--kind", "pretrained", "--checkpoint", checkpoint, "--split", "val",
                "--output-dir", str(output / "validation"), "--max-wall-seconds", str(limit)],
                "timeout_seconds": limit + 120},
        ]
    return [{"name": "training", "command": [python, "-m", "torch.distributed.run", "--nnodes=1", "--node-rank=0",
        "--nproc-per-node=2", "--rdzv-backend=static", "--master-addr=127.0.0.1", f"--master-port={master_port}",
        "-m", "mdlm_sft.train", *common, "--init", checkpoint,
        "--baseline-dir", str(resolve_path(root, baseline_dir)), "--output-dir", str(output / "training"),
        "--max-wall-seconds", str(limit)], "timeout_seconds": limit + 180}]


def configuration_path(root, experiment, variant=None):
    if experiment == "experiment4":
        if variant not in ("low", "high"):
            raise ValueError("Experiment4 requires --variant low or high; use its run.sh for both")
        return root / "experiments" / experiment / f"config-{variant}.json"
    if variant is not None:
        raise ValueError("--variant is only valid for experiment4")
    return root / "experiments" / experiment / "config.json"


def preflight(root, experiment, env):
    if sys.platform != "linux":
        raise RuntimeError("Real experiments require Linux; use --dry-run to inspect commands elsewhere")
    required = 2 if experiment in ("experiment1", "experiment2", "experiment4", "experiment6", "experiment7", "experiment8", "experiment9", "experiment10", "experiment11", "experiment12") else 1
    probe = (
        "import torch; from mdlm_sft.runtime import validate_environment; "
        "from mdlm_sft.scoring import preflight_executor; validate_environment(); "
        "assert torch.version.cuda == '12.4', 'Use the CUDA 12.4 environment created by setup.sh'; "
        f"assert torch.cuda.device_count() >= {required}, 'Insufficient visible CUDA GPUs'; "
        "assert torch.cuda.is_bf16_supported(), 'BF16 is required'; print(preflight_executor())"
    )
    subprocess.run([sys.executable, "-c", probe], cwd=root, env=env, check=True)
    subprocess.run([sys.executable, str(root / "scripts/manage_inputs.py")], cwd=root, env=env, check=True)
    if experiment in ("experiment6", "experiment7", "experiment8", "experiment9", "experiment10", "experiment11", "experiment12"):
        from scripts.manage_inputs import asset_path, verify_file
        lock = read_json(root / "experiments" / experiment / "data.json")
        for entry in lock["files"]:
            verify_file(asset_path(root, entry["path"]), entry)


def run_command(phase, root, env, log):
    """Stream output and terminate the whole torchrun process group on timeout."""
    process = subprocess.Popen(phase["command"], cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    def copy_output():
        for line in iter(process.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace")
            log.write(text)
            log.flush()
            print(text, end="", flush=True)
    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=phase["timeout_seconds"])
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        reader.join(timeout=5)
        process.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, phase["command"])


def completed_result(experiment, output, config, root=ROOT, references=None):
    if experiment == "experiment3":
        from sampling_comparison import check_completion
        return check_completion(root, output, config, references)
    if experiment == "experiment0":
        loss, score = read_json(output / "loss/summary.json"), read_json(output / "validation/summary.json")
        if loss.get("complete") is not True or loss.get("code_samples") != 1456:
            raise ValueError("Incomplete baseline reference-loss evaluation")
        if score.get("complete") is not True or score.get("num_questions") != 639 or score.get("question_count") != 639:
            raise ValueError("Incomplete baseline generation evaluation")
        return {"status": "complete", "accuracy": score["accuracy"], "validation_loss": loss["loss"]}
    result = read_json(output / "training/result.json")
    if result.get("status") != "complete":
        return {"status": result.get("status", "failed"), "training_result": result}
    manifest = read_json(resolve_path(root, config["data_dir"]) / "manifest.json")
    samples = manifest["counts"]["train"]["samples"]
    batch_size = config["global_batch_size"]
    expected_steps = ((samples + batch_size - 1) // batch_size) * config["epochs"]
    if (result.get("smoke") is not False or result.get("epoch") != config["epochs"]
            or result.get("global_step") != expected_steps or result.get("pending_validation") is not None):
        raise ValueError(f"Training completion report must match {config['epochs']} epochs and {expected_steps} steps")
    return {"status": "complete", "training_result": result}


def run(args, root=ROOT):
    root = Path(root).resolve()
    variant = getattr(args, "variant", None)
    config_path = configuration_path(root, args.experiment, variant)
    config = read_json(config_path)
    if args.experiment == "experiment4":
        from run_lr_comparison import validate_configs
        validate_configs(root)
    output = resolve_path(root, args.output_dir or config["output_dir"])
    baseline = resolve_path(root, args.baseline_dir)
    sft_dir = resolve_path(root, getattr(args, "sft_dir", "runs/experiment1"))
    plan = make_plan(root, args.experiment, output, config, master_port=args.master_port,
                     baseline_dir=baseline, sft_dir=sft_dir, variant=variant)
    if args.dry_run:
        print(json.dumps({"experiment": args.experiment, "output": str(output), "phases": plan}, indent=2))
        return 0
    if output.exists():
        raise FileExistsError(f"Use a new output directory; existing results are preserved: {output}")
    from experiment_version import capture_version
    version = capture_version(root)
    sampling = args.experiment == "experiment3"
    if sampling:
        from sampling_comparison import check_references
        reference = check_references(root, baseline, sft_dir, config)
    else:
        reference = check_baseline(root, baseline, config) if args.experiment != "experiment0" else None
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONNOUSERSITE": "1", "GLOO_SOCKET_IFNAME": "lo"}
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    preflight(root, args.experiment, env)
    if sampling:
        subprocess.run([sys.executable, str(root / "scripts/sampling_comparison.py"),
                        "--checkpoint", reference["sft_checkpoint"],
                        "--expected-model-sha256", reference["sft"]["checkpoint_source"]["model_sha256"]],
                       cwd=root, env=env, check=True, timeout=120)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = {"experiment": args.experiment, "git": version, "config": config,
        "config_sha256": sha256(config_path), "inputs_lock_sha256": sha256(root / "inputs.lock.json"),
        "output_dir": str(output), "baseline_reference": reference, "phases": plan,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_policy": ("Reuse 32-step results; evaluate pretrained and experiment1 step 609 at 128 steps"
                            if sampling else "Reuse verified experiment0 results without pretrained generation"
                            if reference else "Evaluate the original pretrained EMA")}
    if variant is not None:
        metadata["variant"] = variant
    write_json(output / "experiment.json", metadata)
    try:
        with (output / "console.log").open("x", encoding="utf-8", newline="\n") as log:
            for phase in plan:
                write_json(output / "status.json", {"status": "running", "stage": phase["name"]})
                run_command(phase, root, env, log)
        result = completed_result(args.experiment, output, config, root, references=reference)
    except BaseException as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        write_json(output / "result.json", {"experiment": args.experiment, **result, "wall_seconds": time.monotonic()-started})
        write_json(output / "status.json", result)
        raise
    result.update(experiment=args.experiment, wall_seconds=time.monotonic()-started)
    write_json(output / "result.json", result)
    write_json(output / "status.json", {"status": result["status"], "stage": "finished"})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("experiment0", "experiment1", "experiment2", "experiment3", "experiment4", "experiment6", "experiment7", "experiment8", "experiment9", "experiment10", "experiment11", "experiment12"))
    parser.add_argument("--variant", choices=("low", "high"), help="Single experiment4 group; its run.sh runs both")
    parser.add_argument("--output-dir", help="Fresh result directory; relative paths use the project root")
    parser.add_argument("--baseline-dir", default="runs/experiment0", help="Completed experiment0 directory required by SFT experiments")
    parser.add_argument("--sft-dir", default="runs/experiment1", help="Completed experiment1 directory for the experiment3 comparison")
    parser.add_argument("--master-port", type=int, default=29501, help="Local two-GPU torchrun rendezvous port")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without checking GPUs, creating results or running jobs")
    args = parser.parse_args()
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
