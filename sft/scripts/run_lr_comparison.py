"""Experiment4: sequential fresh SFT at low and high peak learning rates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import run_experiment as launcher
from experiment_version import capture_version
from sampling_comparison import read_score
from mdlm_sft.baseline import check_baseline, read_json, resolve_path, sha256

ROOT = Path(__file__).resolve().parents[1]
RATES = {"low": 1e-5, "high": 1e-4}


def validate_configs(root):
    """Keep the full experiment1 setup, varying only LR and the output location."""
    root = Path(root).resolve()
    reference = read_json(root / "experiments/experiment1/config.json")
    if (reference["epochs"] != 3 or reference["learning_rate"] != 3e-5
            or reference["evaluation"]["steps"] != 32):
        raise ValueError("Experiment4 requires the original 3-epoch, 3e-5, 32-step experiment1")
    configs = {}
    for variant, rate in RATES.items():
        path = root / "experiments/experiment4" / f"config-{variant}.json"
        config = read_json(path)
        expected = {**reference, "learning_rate": rate, "output_dir": f"runs/experiment4/{variant}"}
        if config != expected:
            raise ValueError(f"{variant}: only learning_rate and output_dir may differ from experiment1")
        configs[variant] = config
    return configs


def summarize_group(output, config):
    """Read saved epoch-end scores and matching loss; do not launch evaluation."""
    output = Path(output)
    result = read_json(output / "result.json")
    training = read_json(output / "training/result.json")
    if (result.get("status") != "complete" or training.get("status") != "complete"
            or training.get("smoke") is not False or training.get("epoch") != 3
            or training.get("global_step") != 609 or training.get("pending_validation") is not None):
        raise ValueError("A group must finish all 3 epochs / 609 updates and validation")
    best = read_json(output / "training/checkpoints/best.json")
    if best.get("step") not in (203, 406, 609) or best["step"] != training.get("best_step"):
        raise ValueError("Best checkpoint must identify a completed epoch-end evaluation")
    summaries = {}
    for label, step in (("final", 609), ("best", best["step"])):
        summary, protocol = read_score(output / "training/validation" / f"step-{step:08d}")
        source = summary.get("checkpoint_source", {})
        if (protocol.get("generation") != config["evaluation"]
                or protocol.get("precision") != config["precision"]
                or source.get("source") != "sft_ordinary" or source.get("global_step") != step):
            raise ValueError("Saved score uses an unexpected checkpoint or evaluation setup")
        loss = read_json(output / "training/validation" / f"step-{step:08d}-loss.json")
        if (loss.get("step") != step or loss.get("label") != "validation"
                or loss.get("code_samples") != 1456 or loss.get("seed") != config["validation_loss_seed"]
                or not math.isfinite(loss["loss"])):
            raise ValueError("Validation loss does not match the selected checkpoint")
        summaries[label] = {"step": step, "accuracy": summary["accuracy"], "counts": summary["counts"],
                            "validation_loss": loss["loss"], "checkpoint_source": source,
                            "protocol_sha256": summary["protocol_sha256"],
                            "predictions_sha256": summary["predictions_sha256"]}
    if (summaries["best"]["accuracy"] != best.get("accuracy")
            or best["accuracy"] != training.get("best_accuracy")):
        raise ValueError("Best checkpoint accuracy does not match its saved predictions")
    return {"learning_rate": config["learning_rate"], **summaries}


def check_control(root, control, config):
    metadata = read_json(control / "experiment.json")
    recorded = dict(metadata.get("config", {}))
    expected = dict(config)
    recorded.pop("output_dir", None)
    expected.pop("output_dir", None)
    if (metadata.get("experiment") != "experiment1" or recorded != expected
            or metadata.get("inputs_lock_sha256") != sha256(root / "inputs.lock.json")):
        raise ValueError("The saved experiment1 control differs from its locked configuration or inputs")
    return {"directory": str(control), "experiment_sha256": sha256(control / "experiment.json"),
            "result_sha256": sha256(control / "result.json"), **summarize_group(control, config)}


def make_group_args(args, variant, output):
    return argparse.Namespace(experiment="experiment4", variant=variant,
        output_dir=str(output / variant), baseline_dir=args.baseline_dir,
        master_port=args.master_port, dry_run=args.dry_run)


def run(args, root=ROOT):
    root = Path(root).resolve()
    configs = validate_configs(root)
    output = resolve_path(root, args.output_dir)
    plans = {variant: launcher.make_plan(root, "experiment4", output / variant, config,
                master_port=args.master_port, baseline_dir=args.baseline_dir, variant=variant)
             for variant, config in configs.items()}
    if args.dry_run:
        print(json.dumps({"experiment": "experiment4", "output": str(output),
                          "group_order": list(RATES), "plans": plans}, indent=2))
        return 0
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory; existing results are preserved: {output}")
    version = capture_version(root)
    original = read_json(root / "experiments/experiment1/config.json")
    control = check_control(root, resolve_path(root, args.control_dir), original)
    baseline = check_baseline(root, args.baseline_dir, original)
    output.mkdir(parents=True, exist_ok=False)
    launcher.write_json(output / "experiment.json", {
        "experiment": "experiment4", "git": version, "group_order": list(RATES), "configs": configs,
        "inputs_lock_sha256": sha256(root / "inputs.lock.json"), "control": control,
        "baseline_reference": baseline, "plans": plans,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "Fresh pretrained EMA initialization; only peak learning rate changes",
        "primary_endpoint": "Fixed step 609 accuracy; also report epoch-end best"})
    started = time.monotonic()
    groups = {}
    variant = None
    try:
        for variant, config in configs.items():
            launcher.write_json(output / "status.json", {"status": "running", "stage": variant,
                "learning_rate": config["learning_rate"], "completed_groups": list(groups)})
            code = launcher.run(make_group_args(args, variant, output), root=root)
            if code:
                raise RuntimeError(f"{variant} stopped without completing; next group will not start")
            groups[variant] = summarize_group(output / variant, config)
            if groups[variant]["final"]["protocol_sha256"] != baseline["protocol_sha256"]:
                raise ValueError("A group changed the original 32-step evaluation protocol")
            launcher.write_json(output / "comparison.partial.json", {"control": control, "groups": groups})
    except BaseException as error:
        failure = {"experiment": "experiment4", "status": "failed", "stage": variant,
                   "error_type": type(error).__name__, "error": str(error), "completed_groups": list(groups),
                   "wall_seconds": time.monotonic() - started}
        launcher.write_json(output / "result.json", failure)
        launcher.write_json(output / "status.json", failure)
        raise
    result = {"experiment": "experiment4", "status": "complete", "control": control,
              "groups": groups, "wall_seconds": time.monotonic() - started}
    launcher.write_json(output / "result.json", result)
    launcher.write_json(output / "status.json", {"status": "complete", "stage": "finished"})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/experiment4")
    parser.add_argument("--baseline-dir", default="runs/experiment0")
    parser.add_argument("--control-dir", default="runs/experiment1")
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
