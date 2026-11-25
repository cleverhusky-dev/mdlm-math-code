"""Freeze and sequentially evaluate all six approved official GSM8K test cases."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_version import capture_version
from scripts.run_experiment import run_command, write_json
from mdlm_sft.baseline import read_json, resolve_path, sha256

SETTINGS = [("t1-s32", 1.0, 32), ("t01-s32", 0.1, 32), ("t01-s128", 0.1, 128)]
STATUSES = ("correct", "format_failure", "execution_failure", "wrong_answer")
EVALUATION_SOURCES = {"evaluation.py", "sampling.py", "scoring.py", "executor.py",
                      "data.py", "diffusion.py", "transformer.py", "runtime.py"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config, base):
    if config.get("schema_version") != 1 or config.get("benchmark") != "gsm8k":
        raise ValueError("Expected the frozen GSM8K benchmark configuration")
    if config["source_config"] != "experiments/experiment12/config.json" or config["test"]["questions"] != 1319:
        raise ValueError("Use experiment12 settings and all 1319 official test questions")
    if [(x["id"], x["temperature"], x["steps"]) for x in config["settings"]] != SETTINGS:
        raise ValueError("The approved three decoding settings must be frozen together")
    if [(m["id"], m["kind"]) for m in config["models"]] != [("pretrained", "pretrained"), ("sft", "sft")]:
        raise ValueError("Benchmark must compare pretrained EMA and experiment12 SFT")
    step = config["models"][1].get("global_step")
    if type(step) is not int or step < 1:
        raise ValueError("Experiment12 must specify the step selected on validation")
    for digest in [config["test"]["sha256"], *(m["sha256"] for m in config["models"]),
                   config["models"][1]["model_sha256"]]:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Every test/model identity requires a SHA256")
    if config["max_wall_seconds_per_case"] <= 0 or base["noise_eps"] != 0:
        raise ValueError("Positive run bounds and the original noise schedule are required")
    if (base["precision"] != "bf16" or base["evaluation"] != {
            "batch_size": 8, "seed": 1, "length": 512, "steps": 32, "temperature": 1.0,
            "use_float64": True, "timeout_s": 15, "memory_mb": 512}):
        raise ValueError("Experiment12 evaluation settings must retain the approved defaults")


def case_specs(root, output, config, base):
    cases = []
    for settings in config["settings"]:
        for model in config["models"]:
            identity = settings["id"] + "-" + model["id"]
            resolved = copy.deepcopy(base)
            resolved["evaluation"].update(temperature=settings["temperature"], steps=settings["steps"])
            resolved["output_dir"] = str(output / "cases" / identity)
            cases.append({"id": identity, "model": model, "setting": settings["id"],
                          "config": resolved, "config_path": str(output / "configs" / f"{identity}.json")})
    return cases


def command_for(root, output, config, case, attempt, python=sys.executable):
    return [python, "-m", "mdlm_sft.evaluate", "--config", case["config_path"],
            "--kind", case["model"]["kind"], "--checkpoint", str(resolve_path(root, case["model"]["checkpoint"])),
            "--split", "test", "--confirm-frozen-protocol", "--final-test-parquet",
            str(resolve_path(root, config["test"]["path"])), "--output-dir", str(attempt),
            "--max-wall-seconds", str(config["max_wall_seconds_per_case"])]


def preflight(root, env):
    if sys.platform != "linux":
        raise RuntimeError("Formal test scoring requires Linux; use --dry-run elsewhere")
    script = (
        "import json,platform,torch; from mdlm_sft.runtime import validate_environment; "
        "from mdlm_sft.scoring import preflight_executor; validate_environment(); "
        "assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); "
        "assert torch.version.cuda == '12.4'; "
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),"
        "'executor':preflight_executor()}))"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def checked_assets(root, config):
    entries = [{"path": config["test"]["path"], "sha256": config["test"]["sha256"]},
               *({"path": m["checkpoint"], "sha256": m["sha256"]} for m in config["models"])]
    checked = []
    for entry in entries:
        path = resolve_path(root, entry["path"])
        if path.is_symlink() or not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen input SHA256 mismatch: {entry['path']}")
        checked.append({**entry, "size_bytes": path.stat().st_size})
    checkpoint = resolve_path(root, config["models"][1]["checkpoint"])
    best = read_json(checkpoint.parent / "best.json")
    if best.get("step") != config["models"][1]["global_step"]:
        raise ValueError("Experiment12 best checkpoint metadata differs from the selected step")
    return checked


def check_case(directory, case, frozen):
    """Read completed artifacts only; never reopen the official test parquet here."""
    summary = read_json(directory / "summary.json")
    protocol = read_json(directory / "protocol.json")
    resolved = read_json(directory / "config.resolved.json")
    test_source = read_json(directory / "test_source.json")
    usage = read_json(directory / "usage.json")
    expected = 1319
    if (summary.get("complete") is not True or summary.get("label") != "final_test"
            or summary.get("question_count") != expected or summary.get("num_questions") != expected
            or usage.get("status") != "complete" or resolved != case["config"]):
        raise ValueError("Case is incomplete, exploratory, or has changed settings")
    if (test_source.get("frozen_protocol_acknowledged") is not True
            or test_source.get("sha256") != frozen["benchmark_config"]["test"]["sha256"]):
        raise ValueError("Case used a different or unfrozen test parquet")
    core = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if fingerprint(core) != protocol.get("protocol_sha256") or summary.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError("Case protocol checksum mismatch")
    if (protocol.get("generation") != case["config"]["evaluation"]
            or protocol.get("precision") != case["config"]["precision"]
            or protocol.get("question_order") != [f"test:{i}" for i in range(expected)]):
        raise ValueError("Case generation settings or full test order changed")
    if set(protocol.get("evaluation_source", {})) != EVALUATION_SOURCES:
        raise ValueError("Case must identify all eight evaluation source modules")
    for name, digest in protocol["evaluation_source"].items():
        if frozen["git"]["tracked_files_sha256"].get("mdlm_sft/" + name) != digest:
            raise ValueError(f"Case used different evaluation source: {name}")
    scorer = frozen["environment"]["executor"]["scorer_version"]
    if summary.get("scorer_version") != scorer or protocol.get("scorer_version") != scorer:
        raise ValueError("Case scorer version changed")
    source, model = summary["checkpoint_source"], case["model"]
    if model["kind"] == "pretrained":
        if source.get("source") != "pretrained_ema" or source.get("checkpoint_sha256") != model["sha256"]:
            raise ValueError("Case pretrained checkpoint differs")
    else:
        file_identity = read_json(directory / "checkpoint_file.json")
        if (source.get("source") != "sft_ordinary" or source.get("global_step") != model["global_step"]
                or source.get("model_sha256") != model["model_sha256"] or file_identity.get("sha256") != model["sha256"]):
            raise ValueError("Case SFT checkpoint differs")
    predictions = directory / "predictions.jsonl"
    if sha256(predictions) != summary.get("predictions_sha256"):
        raise ValueError("Case prediction checksum mismatch")
    counts = Counter()
    with predictions.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row.get("gsm8k_id") != f"test:{index}" or row.get("status") not in STATUSES or row.get("scorer_version") != scorer:
                raise ValueError("Case predictions omit/reorder questions or contain invalid outcomes")
            counts[row["status"]] += 1
    if sum(counts.values()) != expected or summary.get("counts") != {name: counts[name] for name in STATUSES}:
        raise ValueError("Case counts disagree with per-question predictions")
    for field, status in (("accuracy", "correct"), ("format_failure_rate", "format_failure"),
                          ("execution_failure_rate", "execution_failure"), ("wrong_answer_rate", "wrong_answer")):
        if summary.get(field) != counts[status] / expected:
            raise ValueError(f"Case aggregate disagrees with predictions: {field}")
    return summary


def artifacts(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Case evidence must not contain symbolic links")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return result


def shared_protocol(directory):
    """The six protocols may differ only in the two prespecified decoding fields."""
    protocol = read_json(directory / "protocol.json")
    protocol.pop("protocol_sha256")
    protocol["generation"].pop("temperature")
    protocol["generation"].pop("steps")
    return protocol


def verified_receipt(case_root, case, frozen):
    receipt_path = case_root / "receipt.json"
    if not receipt_path.exists():
        return None
    receipt = read_json(receipt_path)
    attempt_name = receipt.get("attempt", "")
    if not re.fullmatch(r"attempt-[0-9]{3,}", attempt_name) or receipt.get("frozen_sha256") != frozen["frozen_sha256"]:
        raise ValueError("Saved case receipt belongs to another frozen protocol")
    directory = case_root / attempt_name
    if receipt.get("artifacts") != artifacts(directory):
        raise ValueError("Previously completed case evidence changed; refusing to replace it")
    summary = check_case(directory, case, frozen)
    if receipt.get("summary") != summary:
        raise ValueError("Saved case summary differs from verified artifacts")
    return receipt


def run(args, root=ROOT):
    root = Path(root).resolve()
    config_path = resolve_path(root, args.config)
    config = read_json(config_path)
    source_path = resolve_path(root, config["source_config"])
    base = read_json(source_path)
    validate_config(config, base)
    output = resolve_path(root, args.output_dir or config["output_dir"])
    cases = case_specs(root, output, config, base)
    if args.dry_run:
        print(json.dumps({"output": str(output), "cases": [
            {"id": case["id"], "evaluation": case["config"]["evaluation"],
             "command": command_for(root, output, config, case, output / "cases" / case["id"] / "attempt-001")}
            for case in cases]}, indent=2))
        return 0
    if output.exists() != bool(args.resume):
        raise FileExistsError("Existing results require --resume; a new run requires an unused output directory")
    version = capture_version(root)  # Dirty/uncommitted source fails before creating output or reading test rows.
    assets = checked_assets(root, config)  # Hash bytes only. The evaluator reads rows after freeze.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    environment = preflight(root, env)
    frozen = {"schema_version": 1, "benchmark": "gsm8k", "git": version,
              "benchmark_config": config, "benchmark_config_sha256": sha256(config_path),
              "source_config_sha256": sha256(source_path), "assets": assets, "cases": cases,
              "environment": environment, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
              "execution": "Six sequential single-process unified evaluations; no parallel model groups"}
    frozen["frozen_sha256"] = fingerprint(frozen)
    if args.resume:
        if read_json(output / "frozen.json") != frozen:
            raise ValueError("Resume inputs, code, settings or environment differ from the frozen protocol")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "configs").mkdir()
        for case in cases:
            write_json(Path(case["config_path"]), case["config"])
        write_json(output / "frozen.json", frozen)
        write_json(output / "started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                                             "frozen_sha256": frozen["frozen_sha256"]})
    results = []
    reference_protocol = None
    started = time.monotonic()
    active = None
    try:
        for case in cases:
            active = case["id"]
            if read_json(Path(case["config_path"])) != case["config"]:
                raise ValueError("Frozen resolved configuration changed")
            if capture_version(root) != version or checked_assets(root, config) != assets:
                raise ValueError("Frozen code or input artifacts changed during the benchmark")
            case_root = output / "cases" / case["id"]
            receipt = verified_receipt(case_root, case, frozen)
            if receipt is None:
                case_root.mkdir(parents=True, exist_ok=True)
                previous = list(case_root.glob("attempt-*"))
                attempt_number = max((int(re.match(r"attempt-(\d+)", path.name).group(1))
                                      for path in previous if re.match(r"attempt-(\d+)", path.name)), default=0) + 1
                attempt = case_root / f"attempt-{attempt_number:03d}"
                # Evaluator owns a fresh output directory; runner log stays beside it.
                command = command_for(root, output, config, case, attempt)
                write_json(output / "status.json", {"status": "running", "case": active, "attempt": attempt.name,
                                                     "completed_cases": len(results)})
                write_json(case_root / f"{attempt.name}-command.json", {"command": command,
                                                                       "frozen_sha256": frozen["frozen_sha256"]})
                with (case_root / f"{attempt.name}.log").open("x", encoding="utf-8", newline="\n") as log:
                    run_command({"command": command, "timeout_seconds": config["max_wall_seconds_per_case"] + 120}, root, env, log)
                summary = check_case(attempt, case, frozen)
                receipt = {"status": "complete", "case": active, "attempt": attempt.name,
                           "frozen_sha256": frozen["frozen_sha256"], "summary": summary, "artifacts": artifacts(attempt)}
                write_json(case_root / "receipt.json", receipt)
            common = shared_protocol(case_root / receipt["attempt"])
            if reference_protocol is not None and common != reference_protocol:
                raise ValueError("Cases differ beyond the prespecified temperature and sampling steps")
            reference_protocol = common
            results.append({"case": active, "setting": case["setting"], "model": case["model"]["id"],
                            "receipt": str((case_root / "receipt.json").relative_to(output)), **receipt["summary"]})
            write_json(output / "summary.json", {"status": "running", "complete": False, "results": results,
                                                  "frozen_sha256": frozen["frozen_sha256"]})
        if capture_version(root) != version or checked_assets(root, config) != assets:
            raise ValueError("Frozen code or inputs changed before final completion")
    except BaseException as error:
        failure = {"status": "failed", "case": active, "completed_cases": len(results),
                   "error_type": type(error).__name__, "error": str(error)}
        write_json(output / "status.json", failure)
        write_json(output / "summary.json", {**failure, "complete": False, "results": results,
                                              "frozen_sha256": frozen["frozen_sha256"]})
        raise
    final = {"status": "complete", "complete": True, "question_count_per_case": 1319,
             "completed_cases": len(results), "results": results, "frozen_sha256": frozen["frozen_sha256"],
             "wall_seconds_this_invocation": time.monotonic() - started}
    write_json(output / "summary.json", final)
    write_json(output / "status.json", {"status": "complete", "completed_cases": len(results), "stage": "finished"})
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmarks/gsm8k/config.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed cases; preserve and retry failed attempts")
    parser.add_argument("--dry-run", action="store_true", help="Show six frozen case commands without reading test rows or running them")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
