"""Pin the validation-selected model from a finished SFT run before test scoring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mdlm_sft.baseline import read_json, resolve_path, sha256
from scripts.run_benchmark import validate_config


def benchmark_for_run(root, run_dir):
    config = read_json(root / "benchmarks/gsm8k/config.json")
    training = run_dir / "training"
    result = read_json(training / "result.json")
    if (read_json(run_dir / "experiment.json").get("experiment") != "experiment12"
            or result.get("status") != "complete" or result.get("smoke") is not False):
        raise ValueError("Expected a completed experiment12 training run")
    best = read_json(training / "checkpoints/best.json")
    step = best["step"]
    if type(step) is not int or step < 1 or step != result["best_step"]:
        raise ValueError("Selected step differs from the training result")
    summary = read_json(training / "validation" / f"step-{step:08d}" / "summary.json")
    source = summary["checkpoint_source"]
    if (summary.get("complete") is not True or summary.get("label") != "validation"
            or summary.get("question_count") != 639 or source.get("source") != "sft_ordinary"
            or source.get("global_step") != step
            or not (summary["accuracy"] == best["accuracy"] == result["best_accuracy"])):
        raise ValueError("The checkpoint selection and validation summary do not agree")

    checkpoint = training / "checkpoints/best.pt"
    location = checkpoint.relative_to(root) if checkpoint.is_relative_to(root) else checkpoint
    config["models"][1].update(checkpoint=location.as_posix(), global_step=step,
                               sha256=sha256(checkpoint), model_sha256=source["model_sha256"])
    validate_config(config, read_json(root / config["source_config"]))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", required=True, help="Completed experiment12 run directory")
    parser.add_argument("--output", required=True, help="New JSON config, relative to sft/ or absolute")
    args = parser.parse_args()
    config = benchmark_for_run(ROOT, resolve_path(ROOT, args.sft_dir))
    output = resolve_path(ROOT, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved {output} for validation-selected step {config['models'][1]['global_step']}")


if __name__ == "__main__":
    main()
