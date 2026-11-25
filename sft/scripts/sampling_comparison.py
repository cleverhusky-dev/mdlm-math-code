"""Keep experiment3 on the original models and vary only generation step count."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mdlm_sft.baseline import check_baseline, experiment1_checkpoint, read_json, resolve_path, sha256


def read_score(directory):
    """Check completed summaries against the saved predictions and protocol."""
    summary = read_json(directory / "summary.json")
    if (summary.get("complete") is not True or summary.get("label") != "validation"
            or summary.get("num_questions") != 639 or summary.get("question_count") != 639):
        raise ValueError(f"Expected complete 639-question validation: {directory}")
    predictions = directory / "predictions.jsonl"
    if sha256(predictions) != summary.get("predictions_sha256"):
        raise ValueError(f"Predictions checksum mismatch: {directory}")
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    counts = {key: 0 for key in ("correct", "format_failure", "execution_failure", "wrong_answer")}
    counts.update(Counter(row["status"] for row in rows))
    if len(rows) != 639 or counts != summary["counts"] or summary["accuracy"] != counts["correct"] / 639:
        raise ValueError(f"Summary differs from prediction counts: {directory}")
    protocol = read_json(directory / "protocol.json")
    digest = protocol.pop("protocol_sha256")
    if (hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != digest
            or summary.get("protocol_sha256") != digest):
        raise ValueError(f"Protocol checksum mismatch: {directory}")
    ids = [row["gsm8k_id"] for row in rows]
    if len(set(ids)) != 639 or ids != protocol["question_order"]:
        raise ValueError(f"Question identity/order differs: {directory}")
    return summary, protocol


def check_references(root, baseline, sft_dir, config):
    root = Path(root).resolve()
    baseline, sft_dir = resolve_path(root, baseline), resolve_path(root, sft_dir)
    if config["evaluation"]["steps"] != 128:
        raise ValueError("Experiment3 requires 128 sampling steps")
    original = deepcopy(config)
    original["evaluation"]["steps"] = 32
    baseline_record = check_baseline(root, baseline, original)
    metadata = read_json(sft_dir / "experiment.json")
    result = read_json(sft_dir / "result.json")
    training = read_json(sft_dir / "training/result.json")
    if (metadata.get("experiment") != "experiment1" or result.get("experiment") != "experiment1"
            or result.get("status") != "complete" or training.get("status") != "complete"
            or training.get("smoke") is not False or training.get("epoch") != 3
            or training.get("global_step") != 609
            or training.get("pending_validation") is not None):
        raise ValueError("Experiment3 requires experiment1 to finish its three epochs at step 609")
    if metadata.get("inputs_lock_sha256") != sha256(root / "inputs.lock.json"):
        raise ValueError("Experiment1 input identity differs")
    source_config = dict(metadata["config"])
    source_config.pop("output_dir", None)
    original.pop("output_dir", None)
    if source_config != original:
        raise ValueError("Only evaluation.steps and output_dir may differ from experiment1")
    pretrained, protocol = read_score(baseline / "validation")
    sft, sft_protocol = read_score(sft_dir / "training/validation/step-00000609")
    if protocol != sft_protocol:
        raise ValueError("The original pretrained and SFT evaluation protocols differ")
    source = sft.get("checkpoint_source", {})
    if source.get("source") != "sft_ordinary" or source.get("global_step") != 609:
        raise ValueError("The SFT reference must identify step 609")
    best = read_json(sft_dir / "training/checkpoints/best.json")
    if best.get("step") != training.get("best_step") or (best.get("step") == 609 and best.get("accuracy") != sft["accuracy"]):
        raise ValueError("Experiment1 selected checkpoint metadata differs from its recorded training results")
    checkpoint = experiment1_checkpoint(sft_dir / "training/checkpoints")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing experiment1 step-609 checkpoint: {checkpoint}")
    return {"baseline": baseline_record, "sft_directory": str(sft_dir),
            "sft_checkpoint": str(checkpoint), "sft_checkpoint_sha256": sha256(checkpoint),
            "pretrained": pretrained, "sft": sft, "protocol_32": protocol,
            "sft_experiment_sha256": sha256(sft_dir / "experiment.json"),
            "sft_result_sha256": sha256(sft_dir / "training/result.json")}


def verify_checkpoint(checkpoint, expected_model_sha256):
    # Load only the user's trusted SFT file; mmap avoids copying optimizer tensors.
    import torch
    from mdlm_sft.checkpoints import load_sft
    state = load_sft(checkpoint)
    if (state["progress"]["global_step"] != 609 or state["config"].get("smoke")
            or state["config"]["epochs"] != 3):
        raise ValueError("Checkpoint is not experiment1's full-data step 609")
    # Same state-dict fingerprint as runtime.model_fingerprint, without building a model.
    digest = hashlib.sha256()
    for name, tensor in state["model"].items():
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    if digest.hexdigest() != expected_model_sha256:
        raise ValueError("Checkpoint weights differ from experiment1's saved 609-step predictions")
    print(json.dumps({"status": "passed", "sft_step": 609, "model_sha256": digest.hexdigest()}))


def check_completion(root, output, config, references):
    expected_protocol = deepcopy(references["protocol_32"])
    expected_protocol["generation"]["steps"] = 128
    if expected_protocol["generation"] != config["evaluation"]:
        raise ValueError("Experiment3 generation configuration changed")
    comparisons = {}
    for kind in ("pretrained", "sft"):
        score, protocol = read_score(output / kind)
        previous = references[kind]
        if protocol != expected_protocol:
            raise ValueError(f"Only sampling steps may change in the {kind} protocol")
        if score["checkpoint_source"] != previous["checkpoint_source"]:
            raise ValueError(f"The {kind} checkpoint changed")
        comparisons[kind] = {
            "checkpoint_source": score["checkpoint_source"],
            "steps_32": {key: previous[key] for key in ("accuracy", "counts", "predictions_sha256")},
            "steps_128": {key: score[key] for key in ("accuracy", "counts", "predictions_sha256")},
            "accuracy_change_percentage_points": 100 * (score["accuracy"] - previous["accuracy"])}
    checkpoint = read_json(output / "sft/checkpoint_file.json")
    if checkpoint["sha256"] != references["sft_checkpoint_sha256"]:
        raise ValueError("SFT checkpoint file changed during the comparison")
    return {"status": "complete", "sampling_steps": 128, "sft_step": 609,
            "comparison": comparisons, "sft_minus_pretrained_percentage_points": {
                key: 100 * (comparisons["sft"][key]["accuracy"] - comparisons["pretrained"][key]["accuracy"])
                for key in ("steps_32", "steps_128")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    args = parser.parse_args()
    verify_checkpoint(args.checkpoint, args.expected_model_sha256)
