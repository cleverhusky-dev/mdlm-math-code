"""Offline checks for the controlled, sequential learning-rate comparison."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_experiment as launcher
import run_lr_comparison as comparison


class LearningRateExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base = json.loads((PROJECT / "experiments/experiment1/config.json").read_text(encoding="utf-8"))
        self.write("experiments/experiment1/config.json", self.base)
        for variant, rate in (("low", 1e-5), ("high", 1e-4)):
            self.write(f"experiments/experiment4/config-{variant}.json", {
                **self.base, "learning_rate": rate, "output_dir": f"runs/experiment4/{variant}"})
        self.write("inputs.lock.json", {"files": []})
        self.write("data/prepared/manifest.json", {"counts": {"train": {"samples": 12953}}})

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        return path

    def args(self, **overrides):
        values = dict(output_dir="runs/experiment4", baseline_dir="runs/experiment0",
                      control_dir="runs/experiment1", master_port=29501, dry_run=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_only_learning_rate_and_output_differ_from_experiment1(self):
        configs = comparison.validate_configs(PROJECT)
        for variant, rate in (("low", 1e-5), ("high", 1e-4)):
            expected = {**self.base, "learning_rate": rate, "output_dir": f"runs/experiment4/{variant}"}
            self.assertEqual(configs[variant], expected)
        self.assertEqual(self.base["epochs"], 3)
        self.assertEqual(self.base["evaluation"]["steps"], 32)

    def test_unintended_control_changes_are_rejected(self):
        for key, value in (("epochs", 5), ("seed", 2), ("warmup_fraction", .1),
                           ("global_batch_size", 32), ("learning_rate", 3e-5)):
            with self.subTest(key=key):
                path = self.root / "experiments/experiment4/config-low.json"
                original = path.read_bytes()
                config = json.loads(original)
                config[key] = value
                self.write(path, config)
                with self.assertRaises(ValueError):
                    comparison.validate_configs(self.root)
                path.write_bytes(original)
        path = self.root / "experiments/experiment4/config-high.json"
        config = json.loads(path.read_text())
        config["evaluation"]["steps"] = 128
        self.write(path, config)
        with self.assertRaises(ValueError):
            comparison.validate_configs(self.root)

    def test_both_groups_use_fresh_pretrained_two_gpu_training_and_reuse_baseline(self):
        for variant, config in comparison.validate_configs(self.root).items():
            with self.subTest(variant=variant):
                child = comparison.make_group_args(self.args(), variant, self.root / "runs/experiment4")
                child.dry_run = True
                stream = io.StringIO()
                with patch.object(launcher, "preflight") as preflight, contextlib.redirect_stdout(stream):
                    self.assertEqual(launcher.run(child, self.root), 0)
                preflight.assert_not_called()
                plan = json.loads(stream.getvalue())
                self.assertEqual([phase["name"] for phase in plan["phases"]], ["training"])
                command = plan["phases"][0]["command"]
                self.assertIn("--nproc-per-node=2", command)
                self.assertNotIn("--resume", command)
                self.assertNotIn("--smoke", command)
                self.assertEqual(command[command.index("--init") + 1], str(self.root / "inputs/mdlm.ckpt"))
                self.assertEqual(command[command.index("--baseline-dir") + 1], str(self.root / "runs/experiment0"))
                self.assertEqual(command[command.index("--config") + 1],
                                 str(self.root / f"experiments/experiment4/config-{variant}.json"))
                self.assertEqual(command[command.index("--output-dir") + 1],
                                 str(self.root / f"runs/experiment4/{variant}/training"))
                self.assertLessEqual(plan["phases"][0]["timeout_seconds"], config["max_wall_seconds"] + 180)
        self.assertFalse((self.root / "runs").exists())

    def group_fixture(self):
        output = self.root / "runs/experiment4/low"
        config = comparison.validate_configs(self.root)["low"]
        progress = {"status": "complete", "smoke": False, "epoch": 3, "global_step": 609,
                    "pending_validation": None, "best_step": 203, "best_accuracy": 20 / 639}
        self.write(output / "result.json", {"experiment": "experiment4", "status": "complete",
                                             "training_result": progress})
        self.write(output / "training/result.json", progress)
        self.write(output / "training/config.resolved.json", config)
        self.write(output / "training/checkpoints/best.json", {"step": 203, "accuracy": 20 / 639})
        for step, correct in ((203, 20), (406, 15), (609, 10)):
            directory = output / f"training/validation/step-{step:08d}"
            ids = [f"train-{i}" for i in range(639)]
            protocol = {"generation": config["evaluation"], "precision": config["precision"], "question_order": ids}
            protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            self.write(directory / "protocol.json", {**protocol, "protocol_sha256": protocol_hash})
            predictions = directory / "predictions.jsonl"
            predictions.write_text("".join(json.dumps({"gsm8k_id": identity,
                "status": "correct" if i < correct else "wrong_answer"}) + "\n" for i, identity in enumerate(ids)), encoding="utf-8")
            self.write(directory / "summary.json", {"complete": True, "label": "validation",
                "num_questions": 639, "question_count": 639, "accuracy": correct / 639,
                "counts": {"correct": correct, "wrong_answer": 639 - correct, "format_failure": 0, "execution_failure": 0},
                "checkpoint_source": {"source": "sft_ordinary", "global_step": step, "model_sha256": "a" * 64},
                "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "protocol_sha256": protocol_hash})
            self.write(output / f"training/validation/step-{step:08d}-loss.json", {
                "step": step, "label": "validation", "loss": step / 10000, "code_samples": 1456,
                "target_tokens": 655019, "seed": 101})
        return output, config, progress

    def test_summary_distinguishes_fixed_final_step_from_best(self):
        output, config, _ = self.group_fixture()
        result = comparison.summarize_group(output, config)
        self.assertEqual(result["final"]["step"], 609)
        self.assertEqual(result["final"]["accuracy"], 10 / 639)
        self.assertEqual(result["final"]["validation_loss"], .0609)
        self.assertEqual(result["best"]["step"], 203)
        self.assertEqual(result["best"]["accuracy"], 20 / 639)
        self.assertEqual(result["best"]["validation_loss"], .0203)

    def test_incomplete_or_mismatched_training_cannot_be_summarized_as_complete(self):
        output, config, progress = self.group_fixture()
        for change in ({"status": "bounded_stop"}, {"global_step": 608}, {"smoke": True},
                       {"pending_validation": {"generate": True}}, {"best_accuracy": .99}):
            with self.subTest(change=change):
                self.write(output / "training/result.json", {**progress, **change})
                with self.assertRaises(ValueError):
                    comparison.summarize_group(output, config)

    def test_parent_preserves_existing_output(self):
        marker = self.write("runs/experiment4/keep.json", {"important": True})
        with patch.object(comparison.launcher, "run") as child, patch.object(comparison, "capture_version") as version:
            with self.assertRaises(FileExistsError):
                comparison.run(self.args(), self.root)
        child.assert_not_called()
        version.assert_not_called()
        self.assertEqual(json.loads(marker.read_text()), {"important": True})

    def test_failed_or_bounded_low_group_never_starts_high(self):
        for name, failure in (("bounded", 2), ("exception", RuntimeError("training failed"))):
            with self.subTest(name=name), contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(comparison, "capture_version", return_value={"commit": "a" * 40}))
                stack.enter_context(patch.object(comparison, "check_control", return_value={}))
                stack.enter_context(patch.object(comparison, "check_baseline", return_value={"protocol_sha256": "p"}))
                kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                child = stack.enter_context(patch.object(comparison.launcher, "run", **kwargs))
                output = f"runs/failure-{name}"
                with self.assertRaises(RuntimeError):
                    comparison.run(self.args(output_dir=output), self.root)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(child.call_args.args[0].variant, "low")
                result = json.loads((self.root / output / "result.json").read_text())
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["stage"], "low")
                self.assertEqual(result["completed_groups"], [])

    def test_parent_completes_only_after_low_then_high(self):
        seen = []
        def child(args, root):
            seen.append(args.variant)
            return 0
        def summary(output, config):
            return {"learning_rate": config["learning_rate"], "final": {"protocol_sha256": "p"}, "best": {}}
        with patch.object(comparison, "capture_version", return_value={"commit": "a" * 40}), \
                patch.object(comparison, "check_control", return_value={"learning_rate": 3e-5}), \
                patch.object(comparison, "check_baseline", return_value={"protocol_sha256": "p"}), \
                patch.object(comparison.launcher, "run", side_effect=child), \
                patch.object(comparison, "summarize_group", side_effect=summary), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(comparison.run(self.args(), self.root), 0)
        self.assertEqual(seen, ["low", "high"])
        result = json.loads((self.root / "runs/experiment4/result.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["groups"]), {"low", "high"})
        self.assertEqual(result["groups"]["low"]["learning_rate"], 1e-5)
        self.assertEqual(result["groups"]["high"]["learning_rate"], 1e-4)


if __name__ == "__main__":
    unittest.main()
