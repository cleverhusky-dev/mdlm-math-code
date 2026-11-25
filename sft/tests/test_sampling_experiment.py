"""Offline contract checks for the fixed-checkpoint 128-step comparison."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sampling_experiment_launcher", PROJECT / "scripts/run_experiment.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
comparison_spec = importlib.util.spec_from_file_location(
    "sampling_comparison_checks", PROJECT / "scripts/sampling_comparison.py"
)
comparison = importlib.util.module_from_spec(comparison_spec)
comparison_spec.loader.exec_module(comparison)


class SamplingExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = json.loads(
            (PROJECT / "experiments/experiment1/config.json").read_text(encoding="utf-8")
        )
        self.config["output_dir"] = "runs/experiment3"
        self.config["evaluation"]["steps"] = 128
        self.config_path = self.root / "experiments/experiment3/config.json"
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.output = self.root / "runs/experiment3"

    def args(self, **overrides):
        values = dict(
            experiment="experiment3", output_dir=None,
            baseline_dir="runs/experiment0", sft_dir="runs/experiment1",
            master_port=29501, dry_run=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_release_config_changes_only_output_and_sampling_steps(self):
        actual = json.loads(
            (PROJECT / "experiments/experiment3/config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(actual, self.config)

    def test_plan_evaluates_pretrained_and_fixed_sft_on_validation(self):
        plan = launcher.make_plan(
            self.root, "experiment3", self.output, self.config, python="fixture-python"
        )
        self.assertEqual([phase["name"] for phase in plan], ["pretrained", "sft"])
        checkpoints = {
            "pretrained": self.root / self.config["pretrained_checkpoint"],
            "sft": self.root / "runs/experiment1/training/checkpoints/step-00000609.pt",
        }
        for phase in plan:
            with self.subTest(phase=phase["name"]):
                command = phase["command"]
                self.assertEqual(command[:3], ["fixture-python", "-m", "mdlm_sft.evaluate"])
                expected_arguments = {
                    "--kind": phase["name"],
                    "--config": str(self.config_path),
                    "--checkpoint": str(checkpoints[phase["name"]].resolve()),
                    "--split": "val",
                    "--output-dir": str(self.output / phase["name"]),
                }
                for option, value in expected_arguments.items():
                    self.assertIn(option, command)
                    self.assertEqual(command[command.index(option) + 1], value)
                self.assertNotIn("--final-test-parquet", command)
                self.assertNotIn("--smoke", command)

    def test_plan_never_launches_training_resume_or_loss(self):
        plan = launcher.make_plan(self.root, "experiment3", self.output, self.config)
        forbidden = (
            "torchrun", "torch.distributed.run", "mdlm_sft.train",
            "--resume", "--init", "evaluate_loss.py",
        )
        for phase in plan:
            for argument in phase["command"]:
                for token in forbidden:
                    with self.subTest(phase=phase["name"], token=token):
                        self.assertNotIn(token, argument)

    def test_explicit_sft_directory_resolves_to_its_best_checkpoint(self):
        for sft_dir in ("runs/verified-experiment1", self.root / "archived/experiment1"):
            with self.subTest(sft_dir=sft_dir):
                plan = launcher.make_plan(
                    self.root, "experiment3", self.output, self.config, sft_dir=sft_dir
                )
                command = next(phase["command"] for phase in plan if phase["name"] == "sft")
                expected = (self.root / sft_dir / "training/checkpoints/step-00000609.pt").resolve()
                self.assertEqual(command[command.index("--checkpoint") + 1], str(expected))

    def test_dry_run_needs_no_inputs_references_git_or_gpu_and_creates_no_output(self):
        capture_version = Mock(side_effect=AssertionError("dry run must not inspect Git"))
        output = io.StringIO()
        before = sorted(path.relative_to(self.root) for path in self.root.rglob("*"))
        with (
            patch.dict(sys.modules, {
                "experiment_version": types.SimpleNamespace(capture_version=capture_version),
            }),
            patch.object(launcher, "preflight") as preflight,
            patch.object(launcher, "run_command") as run_command,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(launcher.run(self.args(dry_run=True), self.root), 0)
        capture_version.assert_not_called()
        preflight.assert_not_called()
        run_command.assert_not_called()
        self.assertEqual(before, sorted(path.relative_to(self.root) for path in self.root.rglob("*")))
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "inputs").exists())
        self.assertFalse((self.root / "inputs.lock.json").exists())
        preview = json.loads(output.getvalue())
        self.assertEqual(preview["experiment"], "experiment3")
        self.assertEqual([phase["name"] for phase in preview["phases"]], ["pretrained", "sft"])

    def test_existing_results_are_rejected_before_any_work(self):
        capture_version = Mock(side_effect=AssertionError("existing output must be rejected first"))
        for output_dir in (None, "runs/custom-comparison"):
            with self.subTest(output_dir=output_dir):
                target = self.root / (output_dir or self.config["output_dir"])
                target.mkdir(parents=True)
                saved = target / "keep.json"
                original = b'{"important": true}\n'
                saved.write_bytes(original)
                with (
                    patch.dict(sys.modules, {
                        "experiment_version": types.SimpleNamespace(capture_version=capture_version),
                    }),
                    patch.object(launcher, "preflight") as preflight,
                    patch.object(launcher, "run_command") as run_command,
                    self.assertRaises(FileExistsError),
                ):
                    launcher.run(self.args(output_dir=output_dir), self.root)
                capture_version.assert_not_called()
                preflight.assert_not_called()
                run_command.assert_not_called()
                self.assertEqual(saved.read_bytes(), original)
                self.assertEqual(list(target.iterdir()), [saved])


class SamplingComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.baseline = self.root / "runs/experiment0"
        self.sft = self.root / "runs/experiment1"
        self.output = self.root / "runs/experiment3"
        original = json.loads(
            (PROJECT / "experiments/experiment1/config.json").read_text(encoding="utf-8")
        )
        self.config = deepcopy(original)
        self.config["output_dir"] = "runs/experiment3"
        self.config["evaluation"]["steps"] = 128
        lock = self.write_json(self.root / "inputs.lock.json", {"fixture": "locked inputs"})
        self.write_json(self.sft / "experiment.json", {
            "experiment": "experiment1", "config": original,
            "inputs_lock_sha256": self.digest(lock),
        })
        self.write_json(self.sft / "result.json", {"experiment": "experiment1", "status": "complete"})
        self.write_json(self.sft / "training/result.json", {
            "status": "complete", "smoke": False, "epoch": 3,
            "global_step": 609, "best_step": 609, "pending_validation": None,
        })
        self.checkpoint = self.sft / "training/checkpoints/best.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"offline fixture: fixed experiment1 checkpoint")
        self.sources = {
            "pretrained": {"source": "pretrained_ema", "checkpoint_sha256": "a" * 64,
                           "model_sha256": "b" * 64},
            "sft": {"source": "sft_ordinary", "global_step": 609, "model_sha256": "c" * 64},
        }
        self.protocol32 = {
            "generation": deepcopy(original["evaluation"]), "precision": original["precision"],
            "scorer_version": "offline-fixture", "tokenizer": {"sha256": "d" * 64},
            "question_order": [f"fixture:{index}" for index in range(639)],
            "questions_sha256": "e" * 64, "evaluation_source": {"sampling.py": "f" * 64},
        }
        self.protocol128 = deepcopy(self.protocol32)
        self.protocol128["generation"]["steps"] = 128
        self.write_score(self.baseline / "validation", self.protocol32, self.sources["pretrained"], 47)
        self.sft_validation = self.sft / "training/validation/step-00000609"
        self.write_score(self.sft_validation, self.protocol32, self.sources["sft"], 147)
        self.write_json(self.sft / "training/checkpoints/best.json", {"step": 609, "accuracy": 147 / 639})
        self.write_score(self.output / "pretrained", self.protocol128, self.sources["pretrained"], 58)
        self.write_score(self.output / "sft", self.protocol128, self.sources["sft"], 163)
        self.write_json(self.output / "sft/checkpoint_file.json", {"sha256": self.digest(self.checkpoint)})
        # Existing baseline integrity behavior is covered in test_experiments.py.
        baseline_check = patch.object(comparison, "check_baseline", return_value={"fixture": "verified baseline"})
        self.baseline_check = baseline_check.start()
        self.addCleanup(baseline_check.stop)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def write_score(self, directory, protocol, source, correct):
        directory.mkdir(parents=True, exist_ok=True)
        records = [
            {"gsm8k_id": identifier, "question": f"Fixture question {identifier}",
             "status": "correct" if index < correct else "wrong_answer"}
            for index, identifier in enumerate(protocol["question_order"])
        ]
        predictions = directory / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        self.write_json(directory / "protocol.json", {**protocol, "protocol_sha256": protocol_digest})
        self.write_json(directory / "summary.json", {
            "complete": True, "label": "validation", "num_questions": 639, "question_count": 639,
            "accuracy": correct / 639,
            "counts": {"correct": correct, "wrong_answer": 639 - correct,
                       "format_failure": 0, "execution_failure": 0},
            "checkpoint_source": source, "protocol_sha256": protocol_digest,
            "predictions_sha256": self.digest(predictions),
        })

    def references(self, config=None):
        return comparison.check_references(self.root, self.baseline, self.sft, config or self.config)

    def test_reference_requires_completed_experiment1_and_consistent_metadata(self):
        changes = (
            (self.sft / "experiment.json", {"experiment": "experiment2"}),
            (self.sft / "result.json", {"experiment": "experiment2"}),
            (self.sft / "training/result.json", {"global_step": 812}),
            (self.sft / "training/result.json", {"best_step": 812}),
            (self.sft / "training/result.json", {"epoch": 5}),
            (self.sft / "training/checkpoints/best.json", {"step": 812}),
            (self.sft_validation / "summary.json", {
                "checkpoint_source": {**self.sources["sft"], "global_step": 812},
            }),
        )
        for path, change in changes:
            with self.subTest(path=path.relative_to(self.root), change=change):
                saved = path.read_bytes()
                value = json.loads(saved)
                self.write_json(path, {**value, **change})
                with self.assertRaises(ValueError):
                    self.references()
                path.write_bytes(saved)

    def test_fixed_checkpoint_when_validation_selected_an_earlier_epoch(self):
        training = self.sft / "training/result.json"
        self.write_json(training, {**json.loads(training.read_bytes()), "best_step": 406})
        self.write_json(self.checkpoint.parent / "best.json", {"step": 406, "accuracy": 0.3})
        final = self.checkpoint.parent / "step-00000609.pt"
        final.write_bytes(b"three-epoch-model")
        references = self.references()
        self.assertEqual(references["sft_checkpoint"], str(final))
        self.assertEqual(references["sft_checkpoint_sha256"], self.digest(final))
        final.unlink()
        with self.assertRaises(FileNotFoundError):
            self.references()

    def test_reference_rejects_changes_besides_sampling_steps(self):
        for change in ({"steps": 32}, {"seed": 2}, {"temperature": 0.8}, {"batch_size": 16}):
            with self.subTest(change=change):
                config = deepcopy(self.config)
                config["evaluation"].update(change)
                with self.assertRaises(ValueError):
                    self.references(config)

    def test_completed_report_compares_both_fixed_models_at_32_and_128_steps(self):
        references = self.references()
        self.assertEqual(references["sft_checkpoint_sha256"], self.digest(self.checkpoint))
        self.assertEqual(references["sft_checkpoint"], str(self.checkpoint))
        self.assertEqual(self.baseline_check.call_args.args[2]["evaluation"]["steps"], 32)
        result = comparison.check_completion(self.root, self.output, self.config, references)
        self.assertEqual(result["status"], "complete")
        self.assertEqual((result["sampling_steps"], result["sft_step"]), (128, 609))
        for kind, previous, current in (("pretrained", 47, 58), ("sft", 147, 163)):
            with self.subTest(kind=kind):
                report = result["comparison"][kind]
                self.assertEqual(report["checkpoint_source"], self.sources[kind])
                self.assertEqual(report["steps_32"]["counts"]["correct"], previous)
                self.assertEqual(report["steps_128"]["counts"]["correct"], current)
                self.assertAlmostEqual(report["accuracy_change_percentage_points"], 100 * (current - previous) / 639)
        self.assertAlmostEqual(result["sft_minus_pretrained_percentage_points"]["steps_32"], 100 * 100 / 639)
        self.assertAlmostEqual(result["sft_minus_pretrained_percentage_points"]["steps_128"], 100 * 105 / 639)

    def test_completion_rejects_changed_model_even_with_valid_prediction_hashes(self):
        references = self.references()
        for kind, correct in (("pretrained", 58), ("sft", 163)):
            with self.subTest(kind=kind):
                self.write_score(self.output / kind, self.protocol128,
                                 {**self.sources[kind], "model_sha256": "9" * 64}, correct)
                with self.assertRaises(ValueError):
                    comparison.check_completion(self.root, self.output, self.config, references)
                self.write_score(self.output / kind, self.protocol128, self.sources[kind], correct)
        self.write_json(self.output / "sft/checkpoint_file.json", {"sha256": "0" * 64})
        with self.assertRaises(ValueError):
            comparison.check_completion(self.root, self.output, self.config, references)

    def test_completion_rejects_changed_questions_or_protocol_even_with_valid_hashes(self):
        references = self.references()
        changes = (
            {"generation": {**self.protocol128["generation"], "steps": 32}},
            {"generation": {**self.protocol128["generation"], "seed": 2}},
            {"question_order": [f"other:{index}" for index in range(639)]},
            {"question_order": list(reversed(self.protocol128["question_order"]))},
            {"questions_sha256": "1" * 64},
            {"precision": "float32"},
            {"evaluation_source": {"sampling.py": "2" * 64}},
        )
        for kind, correct in (("pretrained", 58), ("sft", 163)):
            for change in changes:
                with self.subTest(kind=kind, changed_field=next(iter(change))):
                    protocol = {**deepcopy(self.protocol128), **change}
                    self.write_score(self.output / kind, protocol, self.sources[kind], correct)
                    with self.assertRaises(ValueError):
                        comparison.check_completion(self.root, self.output, self.config, references)
                    self.write_score(self.output / kind, self.protocol128, self.sources[kind], correct)


if __name__ == "__main__":
    unittest.main()
