"""Offline checks for experiment ordering, provenance gates and completion reports."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_experiment", PROJECT / "scripts/run_experiment.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = json.loads((PROJECT / "experiments/experiment1/config.json").read_text(encoding="utf-8"))
        for name in ("experiment0", "experiment1", "experiment2"):
            self.write(f"experiments/{name}/config.json", {**self.config, "output_dir": f"runs/{name}",
                "epochs": 5 if name == "experiment2" else 3})
        self.write("data/prepared/manifest.json", {"counts": {"train": {"samples": 12953}}})
        self.write("inputs.lock.json", {"files": [{"path": "inputs/mdlm.ckpt", "sha256": "a" * 64}]})
        self.version = {"commit": "a" * 40, "tags": ["experiment0-v1", "experiment1-v1"], "dirty": False}
        self.version_module = types.SimpleNamespace(capture_version=lambda root: self.version)
        self.baseline = self.root / "runs/experiment0"

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        return path

    def args(self, experiment="experiment1", **overrides):
        return argparse.Namespace(experiment=experiment, output_dir=overrides.get("output_dir"),
            baseline_dir="runs/experiment0", master_port=29501, dry_run=overrides.get("dry_run", False))

    def baseline_fixture(self):
        self.write("runs/experiment0/result.json", {"experiment": "experiment0", "status": "complete"})
        self.write("runs/experiment0/experiment.json", {"experiment": "experiment0", "config": self.config,
            "inputs_lock_sha256": launcher.sha256(self.root / "inputs.lock.json"), "git": self.version})
        source = {"source": "pretrained_ema", "checkpoint_sha256": "a" * 64}
        self.write("runs/experiment0/loss/summary.json", {"complete": True, "code_samples": 1456,
            "loss": 1.1, "checkpoint_source": source})
        names = ("evaluation.py", "sampling.py", "scoring.py", "executor.py", "data.py", "diffusion.py", "transformer.py", "runtime.py")
        code = {}
        for name in names:
            code[name] = launcher.sha256(self.write(f"mdlm_sft/{name}", "source fixture"))
        protocol = {"generation": self.config["evaluation"], "precision": self.config["precision"], "evaluation_source": code}
        protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        self.write("runs/experiment0/validation/protocol.json", protocol)
        predictions = self.write("runs/experiment0/validation/predictions.jsonl", {"fixture": True})
        self.write("runs/experiment0/validation/summary.json", {"complete": True, "num_questions": 639,
            "question_count": 639, "accuracy": 47 / 639, "counts": {"correct": 47}, "checkpoint_source": source,
            "protocol_sha256": protocol["protocol_sha256"], "predictions_sha256": launcher.sha256(predictions)})

    def test_experiment0_plan_uses_separate_full_validation_processes(self):
        plan = launcher.make_plan(self.root, "experiment0", self.baseline, self.config, python="fixture-python")
        self.assertEqual([p["name"] for p in plan], ["loss", "validation"])
        self.assertIn("pretrained", plan[0]["command"])
        self.assertIn("mdlm_sft.evaluate", plan[1]["command"])
        self.assertEqual(plan[1]["command"][plan[1]["command"].index("--split") + 1], "val")
        self.assertNotIn("--final-test-parquet", plan[1]["command"])

    def test_experiment1_plan_uses_two_gpus_fresh_init_and_static_loopback(self):
        for name in ("experiment1", "experiment2"):
            with self.subTest(experiment=name):
                config_path = self.root / "experiments" / name / "config.json"
                command = launcher.make_plan(self.root, name, self.root / "runs" / name,
                                             launcher.read_json(config_path))[0]["command"]
                for value in ("torch.distributed.run", "--nproc-per-node=2", "--rdzv-backend=static", "--master-addr=127.0.0.1", "--init"):
                    self.assertIn(value, command)
                self.assertNotIn("--resume", command)
                self.assertEqual(command[command.index("--init") + 1], str((self.root / "inputs/mdlm.ckpt").resolve()))
                self.assertEqual(command[command.index("--config") + 1], str(config_path))
                self.assertEqual(command[command.index("--baseline-dir") + 1], str(self.baseline.resolve()))
                self.assertNotIn("--smoke", command)

    def test_explicit_baseline_directory_reaches_training_command(self):
        baseline = self.root / "runs/baseline-copy"
        command = launcher.make_plan(self.root, "experiment1", self.root / "runs/experiment1",
                                     self.config, baseline_dir=baseline)[0]["command"]
        self.assertEqual(command[command.index("--baseline-dir") + 1], str(baseline.resolve()))

    def test_dry_run_does_not_require_gpu_git_inputs_or_create_results(self):
        with patch.object(launcher, "preflight") as preflight, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run(self.args(dry_run=True), self.root), 0)
        preflight.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_existing_output_is_never_overwritten(self):
        self.write("runs/experiment1/keep.json", {"important": True})
        with self.assertRaises(FileExistsError), patch.object(launcher, "preflight") as preflight:
            launcher.run(self.args(), self.root)
        preflight.assert_not_called()
        self.assertTrue((self.root / "runs/experiment1/keep.json").exists())

    def test_dirty_git_stops_before_preflight_or_output(self):
        def dirty(root):
            raise ValueError("dirty Git tree")
        with patch.dict(sys.modules, {"experiment_version": types.SimpleNamespace(capture_version=dirty)}), patch.object(launcher, "preflight") as probe:
            with self.assertRaisesRegex(ValueError, "dirty"):
                launcher.run(self.args("experiment0"), self.root)
        probe.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_experiment1_requires_completed_experiment0(self):
        for name in ("experiment1", "experiment2"):
            with self.subTest(experiment=name):
                with patch.dict(sys.modules, {"experiment_version": self.version_module}), patch.object(launcher, "preflight") as probe:
                    with self.assertRaises(FileNotFoundError):
                        launcher.run(self.args(name), self.root)
                probe.assert_not_called()
                self.assertFalse((self.root / "runs" / name).exists())

    def test_matching_baseline_is_accepted(self):
        self.baseline_fixture()
        record = launcher.check_baseline(self.root, self.baseline, self.config)
        self.assertEqual(record["accuracy"], 47 / 639)
        self.assertEqual(record["summary_sha256"], launcher.sha256(self.baseline / "validation/summary.json"))
        self.assertEqual(record["predictions_sha256"], launcher.sha256(self.baseline / "validation/predictions.jsonl"))
        self.assertEqual(record["checkpoint_sha256"], "a" * 64)

    def test_training_can_use_absolute_paths_for_the_same_inputs(self):
        self.baseline_fixture()
        config = dict(self.config)
        for key in ("data_dir", "tokenizer_path", "pretrained_checkpoint"):
            config[key] = str((self.root / config[key]).resolve())
        record = launcher.check_baseline(self.root, "runs/experiment0", config)
        self.assertEqual(record["accuracy"], 47 / 639)
        config["pretrained_checkpoint"] = str(self.root / "inputs/different.ckpt")
        with self.assertRaisesRegex(ValueError, "pretrained_checkpoint"):
            launcher.check_baseline(self.root, self.baseline, config)

    def test_config_input_prediction_or_evaluation_code_changes_are_rejected(self):
        self.baseline_fixture()
        cases = [
            ("runs/experiment0/result.json", lambda value: value.update(status="failed")),
            ("runs/experiment0/experiment.json", lambda value: value["config"]["evaluation"].update(steps=128)),
            ("runs/experiment0/experiment.json", lambda value: value.update(inputs_lock_sha256="b" * 64)),
            ("runs/experiment0/loss/summary.json", lambda value: value.update(code_samples=16)),
            ("runs/experiment0/validation/summary.json", lambda value: value.update(predictions_sha256="c" * 64)),
            ("runs/experiment0/validation/summary.json", lambda value: value.update(accuracy=0.9)),
            ("runs/experiment0/validation/summary.json", lambda value: value["checkpoint_source"].update(source="sft_ordinary")),
            ("runs/experiment0/validation/summary.json", lambda value: value["checkpoint_source"].update(checkpoint_sha256="b" * 64)),
        ]
        for relative, mutate in cases:
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                value = launcher.read_json(path)
                mutate(value)
                self.write(relative, value)
                with self.assertRaises(ValueError):
                    launcher.check_baseline(self.root, self.baseline, self.config)
                path.write_bytes(original)
        self.write("mdlm_sft/sampling.py", "modified sampling")
        with self.assertRaisesRegex(ValueError, "code changed"):
            launcher.check_baseline(self.root, self.baseline, self.config)

    def test_successful_process_exit_does_not_make_bounded_training_complete(self):
        self.baseline_fixture()
        def bounded(phase, root, env, log):
            self.write("runs/experiment1/training/result.json", {"status": "bounded_stop", "global_step": 20})
        with patch.dict(sys.modules, {"experiment_version": self.version_module}), patch.object(launcher, "preflight"), patch.object(launcher, "run_command", side_effect=bounded), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run(self.args(), self.root), 2)
        result = launcher.read_json(self.root / "runs/experiment1/result.json")
        self.assertEqual(result["status"], "bounded_stop")
        metadata = launcher.read_json(self.root / "runs/experiment1/experiment.json")
        self.assertEqual(metadata["baseline_reference"]["accuracy"], 47 / 639)
        self.assertIn("Reuse verified experiment0", metadata["baseline_policy"])
        self.assertIn("--baseline-dir", metadata["phases"][0]["command"])

    def test_full_training_report_must_match_609_steps_and_no_pending_validation(self):
        output = self.root / "runs/experiment1"
        correct = {"status": "complete", "smoke": False, "epoch": 3, "global_step": 609, "pending_validation": None}
        for change in ({"global_step": 2}, {"smoke": True}, {"pending_validation": {"generate": True}}):
            self.write("runs/experiment1/training/result.json", {**correct, **change})
            with self.assertRaises(ValueError):
                launcher.completed_result("experiment1", output, self.config, self.root)
        self.write("runs/experiment1/training/result.json", correct)
        self.assertEqual(launcher.completed_result("experiment1", output, self.config, self.root)["status"], "complete")

    def test_experiment2_reuses_baseline_and_accepts_only_full_five_epoch_completion(self):
        self.baseline_fixture()
        output = self.root / "runs/experiment2"
        config = launcher.read_json(self.root / "experiments/experiment2/config.json")
        correct = {"status": "complete", "smoke": False, "epoch": 5,
                   "global_step": 1015, "pending_validation": None}
        for change in ({"global_step": 609}, {"epoch": 3}, {"smoke": True},
                       {"pending_validation": {"generate": True}}):
            self.write("runs/experiment2/training/result.json", {**correct, **change})
            with self.assertRaises(ValueError):
                launcher.completed_result("experiment2", output, config, self.root)
        fresh_output = self.root / "runs/five-epochs"
        def completed(phase, root, env, log):
            self.write("runs/five-epochs/training/result.json", correct)
        with patch.dict(sys.modules, {"experiment_version": self.version_module}), patch.object(launcher, "preflight"), patch.object(launcher, "run_command", side_effect=completed), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run(self.args("experiment2", output_dir=str(fresh_output)), self.root), 0)
        metadata = launcher.read_json(fresh_output / "experiment.json")
        self.assertEqual(metadata["baseline_reference"]["accuracy"], 47 / 639)
        self.assertEqual([phase["name"] for phase in metadata["phases"]], ["training"])
        self.assertIn("Reuse verified experiment0", metadata["baseline_policy"])


if __name__ == "__main__":
    unittest.main()
