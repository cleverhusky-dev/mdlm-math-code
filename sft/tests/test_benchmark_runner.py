"""Synthetic final-test orchestration fixtures; never open official test examples."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_benchmark as runner
from scripts.freeze_benchmark import benchmark_for_run

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads((ROOT / "benchmarks/gsm8k/config.json").read_text(encoding="utf-8"))
        self.base = json.loads((ROOT / "experiments/experiment12/config.json").read_text(encoding="utf-8"))
        for identity, payload in [(self.config["test"], b"synthetic parquet placeholder"),
                                  *[(model, model["id"].encode()) for model in self.config["models"]]]:
            path = self.root / identity.get("path", identity.get("checkpoint"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            identity["sha256"] = runner.sha256(path)
        (self.root / self.config["source_config"]).parent.mkdir(parents=True, exist_ok=True)
        (self.root / "benchmarks/gsm8k").mkdir(parents=True)
        runner.write_json(self.root / self.config["source_config"], self.base)
        runner.write_json(self.root / "benchmarks/gsm8k/config.json", self.config)
        runner.write_json((self.root / self.config["models"][1]["checkpoint"]).parent / "best.json", {"step": 960})
        self.output = self.root / self.config["output_dir"]
        self.version = {"commit": "fixture", "dirty": False, "tags": [], "tracked_files_sha256": {
            "mdlm_sft/" + name: runner.fingerprint(name) for name in runner.EVALUATION_SOURCES}}
        self.environment = {"executor": {"scorer_version": "fixture-scorer"}}
        self.args = argparse.Namespace(config="benchmarks/gsm8k/config.json", output_dir=None,
                                       resume=False, dry_run=False)
        self.calls = []

    def test_freeze_selects_retrained_validation_winner_without_opening_test(self):
        run_dir = self.root / "runs/experiment12"
        training = run_dir / "training"
        selected = {"step": 768, "accuracy": 120 / 639}
        runner.write_json(run_dir / "experiment.json", {"experiment": "experiment12"})
        runner.write_json(training / "result.json", {"status": "complete", "smoke": False,
                          "best_step": 768, "best_accuracy": selected["accuracy"]})
        runner.write_json(training / "checkpoints/best.json", selected)
        validation = training / "validation/step-00000768"
        validation.mkdir(parents=True)
        summary = {"complete": True, "label": "validation", "question_count": 639,
                   "accuracy": selected["accuracy"], "checkpoint_source": {
                       "source": "sft_ordinary", "global_step": 768, "model_sha256": "c" * 64}}
        runner.write_json(validation / "summary.json", summary)
        (self.root / self.config["test"]["path"]).unlink()
        config = benchmark_for_run(self.root, run_dir)
        self.assertEqual(config["models"][1], {**self.config["models"][1], "global_step": 768,
                                               "model_sha256": "c" * 64})
        self.assertEqual(config["settings"], self.config["settings"])
        summary["checkpoint_source"]["global_step"] = 960
        runner.write_json(validation / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "do not agree"):
            benchmark_for_run(self.root, run_dir)

    def fake_evaluate(self, phase, root, env, log):
        command = phase["command"]
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir()
        frozen = runner.read_json(self.output / "frozen.json")
        case = next(case for case in frozen["cases"] if case["config_path"] == command[command.index("--config") + 1])
        self.calls.append(case["id"])
        scorer = self.environment["executor"]["scorer_version"]
        protocol = {"generation": case["config"]["evaluation"], "precision": "bf16",
                    "numerics": {"noise_eps": 0.0}, "tokenizer": {"name": "fixture"},
                    "scorer_version": scorer, "questions_sha256": "f" * 64,
                    "question_order": [f"test:{i}" for i in range(1319)],
                    "evaluation_source": {name: self.version["tracked_files_sha256"]["mdlm_sft/" + name]
                                          for name in runner.EVALUATION_SOURCES}}
        protocol["protocol_sha256"] = runner.fingerprint(protocol)
        runner.write_json(destination / "protocol.json", protocol)
        runner.write_json(destination / "config.resolved.json", case["config"])
        runner.write_json(destination / "test_source.json", {"sha256": self.config["test"]["sha256"],
                                                            "frozen_protocol_acknowledged": True})
        runner.write_json(destination / "usage.json", {"status": "complete"})
        predictions = destination / "predictions.jsonl"
        predictions.write_text("".join(json.dumps({"gsm8k_id": f"test:{i}", "status": "correct",
                                                   "scorer_version": scorer}) + "\n" for i in range(1319)), encoding="utf-8")
        model = case["model"]
        if model["kind"] == "pretrained":
            source = {"source": "pretrained_ema", "checkpoint_sha256": model["sha256"]}
        else:
            source = {"source": "sft_ordinary", "global_step": 960, "model_sha256": model["model_sha256"]}
            runner.write_json(destination / "checkpoint_file.json", {"sha256": model["sha256"]})
        summary = {"complete": True, "label": "final_test", "question_count": 1319, "num_questions": 1319,
                   "counts": {status: 1319 if status == "correct" else 0 for status in runner.STATUSES},
                   "accuracy": 1.0, "format_failure_rate": 0.0, "execution_failure_rate": 0.0,
                   "wrong_answer_rate": 0.0, "scorer_version": scorer, "checkpoint_source": source,
                   "protocol_sha256": protocol["protocol_sha256"], "predictions_sha256": runner.sha256(predictions)}
        runner.write_json(destination / "summary.json", summary)
        log.write("synthetic evaluation only\n")

    @contextlib.contextmanager
    def mocked_runtime(self, execute=None):
        with patch.object(runner, "capture_version", return_value=self.version), \
                patch.object(runner, "preflight", return_value=self.environment), \
                patch.object(runner, "run_command", side_effect=execute or self.fake_evaluate) as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            yield launch

    def test_six_cases_inherit_defaults_and_use_frozen_test_flags(self):
        runner.validate_config(self.config, self.base)
        cases = runner.case_specs(self.root, self.output, self.config, self.base)
        self.assertEqual([case["id"] for case in cases], [
            "t1-s32-pretrained", "t1-s32-sft", "t01-s32-pretrained", "t01-s32-sft",
            "t01-s128-pretrained", "t01-s128-sft"])
        for case in cases:
            expected = copy.deepcopy(self.base)
            expected["evaluation"].update(temperature=case["config"]["evaluation"]["temperature"],
                                           steps=case["config"]["evaluation"]["steps"])
            expected["output_dir"] = case["config"]["output_dir"]
            self.assertEqual(case["config"], expected)
            command = runner.command_for(self.root, self.output, self.config, case, self.output / "attempt")
            self.assertIn("--confirm-frozen-protocol", command)
            self.assertEqual(command[command.index("--split") + 1], "test")
            self.assertNotIn("--resume", command)
        changed = copy.deepcopy(self.base)
        changed["evaluation"]["seed"] = 2
        with self.assertRaisesRegex(ValueError, "approved defaults"):
            runner.validate_config(self.config, changed)

    def test_dry_run_has_no_runtime_input_or_output_access(self):
        self.args.dry_run = True
        with patch.object(runner, "checked_assets") as assets, patch.object(runner, "capture_version") as version, \
                patch.object(runner, "preflight") as preflight, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(self.args, self.root), 0)
            assets.assert_not_called()
            version.assert_not_called()
            preflight.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_dirty_checkout_stops_before_input_hashing_or_output(self):
        with patch.object(runner, "capture_version", side_effect=RuntimeError("Uncommitted changes")), \
                patch.object(runner, "checked_assets") as assets:
            with self.assertRaisesRegex(RuntimeError, "Uncommitted"):
                runner.run(self.args, self.root)
            assets.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_complete_resume_checks_evidence_and_changed_frozen_input(self):
        with self.mocked_runtime() as launch:
            self.assertEqual(runner.run(self.args, self.root), 0)
            self.assertEqual(launch.call_count, 6)
            self.assertEqual(runner.read_json(self.output / "status.json")["completed_cases"], 6)
            self.args.resume = True
            launch.reset_mock()
            self.assertEqual(runner.run(self.args, self.root), 0)
            launch.assert_not_called()
            test_path = self.root / self.config["test"]["path"]
            original = test_path.read_bytes()
            test_path.write_bytes(b"changed input")
            with self.assertRaisesRegex(ValueError, "input SHA256 mismatch"):
                runner.run(self.args, self.root)
            test_path.write_bytes(original)
            prediction = self.output / "cases/t1-s32-pretrained/attempt-001/predictions.jsonl"
            prediction.write_text("changed evidence", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed case evidence changed"):
                runner.run(self.args, self.root)
            launch.assert_not_called()

    def test_failure_stops_then_resume_retries_without_overwriting_attempt(self):
        def fail_second(phase, root, env, log):
            if len(self.calls) == 1:
                raise RuntimeError("synthetic interruption")
            self.fake_evaluate(phase, root, env, log)
        with self.mocked_runtime(fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                runner.run(self.args, self.root)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(runner.read_json(self.output / "status.json")["status"], "failed")
        first_log = self.output / "cases/t1-s32-sft/attempt-001.log"
        before = first_log.read_bytes()
        self.args.resume = True
        with self.mocked_runtime() as launch:
            self.assertEqual(runner.run(self.args, self.root), 0)
            self.assertEqual(launch.call_count, 5)
        self.assertEqual(first_log.read_bytes(), before)
        receipt = runner.read_json(self.output / "cases/t1-s32-sft/receipt.json")
        self.assertEqual(receipt["attempt"], "attempt-002")

    def test_incomplete_case_or_cross_case_protocol_change_fails(self):
        def incomplete(phase, root, env, log):
            self.fake_evaluate(phase, root, env, log)
            command = phase["command"]
            destination = Path(command[command.index("--output-dir") + 1])
            runner.write_json(destination / "usage.json", {"status": "failed"})
        with self.mocked_runtime(incomplete):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.run(self.args, self.root)
        self.assertFalse((self.output / "cases/t1-s32-pretrained/receipt.json").exists())
        self.args.resume = True
        def changed_tokenizer(phase, root, env, log):
            self.fake_evaluate(phase, root, env, log)
            command = phase["command"]
            if command[command.index("--kind") + 1] != "sft":
                return
            destination = Path(command[command.index("--output-dir") + 1])
            protocol = runner.read_json(destination / "protocol.json")
            protocol["tokenizer"]["name"] = "different"
            protocol.pop("protocol_sha256")
            protocol["protocol_sha256"] = runner.fingerprint(protocol)
            runner.write_json(destination / "protocol.json", protocol)
            summary = runner.read_json(destination / "summary.json")
            summary["protocol_sha256"] = protocol["protocol_sha256"]
            runner.write_json(destination / "summary.json", summary)
        with self.mocked_runtime(changed_tokenizer):
            with self.assertRaisesRegex(ValueError, "differ beyond"):
                runner.run(self.args, self.root)


if __name__ == "__main__":
    unittest.main()
