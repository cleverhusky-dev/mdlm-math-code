"""Launcher contracts tested without starting workers, GPUs or code execution."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import run_rl
from mdlm_rl.config import apply_run_overrides, load_config


class RLLauncherTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("experiments/experiment1/config.json")

    def test_dry_run_checks_inputs_without_output_or_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            stream = io.StringIO()
            with patch.object(run_rl, "inspect_inputs", return_value={"train_questions": 5747}) as inspect, \
                    patch.object(run_rl, "readiness") as ready, \
                    patch.object(run_rl, "run_child") as launch, contextlib.redirect_stdout(stream):
                status = run_rl.main(["--dry-run", "--output-dir", str(output)])
            self.assertEqual(status, 0)
            self.assertEqual(inspect.call_args.args[0]["initial_checkpoint_sha256"],
                             self.config["initial_checkpoint_sha256"])
            self.assertEqual(json.loads(stream.getvalue())["status"], "inputs_verified")
            ready.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_input_identity_never_launches_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            with patch.object(run_rl, "inspect_inputs", side_effect=ValueError("SHA256 mismatch")), \
                    patch.object(run_rl, "readiness") as ready, patch.object(run_rl, "run_child") as launch:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    run_rl.main(["--output-dir", str(output)])
            ready.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_preflight_verifies_runtime_but_does_not_start_or_lock_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            with patch.object(run_rl, "inspect_inputs", return_value={}) as inspect, \
                    patch.object(run_rl, "readiness", return_value={"world_size": 2}) as ready, \
                    patch.object(run_rl, "run_child") as launch, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_rl.main(["--preflight-only", "--output-dir", str(output)]), 0)
            inspect.assert_called_once()
            ready.assert_called_once()
            self.assertEqual(ready.call_args.args[1], "cuda")
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_command_preserves_explicit_overrides_and_one_two_rank_job(self):
        args = run_rl.argument_parser().parse_args([
            "--resume", "runs/experiment1/checkpoints/last.pt", "--max-steps", "1200",
            "--max-wall-seconds", "3600", "--output-dir", "runs/experiment1"])
        command = run_rl.training_command(args, self.config)
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=2", command)
        self.assertIn("--local-addr=127.0.0.1", command)
        self.assertEqual(command[command.index("--module") + 1], "mdlm_rl.train")
        for flag, expected in (("--max-steps", "1200"), ("--max-wall-seconds", "3600.0"),
                               ("--device", "cuda")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertTrue(Path(command[command.index("--config") + 1]).is_absolute())
        self.assertEqual(command[command.index("--resume") + 1],
                         str(Path(args.resume).resolve()))

    def test_lock_is_exclusive_and_released_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            lock = Path(directory) / ".run.lock"
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                with run_rl.output_lock(output):
                    self.assertFalse(output.exists())
                    self.assertTrue(lock.exists())
                    with self.assertRaisesRegex(RuntimeError, "Another job owns"):
                        with run_rl.output_lock(output):
                            self.fail("Two writers acquired the same run")
                    self.assertTrue(lock.exists())
                    raise RuntimeError("worker failed")
            self.assertFalse(lock.exists())
            with run_rl.output_lock(output):
                self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())

    def test_lock_does_not_delete_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            lock = Path(directory) / ".run.lock"
            with run_rl.output_lock(output):
                lock.write_text(json.dumps({"token": "different-owner"}), encoding="utf-8")
            self.assertEqual(json.loads(lock.read_text())["token"], "different-owner")

    def test_launch_holds_lock_for_entire_child_and_propagates_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            lock = Path(directory) / ".run.lock"

            def child(command):
                self.assertTrue(lock.is_file())
                self.assertIn("--nproc-per-node=2", command)
                return 2

            with patch.object(run_rl, "inspect_inputs", return_value={}), \
                    patch.object(run_rl, "readiness", return_value={}), \
                    patch.object(run_rl, "run_child", side_effect=child) as launch, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_rl.main(["--output-dir", str(output)]), 2)
            launch.assert_called_once()
            self.assertFalse(lock.exists())

    def test_shared_overrides_preserve_original_and_bound_smoke(self):
        original = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            result = apply_run_overrides(self.config, smoke=True, device="cpu",
                                         output_dir=str(Path(directory) / "smoke"))
        self.assertEqual(self.config, original)
        self.assertEqual((result["max_steps"], result["warmup_steps"],
                          result["validation_every_steps"], result["checkpoint_every_steps"]),
                         (8, 2, 4, 4))
        self.assertEqual(result["rollout"], original["rollout"])
        self.assertEqual(result["max_steps"] // result["inner_updates"], 2)
        self.assertEqual(Path(apply_run_overrides(self.config, smoke=True)["output_dir"]).name,
                         "experiment1-smoke")
        with self.assertRaisesRegex(ValueError, "separate from the training directory"):
            apply_run_overrides(self.config, smoke=True, output_dir=self.config["output_dir"])
        with self.assertRaisesRegex(ValueError, "CPU execution"):
            apply_run_overrides(self.config, device="cpu")
        with self.assertRaisesRegex(ValueError, "multiple of inner_updates"):
            apply_run_overrides(self.config, max_steps=999)

    def test_missing_second_gpu_fails_before_executor_runs(self):
        with patch.object(run_rl, "validate_environment"), \
                patch.object(run_rl.platform, "system", return_value="Linux"), \
                patch.object(run_rl.dist, "is_available", return_value=True), \
                patch.object(run_rl.dist, "is_gloo_available", return_value=True), \
                patch.object(run_rl.torch.cuda, "is_available", return_value=True), \
                patch.object(run_rl.torch.cuda, "device_count", return_value=1), \
                patch.object(run_rl, "preflight_executor") as executor:
            with self.assertRaisesRegex(RuntimeError, "Two visible CUDA GPUs"):
                run_rl.readiness(self.config, "cuda")
            executor.assert_not_called()

    def test_child_uses_project_root_and_clean_environment(self):
        process = MagicMock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(run_rl.subprocess, "Popen", return_value=process) as popen, \
                patch.dict(run_rl.os.environ, {"PYTHONPATH": "untrusted", "PYTHONHOME": "wrong"}), \
                patch.object(run_rl.signal, "signal", return_value=run_rl.signal.SIG_DFL):
            self.assertEqual(run_rl.run_child(["python", "mock-worker"]), 0)
        self.assertEqual(popen.call_args.kwargs["cwd"], run_rl.ROOT)
        environment = popen.call_args.kwargs["env"]
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")


if __name__ == "__main__":
    unittest.main()
