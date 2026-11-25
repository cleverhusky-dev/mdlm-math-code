"""Data-only comparisons keep experiment8's training and two-GPU launch policy."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_experiment_launcher", ROOT / "scripts/run_experiment.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class DataExperimentTests(unittest.TestCase):
    def setUp(self):
        self.configs = {
            name: json.loads((ROOT / "experiments" / name / "config.json").read_text(encoding="utf-8"))
            for name in ("experiment8", "experiment9", "experiment10", "experiment11", "experiment12")
        }

    def test_configs_change_only_dataset_and_output_from_experiment8(self):
        base = self.configs["experiment8"]
        for name, dataset in (("experiment10", "data/mix"), ("experiment11", "data/single"),
                              ("experiment12", "data/vars")):
            with self.subTest(experiment=name):
                config = self.configs[name]
                self.assertEqual(set(config), set(base))
                self.assertEqual({key for key in base if config[key] != base[key]},
                                 {"data_dir", "output_dir"})
                self.assertEqual(config["data_dir"], dataset)
                self.assertEqual(config["output_dir"], f"runs/{name}")
                self.assertEqual(config["learning_rate"], 1e-4)
                self.assertEqual(config["epochs"], 5)
                self.assertEqual(config["global_batch_size"], 64)
                self.assertEqual(config["micro_batch_size"], 8)
                self.assertIsNone(config["gradient_accumulation_steps"])
                self.assertEqual(config["global_batch_size"] // (2 * config["micro_batch_size"]), 4)

    def test_two_gpu_plan_fresh_initialization_reuses_baseline_without_evaluation(self):
        for name, port in (("experiment10", 29510), ("experiment11", 29511), ("experiment12", 29512)):
            with self.subTest(experiment=name):
                config = self.configs[name]
                plan = launcher.make_plan(ROOT, name, ROOT / "runs" / name, config, master_port=port)
                self.assertEqual([phase["name"] for phase in plan], ["training"])
                command = plan[0]["command"]
                for argument in ("torch.distributed.run", "--nproc-per-node=2", "--rdzv-backend=static",
                                 "--master-addr=127.0.0.1", f"--master-port={port}"):
                    self.assertIn(argument, command)
                for argument in ("--resume", "--smoke", "mdlm_sft.evaluate", "--nproc-per-node=1"):
                    self.assertNotIn(argument, command)
                self.assertEqual(command[command.index("--init") + 1],
                                 str((ROOT / "inputs/mdlm.ckpt").resolve()))
                self.assertEqual(command[command.index("--baseline-dir") + 1],
                                 str((ROOT / "runs/experiment0").resolve()))
                self.assertEqual(command[command.index("--config") + 1],
                                 str(ROOT / "experiments" / name / "config.json"))

    def test_experiment10_data_metadata_matches_config_and_experiment8_parent(self):
        data = json.loads((ROOT / "experiments/experiment10/data.json").read_text(encoding="utf-8"))
        config = self.configs["experiment10"]
        planned = data["planned_training"]
        self.assertEqual(data["parent"]["experiment"], "experiment8")
        self.assertEqual(planned["base_experiment"], "experiment8")
        for key in ("learning_rate", "epochs", "global_batch_size"):
            self.assertEqual(planned[key], config[key], key)
        self.assertIs(planned["training_config_created"], True)

    def test_cli_accepts_data_experiments(self):
        for name in ("experiment10", "experiment11", "experiment12"):
            with self.subTest(experiment=name), patch.object(launcher.sys, "argv", [
                    "run_experiment.py", name, "--dry-run"]), patch.object(launcher, "run", return_value=0) as run:
                with self.assertRaises(SystemExit) as exit_result:
                    launcher.main()
                self.assertEqual(exit_result.exception.code, 0)
                self.assertEqual(run.call_args.args[0].experiment, name)
                self.assertTrue(run.call_args.args[0].dry_run)

    def test_preflight_requires_two_gpus_and_checks_training_and_validation_locks(self):
        for name, dataset in (("experiment10", "mix"), ("experiment11", "single"), ("experiment12", "vars")):
            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "experiments" / name).mkdir(parents=True)
                data = root / "data" / dataset
                data.mkdir(parents=True)
                files = []
                for filename in ("train.npz", "val.npz"):
                    path = data / filename
                    path.write_bytes(b"fixture")
                    files.append({"path": path.relative_to(root).as_posix(), "size_bytes": 7,
                                  "sha256": launcher.sha256(path)})
                (root / "experiments" / name / "data.json").write_text(
                    json.dumps({"files": files}), encoding="utf-8")
                with patch.object(launcher.sys, "platform", "linux"), patch.object(launcher.subprocess, "run") as run:
                    launcher.preflight(root, name, {})
                    self.assertIn("device_count() >= 2", run.call_args_list[0].args[0][-1])
                    for filename in ("train.npz", "val.npz"):
                        with self.subTest(changed_file=filename):
                            path = data / filename
                            path.write_bytes(b"changed")
                            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                                launcher.preflight(root, name, {})
                            path.write_bytes(b"fixture")

    def test_completion_requires_dataset_derived_steps_including_single_tail_batch(self):
        for name, samples, expected_steps in (("experiment10", 15360, 1200),
                                               ("experiment11", 5747, 450),
                                               ("experiment12", 12288, 960)):
            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = self.configs[name]
                data = root / config["data_dir"]
                data.mkdir(parents=True)
                (data / "manifest.json").write_text(
                    json.dumps({"counts": {"train": {"samples": samples}}}), encoding="utf-8")
                output = root / "runs" / name
                (output / "training").mkdir(parents=True)
                path = output / "training/result.json"
                result = {"status": "complete", "smoke": False, "epoch": 5,
                          "global_step": expected_steps, "pending_validation": None}
                changes = [{"global_step": value} for value in (expected_steps - 1, 450, 960, 1200)
                           if value != expected_steps]
                if name == "experiment11":
                    changes.append({"global_step": (samples // 64) * 5})
                changes += [{"epoch": 4}, {"smoke": True}, {"pending_validation": {"generate": True}}]
                for change in changes:
                    with self.subTest(change=change):
                        path.write_text(json.dumps({**result, **change}), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            launcher.completed_result(name, output, config, root)
                path.write_text(json.dumps({**result, "status": "bounded_stop"}), encoding="utf-8")
                self.assertEqual(launcher.completed_result(name, output, config, root)["status"], "bounded_stop")
                path.write_text(json.dumps(result), encoding="utf-8")
                self.assertEqual(launcher.completed_result(name, output, config, root)["status"], "complete")

    def test_experiment9_keeps_two_gpus_and_existing_default_port(self):
        command = launcher.make_plan(ROOT, "experiment9", ROOT / "runs/experiment9",
                                     self.configs["experiment9"])[0]["command"]
        self.assertIn("--nproc-per-node=2", command)
        self.assertIn("--master-port=29501", command)
        self.assertEqual(self.configs["experiment9"]["learning_rate"], 5e-4)


if __name__ == "__main__":
    unittest.main()
