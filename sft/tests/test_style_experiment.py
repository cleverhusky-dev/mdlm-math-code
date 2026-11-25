"""Style-data experiments retain the baseline protocol and require full runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("style_launcher", ROOT / "scripts/run_experiment.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class StyleExperimentTests(unittest.TestCase):
    def setUp(self):
        self.configs = {name: json.loads((ROOT / "experiments" / name / "config.json").read_text())
                        for name in ("experiment6", "experiment7", "experiment8", "experiment9")}

    def test_only_data_and_output_differ_from_experiment1(self):
        original = json.loads((ROOT / "experiments/experiment1/config.json").read_text())
        config = self.configs["experiment6"]
        changes = {key for key in original if original[key] != config[key]}
        self.assertEqual(changes, {"data_dir", "output_dir"})
        self.assertEqual(config["data_dir"], "data/style")

    def test_experiment7_changes_only_epochs_and_output(self):
        original, extended = self.configs["experiment6"], self.configs["experiment7"]
        self.assertEqual(set(original), set(extended))
        self.assertEqual({key for key in original if original[key] != extended[key]},
                         {"epochs", "output_dir"})
        self.assertEqual(extended["epochs"], 5)
        self.assertEqual(extended["output_dir"], "runs/experiment7")

    def test_experiment8_changes_only_learning_rate_and_output(self):
        original, higher_lr = self.configs["experiment7"], self.configs["experiment8"]
        self.assertEqual(set(original), set(higher_lr))
        self.assertEqual({key for key in original if original[key] != higher_lr[key]},
                         {"learning_rate", "output_dir"})
        self.assertEqual(original["learning_rate"], 3e-5)
        self.assertEqual(higher_lr["learning_rate"], 1e-4)
        self.assertEqual(higher_lr["epochs"], 5)
        self.assertEqual(higher_lr["data_dir"], "data/style")
        self.assertEqual(higher_lr["output_dir"], "runs/experiment8")

    def test_uses_two_gpu_fresh_pretrained_initialization_and_existing_baseline(self):
        for name, config in self.configs.items():
            with self.subTest(experiment=name):
                plan = launcher.make_plan(ROOT, name, ROOT / "runs" / name, config)
                self.assertEqual([phase["name"] for phase in plan], ["training"])
                command = plan[0]["command"]
                self.assertIn("--nproc-per-node=2", command)
                self.assertNotIn("--resume", command)
                self.assertNotIn("--smoke", command)
                self.assertEqual(command[command.index("--init") + 1], str((ROOT / "inputs/mdlm.ckpt").resolve()))
                self.assertEqual(command[command.index("--baseline-dir") + 1], str((ROOT / "runs/experiment0").resolve()))

    def test_experiment9_changes_only_learning_rate_and_output(self):
        original, higher_lr = self.configs["experiment8"], self.configs["experiment9"]
        self.assertEqual(set(original), set(higher_lr))
        self.assertEqual({key for key in original if original[key] != higher_lr[key]},
                         {"learning_rate", "output_dir"})
        self.assertEqual(original["learning_rate"], 1e-4)
        self.assertEqual(higher_lr["learning_rate"], 5e-4)
        self.assertEqual(higher_lr["epochs"], 5)
        self.assertEqual(higher_lr["data_dir"], "data/style")
        self.assertEqual(higher_lr["output_dir"], "runs/experiment9")
        self.assertEqual((ROOT / "experiments/experiment9/data.json").read_bytes(),
                         (ROOT / "experiments/experiment8/data.json").read_bytes())

    def test_completion_requires_all_epochs_and_dataset_derived_updates(self):
        for name, expected_steps, wrong_steps in (("experiment6", 576, (192, 575, 609)),
                                                  ("experiment7", 960, (576, 959, 1015)),
                                                  ("experiment8", 960, (576, 959, 1015)),
                                                  ("experiment9", 960, (576, 959, 1015))):
            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as directory:
                config = self.configs[name]
                root = Path(directory)
                data = root / "data/style"
                output = root / "runs" / name
                data.mkdir(parents=True)
                (output / "training").mkdir(parents=True)
                (data / "manifest.json").write_text(json.dumps({"counts": {"train": {"samples": 12288}}}))
                result = {"status": "complete", "smoke": False, "epoch": config["epochs"],
                          "global_step": expected_steps, "pending_validation": None}
                path = output / "training/result.json"
                changes = [{"global_step": steps} for steps in wrong_steps]
                changes += [{"epoch": config["epochs"] - 1}, {"smoke": True},
                            {"pending_validation": {"generate": True}}]
                for change in changes:
                    with self.subTest(change=change):
                        path.write_text(json.dumps({**result, **change}))
                        with self.assertRaises(ValueError):
                            launcher.completed_result(name, output, config, root)
                path.write_text(json.dumps({**result, "status": "bounded_stop"}))
                self.assertEqual(launcher.completed_result(name, output, config, root)["status"], "bounded_stop")
                path.write_text(json.dumps(result))
                self.assertEqual(launcher.completed_result(name, output, config, root)["status"], "complete")

    def test_preflight_checks_two_gpus_and_rejects_modified_dataset(self):
        for name in self.configs:
            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "experiments" / name).mkdir(parents=True)
                (root / "data/style").mkdir(parents=True)
                data = root / "data/style/train.npz"
                data.write_bytes(b"fixture")
                lock = {"files": [{"path": "data/style/train.npz", "size_bytes": 7,
                                   "sha256": launcher.sha256(data)}]}
                (root / "experiments" / name / "data.json").write_text(json.dumps(lock))
                with patch.object(launcher.sys, "platform", "linux"), patch.object(launcher.subprocess, "run") as run:
                    launcher.preflight(root, name, {})
                    self.assertIn("device_count() >= 2", run.call_args_list[0].args[0][-1])
                    data.write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        launcher.preflight(root, name, {})


if __name__ == "__main__":
    unittest.main()
