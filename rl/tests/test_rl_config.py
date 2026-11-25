"""Configuration contracts protect the agreed RL batch and evaluation protocol."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mdlm_rl.checkpoints import SCHEMA
from mdlm_rl.config import load_config, validate_config
from mdlm_rl.runtime import sha256
from scripts.configure_run import configure


ROOT = Path(__file__).resolve().parents[1]


class RLConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "experiments/experiment1/config.json").read_text(encoding="utf-8"))

    def test_configured_experiment_preserves_agreed_units_and_protocol(self):
        validate_config(self.config)
        config = self.config
        self.assertEqual((config["questions_per_batch"], config["samples_per_question"],
                          config["inner_updates"]), (8, 8, 4))
        self.assertEqual((config["world_size"], config["micro_batch_size"],
                          config["gradient_accumulation_steps"]), (2, 8, 4))
        self.assertEqual(config["max_steps"] // config["inner_updates"], 250)
        self.assertEqual(config["validation_every_steps"], 100)
        self.assertEqual(config["rollout"], {"length": 512, "steps": 128,
                                            "temperature": 0.1, "use_float64": True})
        self.assertEqual(config["estimator_temperature"], 1.0)
        self.assertEqual(config["initial_global_step"], 960)
        self.assertEqual((config["learning_rate"], config["clip_epsilon"], config["kl_beta"]),
                         (3e-6, 0.2, 0.04))
        self.assertEqual(config["data_dir"], "data/gsm8k")
        self.assertEqual(config["tokenizer_path"], "inputs/tokenizer")
        self.assertEqual(config["initial_checkpoint"],
                         "../sft/runs/experiment12/training/checkpoints/best.pt")
        self.assertEqual(config["output_dir"], "runs/experiment1")

    def test_unknown_missing_and_duplicate_keys_fail_closed(self):
        for change in ("unknown", "missing", "nested"):
            with self.subTest(change=change):
                value = copy.deepcopy(self.config)
                if change == "unknown":
                    value["inner_upates"] = 8
                elif change == "missing":
                    del value["inner_updates"]
                else:
                    value["rollout"]["top_k"] = 1
                with self.assertRaisesRegex(ValueError, "keys"):
                    validate_config(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)

    def test_batch_inconsistency_and_mid_rollout_boundaries_rejected(self):
        changes = {"questions_per_batch": 7, "samples_per_question": 1,
                   "gradient_accumulation_steps": 2, "max_steps": 999,
                   "validation_every_steps": 99, "checkpoint_every_steps": 99}
        for key, value in changes.items():
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_invalid_numeric_inputs_are_not_silently_coerced(self):
        for key, values in {
            "inner_updates": (True, 4.0, 0),
            "seed": (False, -1, 2**32),
            "learning_rate": (True, 0, float("nan"), float("inf"), 10**1000),
            "prompt_mask_probability": (-0.01, 1.01, float("nan")),
            "clip_epsilon": (0, 1, float("nan")),
            "kl_beta": (-1, float("inf")),
            "warmup_steps": (-1, 1001),
            "max_wall_seconds": (0, float("inf")),
            "estimator_temperature": (0.1, True),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(self.config)
                    config[key] = value
                    with self.assertRaises(ValueError):
                        validate_config(config)

    def test_decoding_drift_and_invalid_reward_limits_rejected(self):
        for key, value in (("temperature", 1.0), ("length", 256), ("use_float64", False),
                           ("steps", 32), ("timeout_s", 0), ("memory_mb", True)):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config["evaluation"][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_validation_is_pure_and_does_not_require_input_files(self):
        before = copy.deepcopy(self.config)
        with patch("pathlib.Path.open", side_effect=AssertionError("No file reads allowed")):
            validate_config(self.config)
        self.assertEqual(self.config, before)

    def test_load_uses_project_root_and_preserves_explicit_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = copy.deepcopy(self.config)
            explicit_output = str((root / "external-results").resolve())
            config["output_dir"] = explicit_output
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with patch("mdlm_rl.config.ROOT", root):
                loaded = load_config("config.json")
            self.assertEqual(loaded["data_dir"], str((root / "data/gsm8k").resolve()))
            self.assertEqual(loaded["output_dir"], explicit_output)
            validate_config(loaded)

    def test_cpu_fixture_can_keep_the_same_effective_batch(self):
        config = copy.deepcopy(self.config)
        config.update(world_size=1, precision="fp32", gradient_accumulation_steps=8)
        validate_config(config)

    def test_local_config_binds_new_sft_weights_and_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            manifest = data / "manifest.json"
            manifest.write_text('{"experiment": 12}', encoding="utf-8")
            checkpoint = data / "best.pt"
            state = {"schema": SCHEMA, "progress": {"global_step": 768},
                     "identity": {"data_manifest_sha256": sha256(manifest)},
                     "model": {}, "optimizer": {}, "scheduler": {}, "rng_by_rank": [], "config": {}}
            torch.save(state, checkpoint)
            output = data / "local.json"
            configure(checkpoint, data, output)
            configure(checkpoint, data, output)
            config = load_config(output)
            self.assertEqual(Path(config["data_dir"]), data.resolve())
            self.assertEqual(config["initial_global_step"], 768)
            self.assertEqual(config["initial_checkpoint_sha256"], sha256(checkpoint))
            self.assertEqual(config["expected_data_manifest_sha256"], sha256(manifest))

            manifest.write_text('{"other": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset used"):
                configure(checkpoint, data, output)


if __name__ == "__main__":
    unittest.main()
