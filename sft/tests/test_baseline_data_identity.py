"""A training-data change may reuse a baseline only with unchanged held-out data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mdlm_sft.baseline import check_baseline, read_json, sha256, VALIDATION_FILES


class BaselineDataIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        project = Path(__file__).resolve().parents[1]
        self.original_config = read_json(project / "experiments/experiment1/config.json")
        self.config = {**self.original_config, "data_dir": "data/style"}
        questions = [{"gsm8k_id": f"train:{i}", "question": f"Question {i}", "gold_answer": i}
                     for i in range(639)]
        self.write("data/prepared/val_questions.jsonl", "".join(json.dumps(q) + "\n" for q in questions))
        for name in VALIDATION_FILES[:-1]:
            self.write(f"data/prepared/{name}", f"held-out fixture: {name}\n")
        hashes = {name: sha256(self.root / "data/prepared" / name) for name in VALIDATION_FILES}
        manifest = {"counts": {"train": {"samples": 12953},
                              "val": {"questions": 639, "samples": 1456, "target_tokens": 655019}},
                    "sequence_length": 512, "tokenizer": {"model_vocab_size": 49154},
                    "artifacts_sha256": hashes}
        self.write("data/prepared/manifest.json", manifest)
        for name in VALIDATION_FILES:
            self.write(f"data/style/{name}", (self.root / "data/prepared" / name).read_text())
        manifest["counts"]["train"]["samples"] = 12288
        self.write("data/style/manifest.json", manifest)
        self.write("inputs.lock.json", {"files": [
            {"path": "inputs/mdlm.ckpt", "sha256": "a" * 64},
            {"path": "data/prepared/manifest.json", "sha256": sha256(self.root / "data/prepared/manifest.json")}]})
        source = {"source": "pretrained_ema", "checkpoint_sha256": "a" * 64}
        self.write("runs/experiment0/result.json", {"experiment": "experiment0", "status": "complete"})
        self.write("runs/experiment0/experiment.json", {"experiment": "experiment0", "config": self.original_config,
                   "inputs_lock_sha256": sha256(self.root / "inputs.lock.json"), "git": {"commit": "fixture"}})
        self.write("runs/experiment0/loss/summary.json", {
            "complete": True, "code_samples": 1456, "target_tokens": 655019, "loss": 1.1,
            "checkpoint_source": source, "data_manifest_sha256": sha256(self.root / "data/prepared/manifest.json")})
        code = {}
        for name in ("evaluation.py", "sampling.py", "scoring.py", "executor.py", "data.py",
                     "diffusion.py", "transformer.py", "runtime.py"):
            code[name] = sha256(self.write(f"mdlm_sft/{name}", "fixture source"))
        self.protocol = {"generation": self.config["evaluation"], "precision": "bf16", "evaluation_source": code,
                         "tokenizer": manifest["tokenizer"], "question_order": [q["gsm8k_id"] for q in questions],
                         "questions_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        predictions = self.write("runs/experiment0/validation/predictions.jsonl", "fixture predictions\n")
        self.score = {"complete": True, "num_questions": 639, "question_count": 639, "counts": {"correct": 47},
                      "accuracy": 47 / 639, "checkpoint_source": source, "predictions_sha256": sha256(predictions)}
        self.save_protocol()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return path

    def save_protocol(self):
        digest = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
        self.write("runs/experiment0/validation/protocol.json", {**self.protocol, "protocol_sha256": digest})
        self.write("runs/experiment0/validation/summary.json", {**self.score, "protocol_sha256": digest})

    def check(self):
        return check_baseline(self.root, "runs/experiment0", self.config)

    def test_deduplicated_training_with_identical_validation_reuses_baseline(self):
        record = self.check()
        self.assertEqual(record["accuracy"], 47 / 639)
        identity = record["validation_identity"]
        self.assertEqual(identity["data_manifest_sha256"], sha256(self.root / "data/style/manifest.json"))
        self.assertNotEqual(identity["data_manifest_sha256"], identity["baseline_manifest_sha256"])
        self.assertEqual(set(identity["validation_artifacts_sha256"]), set(VALIDATION_FILES))

    def test_each_changed_validation_file_is_rejected_even_if_its_new_manifest_is_updated(self):
        for name in VALIDATION_FILES:
            with self.subTest(name=name):
                path = self.root / "data/style" / name
                original = path.read_bytes()
                manifest = read_json(self.root / "data/style/manifest.json")
                changed = deepcopy(manifest)
                path.write_bytes(original + b"tampered")
                changed["artifacts_sha256"][name] = sha256(path)
                self.write("data/style/manifest.json", changed)
                with self.assertRaisesRegex(ValueError, "Validation artifact"):
                    self.check()
                path.write_bytes(original)
                self.write("data/style/manifest.json", manifest)

    def test_stale_manifest_does_not_hide_changed_file(self):
        self.write("data/style/val.npz", "different cache")
        with self.assertRaisesRegex(ValueError, "val.npz"):
            self.check()

    def test_original_manifest_must_match_recorded_loss_identity(self):
        path = self.root / "runs/experiment0/loss/summary.json"
        value = read_json(path)
        value["data_manifest_sha256"] = "b" * 64
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "recorded, locked"):
            self.check()

    def test_counts_or_encoding_changes_are_rejected(self):
        path = self.root / "data/style/manifest.json"
        original = read_json(path)
        for change in ("counts", "sequence_length", "tokenizer"):
            with self.subTest(change=change):
                value = deepcopy(original)
                if change == "counts":
                    value["counts"]["val"]["samples"] = 1455
                else:
                    value[change] = None
                self.write(path, value)
                with self.assertRaisesRegex(ValueError, "Validation data"):
                    self.check()
                self.write(path, original)

    def test_question_protocol_is_verified_beyond_file_and_manifest_agreement(self):
        self.protocol["question_order"].reverse()
        self.save_protocol()
        with self.assertRaisesRegex(ValueError, "questions differ"):
            self.check()

    def test_generation_and_source_guards_still_apply_to_new_training_data(self):
        self.config["evaluation"] = {**self.config["evaluation"], "steps": 128}
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.check()
        self.config["evaluation"] = self.original_config["evaluation"]
        self.write("mdlm_sft/sampling.py", "changed source")
        with self.assertRaisesRegex(ValueError, "Evaluation code changed"):
            self.check()


if __name__ == "__main__":
    unittest.main()
