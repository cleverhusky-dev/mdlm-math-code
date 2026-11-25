"""CPU-only checks for the outcome-blind, read-only experiment5 probe."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))
import generalization_probe as probe
import run_generalization_probe as launcher
from mdlm_sft.data import SEPARATOR, encode_example


def reference(index, question_tokens=10, code_tokens=20, **changes):
    row = {"gsm8k_id": f"train:{index}", "question_tokens": question_tokens,
           "code_tokens": code_tokens, "total_tokens": question_tokens + code_tokens + 4}
    return {**row, **changes}


class SelectionTests(unittest.TestCase):
    def test_seed_five_uses_numeric_ids_and_is_independent_of_row_order_and_global_rng(self):
        train = [reference(i) for i in range(200, 208)]
        val = [reference(i) for i in (1, 2, 10, 11, 20, 21, 100, 101)]
        rng_before = random.getstate()
        selected = probe.select_questions(train, val, questions_per_split=3)
        self.assertEqual(selected["ids"]["val"], ["train:10", "train:20", "train:21"])
        self.assertEqual(selected["ids"]["train"], ["train:200", "train:201", "train:202"])
        self.assertEqual(selected, probe.select_questions(train[::-1], val[::-1], questions_per_split=3))
        self.assertEqual(random.getstate(), rng_before)
        self.assertTrue(set(selected["ids"]["train"]).isdisjoint(selected["ids"]["val"]))

    def test_length_match_breaks_ties_numerically_and_never_reuses_a_train_question(self):
        train = [reference(10, 10, 21), reference(30, 50, 20), reference(2, 11, 20)]
        val = [reference(101), reference(100)]
        selected = probe.select_questions(train, val, questions_per_split=2)
        self.assertEqual([(p["val_id"], p["train_id"]) for p in selected["pairs"]],
                         [("train:100", "train:2"), ("train:101", "train:10")])
        self.assertEqual([p["l1_distance"] for p in selected["pairs"]], [1, 1])
        self.assertEqual(selected["pairs"][0]["prefix_length_difference"], 1)
        self.assertEqual(selected["pairs"][1]["code_length_difference"], 1)

    def test_shortest_single_reference_controls_inclusive_256_token_eligibility(self):
        train = [reference(1, 10, 250), reference(1, 10, 242), reference(2, 10, 243)]
        val = [reference(100, 10, 241), reference(101, 10, 243)]
        selected = probe.select_questions(train, val, questions_per_split=1)
        self.assertEqual(selected["pool_counts"], {"train": 1, "val": 1})
        self.assertEqual(selected["ids"], {"train": ["train:1"], "val": ["train:100"]})
        self.assertEqual(selected["pairs"][0]["train"],
                         {"prefix_length": 13, "shortest_code_tokens": 242,
                          "shortest_total_tokens": 256})

    def test_multiple_references_do_not_fill_an_undersized_question_pool(self):
        one_question = [reference(1), reference(1, code_tokens=21)]
        two_questions = [reference(100), reference(101)]
        for train, val in ((one_question, two_questions), (two_questions, one_question)):
            with self.subTest(train_ids=[r["gsm8k_id"] for r in train]):
                with self.assertRaises(ValueError):
                    probe.select_questions(train, val, questions_per_split=2)

    def test_overlap_official_test_ids_and_inconsistent_token_audits_are_rejected(self):
        cases = (
            ([reference(1)], [reference(1)]),
            ([reference(1)], [reference(100, gsm8k_id="test:100")]),
            ([reference(1)], [reference(100, total_tokens=35)]),
            ([reference(1)], [reference(100), reference(100, question_tokens=11)]),
        )
        for train, val in cases:
            with self.subTest(train=train, val=val), self.assertRaises(ValueError):
                probe.select_questions(train, val, questions_per_split=1)

    def test_reference_audit_keeps_multiple_solutions_as_separate_loss_examples(self):
        class Tokenizer:
            bos_token_id = eos_token_id = 0
            pad_token_id = 99

            def __len__(self):
                return 100

            def encode(self, text, add_special_tokens=False):
                return [76, 94] if text == SEPARATOR else list(range(10, 10 + len(text.split())))

        class Row(list):
            def tolist(self):
                return list(self)

        class Dataset(SimpleNamespace):
            def __len__(self):
                return len(self.gsm8k_ids)

        tokenizer = Tokenizer()
        rows = {"train": [reference(1, 2, 2, question="Question one", code="a b", gold_answer=3),
                          reference(1, 2, 5, question="Question one", code="a b c d e", gold_answer=3)],
                "val": [reference(100, 2, 3, question="Question two", code="x y z", gold_answer=4)]}
        questions, datasets = {}, {}
        for split, members in rows.items():
            encoded = [encode_example(row, tokenizer) for row in members]
            questions[split] = [{key: members[0][key] for key in ("gsm8k_id", "question", "gold_answer")}]
            datasets[split] = Dataset(
                gsm8k_ids=[r["gsm8k_id"] for r in members],
                input_ids=[Row(r["input_ids"]) for r in encoded],
                target_mask=[Row(r["target_mask"]) for r in encoded],
                prefix_lengths=[r["prefix_length"] for r in encoded])
        selected = probe.select_questions(rows["train"], rows["val"], questions_per_split=1)
        indices, audit = probe.audit_references(None, datasets, questions, rows, selected, tokenizer)
        self.assertEqual(indices, {"train": [0, 1], "val": [0]})
        self.assertEqual(audit["splits"]["train"]["code_samples"], 2)
        self.assertEqual(audit["splits"]["train"]["code_tokens"], 7)
        self.assertEqual(audit["splits"]["train"]["eos_tokens"], 2)
        self.assertEqual(audit["splits"]["train"]["target_tokens"], 2 * (512 - 5))


class PredictionDiagnosticsTests(unittest.TestCase):
    def test_low_format_failure_does_not_hide_execution_failure_or_wrong_answers(self):
        rows = [
            {"status": "execution_failure", "error": "invalid_python: SyntaxError",
             "predicted_answer": None, "generated_token_ids": [0, 1]},
            {"status": "execution_failure", "error": "policy_rejection: Call",
             "predicted_answer": None, "generated_token_ids": [49153]},
            {"status": "wrong_answer", "predicted_answer": 7,
             "generated_token_ids": [2, 3, 4, 49152]},
            {"status": "correct", "predicted_answer": 4.5,
             "generated_token_ids": [6]},
        ]
        result = probe.prediction_diagnostics(rows, {0, 49152, 49153})
        self.assertEqual(result["counts"], {"execution_failure": 2, "wrong_answer": 1, "correct": 1})
        self.assertEqual(result["error_prefix_counts"], {"invalid_python": 1, "policy_rejection": 1})
        self.assertEqual(result["numeric_result_count"], 2)
        self.assertEqual(result["generated_token_count"], 8)
        self.assertEqual(result["generated_special_token_count"], 3)
        self.assertEqual(result["generated_special_token_fraction"], 3 / 8)
        self.assertIsNone(result["ast_parse"])

    def test_optional_ast_check_never_executes_generated_code_or_changes_scorer_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "must-not-exist"
            rows = [{"status": "execution_failure", "predicted_answer": None,
                     "generated_text": f"open({str(marker)!r}, 'w').write('executed')"},
                    {"status": "format_failure", "generated_text": "def broken(:"}]
            before = deepcopy(rows)
            with patch("builtins.exec", side_effect=AssertionError("diagnostics must not execute code")):
                result = probe.prediction_diagnostics(rows, {0}, parse_ast=True)
            self.assertFalse(marker.exists())
            self.assertEqual(rows, before)
            self.assertEqual(result["ast_parse"], {"parsable": 1, "unparsable": 1})
            self.assertEqual(result["counts"], {"execution_failure": 1, "format_failure": 1})
            self.assertEqual(result["numeric_result_count"], 0)
            self.assertIsNone(result["generated_special_token_fraction"])


class LauncherBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in ("experiments/experiment1/config.json", "experiments/experiment5/config.json"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((PROJECT / relative).read_bytes())

    def args(self, **changes):
        return argparse.Namespace(**{"config": "experiments/experiment5/config.json",
            "output_dir": None, "prepare_only": False, "dry_run": False, **changes})

    def test_dry_run_needs_no_inputs_or_gpu_and_writes_nothing(self):
        before = sorted(self.root.rglob("*"))
        output = io.StringIO()
        with patch.object(launcher, "capture_version") as version, contextlib.redirect_stdout(output):
            self.assertEqual(launcher.run(self.args(dry_run=True), self.root), 0)
        version.assert_not_called()
        self.assertEqual(before, sorted(self.root.rglob("*")))
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["model_order"], ["pretrained", "sft_original"])
        self.assertEqual(plan["split_order"], ["train", "val"])
        self.assertEqual(plan["config"]["questions_per_split"], 64)
        self.assertEqual(plan["evaluation"]["steps"], 32)

    def test_existing_results_and_prepare_inside_formal_output_are_rejected(self):
        output = self.root / "runs/experiment5"
        output.mkdir(parents=True)
        marker = output / "result.json"
        marker.write_bytes(b'{"status":"complete"}\n')
        with patch.object(launcher, "capture_version") as version:
            with self.assertRaises(FileExistsError):
                launcher.run(self.args(), self.root)
            with self.assertRaises(ValueError):
                launcher.run(self.args(prepare_only=True, output_dir="runs/experiment5/prepared"), self.root)
        version.assert_not_called()
        self.assertEqual(marker.read_bytes(), b'{"status":"complete"}\n')

    def test_preflight_error_or_timeout_never_leaves_a_complete_result(self):
        for error, expected in ((RuntimeError("fixture preflight failed"), "failed"),
                                (TimeoutError("fixture deadline reached"), "bounded_stop")):
            with self.subTest(status=expected):
                output = f"runs/{expected}"
                with patch.object(launcher, "capture_version", side_effect=error):
                    with self.assertRaises(type(error)):
                        launcher.run(self.args(output_dir=output), self.root)
                result = json.loads((self.root / output / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["completed_groups"], [])
                self.assertEqual(result["stage"], "preflight")
                self.assertFalse((self.root / output / "result.partial.json").exists())


class CompletedGroupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = {"validation_loss_seed": 101, "validation_loss_batch_size": 8,
                          "precision": "bf16"}
        self.selection = {"ids": {"train": [f"train:{i}" for i in range(64)],
                                  "val": [f"train:{i}" for i in range(100, 164)]}}
        self.audit = {"splits": {split: {"code_samples": 64, "target_tokens": 64 * 500}
                                 for split in ("train", "val")}}
        self.sources = {}
        for index, name in enumerate(("pretrained", "sft_original")):
            source = {"source": "pretrained_ema" if index == 0 else "sft_ordinary",
                      "model_sha256": str(index) * 64}
            protocol = {"generation": {"steps": 32, "seed": 1}, "precision": "bf16",
                        "numerics": {"torch": "2.6.0"}, "scorer_version": "fixture",
                        "tokenizer": {"sha256": "tokenizer-fixture"},
                        "evaluation_source": {"sampling.py": "source-fixture"},
                        "question_order": [], "questions_sha256": "full-set-fixture"}
            self.sources[name] = {"checkpoint_source": source, "protocol": protocol}
            self.write(f"{name}-checkpoint.json", {"weights_unchanged": True,
                "model_sha256_before": source["model_sha256"],
                "model_sha256_after": source["model_sha256"], "checkpoint_source": source})
            for split in ("train", "val"):
                key = f"{name}/{split}"
                records = [{"gsm8k_id": identity, "status": "wrong_answer", "predicted_answer": 1,
                            "question": "Unicode separator stays within a JSON record: \u2028"}
                           for identity in self.selection["ids"][split]]
                predictions = self.root / key / "predictions.jsonl"
                predictions.parent.mkdir(parents=True)
                predictions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                                       encoding="utf-8")
                self.write(f"{key}/summary.json", {"complete": True, "label": f"exploratory_{split}",
                    "num_questions": 64, "question_count": 64, "accuracy": 0,
                    "counts": {"correct": 0, "format_failure": 0, "execution_failure": 0, "wrong_answer": 64},
                    "checkpoint_source": source, "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()})
                self.set_protocol(key, {**protocol, "question_order": self.selection["ids"][split],
                                        "questions_sha256": f"selected-{split}"})
                self.write(f"{key}/reference_loss.json", {"complete": True, "label": f"exploratory_{split}",
                    "num_questions": 64, "checkpoint_source": source, "code_samples": 64,
                    "target_tokens": 32000, "loss": 0.5, "loss_sum": 16000, "seed": 101,
                    "precision": "bf16", "batch_size": 8})
                self.write(f"{key}/diagnostics.json", probe.prediction_diagnostics(records, {0, 49152, 49153}))

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def read(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def set_protocol(self, key, protocol):
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        self.write(f"{key}/protocol.json", {**protocol, "protocol_sha256": digest})
        summary = self.read(f"{key}/summary.json")
        self.write(f"{key}/summary.json", {**summary, "protocol_sha256": digest})

    def check(self, sources=None):
        return probe.check_groups(self.root, self.selection, self.audit, sources or self.sources, self.reference)

    def test_only_four_complete_matching_groups_can_finish(self):
        groups = self.check()
        self.assertEqual(set(groups), {f"{name}/{split}" for name in self.sources for split in ("train", "val")})
        with self.assertRaises(ValueError):
            self.check({name: value for name, value in self.sources.items() if name != "sft_original"})
        with self.assertRaises(ValueError):
            self.check({**self.sources, "unexpected_model": self.sources["pretrained"]})

    def test_each_group_rejects_protocol_drift_even_with_recomputed_valid_hashes(self):
        changes = (
            ("pretrained/train", "generation", {"steps": 128, "seed": 1}),
            ("pretrained/val", "tokenizer", {"sha256": "different"}),
            ("sft_original/train", "numerics", {"torch": "different"}),
            ("sft_original/val", "scorer_version", "different"),
            ("sft_original/train", "evaluation_source", {"sampling.py": "different"}),
            ("sft_original/val", "questions_sha256", "different"),
        )
        for key, field, value in changes:
            with self.subTest(group=key, field=field):
                original = self.read(f"{key}/protocol.json")
                original.pop("protocol_sha256")
                self.set_protocol(key, {**original, field: value})
                with self.assertRaises(ValueError):
                    self.check()
                self.set_protocol(key, original)

    def test_failed_generation_loss_or_changed_model_is_not_complete(self):
        changes = (
            ("pretrained/train/summary.json", {"complete": False}),
            ("sft_original/val/summary.json", {"checkpoint_source": {"model_sha256": "different"}}),
            ("sft_original/train/reference_loss.json", {"target_tokens": 31999}),
            ("sft_original/val/reference_loss.json", {"seed": 1}),
            ("sft_original-checkpoint.json", {"model_sha256_after": "different"}),
        )
        for path, change in changes:
            with self.subTest(path=path, change=change):
                original = self.read(path)
                self.write(path, {**original, **change})
                with self.assertRaises(ValueError):
                    self.check()
                self.write(path, original)


if __name__ == "__main__":
    unittest.main()
