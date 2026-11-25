"""Replay exclusions and assembly preserve original data and audit identities."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_mix_data as preparation


class QuestionExclusionTests(unittest.TestCase):
    def test_student_prefix_case_punctuation_and_unicode_variants_are_excluded(self):
        blocked = preparation.QuestionExclusions(["Maya has 12 apples. How many apples does Maya have?"])
        variants = [
            "Student: MAYA has 12 apples! How many apples does Maya have?",
            " question: Maya has １２ apples; how many apples does Maya have. ",
            "Maya   has 12 apples\nHow many apples does Maya have",
        ]
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(blocked.matches(value), "exact")

    def test_long_near_duplicate_with_changed_number_is_conservatively_excluded(self):
        question = ("Maya visits the local market every Saturday morning to buy fresh apples for her family. "
                    "This week she buys 12 apples and gives some to her friends before taking the rest home. "
                    "If she gives away four apples, how many apples does she take home to her family?")
        blocked = preparation.QuestionExclusions([question])
        self.assertNotEqual(preparation.normalized_question(question),
                            preparation.normalized_question(question.replace("12", "15")))
        self.assertEqual(blocked.matches(question.replace("12", "15")), "lexical_near")

    def test_short_different_arithmetic_is_not_mistaken_for_exact_duplicate(self):
        blocked = preparation.QuestionExclusions(["What is 2 plus 3?"])
        self.assertIsNone(blocked.matches("What is 2 plus 8?"))

    def test_different_questions_and_empty_exclusion_pool_are_allowed(self):
        question = "A train travels 70 miles every hour for three hours. How far does it travel?"
        self.assertIsNone(preparation.QuestionExclusions([]).matches(question))
        self.assertIsNone(preparation.QuestionExclusions([
            "Maya grows twelve apple trees in her garden and plants three additional trees. "
            "How many apple trees does Maya have now?"
        ]).matches(question))


class MixAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = ROOT / "inputs/tokenizer"
        cls.tokenizer = preparation.load_tokenizer(cls.tokenizer_path, local_files_only=True)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / "parent"
        self.selection = self.root / "selection"
        self.execution = self.root / "execution"
        self.output = self.root / "mixed"
        for directory in (self.parent, self.selection, self.execution):
            directory.mkdir()
        self.parent_rows = [self.row("train:0", "How many pencils are in a pair?", 2),
                            self.row("train:2", "How many legs does a cat have?", 4)]
        self.manifest = {
            "schema_version": 1, "sequence_length": 512,
            "tokenizer": preparation.tokenizer_identity(self.tokenizer),
            "split": {"seed": 42}, "counts": {}, "checks": {},
            "official_check": {"official_test_accessed": False},
        }
        for split, rows in (("train", self.parent_rows),
                            ("val", [self.row("train:1", "How many days are in one week?", 7)])):
            self.write_parent_split(split, rows)
        self.manifest["artifacts_sha256"] = {
            path.name: preparation.sha256_file(path) for path in self.parent.iterdir()}
        preparation.write_json(self.parent / "manifest.json", self.manifest)
        self.replay = [self.replay_row(5, "A box contains nine marbles. How many marbles are in it?", 9),
                       self.replay_row(3, "Two baskets each hold six eggs. How many eggs are there?", 12)]
        self.candidates = [{key: value for key, value in row.items() if key != "execution_answer"}
                           for row in self.replay]
        extra = self.replay_row(9, "There are eight crayons in a packet. How many crayons are there?", 8)
        self.candidates.append({key: value for key, value in extra.items() if key != "execution_answer"})
        self.checks = [{"gsm8k_id": row["gsm8k_id"], "code_sha256": row["code_sha256"],
                        "accepted": True, "result": {"status": "ok", "predicted_answer": row["execution_answer"]}}
                       for row in self.replay]
        # Synthetic evidence fixtures exercise binding only; no code is executed
        # or claimed to have been run by this local assembly unit test.
        self.refresh_evidence()

    def row(self, identity, question, gold):
        code = f"def simple_math_problem() -> int:\n    result = {gold}\n    return result\n"
        return {"gsm8k_id": identity, "question": question, "gold_answer": gold,
                "code": code, "code_sha256": preparation.text_sha(code)}

    def replay_row(self, index, question, answer):
        row = self.row(f"tinygsm:{index}", question, answer)
        row.update(source="TinyGSM", answer_provenance="not_provided_by_source", gold_answer=None,
                   execution_answer=answer, source_file="train-00000-of-00001.parquet", source_row_0based=index)
        return row

    def write_parent_split(self, split, rows):
        preparation.write_jsonl(self.parent / f"{split}.jsonl", rows)
        preparation.write_json(self.parent / f"{split}_ids.json", [row["gsm8k_id"] for row in rows])
        preparation.write_jsonl(self.parent / f"{split}_questions.jsonl", [
            {key: row[key] for key in ("gsm8k_id", "question", "gold_answer")} for row in rows])
        encoded = [preparation.encode_example(row, self.tokenizer) for row in rows]
        np.savez_compressed(self.parent / f"{split}.npz",
                            input_ids=np.asarray([row["input_ids"] for row in encoded], dtype=np.int32),
                            target_mask=np.asarray([row["target_mask"] for row in encoded], dtype=np.bool_),
                            prefix_lengths=np.asarray([row["prefix_length"] for row in encoded], dtype=np.int32),
                            gsm8k_ids=np.asarray([row["gsm8k_id"] for row in encoded]))
        self.manifest["counts"][split] = {"samples": len(rows), "questions": len(rows),
                                         "target_tokens": sum(sum(row["target_mask"]) for row in encoded)}

    def refresh_evidence(self):
        preparation.write_jsonl(self.selection / "candidates.jsonl", self.candidates)
        preparation.write_jsonl(self.execution / "accepted.jsonl", self.replay)
        preparation.write_jsonl(self.execution / "checks.jsonl", self.checks)
        candidate_sha = preparation.sha256_file(self.selection / "candidates.jsonl")
        preparation.write_json(self.selection / "selection.json", {
            "seed": 8, "parent_manifest_sha256": preparation.sha256_file(self.parent / "manifest.json"),
            "candidates_sha256": candidate_sha, "requested_replay_samples": 2,
            "test_access": "Synthetic test fixture; no test data accessed"})
        preparation.write_json(self.execution / "execution.json", {
            "status": "passed", "candidates_sha256": candidate_sha,
            "accepted_sha256": preparation.sha256_file(self.execution / "accepted.jsonl"),
            "checks_sha256": preparation.sha256_file(self.execution / "checks.jsonl")})

    def assemble(self):
        return preparation.assemble(SimpleNamespace(
            parent_dir=self.parent, selection_dir=self.selection, execution_dir=self.execution,
            output_dir=self.output, tokenizer=self.tokenizer_path))

    def test_assembly_preserves_parent_bytes_arrays_validation_and_null_replay_gold(self):
        before = {path.name: path.read_bytes() for path in self.parent.iterdir()}
        self.assemble()
        rows = preparation.read_jsonl(self.output / "train.jsonl")
        self.assertEqual(rows[:2], self.parent_rows)
        self.assertEqual([row["gsm8k_id"] for row in rows], ["train:0", "train:2", "tinygsm:3", "tinygsm:5"])
        self.assertTrue(all(row["gold_answer"] is None for row in rows[2:]))
        self.assertEqual({row["execution_answer"] for row in rows[2:]}, {9, 12})
        original = preparation.PreparedDataset(self.parent, "train")
        mixed = preparation.PreparedDataset(self.output, "train")
        np.testing.assert_array_equal(mixed.input_ids[:2], original.input_ids)
        np.testing.assert_array_equal(mixed.target_mask[:2], original.target_mask)
        np.testing.assert_array_equal(mixed.prefix_lengths[:2], original.prefix_lengths)
        self.assertEqual(mixed.manifest["mixing"]["replay_fraction"], 0.5)
        for name, contents in before.items():
            self.assertEqual((self.parent / name).read_bytes(), contents)
            if name.startswith("val"):
                self.assertEqual((self.output / name).read_bytes(), contents)

    def test_modified_unselected_candidate_invalidates_selection_identity(self):
        self.candidates[-1]["question"] = "This candidate changed after the selection was recorded."
        preparation.write_jsonl(self.selection / "candidates.jsonl", self.candidates)
        with self.assertRaisesRegex(ValueError, "[Cc]andidate|[Ss]election"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_output_can_be_outside_the_read_only_input_evidence_tree(self):
        # Frozen inputs and a user's rebuild workspace need not be siblings.
        evidence_root = self.root / "snapshot" / "inputs"
        evidence_root.mkdir(parents=True)
        self.selection = self.selection.rename(evidence_root / "selection")
        self.execution = self.execution.rename(evidence_root / "execution")
        self.output = self.root / "rebuild" / "data" / "mixed"
        self.assemble()
        self.assertEqual(len(preparation.read_jsonl(self.output / "train.jsonl")), 4)
        self.assertEqual((self.output / "val.jsonl").read_bytes(),
                         (self.parent / "val.jsonl").read_bytes())

    def test_success_flag_cannot_override_failed_execution_result(self):
        self.checks[0]["result"]["status"] = "execution_failure"
        self.refresh_evidence()
        with self.assertRaisesRegex(ValueError, "execution evidence"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_boolean_execution_answer_is_not_a_valid_numeric_result(self):
        self.replay[0]["execution_answer"] = True
        self.checks[0]["result"]["predicted_answer"] = True
        self.refresh_evidence()
        with self.assertRaisesRegex(ValueError, "execution evidence"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_changed_replay_code_cannot_replace_executed_source(self):
        self.replay[0]["code"] += "\n# Changed after execution\n"
        self.refresh_evidence()
        with self.assertRaisesRegex(ValueError, "Replay text differs"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("existing evidence", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Preserving existing dataset"):
            self.assemble()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing evidence")


if __name__ == "__main__":
    unittest.main()
