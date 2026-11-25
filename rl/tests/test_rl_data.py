"""Minimal RL exports preserve the audited training/validation question split."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_data import export_data, QUESTION_FILES
from mdlm_rl.data import load_questions, sha256_file


class RLDataExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "rl-data"
        self.train = [
            {"gsm8k_id": "train:0", "question": "One plus two?", "gold_answer": 3},
            {"gsm8k_id": "train:2", "question": "Three plus four?", "gold_answer": 7},
        ]
        self.val = [{"gsm8k_id": "train:1", "question": "Two plus three?", "gold_answer": 5}]
        self.write_source()

    def write_source(self, train=None, val=None, counts=None):
        train = self.train if train is None else train
        val = self.val if val is None else val
        for split, rows in (("train", train), ("val", val)):
            # Deliberate CRLF/spacing checks that export never reserializes rows.
            content = "\r\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\r\n"
            (self.source / f"{split}_questions.jsonl").write_bytes(content.encode("utf-8"))
        manifest = {
            "schema_version": 1, "sequence_length": 512,
            "counts": counts or {"train": {"questions": len(train)}, "val": {"questions": len(val)}},
            "artifacts_sha256": {name: sha256_file(self.source / name) for name in QUESTION_FILES},
        }
        (self.source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.parent_hash = sha256_file(self.source / "manifest.json")

    def export(self, **overrides):
        arguments = {"expected_parent_sha256": self.parent_hash,
                     "expected_train_questions": 2, "expected_val_questions": 1}
        arguments.update(overrides)
        return export_data(self.source, self.output, **arguments)

    def test_export_is_minimal_byte_identical_and_repeatable(self):
        manifest = self.export()
        before = {entry.name: entry.read_bytes() for entry in self.output.iterdir()}
        self.assertEqual(set(before), {*QUESTION_FILES, "manifest.json"})
        for name in QUESTION_FILES:
            self.assertEqual(before[name], (self.source / name).read_bytes())
        self.assertEqual(manifest["kind"], "rl_questions")
        self.assertEqual(manifest["parent_manifest_sha256"], self.parent_hash)
        self.assertEqual(manifest["counts"], {"train": {"questions": 2}, "val": {"questions": 1}})
        self.assertEqual(manifest["artifacts_sha256"], manifest["source_files_sha256"])
        self.assertNotIn(str(self.root), before["manifest.json"].decode())
        self.assertEqual(self.export(), manifest)
        self.assertEqual(before, {entry.name: entry.read_bytes() for entry in self.output.iterdir()})
        self.assertEqual(load_questions(self.output, "train"), self.train)
        self.assertEqual(load_questions(self.output, "val"), self.val)

    def test_existing_different_destination_is_never_overwritten(self):
        self.export()
        target = self.output / "train_questions.jsonl"
        target.write_bytes(b"existing unrelated content\n")
        with self.assertRaisesRegex(FileExistsError, "not overwritten"):
            self.export()
        self.assertEqual(target.read_bytes(), b"existing unrelated content\n")

    def test_heldout_overlap_and_official_test_ids_are_rejected(self):
        for val, message in (([copy.deepcopy(self.train[0])], "overlap"),
                             ([{"gsm8k_id": "test:1", "question": "Test?", "gold_answer": 1}],
                              "training IDs only")):
            with self.subTest(message=message):
                self.write_source(val=val)
                with self.assertRaisesRegex(ValueError, message):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_count_checks_cover_source_manifest_and_requested_experiment(self):
        with self.assertRaisesRegex(ValueError, "Expected 3 train questions"):
            self.export(expected_train_questions=3)
        self.write_source(counts={"train": {"questions": 3}, "val": {"questions": 1}})
        with self.assertRaisesRegex(ValueError, "count disagrees"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_source_hash_tampering_is_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            self.export(expected_parent_sha256="0" * 64)
        with (self.source / "train_questions.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_invalid_gold_and_reference_code_are_rejected(self):
        for gold in (True, float("nan"), float("inf"), "3"):
            with self.subTest(gold=gold):
                train = copy.deepcopy(self.train)
                train[0]["gold_answer"] = gold
                self.write_source(train=train)
                with self.assertRaisesRegex(ValueError, "finite numeric gold"):
                    self.export()
        train = copy.deepcopy(self.train)
        train[0]["code"] = "return 3"
        self.write_source(train=train)
        with self.assertRaisesRegex(ValueError, "reference code"):
            self.export()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
