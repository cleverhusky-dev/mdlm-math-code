"""One-solution selection preserves the audited references and evaluation split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_single_data as preparation


class SingleSelectionTests(unittest.TestCase):
    def test_fixed_selection_is_order_independent_and_does_not_touch_global_rng(self):
        rows = [{"gsm8k_id": f"train:{identity}", "question": f"Question {identity}",
                 "gold_answer": identity, "code": f"def simple_math_problem():\n    return {identity} + {offset} - {offset}\n"}
                for identity in (1, 10, 2) for offset in range(5)]
        state = random.getstate()
        kept, indices, mapping = preparation.select_single(rows)
        again, again_indices, again_mapping = preparation.select_single(rows)
        self.assertEqual((kept, indices, mapping), (again, again_indices, again_mapping))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(kept), 3)
        self.assertEqual(len({row["gsm8k_id"] for row in kept}), 3)
        self.assertEqual(indices, sorted(indices))
        reversed_kept, _, _ = preparation.select_single(list(reversed(rows)))
        self.assertEqual({r["gsm8k_id"]: r["code"] for r in kept},
                         {r["gsm8k_id"]: r["code"] for r in reversed_kept})
        numeric_order = [item["gsm8k_id"] for item in sorted(mapping, key=lambda r: r["question_selection_order"])]
        self.assertEqual(numeric_order, ["train:1", "train:2", "train:10"])

    def test_rejects_duplicate_conflicting_and_test_rows(self):
        first = {"gsm8k_id": "train:0", "question": "Count?", "gold_answer": 2,
                 "code": "def simple_math_problem():\n    return 2\n"}
        cases = [([first, first], "Duplicate code"),
                 ([first, {**first, "gold_answer": 3, "code": first["code"] + "\n"}], "Inconsistent"),
                 ([{**first, "gsm8k_id": "test:0"}], "official training"),
                 ([], "Empty")]
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                preparation.select_single(rows)


class SinglePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = ROOT / "inputs/tokenizer"
        cls.tokenizer = preparation.load_tokenizer(cls.tokenizer_path, local_files_only=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.parent, self.output, self.report = root / "parent", root / "single", root / "report"
        self.parent.mkdir()
        self.rows = {"train": [self.row(0, 0), self.row(0, 1), self.row(3, 0)],
                     "val": [self.row(1, 0)]}
        self.manifest = {"schema_version": 1, "sequence_length": 512,
                         "tokenizer": preparation.tokenizer_identity(self.tokenizer),
                         "split": {"seed": 42}, "checks": {}, "counts": {}}
        for split, rows in self.rows.items():
            preparation.write_jsonl(self.parent / f"{split}.jsonl", rows)
            identities = list(dict.fromkeys(r["gsm8k_id"] for r in rows))
            preparation.write_json(self.parent / f"{split}_ids.json", identities)
            preparation.write_jsonl(self.parent / f"{split}_questions.jsonl", [
                next({key: row[key] for key in ("gsm8k_id", "question", "gold_answer")}
                     for row in rows if row["gsm8k_id"] == identity) for identity in identities])
            encoded = [preparation.encode_example(row, self.tokenizer) for row in rows]
            self.write_cache(split, encoded)
            self.manifest["counts"][split] = {
                "questions": len(identities), "samples": len(rows),
                "target_tokens": sum(sum(row["target_mask"]) for row in encoded)}
        self.refresh_manifest()

    def row(self, number, offset):
        code = f"def simple_math_problem() -> int:\n    result = {number} + {offset} - {offset}\n    return result\n"
        return {"gsm8k_id": f"train:{number}", "question": f"How many items in example {number}?",
                "gold_answer": number, "code": code, "source_rows_0based": [number * 10 + offset],
                "code_sha256": hashlib.sha256(code.encode()).hexdigest()}

    def write_cache(self, split, encoded):
        np.savez_compressed(self.parent / f"{split}.npz",
                            input_ids=np.asarray([r["input_ids"] for r in encoded], dtype=np.int32),
                            target_mask=np.asarray([r["target_mask"] for r in encoded], dtype=np.bool_),
                            prefix_lengths=np.asarray([r["prefix_length"] for r in encoded], dtype=np.int32),
                            gsm8k_ids=np.asarray([r["gsm8k_id"] for r in encoded]))

    def refresh_manifest(self):
        self.manifest["artifacts_sha256"] = {
            name: preparation.sha256_file(self.parent / name) for name in preparation.ARTIFACTS}
        preparation.write_json(self.parent / "manifest.json", self.manifest)

    def prepare(self, **kwargs):
        return preparation.prepare(self.parent, self.output, self.report, self.tokenizer_path,
                                   expected_questions=kwargs.pop("expected_questions", 2), **kwargs)

    def test_single_rows_cache_and_validation_preserved_without_touching_source(self):
        source = {path.name: path.read_bytes() for path in self.parent.iterdir()}
        expected, indices, mapping = preparation.select_single(self.rows["train"])
        report = self.prepare()
        self.assertEqual(preparation.read_jsonl(self.output / "train.jsonl"), expected)
        self.assertEqual(preparation.read_jsonl(self.report / "selection.jsonl"), mapping)
        self.assertEqual(json.loads((self.report / "kept_parent_indices.json").read_text()), indices)
        cache = preparation.PreparedDataset(self.output, "train")
        self.assertEqual(len(cache), 2)
        for index, row in enumerate(expected):
            encoded = preparation.encode_example(row, self.tokenizer)
            np.testing.assert_array_equal(cache.input_ids[index], encoded["input_ids"])
            np.testing.assert_array_equal(cache.target_mask[index], encoded["target_mask"])
            self.assertEqual(cache.prefix_lengths[index], encoded["prefix_length"])
            self.assertEqual(cache.gsm8k_ids[index], row["gsm8k_id"])
        for name in preparation.ARTIFACTS:
            if name not in {"train.jsonl", "train.npz"}:
                self.assertEqual((self.output / name).read_bytes(), source[name])
        for name, contents in source.items():
            self.assertEqual((self.parent / name).read_bytes(), contents)
        self.assertEqual(report["removed_samples"], 1)
        self.assertEqual(report["counts"]["val"], self.manifest["counts"]["val"])

    def test_valid_manifest_hashes_do_not_hide_selected_text_cache_mismatch(self):
        encoded = [preparation.encode_example(row, self.tokenizer) for row in self.rows["train"]]
        _, indices, _ = preparation.select_single(self.rows["train"])
        selected = indices[0]
        encoded[selected]["input_ids"][encoded[selected]["prefix_length"]] = 1
        self.write_cache("train", encoded)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Parent text/cache mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_count_or_wrong_question_gold_file(self):
        with self.assertRaisesRegex(ValueError, "Expected 3 training questions"):
            self.prepare(expected_questions=3)
        questions = preparation.read_jsonl(self.parent / "train_questions.jsonl")
        questions[0]["gold_answer"] += 1
        preparation.write_jsonl(self.parent / "train_questions.jsonl", questions)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "question/gold file disagrees"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_outputs_and_overlapping_paths_are_preserved(self):
        for existing in (self.output, self.report):
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Preserving existing"):
                self.prepare()
            self.assertEqual(sentinel.read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            preparation.prepare(self.parent, self.parent / "nested", self.report, self.tokenizer_path,
                                expected_questions=2)


if __name__ == "__main__":
    unittest.main()
