"""Small preparation integration checks using the pinned, local tokenizer."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_style_data as preparation


class StylePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = ROOT / "inputs/tokenizer"
        cls.tokenizer = preparation.load_tokenizer(cls.tokenizer_path, local_files_only=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.parent, self.output, self.report = root / "parent", root / "derived", root / "report"
        self.parent.mkdir()
        self.rows = {
            "train": [self.row("train:0", "# alpha", 2, 0),
                      self.row("train:0", "# beta", 2, 1),
                      self.row("train:3", "# third", 3, 2)],
            "val": [self.row("train:1", "# validation remains unchanged", 7, 3)],
        }
        self.manifest = {
            "schema_version": 1, "sequence_length": 512,
            "tokenizer": preparation.tokenizer_identity(self.tokenizer),
            "split": {"seed": 42}, "checks": {}, "counts": {},
            "artifacts_sha256": {},
        }
        for split, rows in self.rows.items():
            preparation.write_jsonl(self.parent / f"{split}.jsonl", rows)
            ids = list(dict.fromkeys(row["gsm8k_id"] for row in rows))
            preparation.write_json(self.parent / f"{split}_ids.json", ids)
            questions = [next({key: row[key] for key in ("gsm8k_id", "question", "gold_answer")}
                              for row in rows if row["gsm8k_id"] == identity) for identity in ids]
            preparation.write_jsonl(self.parent / f"{split}_questions.jsonl", questions)
            encoded = [preparation.encode_example(row, self.tokenizer) for row in rows]
            self.write_cache(split, encoded)
            self.manifest["counts"][split] = {
                "questions": len(ids), "samples": len(rows),
                "target_tokens": sum(sum(row["target_mask"]) for row in encoded),
            }
        self.refresh_manifest()

    def row(self, identity, comment, answer, index):
        code = f"def simple_math_problem():\n    {comment}\n    return {answer}\n"
        question = f"How many items are there in example {identity}?"
        question_tokens = len(self.tokenizer.encode(question, add_special_tokens=False))
        code_tokens = len(self.tokenizer.encode(code.strip(), add_special_tokens=False))
        return {
            "gsm8k_id": identity, "question": question, "gold_answer": answer,
            "code": code, "code_sha256": preparation.text_sha(code),
            "source_rows_0based": [index], "question_tokens": question_tokens,
            "code_tokens": code_tokens,
            "total_tokens": question_tokens + code_tokens + 2 + len(
                self.tokenizer.encode(preparation.SEPARATOR, add_special_tokens=False)),
        }

    def write_cache(self, split, encoded):
        np.savez_compressed(
            self.parent / f"{split}.npz",
            input_ids=np.asarray([row["input_ids"] for row in encoded], dtype=np.int32),
            target_mask=np.asarray([row["target_mask"] for row in encoded], dtype=np.bool_),
            prefix_lengths=np.asarray([row["prefix_length"] for row in encoded], dtype=np.int32),
            gsm8k_ids=np.asarray([row["gsm8k_id"] for row in encoded]),
        )

    def refresh_manifest(self):
        self.manifest["artifacts_sha256"] = {
            path.name: preparation.sha256_file(path)
            for path in self.parent.iterdir() if path.name != "manifest.json"
        }
        preparation.write_json(self.parent / "manifest.json", self.manifest)

    def prepare(self):
        return preparation.prepare(self.parent, self.output, self.tokenizer_path, self.report)

    def test_preserves_validation_bytes_order_and_duplicate_sampling_weights(self):
        source_bytes = {path.name: path.read_bytes() for path in self.parent.iterdir()}
        report = self.prepare()
        new_rows = preparation.read_jsonl(self.output / "train.jsonl")
        self.assertEqual(len(new_rows), 3)
        self.assertEqual(Counter(row["gsm8k_id"] for row in new_rows), {"train:0": 2, "train:3": 1})
        for old, new in zip(self.rows["train"], new_rows):
            for key in ("gsm8k_id", "question", "gold_answer", "source_rows_0based"):
                self.assertEqual(new[key], old[key])
            self.assertEqual(new["parent_code_sha256"], old["code_sha256"])
            self.assertEqual(new["code_sha256"], preparation.text_sha(new["code"]))
        self.assertEqual(new_rows[0]["code"], new_rows[1]["code"])
        self.assertEqual(report["duplicate_extra_rows_retained"], 1)
        for name in ("val.jsonl", "val.npz", "val_ids.json", "val_questions.jsonl",
                     "train_ids.json", "train_questions.jsonl"):
            self.assertEqual((self.output / name).read_bytes(), source_bytes[name])
        for name, contents in source_bytes.items():
            self.assertEqual((self.parent / name).read_bytes(), contents)
        self.assertEqual(report["runtime_validation"], "pending")

    def test_written_npz_matches_reencoded_text_and_preserves_conditioning(self):
        self.prepare()
        old_cache = preparation.PreparedDataset(self.parent, "train")
        new_cache = preparation.PreparedDataset(self.output, "train")
        for index, row in enumerate(preparation.read_jsonl(self.output / "train.jsonl")):
            encoded = preparation.encode_example(row, self.tokenizer)
            np.testing.assert_array_equal(new_cache.input_ids[index], encoded["input_ids"])
            np.testing.assert_array_equal(new_cache.target_mask[index], encoded["target_mask"])
            self.assertEqual(new_cache.prefix_lengths[index], encoded["prefix_length"])
            self.assertEqual(new_cache.gsm8k_ids[index], row["gsm8k_id"])
            prefix = encoded["prefix_length"]
            np.testing.assert_array_equal(new_cache.input_ids[index, :prefix], old_cache.input_ids[index, :prefix])
            self.assertNotEqual(new_cache.input_ids[index].tolist(), old_cache.input_ids[index].tolist())
        np.testing.assert_array_equal(new_cache.target_mask, old_cache.target_mask)
        self.assertEqual(new_cache.manifest["counts"], old_cache.manifest["counts"])
        digest = hashlib.sha256()
        for split in ("train", "val"):
            with np.load(self.output / f"{split}.npz", allow_pickle=False) as cache:
                for ids in cache["input_ids"]:
                    digest.update(np.asarray(ids, dtype="<u4").tobytes())
        self.assertEqual(new_cache.manifest["encoded_token_sha256"], digest.hexdigest())

    def test_existing_output_or_nonempty_report_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("existing data", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Preserving existing output"):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing data")
        other_output = self.output.with_name("fresh")
        self.report.mkdir()
        (self.report / "keep.txt").write_text("audit evidence", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Preserving existing reports"):
            preparation.prepare(self.parent, other_output, self.tokenizer_path, self.report)
        self.assertFalse(other_output.exists())

    def test_valid_hash_manifest_does_not_hide_text_cache_mismatch(self):
        encoded = [preparation.encode_example(row, self.tokenizer) for row in self.rows["train"]]
        encoded[0]["input_ids"][encoded[0]["prefix_length"]] = 1
        self.write_cache("train", encoded)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Parent text/cache mismatch at row 0"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_overlength_transform_fails_without_truncation_or_silent_row_drop(self):
        long_code = "def simple_math_problem() -> int:\n    result = " + " + ".join(["1"] * 600) + "\n    return result\n"
        with patch.object(preparation, "transform_code", return_value=(long_code, {})):
            with self.assertRaisesRegex(ValueError, "no truncation allowed"):
                self.prepare()
        self.assertFalse(self.output.exists())
        failure = json.loads((self.report / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["stage"], "encode")
        self.assertEqual(failure["row_index"], 0)
        self.assertEqual(preparation.verify_prepared(self.parent), self.manifest)

    def test_parent_and_outputs_cannot_overlap(self):
        for output, report in ((self.parent / "derived", self.report),
                               (self.output, self.output / "reports"),
                               (self.output, self.parent)):
            with self.assertRaisesRegex(ValueError, "separate directories"):
                preparation.prepare(self.parent, output, self.tokenizer_path, report)


if __name__ == "__main__":
    unittest.main()
