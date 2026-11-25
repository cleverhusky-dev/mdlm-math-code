"""Variable normalization must preserve training exposure and frozen evaluation."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_variable_data as preparation


class VariablePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = ROOT / "inputs/tokenizer"
        cls.tokenizer = preparation.load_tokenizer(cls.tokenizer_path, local_files_only=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        temporary = Path(self.temporary.name)
        self.parent, self.output, self.report = temporary / "parent", temporary / "vars", temporary / "reports"
        self.parent.mkdir()
        self.rows = {"train": [self.row(0, "apples", 2, 3, 0),
                               self.row(0, "oranges", 2, 3, 1),
                               self.row(3, "amount", 3, 0, 2)],
                     "val": [self.row(1, "validation_name", 7, 0, 3)]}
        self.manifest = {
            "schema_version": 1, "sequence_length": 512,
            "tokenizer": preparation.tokenizer_identity(self.tokenizer), "split": {"seed": 42},
            "checks": {"exact_training_duplicates_remaining": 0, "inherited_only": True}, "counts": {},
        }
        for split, rows in self.rows.items():
            preparation.write_jsonl(self.parent / f"{split}.jsonl", rows)
            identities = list(dict.fromkeys(row["gsm8k_id"] for row in rows))
            preparation.write_json(self.parent / f"{split}_ids.json", identities)
            preparation.write_jsonl(self.parent / f"{split}_questions.jsonl", [
                next({key: row[key] for key in ("gsm8k_id", "question", "gold_answer")}
                     for row in rows if row["gsm8k_id"] == identity) for identity in identities])
            encoded = [preparation.encode_example(row, self.tokenizer) for row in rows]
            self.write_cache(split, encoded)
            self.manifest["counts"][split] = {
                "questions": len(identities), "samples": len(rows),
                "target_tokens": sum(sum(row["target_mask"]) for row in encoded),
            }
        self.refresh_manifest()

    def row(self, number, variable, value, addition, index):
        code = (f"def simple_math_problem() -> int:\n    {variable} = {value}\n"
                f"    result = {variable} + {addition}\n    return result\n")
        row = {"gsm8k_id": f"train:{number}", "question": f"How many items in question {number}?",
               "gold_answer": value + addition, "code": code, "code_sha256": preparation.text_sha(code),
               "parent_code_sha256": "f" * 64, "source_rows_0based": [index],
               "provenance": {"source": "fixture", "preserve_nested": [index, "unaltered"]}}
        encoded = preparation.encode_example(row, self.tokenizer)
        row.update(question_tokens=len(self.tokenizer.encode(row["question"], add_special_tokens=False)),
                   code_tokens=len(self.tokenizer.encode(code.strip(), add_special_tokens=False)),
                   total_tokens=encoded["sequence_length"])
        return row

    def write_cache(self, split, encoded):
        np.savez_compressed(self.parent / f"{split}.npz",
                            input_ids=np.asarray([row["input_ids"] for row in encoded], dtype=np.int32),
                            target_mask=np.asarray([row["target_mask"] for row in encoded], dtype=np.bool_),
                            prefix_lengths=np.asarray([row["prefix_length"] for row in encoded], dtype=np.int32),
                            gsm8k_ids=np.asarray([row["gsm8k_id"] for row in encoded]))

    def refresh_manifest(self):
        self.manifest["artifacts_sha256"] = {
            name: preparation.sha256_file(self.parent / name) for name in preparation.ARTIFACTS}
        preparation.write_json(self.parent / "manifest.json", self.manifest)

    def prepare(self, **overrides):
        return preparation.prepare(self.parent, self.output, self.tokenizer_path, self.report,
                                   expected_samples=overrides.get("expected_samples", 3),
                                   expected_questions=overrides.get("expected_questions", 2))

    def test_preserves_rows_metadata_and_new_duplicate_weight_without_touching_parent(self):
        original_bytes = {path.name: path.read_bytes() for path in self.parent.iterdir()}
        report = self.prepare()
        rewritten = preparation.read_jsonl(self.output / "train.jsonl")
        self.assertEqual([r["gsm8k_id"] for r in rewritten], [r["gsm8k_id"] for r in self.rows["train"]])
        self.assertEqual(Counter(row["gsm8k_id"] for row in rewritten), {"train:0": 2, "train:3": 1})
        for before, after in zip(self.rows["train"], rewritten):
            for key in before.keys() - {"code", "code_sha256", "code_tokens", "total_tokens"}:
                self.assertEqual(after[key], before[key], key)
            self.assertEqual(after["code_sha256"], preparation.text_sha(after["code"]))
        self.assertEqual(rewritten[0]["code"], rewritten[1]["code"])
        manifest = preparation.verify_prepared(self.output)
        self.assertEqual(manifest["derivation"]["kind"], "training_variable_normalization")
        self.assertEqual(manifest["derivation"]["runtime_validation"],
                         "separate report from validate_style_data.py")
        self.assertEqual(manifest["checks"]["exact_training_duplicates_remaining"], 1)
        self.assertEqual(manifest["checks"]["exact_training_duplicate_groups"], 1)
        self.assertNotIn("inherited_only", manifest["checks"])
        self.assertEqual(report["parent_duplicate_extra_rows"], 0)
        self.assertEqual(report["duplicate_extra_rows_retained"], 1)
        self.assertEqual(report["counts"], self.manifest["counts"])
        self.assertEqual(report["removed_samples"], 0)
        self.assertEqual(report["runtime_validation"], "not_run")
        self.assertFalse(report["training_started"])
        for name in preparation.ARTIFACTS:
            if name not in {"train.jsonl", "train.npz"}:
                self.assertEqual((self.output / name).read_bytes(), original_bytes[name])
        for name, content in original_bytes.items():
            self.assertEqual((self.parent / name).read_bytes(), content)
        changes = preparation.read_jsonl(self.report / "changes.jsonl")
        self.assertEqual(len(changes), 3)
        for index, item in enumerate(changes):
            self.assertEqual(item["row_index"], index)
            self.assertEqual(item["before_sha256"], self.rows["train"][index]["code_sha256"])
            self.assertEqual(item["after_sha256"], rewritten[index]["code_sha256"])
        self.assertTrue(json.loads((self.report / "preview.json").read_text(encoding="utf-8")))

    def test_full_reencoding_matches_written_cache_and_preserves_prefix_and_masks(self):
        self.prepare()
        parent = preparation.PreparedDataset(self.parent, "train")
        output = preparation.PreparedDataset(self.output, "train")
        for index, row in enumerate(preparation.read_jsonl(self.output / "train.jsonl")):
            encoded = preparation.encode_example(row, self.tokenizer)
            np.testing.assert_array_equal(output.input_ids[index], encoded["input_ids"])
            np.testing.assert_array_equal(output.target_mask[index], encoded["target_mask"])
            self.assertEqual(output.prefix_lengths[index], encoded["prefix_length"])
            self.assertEqual(output.gsm8k_ids[index], row["gsm8k_id"])
            prefix = int(output.prefix_lengths[index])
            np.testing.assert_array_equal(output.input_ids[index, :prefix], parent.input_ids[index, :prefix])
            self.assertEqual(row["total_tokens"], encoded["sequence_length"])
            self.assertLessEqual(row["total_tokens"], 512)
        np.testing.assert_array_equal(output.target_mask, parent.target_mask)
        self.assertFalse(np.array_equal(output.input_ids, parent.input_ids))

    def test_parent_text_cache_disagreement_is_rejected_even_with_valid_hashes(self):
        encoded = [preparation.encode_example(row, self.tokenizer) for row in self.rows["train"]]
        encoded[1]["input_ids"][encoded[1]["prefix_length"]] = 1
        self.write_cache("train", encoded)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Parent text/cache mismatch"):
            self.prepare()
        failure = json.loads((self.report / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["stage"], "parent_text_cache")
        self.assertEqual(failure["row_index"], 1)
        self.assertFalse(self.output.exists())

    def test_overlong_transformed_code_fails_instead_of_dropping_or_truncating(self):
        overlong = self.rows["train"][0]["code"] + "#" + " long_word" * 700 + "\n"
        with patch.object(preparation, "transform_code", return_value=(overlong, {})), \
                patch.object(preparation, "verify_transform"), \
                self.assertRaisesRegex(ValueError, "no truncation allowed"):
            self.prepare()
        failure = json.loads((self.report / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["stage"], "encode")
        self.assertFalse(self.output.exists())
        self.assertEqual(len(preparation.read_jsonl(self.parent / "train.jsonl")), 3)

    def test_failed_transform_verification_is_not_accepted(self):
        with patch.object(preparation, "verify_transform", side_effect=ValueError("semantic mismatch")), \
                self.assertRaisesRegex(ValueError, "semantic mismatch"):
            self.prepare()
        failure = json.loads((self.report / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["stage"], "transform")
        self.assertFalse(self.output.exists())

    def test_existing_or_overlapping_outputs_and_project_reports_are_rejected(self):
        for existing in (self.output, self.report):
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Preserving existing"):
                self.prepare()
            self.assertEqual(sentinel.read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            preparation.prepare(self.parent, self.parent / "nested", self.tokenizer_path, self.report)
        with self.assertRaisesRegex(ValueError, "outside the source directory"):
            preparation.prepare(self.parent, self.output, self.tokenizer_path, ROOT / "runs/forbidden-report")

    def test_missing_artifact_declaration_and_wrong_counts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 samples"):
            self.prepare(expected_samples=4)
        del self.manifest["artifacts_sha256"]["val.npz"]
        preparation.write_json(self.parent / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "all eight"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_corrupted_new_cache_is_rejected_and_staging_stays_in_external_report(self):
        save = np.savez_compressed
        def corrupted(path, **values):
            values = copy.deepcopy(values)
            values["input_ids"][0, values["prefix_lengths"][0]] = 1
            save(path, **values)
        with patch.object(np, "savez_compressed", side_effect=corrupted), \
                self.assertRaisesRegex(AssertionError, "Written cache/text mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())
        staging = list(self.report.glob("data.building-*"))
        self.assertEqual(len(staging), 1)
        self.assertTrue(staging[0].resolve().is_relative_to(self.report.resolve()))
        self.assertFalse(list(self.output.parent.glob("data.building-*")))


if __name__ == "__main__":
    unittest.main()
