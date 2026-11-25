"""Manifest-driven training counts still reject altered or overlapping inputs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mdlm_sft.data import PreparedDataset, load_questions, sha256_file
from mdlm_sft.dataset_checks import (
    labeled_training_questions, load_training_questions,
    training_question_sort_key, validate_training_data,
)


class DatasetChecksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {"schema_version": 1, "sequence_length": 512,
                         "tokenizer": {"vocab_without_mask": 49153}, "counts": {}}
        self.write_split("train", ["train:0", "train:0", "train:2"])
        self.write_split("val", ["train:1"])
        self.refresh_manifest()

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def write_split(self, split, ids):
        rows = [{"gsm8k_id": identity, "question": f"Question {identity}",
                 "gold_answer": 2, "code": "def simple_math_problem():\n    return 2"}
                for identity in ids]
        self.write_json(f"{split}_ids.json", sorted(set(ids)))
        questions = [{key: row[key] for key in ("gsm8k_id", "question", "gold_answer")}
                     for index, row in enumerate(rows) if row["gsm8k_id"] not in ids[:index]]
        for name, values in ((f"{split}.jsonl", rows), (f"{split}_questions.jsonl", questions)):
            (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        prefixes = np.full(len(ids), 4, dtype=np.int32)
        np.savez_compressed(self.root / f"{split}.npz",
                            input_ids=np.ones((len(ids), 512), dtype=np.int32),
                            target_mask=np.arange(512)[None, :] >= prefixes[:, None],
                            prefix_lengths=prefixes, gsm8k_ids=np.asarray(ids))
        self.manifest["counts"][split] = {"samples": len(ids), "questions": len(set(ids)),
                                          "target_tokens": len(ids) * 508}

    def refresh_manifest(self):
        self.manifest["artifacts_sha256"] = {
            path.name: sha256_file(path) for path in self.root.iterdir() if path.name != "manifest.json"}
        self.write_json("manifest.json", self.manifest)

    def validate(self):
        return validate_training_data(self.root, PreparedDataset(self.root, "train"),
                                      PreparedDataset(self.root, "val"), load_questions(self.root, "val"))

    def test_accepts_manifest_counts_and_reduced_training_rows(self):
        self.assertEqual(self.validate()["train"]["samples"], 3)
        self.write_split("train", ["train:0", "train:2"])
        self.refresh_manifest()
        self.assertEqual(self.validate()["train"]["samples"], 2)

    def test_rejects_target_token_count_mismatch(self):
        self.manifest["counts"]["train"]["target_tokens"] -= 1
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "target-token count"):
            self.validate()

    def test_rejects_question_overlap_even_with_valid_hashes(self):
        self.write_split("val", ["train:0"])
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.validate()

    def test_rejects_missing_artifact_hash(self):
        del self.manifest["artifacts_sha256"]["train.jsonl"]
        self.write_json("manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "all eight"):
            self.validate()

    def test_rejects_modified_checksummed_file(self):
        with (self.root / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Prepared artifact changed"):
            self.validate()

    def test_rejects_reference_answer_mismatch(self):
        path = self.root / "train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["gold_answer"] = 3
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "question/answer differs"):
            self.validate()

    def test_rejects_cache_reference_order_mismatch(self):
        path = self.root / "train.jsonl"
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n", encoding="utf-8")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "cache count/order"):
            self.validate()

    def test_rejects_official_test_questions(self):
        self.write_split("val", ["test:1"])
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "official training questions only"):
            self.validate()

    def mixed_fixture(self):
        self.write_split("train", ["train:0", "train:0", "train:2", "tinygsm:7", "tinygsm:12"])
        for name in ("train.jsonl", "train_questions.jsonl"):
            values = [json.loads(line) for line in (self.root / name).read_text().splitlines()]
            for row in values:
                if row["gsm8k_id"].startswith("tinygsm:"):
                    row.update(gold_answer=None, source="TinyGSM",
                               answer_provenance="not_provided_by_source", execution_answer=2)
            if name == "train_questions.jsonl":
                values.sort(key=lambda row: training_question_sort_key(row["gsm8k_id"]))
            (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        self.write_json("train_ids.json", ["train:0", "train:2", "tinygsm:7", "tinygsm:12"])
        self.manifest["derivation"] = {"kind": "tinygsm_replay_mix"}
        self.manifest["mixing"] = {"parent_samples": 3, "replay_samples": 2}
        self.refresh_manifest()

    def change_mixed_rows(self, update, names=("train.jsonl", "train_questions.jsonl")):
        for name in names:
            path = self.root / name
            values = [json.loads(line) for line in path.read_text().splitlines()]
            for row in values:
                if row["gsm8k_id"].startswith("tinygsm:"):
                    update(row)
            path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        self.refresh_manifest()

    def test_declared_replay_mix_accepts_missing_gold_but_smoke_uses_only_real_labels(self):
        self.mixed_fixture()
        self.assertEqual(self.validate()["train"]["samples"], 5)
        self.assertEqual(len(load_training_questions(self.root)), 4)
        self.assertEqual([row["gsm8k_id"] for row in labeled_training_questions(self.root)],
                         ["train:0", "train:2"])
        self.assertEqual(len(labeled_training_questions(self.root, limit=1)), 1)
        # The frozen evaluation loader remains official-only, so unlabelled replay
        # cannot accidentally become a generation benchmark via the old API.
        with self.assertRaisesRegex(ValueError, "Invalid official question ID"):
            load_questions(self.root, "train")

    def test_replay_requires_explicit_mix_manifest(self):
        self.mixed_fixture()
        self.manifest.pop("derivation")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Invalid official question ID"):
            self.validate()

    def test_replay_execution_result_cannot_be_promoted_to_gold(self):
        self.mixed_fixture()
        self.change_mixed_rows(lambda row: row.update(gold_answer=row["execution_answer"]))
        with self.assertRaisesRegex(ValueError, "explicitly null gold_answer"):
            self.validate()

    def test_replay_source_markers_are_required_on_actual_training_references(self):
        self.mixed_fixture()
        self.change_mixed_rows(lambda row: row.pop("answer_provenance"), names=("train.jsonl",))
        with self.assertRaisesRegex(ValueError, "source markers"):
            self.validate()

    def test_replay_never_enters_validation(self):
        self.mixed_fixture()
        self.write_split("val", ["tinygsm:99"])
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Invalid official question ID"):
            self.validate()

    def test_mix_does_not_admit_official_test_ids(self):
        self.mixed_fixture()
        self.write_split("train", ["test:0"])
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "official training questions only"):
            self.validate()

    def test_replay_text_overlap_is_rejected_even_with_a_different_id(self):
        self.mixed_fixture()
        self.change_mixed_rows(lambda row: row.update(question="  QUESTION   train:1  ")
                               if row["gsm8k_id"] == "tinygsm:7" else None)
        with self.assertRaisesRegex(ValueError, "overlaps official"):
            self.validate()

    def test_replay_duplicate_question_text_is_rejected(self):
        self.mixed_fixture()
        self.change_mixed_rows(lambda row: row.update(question="same replay question"))
        with self.assertRaisesRegex(ValueError, "question text is duplicated"):
            self.validate()

    def test_mixture_counts_are_checked_against_actual_sources(self):
        self.mixed_fixture()
        self.manifest["mixing"] = {"parent_samples": 4, "replay_samples": 1}
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "mixture manifest"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
