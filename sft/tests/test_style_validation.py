"""Data identity and behavior-audit tests; fixtures never run unrestricted code."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_style_data as audit


def pair(original="def simple_math_problem():\n    return 2.5\n", new=None):
    new = new or "def simple_math_problem() -> int:\n    result = 2.5\n    return result\n"
    identity = {"split": "train", "row_index": 0, "gsm8k_id": "train:0",
                "question_sha256": audit.code_digest("question"), "gold_answer": 2.5,
                "original_code_sha256": audit.code_digest(original),
                "transformed_code_sha256": audit.code_digest(new)}
    return {**identity, "row_sha256": audit.digest(identity),
            "original_code": original, "transformed_code": new}


def result(answer=2.5, raw_type="float"):
    return {"status": "ok", "predicted_answer": answer, "raw_type": raw_type,
            "used_stdout_fallback": False, "stdout": "", "error": None}


class StyleValidationTests(unittest.TestCase):
    def test_raw_code_is_sent_without_extraction_and_types_are_preserved(self):
        value = pair()
        with patch.object(audit, "run_program", return_value=result()) as run:
            checked = audit.check_pair(value, audit.ExecutionConfig())
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         [value["original_code"], value["transformed_code"]])
        self.assertTrue(checked["behavior_equal"])
        self.assertTrue(checked["transformed"]["correct"])

    def test_equal_gold_score_does_not_hide_a_changed_numeric_result(self):
        with patch.object(audit, "run_program", side_effect=[result(), result(2.5005)]):
            checked = audit.check_pair(pair(), audit.ExecutionConfig())
        self.assertTrue(checked["original"]["correct"])
        self.assertTrue(checked["transformed"]["correct"])
        self.assertFalse(checked["behavior_equal"])
        self.assertIn("predicted_answer", checked["changed_fields"])

    def test_wrong_original_answer_is_retained_and_reported(self):
        with patch.object(audit, "run_program", return_value=result(7.0)):
            checked = audit.check_pair(pair(), audit.ExecutionConfig())
        report = audit.summarize([checked], 1, 1)
        self.assertTrue(report["all_observed_behaviors_preserved"])
        self.assertFalse(report["all_transformed_answers_correct"])
        self.assertEqual(report["splits"]["train"]["original"]["counts"], {"wrong_answer": 1})

    def test_identical_validation_code_runs_once(self):
        value = pair(new="def simple_math_problem():\n    return 2.5\n")
        value["split"] = "val"
        with patch.object(audit, "run_program", return_value=result()) as run:
            checked = audit.check_pair(value, audit.ExecutionConfig())
        self.assertEqual(run.call_count, 1)
        self.assertTrue(checked["shared_execution"])

    def test_infrastructure_error_cannot_be_aggregated_as_model_failure(self):
        with patch.object(audit, "run_program", side_effect=audit.InfrastructureError("seccomp unavailable")):
            checked = audit.check_pair(pair(), audit.ExecutionConfig())
        report = audit.summarize([checked], 1, 1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["splits"]["train"]["infrastructure_errors"], 1)
        self.assertEqual(report["splits"]["train"]["transformed"]["incorrect_or_failed"], 0)

    def test_partial_run_is_not_reported_as_full_dataset(self):
        with patch.object(audit, "run_program", return_value=result()):
            checked = audit.check_pair(pair(), audit.ExecutionConfig())
        report = audit.summarize([checked], 1, 14409)
        self.assertEqual(report["scope"], "limited_self_check")
        self.assertEqual(report["completed_samples"], 1)

    def test_resume_rejects_different_row_even_with_same_question_id(self):
        value = pair()
        with patch.object(audit, "run_program", return_value=result()):
            checked = audit.check_pair(value, audit.ExecutionConfig())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checks.jsonl"
            path.write_text(json.dumps(checked) + "\n", encoding="utf-8")
            self.assertEqual(len(audit.load_completed(path, [value])), 1)
            altered = pair(new="def simple_math_problem():\n    return 3\n")
            with self.assertRaisesRegex(ValueError, "different row"):
                audit.load_completed(path, [altered])

    def test_load_pairs_rejects_code_checksum_and_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, current = Path(directory) / "original", Path(directory) / "new"
            parent.mkdir()
            current.mkdir()
            code = "def simple_math_problem():\n    return 2.5\n"
            row = {"gsm8k_id": "train:0", "question": "question", "gold_answer": 2.5,
                   "code": code, "code_sha256": audit.code_digest(code)}
            for folder in (parent, current):
                (folder / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                (folder / "train_ids.json").write_text('["train:0"]', encoding="utf-8")
                (folder / "train_questions.jsonl").write_text("fixture\n", encoding="utf-8")
            manifest = {"tokenizer": {"fixed": True}, "counts": {"train": {"samples": 1}}}
            with patch.object(audit, "verify_prepared", return_value=manifest):
                self.assertEqual(len(audit.load_pairs(parent, current, ("train",))), 1)
                (current / "train.jsonl").write_text(json.dumps({**row, "code_sha256": "wrong"}) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Code checksum"):
                    audit.load_pairs(parent, current, ("train",))


if __name__ == "__main__":
    unittest.main()
