"""Exact per-question deduplication preserves first-occurrence provenance."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from deduplicate_data import deduplicate


def row(identity="train:0", code="def simple_math_problem() -> int:\n    result = 4\n    return result\n", source=0):
    return {
        "gsm8k_id": identity, "question": f"Question {identity}", "gold_answer": 4,
        "code": code, "source_rows_0based": [source],
        "metadata": {"original_source": source},
    }


class DataDeduplicationTests(unittest.TestCase):
    def test_identical_code_for_same_question_retains_first_row_without_metadata_merge(self):
        original = [row(source=12), row(source=29), row(source=44)]
        snapshot = copy.deepcopy(original)
        kept, indices, removed = deduplicate(original)
        self.assertEqual(kept, [snapshot[0]])
        self.assertEqual(indices, [0])
        self.assertEqual(original, snapshot)
        self.assertEqual(kept[0]["source_rows_0based"], [12])
        self.assertEqual(kept[0]["metadata"], {"original_source": 12})
        self.assertEqual(len(removed), 2)
        for record, removed_index in zip(removed, (1, 2)):
            self.assertEqual(record["removed_row_index"], removed_index)
            self.assertEqual(record["kept_parent_row_index"], 0)
            self.assertEqual(record["kept_output_row_index"], 0)
            self.assertEqual(record["gsm8k_id"], "train:0")

    def test_identical_code_across_questions_remains_separate(self):
        original = [row("train:0"), row("train:1")]
        kept, indices, removed = deduplicate(original)
        self.assertEqual(kept, original)
        self.assertEqual(indices, [0, 1])
        self.assertEqual(removed, [])

    def test_integer_cast_difference_is_not_deduplicated(self):
        direct = "def simple_math_problem() -> int:\n    result = 5 / 2\n    return result\n"
        cast = "def simple_math_problem() -> int:\n    result = int(5 / 2)\n    return result\n"
        original = [row(code=direct), row(code=cast)]
        kept, indices, removed = deduplicate(original)
        self.assertEqual(kept, original)
        self.assertEqual(indices, [0, 1])
        self.assertEqual(removed, [])

    def test_full_string_matching_does_not_normalize_whitespace(self):
        first = row()
        original = [first, {**first, "code": first["code"].rstrip("\n")}]
        kept, indices, removed = deduplicate(original)
        self.assertEqual(kept, original)
        self.assertEqual(indices, [0, 1])
        self.assertEqual(removed, [])

    def test_duplicate_key_with_conflicting_question_or_answer_is_rejected(self):
        for field, value in (("question", "A different question"), ("gold_answer", 5)):
            with self.subTest(field=field):
                first = row()
                original = [first, {**first, field: value}]
                snapshot = copy.deepcopy(original)
                with self.assertRaises(ValueError):
                    deduplicate(original)
                self.assertEqual(original, snapshot)

    def test_first_occurrence_order_and_parent_to_output_mapping_are_stable(self):
        first, second, third = row("train:8"), row("train:2"), row("train:5")
        original = [first, second, {**first, "source_rows_0based": [9]},
                    third, {**second, "source_rows_0based": [10]}]
        kept, indices, removed = deduplicate(original)
        self.assertEqual(kept, [first, second, third])
        self.assertEqual(indices, [0, 1, 3])
        self.assertEqual([record["removed_row_index"] for record in removed], [2, 4])
        self.assertEqual([record["kept_parent_row_index"] for record in removed], [0, 1])
        self.assertEqual([record["kept_output_row_index"] for record in removed], [0, 1])


if __name__ == "__main__":
    unittest.main()
