"""Behavior and source-preservation checks for the answer-style transform."""
from __future__ import annotations

import unittest

from mdlm_sft.style_transform import StyleTransformError, transform_code, verify_transform


class StyleTransformTests(unittest.TestCase):
    def value(self, code):
        namespace = {}
        exec(code, namespace)  # Only small, hand-written test fixtures are executed.
        return namespace["simple_math_problem"]()

    def test_preserves_cast_and_original_expression(self):
        code = "def simple_math_problem():\n    # explanation\n    total = 7 / 2\n    return (int(total))\n"
        output, meta = transform_code(code)
        self.assertEqual(output, "def simple_math_problem() -> int:\n    total = 7 / 2\n    result = (int(total))\n    return result\n")
        self.assertEqual(self.value(code), self.value(output))
        self.assertEqual(meta["comments_removed"], 1)
        self.assertTrue(meta["ast_verified"])

    def test_annotation_does_not_cast_float(self):
        output, _ = transform_code("def simple_math_problem():\n    return (7 / 2)\n")
        self.assertEqual(self.value(output), 3.5)
        self.assertIsInstance(self.value(output), float)

    def test_strings_hashes_unicode_and_numeric_spelling_are_preserved(self):
        code = 'def simple_math_problem():\n    标签 = "# not a comment\\n"\n    cost = 1.40  # trailing comment\n    return (cost + len(标签))\n'
        output, _ = transform_code(code)
        self.assertIn('标签 = "# not a comment\\n"', output)
        self.assertIn("cost = 1.40", output)
        self.assertNotIn("trailing comment", output)
        self.assertEqual(self.value(code), self.value(output))

    def test_multiline_strings_and_return_survive(self):
        code = 'def simple_math_problem():\n    text = """a\n\n    # literal text\n"""\n    return (\n        len(text) + 1 # explanation\n    )\n'
        output, _ = transform_code(code)
        self.assertIn('"""a\n\n    # literal text\n"""', output)
        self.assertEqual(self.value(code), self.value(output))

    def test_existing_result_is_reused_and_idempotent(self):
        code = "def simple_math_problem():\n    result = 4.75\n    return (int(result))\n"
        output, _ = transform_code(code)
        self.assertIn("result = (int(result))", output)
        again, metadata = transform_code(output)
        self.assertEqual(output, again)
        self.assertFalse(metadata["return_annotation_added"])
        self.assertEqual(metadata["returns_rewritten"], 0)
        self.assertEqual(self.value(code), self.value(output))

    def test_nested_helper_signature_and_return_are_untouched(self):
        code = "def simple_math_problem():\n    def twice(n):\n        # helper\n        return n * 2\n    return (twice(6))\n"
        output, metadata = transform_code(code)
        self.assertIn("def twice(n):\n        return n * 2", output)
        self.assertIn("result = (twice(6))", output)
        self.assertEqual(metadata["nested_functions_preserved"], 1)
        self.assertEqual(self.value(code), self.value(output))

    def test_branch_returns_keep_control_flow(self):
        for value in (0, 1):
            code = f"def simple_math_problem():\n    value = {value}\n    if value:\n        return ('0')\n    else:\n        return (2.5)\n"
            output, metadata = transform_code(code)
            self.assertEqual(metadata["returns_rewritten"], 2)
            self.assertEqual(self.value(code), self.value(output))
            self.assertIn("        return result", output)

    def test_ast_audit_rejects_calculation_change(self):
        code = "def simple_math_problem():\n    return 2 + 3\n"
        output, _ = transform_code(code)
        with self.assertRaisesRegex(StyleTransformError, "ast_mismatch"):
            verify_transform(code, output.replace("2 + 3", "2 - 3"))

    def test_unsupported_scope_and_docstring_are_explicit_errors(self):
        cases = {
            "existing_docstring": 'def simple_math_problem():\n    """question"""\n    return 1\n',
            "result_capture": "def simple_math_problem():\n    def helper():\n        return result\n    return helper()\n",
            "result_binding": "def simple_math_problem():\n    x = result\n    return x\n",
            "unsupported_control_flow": "def simple_math_problem():\n    global result\n    return 1\n",
            "namespace_observation": "def simple_math_problem():\n    x = locals()\n    return len(x)\n",
        }
        for reason, code in cases.items():
            with self.subTest(reason=reason):
                with self.assertRaises(StyleTransformError) as raised:
                    transform_code(code)
                self.assertEqual(raised.exception.reason, reason)

    def test_finally_cannot_observe_new_result_assignment(self):
        code = "def simple_math_problem():\n    try:\n        return 2\n    finally:\n        print('done')\n"
        with self.assertRaisesRegex(StyleTransformError, "unsupported_control_flow"):
            transform_code(code)

    def test_generator_cannot_capture_new_result_binding(self):
        code = "def simple_math_problem():\n    return sum(result for item in range(2))\n"
        with self.assertRaisesRegex(StyleTransformError, "result_capture"):
            transform_code(code)


if __name__ == "__main__":
    unittest.main()
