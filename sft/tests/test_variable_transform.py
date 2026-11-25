"""Scope, collision and source-preservation fixtures for variable alpha renaming."""
import unittest

from mdlm_sft.variable_transform import VariableTransformError, transform_code, verify_transform


class VariableTransformTests(unittest.TestCase):
    def value(self, code, **globals_):
        namespace = dict(globals_)
        exec(code, namespace)  # Hand-written fixtures only; dataset code is never executed here.
        return namespace["simple_math_problem"]()

    def checked(self, original, **globals_):
        transformed, metadata = transform_code(original)
        verify_transform(original, transformed)
        self.assertEqual(self.value(original, **globals_), self.value(transformed, **globals_))
        self.assertEqual(transform_code(transformed)[0], transformed)
        return transformed, metadata

    def test_first_binding_order_and_result_preserved_with_existing_x_names(self):
        original = "def simple_math_problem() -> int:\n    apples = 7\n    x_1 = apples / 2\n    apples += x_1\n    result = (int(apples))\n    return result\n"
        transformed, metadata = self.checked(original)
        self.assertEqual(transformed, "def simple_math_problem() -> int:\n    x_1 = 7\n    x_2 = x_1 / 2\n    x_1 += x_2\n    result = (int(x_1))\n    return result\n")
        self.assertEqual(metadata["scopes"][0]["mapping"], {"apples": "x_1", "x_1": "x_2"})

    def test_helper_name_can_equal_its_own_local_variable(self):
        original = "def simple_math_problem() -> int:\n    def total(count):\n        total = count * 2\n        return total\n    result = total(5)\n    return result\n"
        transformed, metadata = self.checked(original)
        self.assertIn("def total(x_1):", transformed)
        self.assertIn("x_2 = x_1 * 2", transformed)
        self.assertIn("result = total(5)", transformed)
        self.assertEqual(metadata["scopes"][1]["mapping"], {"count": "x_1", "total": "x_2"})

    def test_nested_parameters_shadow_outer_locals_and_keyword_calls_follow_binding(self):
        original = "def simple_math_problem() -> int:\n    amount = 7\n    def helper(amount, extra=amount):\n        subtotal = amount + extra\n        return subtotal\n    result = helper(amount=2, extra=3) + amount\n    return result\n"
        transformed, metadata = self.checked(original)
        self.assertIn("def helper(x_2, x_3=x_1):", transformed)
        self.assertIn("helper(x_2=2, x_3=3) + x_1", transformed)
        self.assertEqual(metadata["scopes"][1]["parameters"], ["amount", "extra"])

    def test_closure_import_alias_and_attribute_spelling(self):
        original = "def simple_math_problem() -> int:\n    import math\n    from math import ceil as round_up\n    amount = 7\n    def helper(divisor):\n        return math.ceil(amount / divisor) + round_up(1.2)\n    result = helper(2)\n    return result\n"
        transformed, metadata = self.checked(original)
        self.assertIn("import math as x_1", transformed)
        self.assertIn("from math import ceil as x_2", transformed)
        self.assertIn("x_1.ceil(x_3 / x_4) + x_2(1.2)", transformed)
        self.assertEqual(metadata["imports_aliased"], 1)

    def test_comprehension_iterable_scope_shadowing_and_closure(self):
        original = "def simple_math_problem() -> int:\n    numbers = [2, 3]\n    increment = 4\n    values = [numbers + increment for numbers in numbers]\n    mapping = {key: value + increment for key, value in enumerate(values)}\n    result = sum(value for value in mapping.values())\n    return result\n"
        transformed, metadata = self.checked(original)
        self.assertIn("[x_4 + x_2 for x_4 in x_1]", transformed)
        replacements = [new for scope in metadata["scopes"] for new in scope["mapping"].values()]
        self.assertEqual(len(replacements), len(set(replacements)))
        self.assertEqual(sum(s["kind"] == "comprehension" for s in metadata["scopes"]), 3)

    def test_unresolved_global_x_name_is_not_captured(self):
        original = "def simple_math_problem() -> int:\n    amount = 3\n    def helper():\n        return amount + x_1\n    result = helper()\n    return result\n"
        transformed, metadata = self.checked(original, x_1=10)
        self.assertEqual(metadata["scopes"][0]["mapping"], {"amount": "x_2"})
        self.assertIn("return x_2 + x_1", transformed)

    def test_unicode_comments_strings_and_numeric_spelling_remain_exact(self):
        original = 'def simple_math_problem() -> int:\n    标签 = "标签 # amount"  # 标签 remains in comment\n    金额 = 1.40\n    result = (金额 + len(标签))\n    return result\n'
        transformed, _ = self.checked(original)
        self.assertIn('x_1 = "标签 # amount"  # 标签 remains in comment', transformed)
        self.assertIn("x_2 = 1.40", transformed)

    def test_verifier_rejects_calculation_binding_and_format_changes(self):
        original = "def simple_math_problem() -> int:\n    alpha = 2\n    beta = 3\n    result = alpha + beta\n    return result\n"
        transformed, _ = self.checked(original)
        for changed in (transformed.replace("x_1 + x_2", "x_1 - x_2"),
                        transformed.replace("x_1 + x_2", "x_1 + x_1"), transformed + "\n"):
            with self.subTest(changed=changed), self.assertRaises(VariableTransformError):
                verify_transform(original, changed)

    def test_unsupported_namespace_or_dynamic_binding_is_an_explicit_error(self):
        bodies = ["global value\n    result = 1", "result = len(locals())",
                  "import xml.etree\n    result = 1", "result = helper(**{})",
                  "result = (lambda value: value)(1)"]
        for body in bodies:
            code = "def simple_math_problem() -> int:\n    " + body + "\n    return result\n"
            with self.subTest(body=body), self.assertRaises(VariableTransformError):
                transform_code(code)


if __name__ == "__main__":
    unittest.main()
