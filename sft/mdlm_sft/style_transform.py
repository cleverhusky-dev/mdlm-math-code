"""Normalize answer-code presentation without changing its calculation."""
from __future__ import annotations

import ast
import copy
import io
import symtable
import tokenize


class StyleTransformError(ValueError):
    """The source does not meet the supported, auditable transformation rules."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def _parse(code: str) -> ast.Module:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        raise StyleTransformError("invalid_syntax", str(exc)) from exc


def _tokens(code: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise StyleTransformError("invalid_tokens", str(exc)) from exc


def _offset(code: str, row: int, column: int, *, ast_column: bool = False) -> int:
    # AST columns count UTF-8 bytes; tokenize columns count Unicode characters.
    lines = code.split("\n")
    if ast_column:
        column = len(lines[row - 1].encode("utf-8")[:column].decode("utf-8"))
    return sum(len(line) + 1 for line in lines[:row - 1]) + column


def _edit(code: str, edits: list[tuple[int, int, str]]) -> str:
    previous_start = len(code) + 1
    for start, end, replacement in sorted(edits, reverse=True):
        if end > previous_start:
            raise StyleTransformError("overlapping_edits", "source edits overlap")
        code = code[:start] + replacement + code[end:]
        previous_start = start
    return code


def _outer_returns(function: ast.FunctionDef) -> list[ast.Return]:
    returns: list[ast.Return] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Return):
            returns.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in function.body:
        visit(statement)
    return returns


def _already_result(node: ast.Return) -> bool:
    return isinstance(node.value, ast.Name) and node.value.id == "result"


def _validate_source(tree: ast.Module, code: str) -> tuple[ast.FunctionDef, list[ast.Return]]:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise StyleTransformError("module_shape", "expected one top-level function")
    function = tree.body[0]
    args = function.args
    if (function.name != "simple_math_problem" or function.decorator_list
            or args.posonlyargs or args.args or args.kwonlyargs or args.vararg or args.kwarg
            or getattr(function, "type_params", [])):
        raise StyleTransformError("signature", "expected an undecorated, zero-argument simple_math_problem")
    if function.returns is not None and not (
        isinstance(function.returns, ast.Name) and function.returns.id == "int"
    ):
        raise StyleTransformError("return_annotation", "only an absent annotation or bare int is supported")

    for node in ast.walk(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node, clean=False) is not None:
                raise StyleTransformError("existing_docstring", "docstrings require an explicit policy")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.Try, ast.TryStar, ast.With,
                             ast.AsyncWith, ast.Yield, ast.YieldFrom, ast.Await,
                             ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            raise StyleTransformError("unsupported_control_flow", type(node).__name__)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in {
            "locals", "globals", "vars", "dir", "eval", "exec", "compile", "__import__",
            "simple_math_problem",
        }:
            raise StyleTransformError("namespace_observation", node.id)

    returns = _outer_returns(function)
    if not returns or any(node.value is None for node in returns):
        raise StyleTransformError("return_shape", "expected value-returning statements")
    if any(not _already_result(node) for node in returns):
        for node in ast.walk(function):
            if (isinstance(node, ast.FunctionDef) and node is not function) or isinstance(
                node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                if any(isinstance(child, ast.Name) and child.id == "result" for child in ast.walk(node)):
                    raise StyleTransformError("result_capture", "a nested scope refers to result")
        symbols = next(
            child for child in symtable.symtable(code, "<style-source>", "exec").get_children()
            if child.get_name() == function.name
        )
        if "result" in symbols.get_identifiers() and not symbols.lookup("result").is_local():
            raise StyleTransformError("result_binding", "introducing a local result would change name resolution")

    depth = 0
    for token in _tokens(code):
        if token.type == tokenize.INDENT:
            depth += 1
            if token.string != "    " * depth:
                raise StyleTransformError("indentation", "expected four spaces per indentation level")
        elif token.type == tokenize.DEDENT:
            depth -= 1
        elif token.type == tokenize.OP and token.string == ";":
            raise StyleTransformError("semicolon", "one statement per physical line is required")
    return function, returns


def _expected_tree(tree: ast.Module) -> ast.Module:
    expected = copy.deepcopy(tree)
    function = expected.body[0]
    function.returns = ast.Name(id="int", ctx=ast.Load())

    class NormalizeReturns(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node  # Nested helper functions keep their original returns.

        def visit_Return(self, node):
            if _already_result(node):
                return node
            return [
                ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=node.value),
                ast.Return(value=ast.Name(id="result", ctx=ast.Load())),
            ]

    normalizer = NormalizeReturns()
    body = []
    for statement in function.body:
        normalized = normalizer.visit(statement)
        body.extend(normalized if isinstance(normalized, list) else [normalized])
    function.body = body
    return expected


def verify_transform(original: str, transformed: str) -> None:
    """Reject any AST difference beyond the approved annotation/return changes."""
    original_tree = _parse(original)
    _validate_source(original_tree, original)
    actual = _parse(transformed)
    expected = _expected_tree(original_tree)
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise StyleTransformError("ast_mismatch", "a computation, literal, scope, or statement changed")


def transform_code(code: str) -> tuple[str, dict]:
    """Remove comments, annotate the outer function, and normalize its returns.

    Expressions retain their original spelling, casts, and parentheses. Nested
    helper functions keep their signatures and control flow. No sample is run.
    Unsupported sources raise StyleTransformError rather than being discarded.
    """
    tree = _parse(code)
    function, returns = _validate_source(tree, code)
    source_tokens = _tokens(code)
    comment_count = sum(token.type == tokenize.COMMENT for token in source_tokens)

    header_start = _offset(code, function.lineno, function.col_offset, ast_column=True)
    header_end = None
    for token in source_tokens:
        if _offset(code, *token.start) >= header_start and token.type == tokenize.OP and token.string == ":":
            header_end = _offset(code, *token.end)
            header_line = token.end[0]
            break
    if header_end is None or function.body[0].lineno <= header_line:
        raise StyleTransformError("inline_body", "a multiline function body is required")
    edits = [(header_start, header_end, "def simple_math_problem() -> int:")]
    for node in returns:
        start = _offset(code, node.lineno, node.col_offset, ast_column=True)
        end = _offset(code, node.end_lineno, node.end_col_offset, ast_column=True)
        if _already_result(node):
            replacement = "return result"
        else:
            expression_source = code[start + len("return"):end]
            indentation = " " * node.col_offset
            replacement = f"result ={expression_source}\n{indentation}return result"
        edits.append((start, end, replacement))
    transformed = _edit(code, edits)

    comment_edits = []
    for token in _tokens(transformed):
        if token.type != tokenize.COMMENT:
            continue
        start = _offset(transformed, *token.start)
        end = _offset(transformed, *token.end)
        line_start = _offset(transformed, token.start[0], 0)
        if not transformed[line_start:start].strip():
            start = line_start
            if transformed[end:end + 1] == "\n":
                end += 1
        comment_edits.append((start, end, ""))
    transformed = _edit(transformed, comment_edits)
    if not transformed.endswith("\n"):
        transformed += "\n"
    verify_transform(code, transformed)
    return transformed, {
        "comments_removed": comment_count,
        "return_annotation_added": function.returns is None,
        "outer_returns": len(returns),
        "returns_rewritten": sum(not _already_result(node) for node in returns),
        "nested_functions_preserved": sum(
            isinstance(node, ast.FunctionDef) and node is not function for node in ast.walk(function)
        ),
        "docstring_added": False,
        "ast_verified": True,
    }
