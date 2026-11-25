"""Binding-aware, source-preserving alpha renaming of mathematical answer code."""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import io
import tokenize
import unicodedata


class VariableTransformError(ValueError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


@dataclass(eq=False)
class Binding:
    name: str
    scope: "Scope"
    position: tuple[int, int]
    kinds: set[str] = field(default_factory=set)
    function: ast.FunctionDef | None = None
    replacement: str | None = None


@dataclass(eq=False)
class Scope:
    index: int
    label: str
    kind: str
    lineno: int
    parent: "Scope | None"
    bindings: dict[str, Binding] = field(default_factory=dict)
    parameters: list[str] = field(default_factory=list)

    def bind(self, name, node, kind, function=None):
        position = (node.lineno, node.col_offset)
        binding = self.bindings.setdefault(name, Binding(name, self, position))
        binding.position = min(binding.position, position)
        binding.kinds.add(kind)
        if function is not None:
            if binding.function is not None:
                raise VariableTransformError("duplicate_helper", name)
            binding.function = function
        return binding

    def resolve(self, name):
        scope = self
        while scope is not None:
            if name in scope.bindings:
                return scope.bindings[name]
            scope = scope.parent
        return None


def _parse(code):
    try:
        tree = ast.parse(code)
        compile(tree, "<variable-transform>", "exec")  # Syntax/scope check only.
    except (SyntaxError, ValueError, TypeError) as error:
        raise VariableTransformError("invalid_syntax", str(error)) from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise VariableTransformError("module_shape", "expected one top-level function")
    function = tree.body[0]
    if (function.name != "simple_math_problem" or function.args.posonlyargs or function.args.args
            or function.args.kwonlyargs or function.args.vararg or function.args.kwarg
            or function.decorator_list or not isinstance(function.returns, ast.Name)
            or function.returns.id != "int"):
        raise VariableTransformError("signature", "expected simple_math_problem() -> int")
    unsupported = (ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal, ast.NamedExpr,
                   ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
                   ast.Yield, ast.YieldFrom, ast.Try, ast.TryStar, ast.With,
                   ast.Match, ast.JoinedStr)
    observed = {"locals", "globals", "vars", "dir", "eval", "exec", "compile", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, unsupported):
            raise VariableTransformError("unsupported_scope", type(node).__name__)
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or getattr(node, "type_params", [])):
            raise VariableTransformError("unsupported_signature", node.name)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in observed:
            raise VariableTransformError("namespace_observation", node.id)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise VariableTransformError("namespace_observation", node.attr)
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            raise VariableTransformError("star_import", "wildcard bindings cannot be audited")
        if isinstance(node, ast.Import) and any("." in alias.name and alias.asname is None for alias in node.names):
            raise VariableTransformError("dotted_import", "adding an alias would bind a different module")
        if isinstance(node, ast.Call) and any(keyword.arg is None for keyword in node.keywords):
            raise VariableTransformError("dynamic_keywords", "**kwargs cannot be statically rebound")
    return tree


class Analysis:
    def __init__(self, tree):
        self.tree = tree
        self.scopes = [Scope(0, "<module>", "module", 1, None)]
        self.scope_for = {}
        self.occurrences = []  # node, field, lexical scope, role
        self.calls = []
        self.visit(tree, self.scopes[0])
        for scope in self.scopes:
            for binding in scope.bindings.values():
                if "function" in binding.kinds and binding.kinds != {"function"}:
                    raise VariableTransformError("helper_rebinding", binding.name)
        self.keyword_bindings = {}
        for call, scope in self.calls:
            if not call.keywords:
                continue
            binding = scope.resolve(call.func.id) if isinstance(call.func, ast.Name) else None
            if binding is not None and binding.function is not None:
                function_scope = self.scope_for[binding.function]
                for keyword in call.keywords:
                    if keyword.arg not in function_scope.parameters:
                        raise VariableTransformError("unknown_helper_keyword", keyword.arg)
                    self.keyword_bindings[keyword] = function_scope.bindings[keyword.arg]
            elif binding is not None and "import" not in binding.kinds:
                raise VariableTransformError("indirect_keyword_call", ast.dump(call.func))

    def child(self, node, parent, kind, label):
        scope = Scope(len(self.scopes), f"{parent.label}/{label}", kind, node.lineno, parent)
        self.scopes.append(scope)
        self.scope_for[node] = scope
        return scope

    def visit(self, node, scope):
        if isinstance(node, ast.FunctionDef):
            scope.bind(node.name, node, "function", node)
            self.occurrences.append((node, "name", scope, "function"))
            args = node.args
            for value in [*args.defaults, *(x for x in args.kw_defaults if x is not None)]:
                self.visit(value, scope)
            all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            if args.vararg:
                all_args.append(args.vararg)
            if args.kwarg:
                all_args.append(args.kwarg)
            for argument in all_args:
                if argument.annotation is not None:
                    self.visit(argument.annotation, scope)
            if node.returns is not None:
                self.visit(node.returns, scope)
            child = self.child(node, scope, "function", f"{node.name}@{node.lineno}")
            for argument in all_args:
                child.bind(argument.arg, argument, "parameter")
                child.parameters.append(argument.arg)
                self.occurrences.append((argument, "arg", child, "parameter"))
            for statement in node.body:
                self.visit(statement, child)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if any(generator.is_async for generator in node.generators):
                raise VariableTransformError("unsupported_scope", "async comprehension")
            # Python evaluates the first iterable in the enclosing scope.
            self.visit(node.generators[0].iter, scope)
            child = self.child(node, scope, "comprehension", f"{type(node).__name__}@{node.lineno}:{node.col_offset}")
            for index, generator in enumerate(node.generators):
                if index:
                    self.visit(generator.iter, child)
                self.visit(generator.target, child)
                for condition in generator.ifs:
                    self.visit(condition, child)
            for value in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
                self.visit(value, child)
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                scope.bind(node.id, node, "local")
            self.occurrences.append((node, "id", scope, "name"))
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name
                scope.bind(name, alias, "import")
                self.occurrences.append((alias, "asname", scope, "import"))
            return
        if isinstance(node, ast.Call):
            self.calls.append((node, scope))
        for child in ast.iter_child_nodes(node):
            self.visit(child, scope)

    def binding_for(self, node, field, scope, role):
        name = (node.asname or node.name) if role == "import" else getattr(node, field)
        return scope.resolve(name)

    def canonical(self):
        expected = copy.deepcopy(self.tree)
        copies = {id(old): new for old, new in zip(ast.walk(self.tree), ast.walk(expected))}
        canonical = {}
        for scope in self.scopes:
            for order, binding in enumerate(sorted(scope.bindings.values(), key=lambda b: b.position)):
                canonical[binding] = f"__binding_{scope.index}_{order}"
        for node, field, scope, role in self.occurrences:
            binding = self.binding_for(node, field, scope, role)
            if binding is not None:
                setattr(copies[id(node)], field, canonical[binding])
        for keyword, binding in self.keyword_bindings.items():
            copies[id(keyword)].arg = canonical[binding]
        return ast.dump(expected, include_attributes=False)


def _offset(code, row, column):
    lines = code.split("\n")
    prefix = lines[row - 1].encode("utf-8")[:column].decode("utf-8")
    return sum(len(line) + 1 for line in lines[:row - 1]) + len(prefix)


def _plan(code):
    tree = _parse(code)
    analysis = Analysis(tree)
    reserved = {"result"}
    for node, field, scope, role in analysis.occurrences:
        binding = analysis.binding_for(node, field, scope, role)
        if binding is None:
            reserved.add(getattr(node, field))
        elif "function" in binding.kinds or binding.scope.kind == "module":
            reserved.add(binding.name)
    bindings = sorted((b for scope in analysis.scopes for b in scope.bindings.values()
                       if scope.kind != "module" and "function" not in b.kinds and b.name != "result"),
                      key=lambda b: b.position)
    number = 1
    for binding in bindings:
        while f"x_{number}" in reserved:
            number += 1
        binding.replacement = f"x_{number}"
        reserved.add(binding.replacement)
        number += 1
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    lines = code.split("\n")
    line_offsets = [0]
    for line in lines[:-1]:
        line_offsets.append(line_offsets[-1] + len(line) + 1)
    name_tokens = [(token, line_offsets[token.start[0]-1] + token.start[1],
                    line_offsets[token.end[0]-1] + token.end[1])
                   for token in tokens if token.type == tokenize.NAME]
    edits = []
    occurrences = 0
    imports_aliased = 0

    def token_edit(node, old, new, *, last=False):
        start = _offset(code, node.lineno, node.col_offset)
        end = _offset(code, node.end_lineno, node.end_col_offset)
        candidates = [(token, begin, finish) for token, begin, finish in name_tokens
                      if start <= begin < end and unicodedata.normalize("NFKC", token.string) == old]
        if not candidates:
            raise VariableTransformError("source_span", old)
        _, begin, finish = candidates[-1] if last else candidates[0]
        edits.append((begin, finish, new))

    for node, field, scope, role in analysis.occurrences:
        binding = analysis.binding_for(node, field, scope, role)
        if binding is None or binding.replacement is None or binding.replacement == binding.name:
            continue
        replacement = binding.replacement
        if role == "import" and node.asname is None:
            end = _offset(code, node.end_lineno, node.end_col_offset)
            edits.append((end, end, f" as {replacement}"))
            imports_aliased += 1
        else:
            token_edit(node, binding.name, replacement, last=role == "import")
        occurrences += 1
    for keyword, binding in analysis.keyword_bindings.items():
        if binding.replacement is not None and binding.replacement != binding.name:
            token_edit(keyword, keyword.arg, binding.replacement)
            occurrences += 1
    transformed = code
    previous = len(code) + 1
    for start, end, replacement in sorted(edits, reverse=True):
        if end > previous:
            raise VariableTransformError("overlapping_edits", "identifier source spans overlap")
        transformed = transformed[:start] + replacement + transformed[end:]
        previous = start
    metadata = {"scopes": [{"scope": scope.label, "kind": scope.kind, "lineno": scope.lineno,
                             "mapping": {b.name: b.replacement for b in sorted(scope.bindings.values(), key=lambda b: b.position)
                                         if b.replacement is not None}, "parameters": scope.parameters}
                            for scope in analysis.scopes if scope.kind != "module"],
                "bindings_renamed": sum(b.replacement != b.name for b in bindings),
                "name_occurrences_changed": occurrences, "imports_aliased": imports_aliased,
                "ast_verified": True}
    return transformed, analysis, metadata


def verify_transform(original: str, transformed: str) -> None:
    """Require the prescribed identifier-only edits and equivalent lexical bindings."""
    expected, original_analysis, _ = _plan(original)
    new_analysis = Analysis(_parse(transformed))
    if original_analysis.canonical() != new_analysis.canonical():
        raise VariableTransformError("binding_mismatch", "a computation, literal, or lexical binding changed")
    if transformed != expected:
        raise VariableTransformError("source_mismatch", "changes exceed the prescribed alpha renaming")


def transform_code(code: str) -> tuple[str, dict]:
    transformed, analysis, metadata = _plan(code)
    actual = Analysis(_parse(transformed))
    if analysis.canonical() != actual.canonical():
        raise VariableTransformError("binding_mismatch", "renaming changed lexical resolution")
    return transformed, metadata
