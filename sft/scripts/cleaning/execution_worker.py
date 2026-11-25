#!/usr/bin/env python3
"""Execute one audited original/wrapped code pair under explicit resource/API limits.

JSON stdin: {"original_code": "...", "wrapped_code": "..."}.
JSON stdout: {"original": {...}, "wrapped": {...}, "upstream": {...}}.
The parent must also enforce a wall-clock timeout and use an isolated cwd/env.
This worker never modifies source data and does not execute code without OS limits.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import importlib.util
import io
import json
import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

MEMORY_BYTES = 512 * 1024 * 1024
MAX_INPUT_BYTES = 1024 * 1024
MAX_CODE_CHARS = 200_000
MAX_OUTPUT_CHARS = 8192
_JOB_HANDLE = None


class PolicyError(Exception):
    pass


class OutputError(Exception):
    pass


def install_resource_limits():
    """Keep a kill-on-close, single-process Windows job alive until process exit."""
    global _JOB_HANDLE
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise PolicyError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")
        limits = ExtendedLimits()
        # ACTIVE_PROCESS (0x8), PROCESS_MEMORY (0x100), KILL_ON_JOB_CLOSE (0x2000).
        limits.BasicLimitInformation.LimitFlags = 0x8 | 0x100 | 0x2000
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        limits.ProcessMemoryLimit = MEMORY_BYTES
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            kernel.CloseHandle(handle)
            raise PolicyError(f"SetInformationJobObject failed: {error}")
        if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
            error = ctypes.get_last_error()
            kernel.CloseHandle(handle)
            raise PolicyError(f"AssignProcessToJobObject failed: {error}")
        _JOB_HANDLE = handle
        return
    if sys.platform.startswith("linux"):
        import resource
        for kind, value in (
            (resource.RLIMIT_AS, MEMORY_BYTES), (resource.RLIMIT_CPU, 4),
            (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NPROC, 0),
        ):
            resource.setrlimit(kind, (value, value))
        return
    raise PolicyError("No supported OS resource limiter; execution refused")


MATH_API = {"ceil", "sqrt", "log", "log2"}
SYMPY_API = {"symbols", "Symbol", "Eq", "solve"}
DATETIME_API = {"datetime", "timedelta", "date"}
METHOD_API = {"is_integer", "index", "values", "items", "sort", "append", "total_seconds", "strptime"}
BUILTIN_API = {"abs", "min", "max", "sum", "len", "range", "enumerate", "int", "float", "str", "bool", "round", "print", "zip"}
MODULE_API = {"math": MATH_API, "sympy": SYMPY_API, "datetime": DATETIME_API}
ALLOWED_NODES = {
    ast.Module, ast.Assign, ast.AugAssign, ast.Expr, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.DictComp,
    ast.SetComp, ast.GeneratorExp, ast.comprehension, ast.If, ast.For, ast.While,
    ast.Break, ast.Continue, ast.Pass, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Return, ast.Lambda, ast.Import, ast.ImportFrom, ast.alias,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
}


def check_identifier(name):
    if "__" in name:
        raise PolicyError(f"Double-underscore identifier blocked: {name}")


def validate_ast(code):
    if not isinstance(code, str) or not code.strip() or len(code) > MAX_CODE_CHARS:
        raise PolicyError("Invalid or overlong source code")
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    if len(nodes) > 25000:
        raise PolicyError("AST too large")
    imported_names = set()
    module_aliases = {"math": "math"}
    explicit_module_aliases = set()
    function_names = set()
    for node in nodes:
        if type(node) not in ALLOWED_NODES:
            raise PolicyError(f"AST node blocked: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in MODULE_API:
                    raise PolicyError(f"Import blocked: {alias.name}")
                local = alias.asname or alias.name
                check_identifier(local)
                if local in BUILTIN_API:
                    raise PolicyError("Import would overwrite a builtin")
                module_aliases[local] = alias.name
                explicit_module_aliases.add(local)
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in MODULE_API:
                raise PolicyError(f"Import blocked: {node.module}")
            for alias in node.names:
                if alias.name not in MODULE_API[node.module]:
                    raise PolicyError(f"Imported symbol blocked: {alias.name}")
                local = alias.asname or alias.name
                check_identifier(local)
                if local in BUILTIN_API:
                    raise PolicyError("Import would overwrite a builtin")
                imported_names.add(local)
        if isinstance(node, ast.FunctionDef):
            check_identifier(node.name)
            if node.name == "print" or node.decorator_list or node.returns or node.type_params:
                raise PolicyError("Decorated/annotated/builtin-rebinding function blocked")
            function_names.add(node.name)

    # Numeric builtins may legitimately be shadowed by scalar/loop variables.
    # The implicit math convenience binding is also not an explicit import.
    # Print and explicitly imported capabilities remain protected from rebinding.
    protected = {"print"} | imported_names | explicit_module_aliases
    callable_names = BUILTIN_API | imported_names | function_names
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in protected:
            raise PolicyError(f"Rebinding protected function name: {node.name}")
        if isinstance(node, ast.Name):
            check_identifier(node.id)
            if isinstance(node.ctx, ast.Store) and node.id in protected:
                raise PolicyError(f"Rebinding protected name: {node.id}")
        elif isinstance(node, ast.arg):
            check_identifier(node.arg)
            if node.arg in protected or node.annotation:
                raise PolicyError("Protected/annotated parameter blocked")
        elif isinstance(node, ast.keyword):
            if node.arg is None:
                raise PolicyError("Keyword unpacking blocked")
            check_identifier(node.arg)
        elif isinstance(node, ast.Attribute):
            check_identifier(node.attr)
            if not isinstance(node.ctx, ast.Load):
                raise PolicyError("Attribute mutation blocked")
            if isinstance(node.value, ast.Name) and node.value.id in module_aliases:
                if node.attr not in MODULE_API[module_aliases[node.value.id]]:
                    raise PolicyError(f"Module attribute blocked: {node.attr}")
            elif node.attr not in METHOD_API:
                raise PolicyError(f"Method/attribute blocked: {node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in callable_names:
                    raise PolicyError(f"Call blocked: {node.func.id}")
                if node.func.id == "print" and (len(node.args) != 1 or node.keywords):
                    raise PolicyError("Only one-argument print without keywords is allowed")
            elif not isinstance(node.func, ast.Attribute):
                raise PolicyError("Indirect call blocked")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, str, bool, type(None))):
                raise PolicyError("Constant type blocked")
            if isinstance(node.value, str) and len(node.value) > 4096:
                raise PolicyError("String literal too long")
            if isinstance(node.value, int) and node.value.bit_length() > 4096:
                raise PolicyError("Integer literal too large")
    return tree


class LimitedStringIO(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.characters = 0

    def write(self, value):
        self.characters += len(value)
        if self.characters > MAX_OUTPUT_CHARS:
            raise OutputError("Captured output exceeds limit")
        return super().write(value)


class PrintCapture:
    def __init__(self):
        self.values = []
        self.text = LimitedStringIO()

    def print(self, *args, **kwargs):
        if len(args) != 1 or kwargs:
            raise PolicyError("Only one-argument print without keywords is allowed")
        if self.values:
            raise OutputError("More than one print output")
        self.values.append(args[0])
        self.text.write(str(args[0]))
        self.text.write("\n")


def safe_sympy_facade():
    import sympy

    identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

    def safe_names(names, *, one=False):
        if not isinstance(names, str) or len(names) > 256:
            raise PolicyError("Symbol names must be a short identifier string")
        parts = re.split(r"[\s,]+", names.strip())
        if not parts or len(parts) > 16 or (one and len(parts) != 1):
            raise PolicyError("Invalid symbol-name list")
        if any(not identifier.fullmatch(part) or "__" in part for part in parts):
            raise PolicyError("Invalid symbol identifier")
        return names

    def assumptions(kwargs):
        if any(k not in {"real", "positive", "integer", "nonnegative", "nonzero"} or not isinstance(v, bool) for k, v in kwargs.items()):
            raise PolicyError("Unsupported symbol assumption")
        return kwargs

    def symbols(names, **kwargs):
        return sympy.symbols(safe_names(names), **assumptions(kwargs))

    def Symbol(name, **kwargs):
        return sympy.Symbol(safe_names(name, one=True), **assumptions(kwargs))

    def expressions(value, depth=0):
        if depth > 8:
            raise PolicyError("Nested symbolic arguments too deep")
        if isinstance(value, (int, float, bool, sympy.Basic)):
            return
        if isinstance(value, (list, tuple)) and len(value) <= 256:
            for item in value:
                expressions(item, depth + 1)
            return
        raise PolicyError("Symbolic API accepts numeric/symbolic expressions, never strings")

    def Eq(lhs, rhs):
        expressions(lhs)
        expressions(rhs)
        return sympy.Eq(lhs, rhs)

    def solve(*args, **kwargs):
        if not args or kwargs:
            raise PolicyError("Unsupported solve signature")
        for arg in args:
            expressions(arg)
        return sympy.solve(*args)

    return SimpleNamespace(symbols=symbols, Symbol=Symbol, Eq=Eq, solve=solve)


def math_environment(capture):
    math_facade = SimpleNamespace(**{name: getattr(math, name) for name in MATH_API})

    def strptime(value, format):
        # CPython's datetime.strptime imports _strptime using its calling frame.
        # This trusted wrapper keeps that internal import outside sample builtins.
        if not isinstance(value, str) or not isinstance(format, str) or max(len(value), len(format)) > 512:
            raise PolicyError("strptime requires short string arguments")
        return datetime.datetime.strptime(value, format)

    facades = {
        "math": math_facade,
        "datetime": SimpleNamespace(datetime=SimpleNamespace(strptime=strptime), timedelta=datetime.timedelta, date=datetime.date),
    }

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in MODULE_API:
            raise PolicyError(f"Import blocked: {name}")
        if any(item not in MODULE_API[name] for item in (fromlist or ())):
            raise PolicyError("Imported symbol blocked")
        if name == "sympy" and name not in facades:
            facades[name] = safe_sympy_facade()
        return facades[name]

    builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "enumerate": enumerate, "int": int, "float": float,
        "str": str, "bool": bool, "round": round, "zip": zip,
        "print": capture.print, "__import__": restricted_import,
    }
    return {"__builtins__": builtins, "math": math_facade}


NUMERIC_TEXT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
FRACTION_TEXT = re.compile(r"[+-]?\d+/[+-]?\d+\Z")


def finite_value(value):
    if isinstance(value, bool):
        raise OutputError("Boolean is not an accepted numeric answer")
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise OutputError("Numeric answer too large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OutputError("Non-finite numeric answer")
        return value
    sympy = sys.modules.get("sympy")
    if sympy is not None and isinstance(value, sympy.Basic) and value.is_number and value.is_real:
        converted = float(value)
        if not math.isfinite(converted):
            raise OutputError("Non-finite symbolic answer")
        return converted
    raise OutputError(f"Unsupported numeric answer type: {type(value).__name__}")


def parse_numeric_output(text):
    text = text.strip()
    if len(text) > 1024:
        raise OutputError("Numeric output too long")
    if NUMERIC_TEXT.fullmatch(text):
        return finite_value(float(text) if any(c in text for c in ".eE") else int(text))
    if FRACTION_TEXT.fullmatch(text):
        return finite_value(float(Fraction(text)))
    raise OutputError("Printed output is not one complete numeric value")


def failure(error, status=None):
    if status is None:
        status = "policy_error" if isinstance(error, PolicyError) else "output_error" if isinstance(error, OutputError) else "execution_error"
    return {"status": status, "value": None, "error": f"{type(error).__name__}: {str(error)[:1000]}"}


def execute_math(code, *, wrapped):
    capture = PrintCapture()
    ns = math_environment(capture)
    try:
        with contextlib.redirect_stdout(LimitedStringIO()), contextlib.redirect_stderr(LimitedStringIO()):
            exec(compile(code, "<wrapped>" if wrapped else "<original>", "exec"), ns, ns)
            if wrapped:
                fn = ns.get("simple_math_problem")
                if not callable(fn):
                    raise OutputError("Missing simple_math_problem function")
                out = fn()
                value = finite_value(out)
            else:
                if len(capture.values) != 1:
                    raise OutputError("Expected exactly one print output")
                out = capture.values[0]
                if isinstance(out, bool):
                    raise OutputError("Boolean is not an accepted numeric answer")
                value = parse_numeric_output(capture.text.getvalue())
        return {"status": "ok", "value": value, "raw_type": type(out).__name__}
    except Exception as error:
        return failure(error)


def execute_upstream(code, upstream_path):
    try:
        spec = importlib.util.spec_from_file_location("fixed_upstream_gsm8k", upstream_path)
        if spec is None or spec.loader is None:
            raise PolicyError("Cannot load upstream scorer")
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(LimitedStringIO()), contextlib.redirect_stderr(LimitedStringIO()):
            spec.loader.exec_module(module)
        # Only tighten output resource bounds; do not alter upstream builtins,
        # math-only imports, extraction, numeric conversion, or fallback logic.
        module.io = SimpleNamespace(StringIO=LimitedStringIO)
        extracted = module._extract_code(code)
        validate_ast(extracted)
        ns = module._safe_exec_no_timer(extracted)
        fn = ns.get("simple_math_problem")
        if not callable(fn):
            raise OutputError("Missing upstream simple_math_problem function")
        stdout, stderr = LimitedStringIO(), LimitedStringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            out = fn()
        value = module._to_number(out)
        fallback = value is None
        if fallback:
            value = module._extract_text_answer(stdout.getvalue())
        if value is None:
            raise OutputError("Upstream scorer could not extract a number")
        # Upstream _to_number already turns booleans into int. Keep that exact
        # behavior here; strict mathematical checks above reject booleans.
        value = finite_value(value)
        return {"status": "ok", "value": value, "raw_type": type(out).__name__, "used_stdout_fallback": fallback}
    except Exception as error:
        return failure(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--upstream-file", type=Path, required=True)
    args = parser.parse_args()
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    try:
        install_resource_limits()
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise PolicyError("Input JSON too large")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"original_code", "wrapped_code"}:
            raise PolicyError("Expected only original_code and wrapped_code")
        validate_ast(payload["original_code"])
        validate_ast(payload["wrapped_code"])
    except Exception as error:
        result = {name: failure(error, "policy_error") for name in ("original", "wrapped", "upstream")}
    else:
        result = {
            "original": execute_math(payload["original_code"], wrapped=False),
            "wrapped": execute_math(payload["wrapped_code"], wrapped=True),
            "upstream": execute_upstream(payload["wrapped_code"], args.upstream_file.resolve()),
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
