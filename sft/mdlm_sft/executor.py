"""Linux isolated math worker, invoked by scoring.py; never import model code.

This is a narrow evaluator for arithmetic programs, not a general-purpose hostile
code service. It refuses execution unless all OS resource/seccomp limits install.
"""
from __future__ import annotations

import ast
import contextlib
import ctypes
import ctypes.util
import errno
import io
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# CPython lazily uses unicodedata for non-ASCII identifier normalization,
# Unicode syntax diagnostics, and named (\\N{...}) string escapes. Load the
# extension while trusted startup can still open libraries, before seccomp.
# This does not expose the module to generated code: imports remain math-only.
import unicodedata

# -I intentionally removes the working directory. Only import our trusted sibling.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scoring import to_number, extract_text_answer, syntax_failure_status
else:
    from .scoring import to_number, extract_text_answer, syntax_failure_status

MAX_CODE_CHARS = 200_000
MAX_INPUT_BYTES = 1024 * 1024
MATH_API = {name for name in dir(math) if not name.startswith("_")}
METHOD_API = {"is_integer", "index", "values", "items", "sort", "append", "count"}
BUILTIN_API = {"abs", "min", "max", "sum", "len", "range", "enumerate",
               "int", "float", "str", "bool", "round", "print"}
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


class PolicyError(Exception):
    pass


class OutputLimitError(Exception):
    pass


def _identifier(name):
    if "__" in name:
        raise PolicyError(f"Double-underscore identifier blocked: {name}")


def validate_ast(code: str):
    if not isinstance(code, str) or len(code) > MAX_CODE_CHARS:
        raise PolicyError("Invalid or overlong program")
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    if len(nodes) > 25000:
        raise PolicyError("AST exceeds node limit")
    imported_names, module_aliases, explicit_aliases, function_names = set(), {"math"}, set(), set()
    for node in nodes:
        if type(node) not in ALLOWED_NODES:
            raise PolicyError(f"AST node blocked: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "math":
                    raise PolicyError(f"Import blocked: {alias.name}")
                local = alias.asname or alias.name
                _identifier(local)
                if local in BUILTIN_API:
                    raise PolicyError("Import rebinds a builtin")
                module_aliases.add(local)
                explicit_aliases.add(local)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module != "math":
                raise PolicyError(f"Import blocked: {node.module}")
            for alias in node.names:
                if alias.name not in MATH_API:
                    raise PolicyError(f"Imported math symbol blocked: {alias.name}")
                local = alias.asname or alias.name
                _identifier(local)
                if local in BUILTIN_API:
                    raise PolicyError("Import rebinds a builtin")
                imported_names.add(local)
        elif isinstance(node, ast.FunctionDef):
            _identifier(node.name)
            if node.decorator_list or getattr(node, "type_params", []):
                raise PolicyError("Decorators and generic functions blocked")
            # Permit only the bare builtin-name annotation in baseline outputs.
            # Python annotations do not coerce the function's returned value.
            if node.returns is not None and not (
                    isinstance(node.returns, ast.Name) and node.returns.id == "int"):
                raise PolicyError("Only the bare int return annotation is allowed")
            function_names.add(node.name)
    protected = {"print"} | imported_names | explicit_aliases
    callable_names = BUILTIN_API | imported_names | function_names
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in protected:
            raise PolicyError("Rebinding protected function")
        elif isinstance(node, ast.Name):
            _identifier(node.id)
            if isinstance(node.ctx, ast.Store) and node.id in protected:
                raise PolicyError("Rebinding protected name")
        elif isinstance(node, ast.arg):
            _identifier(node.arg)
            if node.arg in protected or node.annotation:
                raise PolicyError("Protected or annotated argument")
        elif isinstance(node, ast.keyword):
            if node.arg is None:
                raise PolicyError("Keyword unpacking blocked")
            _identifier(node.arg)
        elif isinstance(node, ast.Attribute):
            _identifier(node.attr)
            if not isinstance(node.ctx, ast.Load):
                raise PolicyError("Attribute mutation blocked")
            if isinstance(node.value, ast.Name) and node.value.id in module_aliases:
                if node.attr not in MATH_API:
                    raise PolicyError(f"Math attribute blocked: {node.attr}")
            elif node.attr not in METHOD_API:
                raise PolicyError(f"Attribute blocked: {node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in callable_names:
                    raise PolicyError(f"Call blocked: {node.func.id}")
                if node.func.id == "print" and any(k.arg not in {"sep", "end", "flush"} for k in node.keywords):
                    raise PolicyError("Unsupported print keyword")
            elif not isinstance(node.func, ast.Attribute):
                raise PolicyError("Indirect call blocked")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, str, bool, type(None))):
                raise PolicyError("Unsupported constant type")
            if isinstance(node.value, str) and len(node.value) > 4096:
                raise PolicyError("String literal exceeds limit")
            if isinstance(node.value, int) and node.value.bit_length() > 4096:
                raise PolicyError("Integer literal exceeds limit")
    return tree


class LimitedStringIO(io.StringIO):
    def __init__(self, limit):
        super().__init__()
        self.limit, self.characters = limit, 0

    def write(self, value):
        self.characters += len(value)
        if self.characters > self.limit:
            raise OutputLimitError("Captured output exceeds limit")
        return super().write(value)


def install_limits(timeout_s, memory_mb):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Linux required")
    import resource
    # Load the library before denying all filesystem opens. Absence is fatal.
    seccomp = ctypes.CDLL(ctypes.util.find_library("seccomp") or "libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes, seccomp.seccomp_init.restype = [ctypes.c_uint32], ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes, seccomp.seccomp_load.restype = [ctypes.c_void_p], ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "Cannot set no_new_privs")
    for kind, soft, hard in (
        (resource.RLIMIT_AS, memory_mb * 1024**2, memory_mb * 1024**2),
        (resource.RLIMIT_CPU, max(1, math.ceil(timeout_s)), max(1, math.ceil(timeout_s)) + 1),
        (resource.RLIMIT_FSIZE, 0, 0), (resource.RLIMIT_NPROC, 0, 0),
        (resource.RLIMIT_NOFILE, 16, 16), (resource.RLIMIT_CORE, 0, 0),
    ):
        resource.setrlimit(kind, (soft, hard))
    context = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    # AST restrictions are the primary capability boundary. This additional
    # kernel filter denies filesystem mutation/read opens, network and processes,
    # including under root where RLIMIT_NPROC alone is not effective.
    denied = (
        "open openat openat2 creat name_to_handle_at open_by_handle_at "
        "socket socketpair connect accept accept4 bind listen sendto sendmsg sendmmsg "
        "recvfrom recvmsg recvmmsg execve execveat fork vfork clone clone3 "
        "ptrace process_vm_readv process_vm_writev pidfd_open pidfd_getfd pidfd_send_signal "
        "mount umount2 pivot_root chroot unshare setns bpf perf_event_open userfaultfd "
        "io_uring_setup io_uring_enter io_uring_register memfd_create "
        "unlink unlinkat rename renameat renameat2 mkdir mkdirat rmdir link linkat "
        "symlink symlinkat mknod mknodat truncate ftruncate chmod fchmod fchmodat "
        "chown fchown lchown fchownat setxattr lsetxattr fsetxattr removexattr "
        "lremovexattr fremovexattr kill tkill tgkill reboot kexec_load init_module "
        "finit_module delete_module setuid setgid setresuid setresgid capset"
    ).split()
    try:
        for name in denied:
            syscall = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if syscall >= 0:
                result = seccomp.seccomp_rule_add(context, 0x00050000 | errno.EPERM, syscall, 0)
                if result != 0:
                    raise RuntimeError(f"Cannot deny syscall {name}: {result}")
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


def execute(code, max_output_chars):
    try:
        tree = validate_ast(code)
    except (SyntaxError, ValueError, RecursionError) as error:
        return {"status": syntax_failure_status(code), "error": f"invalid_python: {str(error)[:500]}",
                "predicted_answer": None, "used_stdout_fallback": False}
    except PolicyError as error:
        return {"status": "execution_failure", "error": f"policy_rejection: {error}",
                "predicted_answer": None, "used_stdout_fallback": False}
    math_facade = SimpleNamespace(**{name: getattr(math, name) for name in MATH_API})

    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name != "math" or any(v not in MATH_API for v in (fromlist or ())):
            raise ImportError("Only public math imports are permitted")
        return math_facade

    safe_builtins = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
                     "range": range, "enumerate": enumerate, "int": int, "float": float,
                     "str": str, "bool": bool, "round": round, "print": print,
                     "__import__": limited_import}
    namespace = {"__builtins__": safe_builtins, "math": math_facade}
    try:
        # Top-level stdout is discarded, matching upstream. Only fn stdout is fallback.
        with contextlib.redirect_stdout(LimitedStringIO(max_output_chars)), contextlib.redirect_stderr(LimitedStringIO(max_output_chars)):
            exec(compile(tree, "<sample>", "exec"), namespace, namespace)
        function = namespace.get("simple_math_problem")
        if not callable(function):
            return {"status": "format_failure", "error": "missing_simple_math_problem",
                    "predicted_answer": None, "used_stdout_fallback": False}
        stdout, stderr = LimitedStringIO(max_output_chars), LimitedStringIO(max_output_chars)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            output = function()
        value = to_number(output)
        fallback = value is None
        if fallback:
            value = extract_text_answer(stdout.getvalue())
        if isinstance(value, int) and value.bit_length() > 4096:
            raise OutputLimitError("Numeric result exceeds 4096 bits")
        return {"status": "ok" if value is not None else "format_failure",
                "predicted_answer": value, "used_stdout_fallback": fallback,
                "stdout": stdout.getvalue(), "raw_type": type(output).__name__,
                "error": None if value is not None else "no_numeric_result"}
    except SyntaxError as error:
        return {"status": syntax_failure_status(code), "error": f"invalid_python: {str(error)[:500]}",
                "predicted_answer": None, "used_stdout_fallback": False}
    except Exception as error:
        return {"status": "execution_failure", "error": f"{type(error).__name__}: {str(error)[:500]}",
                "predicted_answer": None, "used_stdout_fallback": False}


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("Worker input exceeds limit")
        payload = json.loads(raw)
        install_limits(payload["timeout_s"], payload["memory_mb"])
    except Exception as error:
        print(json.dumps({"event": "infrastructure_error", "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 2
    print(json.dumps({"event": "ready"}), flush=True)
    result = execute(payload["code"], payload["max_output_chars"])
    print(json.dumps({"event": "result", "result": result}, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
