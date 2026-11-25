"""Program scoring; generated code runs only in the isolated subprocess.

Parsing and numeric rules follow s-flm 30def799 sandbox_gsm8k.py. The execution
boundary is deliberately stricter: Linux-only, resource limits, seccomp and a
small AST capability policy. Rejections are recorded, never silently retried in
an unrestricted process.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterable

SCORER_VERSION = "sflm-30def799-numeric-v1-linux-policy-v2-return-int-status-v2"
STATUSES = ("correct", "format_failure", "execution_failure", "wrong_answer")
MAX_CODE_CHARS = 200_000


class InfrastructureError(RuntimeError):
    """Scoring is unavailable or its protocol failed; abort aggregate scoring."""


def syntax_failure_status(code: str) -> str:
    """Classify unparseable text by whether an intended entrypoint is visible.

    Pure explanatory output has no program entrypoint (format failure). A
    malformed definition/assignment of that entrypoint is an execution failure.
    This is only diagnostic classification: it never repairs or executes code.
    """
    hint = re.search(r"\bdef\s+simple_math_problem\b|\bsimple_math_problem\s*=(?!=)", code)
    return "execution_failure" if hint else "format_failure"


def extract_code(text: str) -> str:
    """Exactly the upstream first-fence/first-def/trailing-syntax recovery rule."""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text,
                      flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    i = text.find("def ")
    if i != -1:
        text = text[i:]
    text = text.strip()
    if not text:
        return text
    try:
        compile(text, "<sample>", "exec")
        return text
    except SyntaxError as error:
        if error.lineno is None or error.lineno <= 1:
            return text
        error_lineno = error.lineno
    candidate = "\n".join(text.splitlines()[:error_lineno - 1]).strip()
    if not candidate:
        return text
    try:
        compile(candidate, "<sample>", "exec")
        return candidate
    except SyntaxError:
        return text


def to_number(value):
    """Upstream conversion, including bool-as-int and first numeric substring.

    Generated programs cannot import NumPy, so NumPy scalar handling is not
    needed in the isolated worker. Nonfinite/oversized pathological conversions
    are rejected instead of escaping the scorer as exceptions.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if abs(value - round(value)) < 1e-6:
            return int(round(value))
        return value
    if isinstance(value, str):
        match = re.search(r"[-+]?\$?\d[\d,]*\.?\d*", value)
        if not match:
            return None
        text = match.group(0).replace(",", "").replace("$", "")
        try:
            return to_number(float(text)) if text.count(".") == 1 else int(text)
        except (ValueError, OverflowError):
            return None
    return None


def extract_gold_answer(answer):
    if isinstance(answer, str):
        match = re.search(r"####\s*(.+)", answer)
        if match:
            answer = match.group(1)
    return to_number(answer)


def extract_text_answer(text: str):
    if not text:
        return None
    for pattern in (
        r"(?:final answer|answer|result)\s*(?:is|=|:)\s*([-+]?\$?\d[\d,]*\.?\d*)",
        r"####\s*([-+]?\$?\d[\d,]*\.?\d*)",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return to_number(matches[-1])
    matches = re.findall(r"[-+]?\$?\d[\d,]*\.?\d*", text)
    return to_number(matches[-1]) if matches else None


def numbers_equal(predicted, gold) -> bool:
    if predicted is None or gold is None:
        return False
    if isinstance(predicted, float) or isinstance(gold, float):
        try:
            return abs(float(predicted) - float(gold)) <= 1e-3
        except OverflowError:
            return False
    return int(predicted) == int(gold)


@dataclass(frozen=True)
class ExecutionConfig:
    timeout_s: float = 15.0
    memory_mb: int = 512
    max_output_chars: int = 8192

    def validate(self):
        if not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 300:
            raise ValueError("timeout_s must be finite and in (0, 300]")
        if not 64 <= self.memory_mb <= 4096:
            raise ValueError("memory_mb must be in [64, 4096]")
        if not 256 <= self.max_output_chars <= 65536:
            raise ValueError("max_output_chars must be in [256, 65536]")


def _decode_protocol(stdout: str):
    try:
        records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeError) as error:
        raise InfrastructureError("Invalid executor response") from error
    if any(not isinstance(record, dict) for record in records):
        raise InfrastructureError("Executor protocol records must be JSON objects")
    if records and records[0].get("event") == "infrastructure_error":
        raise InfrastructureError(records[0].get("error", "Executor setup failed"))
    if not records or records[0] != {"event": "ready"}:
        raise InfrastructureError("Executor did not confirm its security/resource setup")
    return records


def run_program(code: str, config: ExecutionConfig | None = None) -> dict:
    """Run a program in the Linux worker. Startup failures always raise."""
    config = config or ExecutionConfig()
    config.validate()
    if not sys.platform.startswith("linux"):
        raise InfrastructureError("Program execution requires Linux resource/seccomp support")
    worker = Path(__file__).with_name("executor.py").resolve()
    payload = json.dumps({"code": code, "timeout_s": config.timeout_s,
                          "memory_mb": config.memory_mb,
                          "max_output_chars": config.max_output_chars})
    try:
        with tempfile.TemporaryDirectory(prefix="mdlm-exec-") as workdir:
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=workdir, env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                                  "PYTHONHASHSEED": "0"},
                text=True, encoding="utf-8", errors="strict", start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(payload, timeout=config.timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process finished while the deadline was being handled.
                stdout, stderr = process.communicate()
                _decode_protocol(stdout)
                return {"status": "execution_failure", "error": "wall_clock_timeout",
                        "predicted_answer": None, "used_stdout_fallback": False}
            except BaseException:
                # User interruption must not leave a program running in the background.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise
    except (OSError, UnicodeError) as error:
        raise InfrastructureError(f"Executor process failed: {type(error).__name__}: {error}") from error
    records = _decode_protocol(stdout)
    if process.returncode != 0:
        if process.returncode == -signal.SIGXCPU:
            return {"status": "execution_failure", "error": "cpu_limit",
                    "predicted_answer": None, "used_stdout_fallback": False}
        raise InfrastructureError(
            f"Executor terminated unexpectedly (code {process.returncode}); {stderr[:1000]}"
        )
    if len(records) != 2 or records[1].get("event") != "result":
        raise InfrastructureError("Executor omitted or duplicated its result")
    result = records[1].get("result")
    if (not isinstance(result, dict)
            or result.get("status") not in {"ok", "format_failure", "execution_failure"}
            or "predicted_answer" not in result):
        raise InfrastructureError("Invalid executor result schema")
    if result["status"] == "ok" and to_number(result["predicted_answer"]) is None:
        raise InfrastructureError("Executor reported success without a finite numeric answer")
    return result


def score_program(sample: str, gold_answer, *, timeout_s: float = 15.0,
                  memory_mb: int = 512, max_output_chars: int = 8192) -> dict:
    """Four mutually exclusive model outcomes; bad gold or broken worker raises."""
    started = time.monotonic()
    gold = extract_gold_answer(gold_answer)
    if gold is None:
        raise InfrastructureError("Dataset gold answer is not a finite upstream-compatible number")
    if not isinstance(sample, str):
        raise TypeError("sample must be decoded text")
    if len(sample) > MAX_CODE_CHARS:
        result = {"status": "format_failure", "error": "overlong_generated_text",
                  "predicted_answer": None, "used_stdout_fallback": False}
        code = ""
    else:
        try:
            code = extract_code(sample)
        except (ValueError, RecursionError, MemoryError) as error:
            code = ""
            result = {"status": syntax_failure_status(sample), "error": f"invalid_python: {type(error).__name__}",
                      "predicted_answer": None, "used_stdout_fallback": False}
        else:
            # Extraction compiles for syntax only; never executes generated text.
            result = run_program(code, ExecutionConfig(timeout_s, memory_mb, max_output_chars))
    if result["status"] == "ok":
        result["status"] = "correct" if numbers_equal(result["predicted_answer"], gold) else "wrong_answer"
    return {**result, "code": code, "gold_answer": gold, "scorer_version": SCORER_VERSION,
            "elapsed_seconds": time.monotonic() - started}


def summarize_results(results: Iterable[dict]) -> dict:
    counts = Counter()
    for result in results:
        status = result.get("status")
        if status not in STATUSES:
            raise InfrastructureError(f"Cannot aggregate non-model outcome: {status!r}")
        counts[status] += 1
    total = sum(counts.values())
    if not total:
        raise ValueError("Cannot score an empty evaluation")
    return {"num_questions": total, "counts": {s: counts[s] for s in STATUSES},
            "accuracy": counts["correct"] / total,
            "format_failure_rate": counts["format_failure"] / total,
            "execution_failure_rate": counts["execution_failure"] / total,
            "wrong_answer_rate": counts["wrong_answer"] / total,
            "scorer_version": SCORER_VERSION}


def preflight_executor(config: ExecutionConfig | None = None) -> dict:
    # Each probe starts a fresh restricted worker; an ASCII-only probe cannot
    # detect parser extensions that would otherwise load after open is denied.
    probes = (
        ("ascii", "def simple_math_problem():\n    return 17", "ok", 17, None),
        ("unicode_identifiers", "def simple_math_problem():\n    \u91d1\u989d = 17\n    \uff2b = 0\n    return \u91d1\u989d + K", "ok", 17, None),
        ("named_unicode_escape", "def simple_math_problem():\n    return int('\\N{DIGIT ONE}\\N{DIGIT SEVEN}')", "ok", 17, None),
        ("unicode_syntax_error", "def simple_math_problem():\n    return 1 + \u00b2", "execution_failure", None, "invalid_python:"),
        ("return_int", "def simple_math_problem() -> int:\n    return 9 + 8", "ok", 17, None),
        ("return_int_no_coercion", "def simple_math_problem() -> int:\n    return 2.5", "ok", 2.5, None),
        ("return_annotation_call_rejected", "def simple_math_problem() -> int():\n    return 17", "execution_failure", None, "policy_rejection:"),
    )
    checked = []
    for name, code, status, answer, error_prefix in probes:
        result = run_program(code, config)
        if (result["status"] != status or result["predicted_answer"] != answer
                or (error_prefix is not None and not result.get("error", "").startswith(error_prefix))):
            raise InfrastructureError(f"Executor preflight failed ({name}): {result}")
        checked.append(name)
    return {"status": "passed", "scorer_version": SCORER_VERSION,
            "security": "Linux RLIMIT + seccomp + AST capability policy", "checks": checked}
