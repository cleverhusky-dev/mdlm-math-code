"""去重、核对执行答案并按 s-flm 的分段编码规则检查长度。

只读取 GSM8K train；不读取测试题，不启动训练，也不预先确定训练/验证划分。
原始数据不改动。执行样本必须通过 execution_worker.py，绝不在本进程 exec。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE = "data/PoT/gsm_gpt4.json"
TOKENIZER_ID = "HuggingFaceTB/SmolLM-135M"
TOKENIZER_REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
UPSTREAM_COMMIT = "30def79976ca3b1cdcb5fcbfa0165524e44001de"
SEPARATOR = r"\n"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_print(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print" and len(node.args) == 1 and not node.keywords


def wrap_program(code):
    """只把明确的末尾 print 改为 return；保留其余原文、注释、计算和 imports。"""
    code = code.strip()
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == "print":
            raise ValueError("print was rebound")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == "print":
            raise ValueError("print was rebound")
        if isinstance(node, ast.arg) and node.arg == "print":
            raise ValueError("print was rebound")
    replacements = []
    def tail(node):
        if isinstance(node, ast.Expr) and is_print(node.value):
            value = ast.get_source_segment(code, node.value.args[0])
            replacements.append((node, "return (" + value + ")"))
        elif isinstance(node, ast.If) and node.body and node.orelse:
            tail(node.body[-1])
            tail(node.orelse[-1])
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.IfExp) and is_print(node.value.body) and is_print(node.value.orelse):
            v = node.value
            text = "return ((" + ast.get_source_segment(code, v.body.args[0]) + ") if (" + ast.get_source_segment(code, v.test) + ") else (" + ast.get_source_segment(code, v.orelse.args[0]) + "))"
            replacements.append((node, text))
        else:
            raise ValueError("unsupported terminal output")
    tail(tree.body[-1])
    # AST columns are UTF-8 byte offsets, not Python character offsets.
    lines = code.encode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    raw = code.encode("utf-8")
    changes = [(offsets[n.lineno-1]+n.col_offset, offsets[n.end_lineno-1]+n.end_col_offset, value.encode("utf-8")) for n, value in replacements]
    for start, end, value in sorted(changes, reverse=True):
        raw = raw[:start] + value + raw[end:]
    wrapped = "def simple_math_problem():\n" + textwrap.indent(raw.decode("utf-8"), "    ") + "\n"
    ast.parse(wrapped)
    return wrapped


def jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def worker_run(row, args, worker, upstream):
    command = [sys.executable, "-I", "-B", str(worker), "--upstream-file", str(upstream)]
    if args.dependency_dir:
        command += ["--dependency-dir", str(args.dependency_dir.resolve())]
    payload = json.dumps({"original_code": row["original_code"], "wrapped_code": row["code"]}, ensure_ascii=True)
    # Minimal environment: no HF tokens, proxy settings or user PYTHONPATH are inherited.
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    started = time.perf_counter()
    attempts = []
    for timeout_limit in (args.timeout, max(30.0, args.timeout * 3)):
        try:
            work_root = args.output_dir.resolve() / "execution_tmp"
            work_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="mdlm_exec_", dir=work_root) as work:
                env.update({"TEMP": work, "TMP": work})
                completed = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=work, env=env, timeout=timeout_limit, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), close_fds=True)
            if completed.returncode != 0:
                result = {"worker_status": "worker_error", "error": completed.stderr[-1200:], "returncode": completed.returncode}
            else:
                result = json.loads(completed.stdout)
                result["worker_status"] = "completed"
        except subprocess.TimeoutExpired:
            result = {"worker_status": "timeout", "error": f"Whole worker exceeded {timeout_limit}s"}
        except Exception as error:
            result = {"worker_status": "worker_error", "error": f"{type(error).__name__}: {error}"}
        attempts.append({"timeout_limit_seconds":timeout_limit,"worker_status":result["worker_status"]})
        if result["worker_status"] != "timeout":
            break
    result["attempts"] = attempts
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return result


def numeric_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return left == right or math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    work = root.parent / ".mdlm-data-work/data/cleaning"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "inputs/source")
    parser.add_argument("--audit-dir", type=Path, default=work / "audit")
    parser.add_argument("--output-dir", type=Path, default=work / "results")
    parser.add_argument("--tokenizer-dir", type=Path, default=root / "inputs/tokenizer")
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--execution-cache", type=Path, default=work / "execution_cache.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--limit", type=int, help="仅用于小规模检查；报告将标记非全量")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("workers、timeout 和 limit 必须为正数")
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    import importlib.metadata
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    started = time.perf_counter()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_path = args.data_root / "MathInstruct.json"
    train_path = args.data_root / "train.parquet"
    audit_summary_path = args.audit_dir / "audit_summary.json"
    mapping_path = args.audit_dir / "sample_matches.csv"
    previous = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    inputs = {"mathinstruct": {"path": str(original_path.resolve()), "sha256": file_sha(original_path)}, "gsm8k_train": {"path": str(train_path.resolve()), "sha256": file_sha(train_path)}}
    for name, info in inputs.items():
        if info["sha256"] != previous["inputs"][name]["sha256"]:
            raise RuntimeError("输入文件与题目对应审计的版本不一致，请先重跑对应审计。")
    data = json.loads(original_path.read_text(encoding="utf-8"))
    train = pq.read_table(train_path).to_pylist()
    upstream_path = here / "upstream/sandbox_gsm8k.py"
    grader = load_module("upstream_grader", upstream_path)
    auditor = load_module("audit_mapping", here / "audit_gsm8k_pot.py")
    with mapping_path.open(encoding="utf-8-sig", newline="") as f:
        mapping = list(csv.DictReader(f))
    all_target_indices = {i for i, r in enumerate(data) if r.get("source") == SOURCE}
    if len(mapping) != len(all_target_indices) or {int(r["mathinstruct_row_0based"]) for r in mapping} != all_target_indices:
        raise RuntimeError("对应明细没有恰好覆盖全部目标子集。")
    groups = {}
    for match in mapping:
        if match["status"] != "matched" or match["gsm8k_split"] != "train":
            raise RuntimeError("只接受已确认对应官方 train 的样本。")
        row_index = int(match["mathinstruct_row_0based"])
        official_index = int(match["gsm8k_row_0based"])
        source_row = data[row_index]
        official = train[official_index]
        if match["removed_suffix"] not in auditor.SUFFIXES or source_row["instruction"] != official["question"] + match["removed_suffix"]:
            raise RuntimeError(f"重新核对原题失败：{row_index}")
        key = (official_index, source_row["output"])
        if key not in groups:
            gold = grader._extract_gold_answer(official["answer"])
            if gold is None or not math.isfinite(gold):
                raise RuntimeError(f"官方标答解析失败：{official_index}")
            groups[key] = {"gsm8k_id": f"train:{official_index}", "gsm8k_train_row_0based": official_index, "question": official["question"], "gold_answer": gold, "original_code": source_row["output"], "source_rows_0based": [], "original_code_sha256": sha(source_row["output"])}
        groups[key]["source_rows_0based"].append(row_index)
    rows = list(groups.values())
    total_deduplicated = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    jsonl(out / "deduplicated_original.jsonl", rows)
    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id("<|endoftext|>") != 0:
        raise RuntimeError("意外的 BOS/EOS token 身份")
    separator_tokens = len(tokenizer.encode(SEPARATOR, add_special_tokens=False).ids)
    for row in rows:
        row["normalization_error"] = ""
        try:
            row["code"] = wrap_program(row["original_code"])
        except Exception as error:
            row["code"] = ""
            row["normalization_error"] = f"{type(error).__name__}: {error}"
        row["code_sha256"] = sha(row["code"])
    questions = {row["gsm8k_id"]: row["question"].strip() for row in rows}
    question_lengths = dict(zip(questions, [len(x.ids) for x in tokenizer.encode_batch(list(questions.values()), add_special_tokens=False)]))
    raw_lengths = [len(x.ids) for x in tokenizer.encode_batch([r["original_code"].strip() for r in rows], add_special_tokens=False)]
    code_lengths = [len(x.ids) for x in tokenizer.encode_batch([r["code"].strip() for r in rows], add_special_tokens=False)]
    for row, raw_length, code_length in zip(rows, raw_lengths, code_lengths):
        row["question_tokens"] = question_lengths[row["gsm8k_id"]]
        row["original_code_tokens"] = raw_length
        row["code_tokens"] = code_length
        row["total_tokens"] = 2 + row["question_tokens"] + separator_tokens + code_length
        row["length_ok"] = row["total_tokens"] <= 512
    worker = here / "execution_worker.py"
    dependencies = {name: importlib.metadata.version(name) for name in ("pyarrow", "tokenizers", "numpy", "sympy", "mpmath")}
    execution_identity = {"worker_sha256": file_sha(worker), "launcher_sha256": sha(inspect.getsource(worker_run)), "upstream_sha256": file_sha(upstream_path), "python": sys.version, "dependencies": dependencies, "timeout_seconds": args.timeout, "retry_timeout_seconds": max(30.0,args.timeout*3)}
    identity_hash = sha(json.dumps(execution_identity, sort_keys=True))
    cache = {}
    if args.execution_cache.exists():
        with args.execution_cache.open(encoding="utf-8") as f:
            for line in f:
                try:
                    cached = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 可忽略异常中断留下的半行；不将半行视为执行成功。
                if cached.get("identity_hash") == identity_hash and cached.get("execution", {}).get("worker_status") == "completed":
                    cache[cached["key"]] = cached["execution"]
    pending = {}
    for row in rows:
        row["execution_key"] = sha(row["original_code"] + "\0" + row["code"])
        if not row["normalization_error"] and row["execution_key"] not in cache:
            pending.setdefault(row["execution_key"], row)
    print(json.dumps({"stage":"execute", "samples":len(rows), "unique_uncached_programs":len(pending), "workers":args.workers}), flush=True)
    args.execution_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.execution_cache.exists() and args.execution_cache.stat().st_size:
        with args.execution_cache.open("rb+") as cache_file:
            cache_file.seek(-1, 2)
            if cache_file.read(1) != b"\n":
                cache_file.seek(0, 2)
                cache_file.write(b"\n")
    with args.execution_cache.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(worker_run, row, args, worker, upstream_path): key for key, row in pending.items()}
        for completed, future in enumerate(as_completed(tasks), 1):
            key = tasks[future]
            result = future.result()
            cache[key] = result
            log.write(json.dumps({"identity_hash":identity_hash, "key":key, "execution":result}, ensure_ascii=True, allow_nan=False) + "\n")
            log.flush()
            if completed % 100 == 0 or completed == len(tasks):
                print(json.dumps({"completed":completed, "total":len(tasks), "elapsed_seconds":round(time.perf_counter()-started,1)}), flush=True)

    details, candidates = [], []
    accepted_code_pairs = set()
    failure_counts = Counter()
    status_counts = {stage:Counter() for stage in ("original", "wrapped", "upstream")}
    for row in rows:
        result = cache.get(row["execution_key"], {"worker_status":"not_executed"})
        checks = {}
        failures = []
        if row["normalization_error"]:
            failures.append("normalization_error")
        if result["worker_status"] != "completed":
            failures.append(result["worker_status"])
        for stage in ("original", "wrapped", "upstream"):
            stage_result = result.get(stage, {})
            status = stage_result.get("status", result["worker_status"])
            status_counts[stage][status] += 1
            value = stage_result.get("value")
            checks[f"{stage}_correct"] = status == "ok" and isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and grader._numbers_equal(value, row["gold_answer"])
            if status == "ok" and not checks[f"{stage}_correct"]:
                failures.append(f"{stage}_wrong_answer")
            elif status != "ok":
                failures.append(f"{stage}_{status}")
        checks["conversion_agrees"] = result.get("original",{}).get("status") == "ok" and result.get("wrapped",{}).get("status") == "ok" and numeric_equal(result["original"].get("value"), result["wrapped"].get("value"))
        if not checks["conversion_agrees"]:
            failures.append("conversion_not_verified")
        if not row["length_ok"]:
            failures.append("over_512_tokens")
        passed = all(checks.values()) and row["length_ok"] and not row["normalization_error"] and result["worker_status"] == "completed"
        candidate_key = (row["gsm8k_id"], row["code"])
        if passed and candidate_key in accepted_code_pairs:
            failures.append("duplicate_after_wrapping")
            passed = False
        if passed:
            accepted_code_pairs.add(candidate_key)
            candidates.append({key:row[key] for key in ("gsm8k_id","gsm8k_train_row_0based","question","gold_answer","code","source_rows_0based","original_code_sha256","code_sha256","question_tokens","code_tokens","total_tokens")})
        failure_counts.update(set(failures))
        details.append({"gsm8k_id":row["gsm8k_id"],"source_rows_0based":row["source_rows_0based"],"original_code_sha256":row["original_code_sha256"],"code_sha256":row["code_sha256"],"gold_answer":row["gold_answer"],"question_tokens":row["question_tokens"],"original_code_tokens":row["original_code_tokens"],"code_tokens":row["code_tokens"],"total_tokens":row["total_tokens"],"length_ok":row["length_ok"],"checks":checks,"keep":passed,"reasons":failures,"normalization_error":row["normalization_error"],"execution":result})
    jsonl(out/"checks.jsonl", details)
    jsonl(out/"clean_candidates.jsonl", candidates)
    question_stats = {}
    for row, detail in zip(rows, details):
        stats = question_stats.setdefault(row["gsm8k_id"], {"gsm8k_id":row["gsm8k_id"],"deduplicated_samples":0,"kept_samples":0})
        stats["deduplicated_samples"] += 1
        stats["kept_samples"] += int(detail["keep"])
    with (out/"question_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["gsm8k_id","deduplicated_samples","kept_samples"])
        writer.writeheader()
        writer.writerows(question_stats.values())
    input_unchanged = all(file_sha(info["path"]) == info["sha256"] for info in inputs.values())
    if not input_unchanged:
        raise RuntimeError("执行期间输入文件发生变化，不能使用本轮结果")
    token_files = {p.name:file_sha(p) for p in sorted(args.tokenizer_dir.glob("*")) if p.is_file()}
    summary = {
        "generated_at":datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "full_run":args.limit is None,"input_rows":len(mapping),"deduplicated_rows_full_pool":total_deduplicated,"duplicate_rows_removed":len(mapping)-total_deduplicated,"processed_rows":len(rows),"processed_questions":len(question_stats),
        "kept_samples":len(candidates),"kept_questions":sum(q["kept_samples"]>0 for q in question_stats.values()),"questions_without_kept_sample":sum(q["kept_samples"]==0 for q in question_stats.values()),
        "original_correct":sum(d["checks"]["original_correct"] for d in details),"wrapped_correct":sum(d["checks"]["wrapped_correct"] for d in details),"upstream_correct":sum(d["checks"]["upstream_correct"] for d in details),"conversion_agrees":sum(d["checks"]["conversion_agrees"] for d in details),
        "normalization_errors":sum(bool(r["normalization_error"]) for r in rows),"over_512_tokens":sum(not r["length_ok"] for r in rows),"max_total_tokens":max(r["total_tokens"] for r in rows),"failure_counts_nonexclusive":dict(failure_counts),"execution_status_counts":{k:dict(v) for k,v in status_counts.items()},
        "inputs":inputs,"mapping_sha256":file_sha(mapping_path),"input_unchanged":input_unchanged,"code_sha256":{p.name:file_sha(p) for p in (Path(__file__),worker)},"execution_identity":execution_identity,
        "tokenizer":{"repo":TOKENIZER_ID,"revision":TOKENIZER_REVISION,"files_sha256":token_files,"separator":SEPARATOR,"separator_token_count":separator_tokens,"bos_id":0,"eos_id":0,"max_length":512,"formula":"1 + encode(question.strip(),False) + encode(separator,False) + encode(code.strip(),False) + 1 (token counts)","history_caveat":"上游训练脚本未固定 tokenizer revision；此处固定当前已核对版本以复现清洗。"},
        "upstream_commit":UPSTREAM_COMMIT,"invocation":{"python":sys.executable,"argv":sys.argv,"resolved_data_root":str(args.data_root.resolve()),"output_dir":str(out)},"elapsed_seconds":round(time.perf_counter()-started,3),"dependencies":dependencies,
        "scope":"仅官方训练题候选池；未划分训练/验证，未执行任何官方测试题，未启动训练。"
    }
    (out/"cleaning_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report = f"""# GSM8K-PoT 数据清洗报告

时间：{summary['generated_at']}。本次为{'全量清洗' if summary['full_run'] else '小规模检查，非全量结果'}。

## 结果

| 项目 | 数量 |
|---|---:|
| 原始目标样本 | {summary['input_rows']:,} |
| 去重后的完整候选池 | {summary['deduplicated_rows_full_pool']:,} |
| 本次实际处理样本 | {summary['processed_rows']:,} |
| 原始程序执行答案正确 | {summary['original_correct']:,} |
| 转换前后数值一致 | {summary['conversion_agrees']:,} |
| 函数版在原评分器下答案正确 | {summary['upstream_correct']:,} |
| 函数版输入超过 512 token | {summary['over_512_tokens']:,} |
| 最终保留程序样本 | **{summary['kept_samples']:,}** |
| 最终覆盖独立题目 | **{summary['kept_questions']:,} / {summary['processed_questions']:,}** |

## 处理规则

1. 按“官方原题＋原始代码文本”去重，保留首次出现的样本并记录全部来源行号；完整候选池移除 {summary['duplicate_rows_removed']} 条额外重复。
2. 将明确的末尾 `print` 改为 `return`，整体包装为 `simple_math_problem()`，保留计算、注释和导入；分别执行原始版、函数版和上游评分环境，检查数值与官方 `####` 标答。未修改计算逻辑来迎合答案。
3. 每程序独立子进程、限时 {args.timeout:g} 秒；超时后以 {max(30.0,args.timeout*3):g} 秒上限重试一次并记录。另做 AST 白名单、输出和内存限制。数学执行允许实际出现的 math/sympy/datetime；原评分器仍只允许 math。环境不支持、运行失败、数值错误分别记录。
4. 长度针对最终保存的函数代码，使用 {TOKENIZER_ID}：BOS＋题目＋字面反斜杠 n 分隔符＋代码＋EOS，分段 tokenize 后相加。超过 512 直接排除，不截断代码。
5. 仅保留三次执行均对标答正确、转换结果一致且长度合格的样本；转换后若出现同题同代码重复，再去重并另计。

排除原因可以重叠，不能直接相加：

```json
{json.dumps(summary['failure_counts_nonexclusive'],ensure_ascii=False,indent=2)}
```

## 文件与下一步

- `clean_candidates.jsonl`：通过本轮检查的候选题库；包含原题、最终代码、标答、原始行号、token 数。**尚未划分训练／验证，不能把全文件直接作为训练集。**
- `deduplicated_original.jsonl`：去重后原始代码，便于检查格式转换。
- `checks.jsonl`：逐条执行值、错误原因、是否保留及长度。
- `question_summary.csv`：每题处理／保留样本数量。
- `cleaning_summary.json`：精确配置、文件哈希、依赖和复现参数；运行说明见 SFT 工程根目录 `README.md`。

本轮只处理官方训练题，未使用测试题筛选或调参。通过样本执行不等于证明程序能正确泛化；后续仍需按原题分组划分验证集并开展训练评测。原始下载文件未改动。
"""
    (out/"清洗报告.md").write_text(report,encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("full_run","processed_rows","kept_samples","kept_questions","original_correct","upstream_correct","over_512_tokens","failure_counts_nonexclusive","elapsed_seconds")},ensure_ascii=True,indent=2),flush=True)


if __name__ == "__main__":
    main()
