#!/usr/bin/env python3
"""只读核查 MathInstruct GSM8K-PoT 的题目来源；不执行数据中的代码。

默认从本脚本上一级目录读取数据，在本脚本所在目录生成报告和明细。
依赖：Python >= 3.10，以及 requirements.txt 中的 PyArrow。
所有明细中的行号从 0 开始；输入数据不会被修改。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


SOURCE = "data/PoT/gsm_gpt4.json"
# 对当前发布文件逐条检查后确认的完整后缀；只在题目末尾删除一次。
SUFFIXES = (
    " Let's write a Python program.",
    " Please write a program to solve it",
    " Let's write a program.",
    " Let's program in Python in the response.",
    " Please respond by writing a program in Python.",
    " Let's write a Python program to solve it.",
)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_edges(text):
    """仅规范换行与首尾空白，不改题干中的数字、单位、标点或内部空格。"""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def make_index(references, normalize=lambda text: text):
    result = defaultdict(list)
    for ref in references:
        result[normalize(ref["question"])].append(ref["id"])
    return result


def match_instruction(instruction, raw_index, normalized_index):
    """分层严格匹配；重复官方题面保留为歧义，不擅自选择一个题号。"""
    levels = [("raw_exact", [(instruction, "")], raw_index)]
    stripped = [(instruction[:-len(s)], s) for s in SUFFIXES if instruction.endswith(s)]
    levels.append(("suffix_exact", stripped, raw_index))
    clean = normalize_edges(instruction)
    levels.append(("edge_whitespace_exact", [(clean, "")], normalized_index))
    stripped_clean = [(normalize_edges(clean[:-len(s)]), s) for s in SUFFIXES if clean.endswith(s)]
    levels.append(("suffix_edge_whitespace_exact", stripped_clean, normalized_index))
    for rule, variants, index in levels:
        hits = defaultdict(set)
        for question, suffix in variants:
            for ref_id in index.get(question, []):
                hits[ref_id].add(suffix)
        if hits:
            candidates = sorted(hits)
            suffixes = sorted({s for suffix_set in hits.values() for s in suffix_set})
            return {
                "status": "matched" if len(candidates) == 1 else "ambiguous",
                "rule": rule,
                "candidate_ids": candidates,
                "suffixes": suffixes,
            }
    return {"status": "unmatched", "rule": "", "candidate_ids": [], "suffixes": []}


def write_csv(path, fields, rows):
    # UTF-8 BOM 便于在 Windows 表格软件中查看；字段均为审计元数据，不复制可执行代码。
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def duplicate_stats(groups):
    sizes = [len(rows) for rows in groups.values() if len(rows) > 1]
    return {
        "groups": len(sizes),
        "rows_in_duplicate_groups": sum(sizes),
        "extra_rows": sum(size - 1 for size in sizes),
    }


def audit(args):
    started = time.perf_counter()
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    try:
        import pyarrow
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("缺少 PyArrow，请安装 requirements.txt，或用 --dependency-dir 指定已有依赖目录。") from error

    data_root = args.data_root.resolve()
    out = args.output_dir.resolve()
    files = {
        "mathinstruct": data_root / "MathInstruct.json",
        "gsm8k_train": data_root / "train.parquet",
        "gsm8k_test": args.official_test.resolve(),
    }
    # 前后核对文件身份，避免在读取期间文件被替换而误记录版本。
    hashes_before = {key: file_hash(path) for key, path in files.items()}
    with files["mathinstruct"].open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("MathInstruct 必须是由对象组成的 JSON 数组。")
    selected = [(index, row) for index, row in enumerate(rows) if row.get("source") == SOURCE]
    if not selected:
        raise ValueError(f"未找到目标 source：{SOURCE}")
    for index, row in selected:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("instruction", "output")):
            raise ValueError(f"目标样本 {index} 的 instruction 或 output 无效。")

    references = []
    for split in ("train", "test"):
        official = pq.read_table(files[f"gsm8k_{split}"]).to_pylist()
        for index, row in enumerate(official):
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "answer")):
                raise ValueError(f"GSM8K {split}:{index} 的 question 或 answer 无效。")
            references.append({"id": f"{split}:{index}", "split": split, "row": index, "question": row["question"]})
    raw_index = make_index(references)
    normalized_index = make_index(references, normalize_edges)
    ref_by_id = {ref["id"]: ref for ref in references}
    by_question = defaultdict(list)
    raw_record_groups = defaultdict(list)
    question_code_groups = defaultdict(list)
    matches = []
    issues = []
    suffix_counts = Counter()
    rule_counts = Counter()
    for pot_row, (original_row, row) in enumerate(selected):
        found = match_instruction(row["instruction"], raw_index, normalized_index)
        ref = ref_by_id[found["candidate_ids"][0]] if found["status"] == "matched" else None
        record_key = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        record_hash = text_hash(record_key)
        code_hash = text_hash(row["output"])
        raw_record_groups[record_key].append(original_row)
        if ref:
            # 用原始代码字符串分组，不把空白变化或语义等价程序擅自合并。
            by_question[ref["id"]].append((original_row, row["output"]))
            question_code_groups[(ref["id"], row["output"])].append(original_row)
        else:
            issues.append({"mathinstruct_row_0based": original_row, "instruction": row["instruction"], **found})
        rule_counts[f'{found["status"]}:{found["rule"]}'] += 1
        suffix_counts.update(found["suffixes"])
        matches.append({
            "pot_row_0based": pot_row,
            "mathinstruct_row_0based": original_row,
            "status": found["status"],
            "match_rule": found["rule"],
            "removed_suffix": " | ".join(found["suffixes"]),
            "gsm8k_id": ref["id"] if ref else "",
            "gsm8k_split": ref["split"] if ref else "",
            "gsm8k_row_0based": ref["row"] if ref else "",
            "question_sha256": text_hash(ref["question"]) if ref else "",
            "output_sha256": code_hash,
            "record_sha256": record_hash,
            "candidate_ids": "|".join(found["candidate_ids"]),
        })

    coverage_rows = []
    for ref in references:
        samples = by_question.get(ref["id"], [])
        coverage_rows.append({
            "gsm8k_id": ref["id"], "split": ref["split"], "gsm8k_row_0based": ref["row"],
            "sample_count": len(samples), "distinct_code_texts": len({code for _, code in samples}),
            "question_sha256": text_hash(ref["question"]),
        })
    coverage = {}
    for split in ("train", "test"):
        split_rows = [row for row in coverage_rows if row["split"] == split]
        covered = sum(row["sample_count"] > 0 for row in split_rows)
        coverage[split] = {
            "official_questions": len(split_rows), "matched_samples": sum(row["sample_count"] for row in split_rows),
            "covered_questions": covered, "missing_questions": len(split_rows) - covered,
            "coverage_percent": round(100 * covered / len(split_rows), 6) if split_rows else 0,
            "sample_count_distribution_on_covered_questions": dict(sorted(Counter(row["sample_count"] for row in split_rows if row["sample_count"]).items())),
            "distinct_code_count_distribution_on_covered_questions": dict(sorted(Counter(row["distinct_code_texts"] for row in split_rows if row["sample_count"]).items())),
        }

    order = {}
    for split in ("train", "test"):
        sequence = [row["gsm8k_row_0based"] for row in matches if row["gsm8k_split"] == split]
        if not sequence:
            order[split] = None
            continue
        pairs = list(zip(sequence, sequence[1:]))
        first_occurrences = list(dict.fromkeys(sequence))
        order[split] = {
            "first_15_official_rows_0based": sequence[:15],
            "adjacent_descents": sum(right < left for left, right in pairs),
            "adjacent_equals": sum(right == left for left, right in pairs),
            "adjacent_ascents": sum(right > left for left, right in pairs),
            "nondecreasing": all(left <= right for left, right in pairs),
            "first_occurrences_in_official_order": first_occurrences == sorted(first_occurrences),
        }

    duplicate_rows = []
    for record, indices in raw_record_groups.items():
        if len(indices) > 1:
            duplicate_rows.append({"kind": "exact_record", "gsm8k_id": "", "key_sha256": text_hash(record), "count": len(indices), "extra_rows": len(indices) - 1, "mathinstruct_rows_0based": "|".join(map(str, indices))})
    for (ref_id, code), indices in question_code_groups.items():
        if len(indices) > 1:
            duplicate_rows.append({"kind": "same_question_code", "gsm8k_id": ref_id, "key_sha256": text_hash(code), "count": len(indices), "extra_rows": len(indices) - 1, "mathinstruct_rows_0based": "|".join(map(str, indices))})

    # 原样题面重复检查不依赖 PoT 匹配结果。
    question_sets = {split: {ref["question"] for ref in references if ref["split"] == split} for split in ("train", "test")}
    status_counts = Counter(row["status"] for row in matches)
    if sum(item["matched_samples"] for item in coverage.values()) != status_counts["matched"]:
        raise RuntimeError("计数不一致：覆盖明细与匹配明细未能对账。")
    inputs = {}
    for key, path in files.items():
        after = file_hash(path)
        if after != hashes_before[key]:
            raise RuntimeError(f"输入文件在核查期间发生变化：{path}")
        inputs[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": after}

    summary = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "source_filter": SOURCE, "inputs": inputs,
        "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "pyarrow": pyarrow.__version__},
        "invocation": {"script": str(Path(__file__).resolve()), "arguments": sys.argv[1:], "data_root": str(data_root), "output_dir": str(out)},
        "script_sha256": file_hash(Path(__file__)),
        "mathinstruct_total_rows": len(rows), "all_source_counts": dict(sorted(Counter(row.get("source", "<missing>") for row in rows).items())),
        "pot_sample_rows": len(selected), "matched": status_counts["matched"], "unmatched": status_counts["unmatched"], "ambiguous": status_counts["ambiguous"],
        "match_rule_counts": dict(rule_counts), "removed_suffix_counts": dict(sorted(suffix_counts.items())),
        "coverage": coverage, "order": order,
        "official_raw_question_duplicate_excess": {split: coverage[split]["official_questions"] - len(question_sets[split]) for split in ("train", "test")},
        "official_train_test_raw_question_overlap": len(question_sets["train"] & question_sets["test"]),
        "duplicates": {"exact_record": duplicate_stats(raw_record_groups), "same_question_code": duplicate_stats(question_code_groups), "distinct_question_code_pairs": len(question_code_groups)},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scope": "仅核查指定 PoT 来源的题面对应；未执行程序、验证数学正确性、检查 token 长度或做全库语义去重。",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "sample_matches.csv", list(matches[0]), matches)
    write_csv(out / "question_coverage.csv", ["gsm8k_id", "split", "gsm8k_row_0based", "sample_count", "distinct_code_texts", "question_sha256"], coverage_rows)
    write_csv(out / "duplicate_groups.csv", ["kind", "gsm8k_id", "key_sha256", "count", "extra_rows", "mathinstruct_rows_0based"], duplicate_rows)
    with (out / "review_needed.jsonl").open("w", encoding="utf-8") as stream:
        for issue in issues:
            stream.write(json.dumps(issue, ensure_ascii=False) + "\n")
    (out / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out, summary)
    print(json.dumps({"output_dir": str(out), "samples": len(selected), "coverage": coverage, "unmatched": status_counts["unmatched"], "ambiguous": status_counts["ambiguous"]}, ensure_ascii=True, indent=2))


def write_report(out, summary):
    train, test = summary["coverage"]["train"], summary["coverage"]["test"]
    dup = summary["duplicates"]
    train_order = summary["order"]["train"]
    count = summary["pot_sample_rows"]
    all_train = summary["matched"] == count and train["matched_samples"] == count
    no_issues = summary["unmatched"] == 0 and summary["ambiguous"] == 0
    suffix_only = summary["match_rule_counts"] == {"matched:suffix_exact": count}
    conclusion = (
        f"**不与 GSM8K 训练集＋测试集逐题一一对应。** 本地这份 GSM8K-PoT 有 {count:,} 条程序样本，"
        f"全部对应官方训练集中的 **{train['covered_questions']:,} 道题**；未匹配到官方测试题。"
        if all_train and no_issues and test["official_questions"] > 0
        else "本次核查结果见下表；不能仅凭数据集名称推定完整覆盖或一一对应。"
    )
    method = (
        f"全部 {count:,} 条样本均为“官方题目原文＋六种固定编程指令之一”。只删除末尾这段指令，就能唯一匹配；没有修改题干、规范空白或使用近似匹配。"
        if suffix_only else "按原文、固定末尾指令、换行及首尾空白的顺序分层匹配。规则只处理格式，不改数字、单位、标点或内部空格；未匹配和歧义样本需另行复核。"
    )
    ordering = (
        f"**不保留官方题目顺序。** 按 PoT 原始顺序检查，相邻映射题号下降 {train_order['adjacent_descents']:,} 次。"
        if train_order and train_order["adjacent_descents"] > 0
        else "顺序统计见 audit_summary.json；未观察到下降不等于完整覆盖或逐题一一对应。"
    )
    distribution = "；".join(f"{n} 条样本：{questions:,} 道题" for n, questions in train["sample_count_distribution_on_covered_questions"].items())
    contents = f"""# GSM8K 与 GSM8K-PoT 核查报告

核查时间：{summary['generated_at']}。范围仅限 MathInstruct 中 `source={SOURCE}` 的样本。

## 结论

{conclusion}

| 项目 | 结果 |
|---|---:|
| MathInstruct 全部样本 | {summary['mathinstruct_total_rows']:,} |
| GSM8K-PoT 程序样本 | {count:,} |
| 覆盖训练题 | {train['covered_questions']:,} / {train['official_questions']:,}（{train['coverage_percent']:.2f}%） |
| 未覆盖训练题 | {train['missing_questions']:,} |
| 匹配测试题 | {test['covered_questions']:,} / {test['official_questions']:,} |
| 未匹配／歧义样本 | {summary['unmatched']:,} / {summary['ambiguous']:,} |

## 如何确认

{method}

匹配同时检查了官方 train 和 test；官方两部分题目原文的交集为 {summary['official_train_test_raw_question_overlap']}。输入文件读取前后的 SHA-256 一致，输入未改动。逐条对应、使用的后缀与原始位置保存在明细中。

## 重复与顺序

- 训练题的样本条数分布：{distribution}。
- 按“原题＋代码文本”去重后共 {dup['distinct_question_code_pairs']:,} 对；存在 {dup['same_question_code']['groups']:,} 组重复，合计 {dup['same_question_code']['extra_rows']:,} 条额外重复记录。
- 其中，整条记录完全相同的重复有 {dup['exact_record']['groups']:,} 组、{dup['exact_record']['extra_rows']:,} 条额外记录。以上两类重复有包含关系，不能相加。
- {ordering}

不同代码按原始文本区分，不代表算法不同或解答正确。额外重复记录指每组保留一条后剩余的条数。

## 对本项目的含义与边界

这份子集提供的是部分 GSM8K 训练题的多份程序解答，不能直接按行号与官方全集对齐。后续应按原题划分训练／验证，再归入该题的全部代码版本。

本次结果不保证 MathInstruct 其他来源或预训练模型没有数据重叠，也不排除不同题面间的语义近似。本轮未执行代码、核验数值正确性或检查 512 token 长度；这些是后续训练数据筛选工作。

## 文件与复现

- `scripts/cleaning/audit_gsm8k_pot.py`：核查入口，只读原始数据，重新运行会更新指定输出目录的报告与明细。
- `sample_matches.csv`：每条 PoT 样本到官方题号的对应、匹配规则和哈希。
- `question_coverage.csv`：全部官方题目的样本数与不同代码数。筛选 `split=train` 且 `sample_count=0` 可查看缺失训练题。
- `duplicate_groups.csv`：两种重复的分组及对应原始行号。
- `review_needed.jsonl`：未匹配或歧义记录；当前 {summary['unmatched'] + summary['ambiguous']} 条，文件为空即无需此类复核。
- `audit_summary.json`：完整统计、六种指令及频次、输入 SHA-256、运行环境与脚本版本。

所有明细的行号从 **0** 开始；`train:0` 表示官方训练集第一条。代码正文仍在原始数据中，可通过行号追溯。哈希为 UTF-8 文本／文件的 SHA-256，整条记录哈希使用排序键的紧凑 JSON。

在 SFT 工程目录的 PowerShell 中运行（独立清洗环境）：

```powershell
python -m pip install -r scripts/cleaning/requirements.txt
python scripts/cleaning/audit_gsm8k_pot.py
```

脚本默认读取工程内 `inputs/source/` 的 MathInstruct 与 GSM8K 训练文件，以及 `inputs/gsm8k/test.parquet` 的测试题，输出写到本地工作目录。可用 `--data-root`、`--official-test`、`--output-dir` 改变路径；已有独立依赖目录时，可用 `--dependency-dir` 指定。此次实际参数记录在 `audit_summary.json`。
"""
    (out / "核查报告.md").write_text(contents, encoding="utf-8")


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "inputs/source")
    parser.add_argument("--official-test", type=Path, default=root / "inputs/gsm8k/test.parquet")
    parser.add_argument("--output-dir", type=Path, default=root.parent / ".mdlm-data-work/data/cleaning/audit")
    parser.add_argument("--dependency-dir", type=Path, help="可选：已有 PyArrow 安装目录，只对本进程生效。")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
