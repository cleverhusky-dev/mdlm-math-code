"""Deterministic, outcome-blind selection and read-only prediction diagnostics."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re


def _id_key(value):
    if not isinstance(value, str) or re.fullmatch(r"train:(0|[1-9][0-9]*)", value) is None:
        raise ValueError("Only official training-question IDs are allowed; test is excluded")
    return int(value.split(":")[1])


def select_questions(train_rows, val_rows, *, questions_per_split=64,
                     max_reference_tokens=256, seed=5):
    """Sample short held-out questions, then greedily length-match unused train IDs."""
    if (type(questions_per_split) is not int or questions_per_split <= 0
            or type(max_reference_tokens) is not int or not 0 < max_reference_tokens <= 512
            or type(seed) is not int):
        raise ValueError("Invalid selection count, token bound or seed")
    pools, all_ids = {}, {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        groups = {}
        for row in rows:
            key = row["gsm8k_id"]
            _id_key(key)
            q, c, total = (row[name] for name in ("question_tokens", "code_tokens", "total_tokens"))
            if (any(type(x) is not int or x < 0 for x in (q, c, total))
                    or total != q + c + 4 or total > 512):
                raise ValueError(f"Inconsistent audited token lengths: {key}")
            feature = {"prefix_length": q + 3, "shortest_code_tokens": c,
                       "shortest_total_tokens": total}
            if key in groups and groups[key]["prefix_length"] != q + 3:
                raise ValueError(f"One question has multiple prefix lengths: {key}")
            if key not in groups or total < groups[key]["shortest_total_tokens"]:
                groups[key] = feature
        all_ids[split] = set(groups)
        pools[split] = {key: value for key, value in groups.items()
                        if value["shortest_total_tokens"] <= max_reference_tokens}
        if len(pools[split]) < questions_per_split:
            raise ValueError(f"Insufficient short {split} questions")
    if all_ids["train"] & all_ids["val"]:
        raise ValueError("Train and validation question IDs overlap")
    val_ids = sorted(random.Random(seed).sample(sorted(pools["val"], key=_id_key),
                                                questions_per_split), key=_id_key)
    unused, pairs = set(pools["train"]), []
    for val_id in val_ids:
        val = pools["val"][val_id]
        def distance(key):
            train = pools["train"][key]
            return (abs(train["prefix_length"] - val["prefix_length"])
                    + abs(train["shortest_code_tokens"] - val["shortest_code_tokens"]), _id_key(key))
        train_id = min(unused, key=distance)
        unused.remove(train_id)
        train = pools["train"][train_id]
        pairs.append({"val_id": val_id, "train_id": train_id, "val": val, "train": train,
                      "prefix_length_difference": train["prefix_length"] - val["prefix_length"],
                      "code_length_difference": train["shortest_code_tokens"] - val["shortest_code_tokens"],
                      "l1_distance": distance(train_id)[0]})
    return {"seed": seed, "questions_per_split": questions_per_split,
            "max_reference_tokens": max_reference_tokens,
            "rule": "shortest reference total <= bound; sample numeric-ID-sorted val pool with random.Random(seed).sample; numeric-ID-sort selected val; greedily minimize L1(prefix length, shortest code length) over unused short train questions; numeric-ID tie break",
            "selection_uses_model_results": False, "test_accessed": False,
            "pool_counts": {split: len(pool) for split, pool in pools.items()},
            "ids": {"val": val_ids, "train": [pair["train_id"] for pair in pairs]}, "pairs": pairs}


def prediction_diagnostics(rows, special_token_ids, *, parse_ast=False):
    """Aggregate saved scorer outcomes/tokens; never execute generated programs."""
    special = set(special_token_ids)
    counts, errors, tokens, syntax = Counter(), Counter(), Counter(), Counter()
    numeric, total = 0, 0
    for row in rows:
        counts[row["status"]] += 1
        error = row.get("error")
        if error:
            errors[str(error).partition(":")[0].strip()] += 1
        answer = row.get("predicted_answer")
        numeric += type(answer) is int or (type(answer) is float and math.isfinite(answer))
        values = row.get("generated_token_ids", [])
        total += len(values)
        tokens.update(value for value in values if value in special)
        if parse_ast:
            try:
                ast.parse(row.get("generated_text", ""))
                syntax["parsable"] += 1
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                syntax["unparsable"] += 1
    return {"counts": dict(sorted(counts.items())), "error_prefix_counts": dict(sorted(errors.items())),
            "numeric_result_count": numeric, "generated_token_count": total,
            "generated_special_token_count": sum(tokens.values()),
            "generated_special_token_fraction": sum(tokens.values()) / total if total else None,
            "special_token_counts": {str(key): tokens[key] for key in sorted(special)},
            "ast_parse": dict(syntax) if parse_ast else None,
            "ast_note": "Optional syntax-only check of raw generated text; no extraction or execution"}


def audit_references(directory, datasets, questions, references, selection, tokenizer):
    """Re-encode every selected reference and compare the frozen cache and prompt."""
    from mdlm_sft.data import encode_example, encode_prompt
    selected, audit = {}, {}
    for split in ("train", "val"):
        data, rows = datasets[split], references[split]
        by_id = {row["gsm8k_id"]: row for row in questions[split]}
        if len(rows) != len(data) or [row["gsm8k_id"] for row in rows] != data.gsm8k_ids:
            raise ValueError(f"Reference rows/cache IDs differ: {split}")
        if set(data.gsm8k_ids) != set(by_id):
            raise ValueError(f"Question/reference identity differs: {split}")
        ids = set(selection["ids"][split])
        indices, totals, lengths = [], Counter(), {name: [] for name in ("code_tokens", "total_tokens")}
        for index, row in enumerate(rows):
            if row["gsm8k_id"] not in ids:
                continue
            question = by_id[row["gsm8k_id"]]
            if any(row[key] != question[key] for key in ("question", "gold_answer")):
                raise ValueError("Reference question text or gold answer differs")
            encoded = encode_example(row, tokenizer)
            prefix = encoded["prefix_length"]
            if (encoded["input_ids"] != data.input_ids[index].tolist()
                    or encoded["target_mask"] != data.target_mask[index].tolist()
                    or prefix != int(data.prefix_lengths[index])):
                raise ValueError(f"Re-encoding differs from cache: {split}/{index}")
            prompt = encode_prompt(question["question"], tokenizer)
            if (prompt["prefix_length"] != prefix or prompt["target_mask"] != encoded["target_mask"]
                    or prompt["input_ids"][:prefix] != encoded["input_ids"][:prefix]):
                raise ValueError("Training and generation prefixes differ")
            indices.append(index)
            for name in lengths:
                lengths[name].append(row[name])
            totals.update({"code_tokens": row["code_tokens"], "eos_tokens": 1,
                           "pad_tokens": 512 - encoded["sequence_length"],
                           "target_tokens": sum(encoded["target_mask"])})
        if set(data.gsm8k_ids[i] for i in indices) != ids:
            raise ValueError("A selected question has no reference")
        if totals["target_tokens"] != totals["code_tokens"] + totals["eos_tokens"] + totals["pad_tokens"]:
            raise AssertionError("Target token accounting differs")
        selected[split] = indices
        audit[split] = {"num_questions": len(ids), "code_samples": len(indices), **totals,
                        "reference_length_ranges": {name: {"min": min(values), "max": max(values)}
                                                    for name, values in lengths.items()},
                        "reference_counts_by_question": dict(Counter(data.gsm8k_ids[i] for i in indices)),
                        "reference_indices": indices, "all_references_reencoded": True,
                        "cache_tokens_masks_prefixes_equal": True, "generation_prefix_equal": True,
                        "truncation": False}
    return selected, {"status": "passed", "splits": audit, "test_accessed": False,
                      "loss_mask_unchanged": True,
                      "scope": "Encoding consistency only; does not establish that the training objective is suitable",
                      "target_accounting": "Token roles: reference code, appended EOS, trailing PAD; all remain loss targets"}


def check_groups(output, selection, audit, sources, reference):
    """Fail closed unless all four saved generation/loss groups match their inputs."""
    from mdlm_sft.baseline import read_json, sha256
    from mdlm_sft.data import read_jsonl
    output, groups, protocols = Path(output), {}, {}
    if set(sources) != {"pretrained", "sft_original"}:
        raise ValueError("Experiment5 requires pretrained and original SFT references")
    for name, saved in sources.items():
        checkpoint = read_json(output / f"{name}-checkpoint.json")
        if (checkpoint.get("weights_unchanged") is not True
                or checkpoint["model_sha256_before"] != checkpoint["model_sha256_after"]
                or checkpoint["model_sha256_before"] != saved["checkpoint_source"].get("model_sha256", checkpoint["model_sha256_before"])
                or checkpoint["checkpoint_source"] != saved["checkpoint_source"]):
            raise ValueError("Missing or inconsistent read-only checkpoint audit")
        for split in ("train", "val"):
            key = f"{name}/{split}"
            directory = output / key
            summary, protocol = read_json(directory / "summary.json"), read_json(directory / "protocol.json")
            rows = read_jsonl(directory / "predictions.jsonl")
            counts = {status: 0 for status in ("correct", "format_failure", "execution_failure", "wrong_answer")}
            counts.update(Counter(row["status"] for row in rows))
            digest = protocol.pop("protocol_sha256")
            if (summary.get("complete") is not True or summary["label"] != f"exploratory_{split}"
                    or summary["num_questions"] != 64 or summary["question_count"] != 64 or len(rows) != 64
                    or summary["counts"] != counts or summary["accuracy"] != counts["correct"] / 64
                    or summary["checkpoint_source"] != saved["checkpoint_source"]
                    or summary["predictions_sha256"] != sha256(directory / "predictions.jsonl")
                    or summary["protocol_sha256"] != digest
                    or set(protocol) != set(saved["protocol"])
                    or hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != digest
                    or protocol["question_order"] != selection["ids"][split]
                    or [row["gsm8k_id"] for row in rows] != selection["ids"][split]):
                raise ValueError(f"Incomplete or inconsistent saved generation group: {key}")
            for field in set(saved["protocol"]) - {"question_order", "questions_sha256"}:
                if protocol[field] != saved["protocol"][field]:
                    raise ValueError(f"Subset protocol changed: {key}/{field}")
            if split in protocols and protocol != protocols[split]:
                raise ValueError(f"Models did not use the same {split} protocol")
            protocols[split] = protocol
            loss = read_json(directory / "reference_loss.json")
            if (loss.get("complete") is not True or loss["label"] != f"exploratory_{split}"
                    or loss["num_questions"] != 64 or loss["checkpoint_source"] != saved["checkpoint_source"]
                    or loss["code_samples"] != audit["splits"][split]["code_samples"]
                    or loss["target_tokens"] != audit["splits"][split]["target_tokens"]
                    or loss["seed"] != reference["validation_loss_seed"]
                    or loss["precision"] != reference["precision"]
                    or loss["batch_size"] != reference["validation_loss_batch_size"]
                    or not math.isfinite(loss["loss"]) or loss["loss"] != loss["loss_sum"] / loss["target_tokens"]):
                raise ValueError(f"Incomplete or inconsistent reference loss: {key}")
            groups[key] = {"generation": summary, "reference_loss": loss,
                           "diagnostics": read_json(directory / "diagnostics.json")}
    if len(groups) != 4:
        raise ValueError("Experiment5 requires exactly four completed groups")
    return groups
