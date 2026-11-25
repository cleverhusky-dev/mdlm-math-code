"""One effective training curve, with durable evidence of rolled-back attempts.

Only rank zero writes these files. A resume never guesses a branch from an
already duplicated/corrupt committed prefix: it fails with the source intact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


class LogStorageError(RuntimeError):
    """Storage timeouts must not enter the trainer's bounded-stop save path."""


def _sync_directory(path):
    # Windows cannot fsync directory handles through this API. Atomic replace
    # and file fsync still apply; Linux deployment also persists directory entries.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_bytes(path, content):
    try:
        _write_and_replace(path, content)
    except TimeoutError as error:
        raise LogStorageError("Training log storage timed out; resume the previous checkpoint") from error


def _write_and_replace(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _sync_directory(path.parent.parent)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def append_durable_jsonl(path, record, *, atomic=False):
    """Persist the row before a later checkpoint can commit its step."""
    path = Path(path)
    content = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    try:
        if atomic:
            # Small invocation/usage journals use whole-file replacement so
            # a killed append cannot merge two attempts into one partial line.
            previous = path.read_bytes() if path.exists() else b""
            if previous and not previous.endswith(b"\n"):
                raise ValueError("Incomplete audit journal; preserve and inspect before appending")
            _atomic_bytes(path, previous + content)
        else:
            with path.open("ab") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(path.parent)
    except TimeoutError as error:
        # The optimizer has already updated when train.jsonl is appended.
        # Saving that cursor with a torn row would make recovery impossible.
        raise LogStorageError("Training log storage timed out; resume the previous checkpoint") from error


def _seconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid recorded wall time; preserve and inspect run evidence")
    return value


def recorded_wall_seconds(output):
    """Cumulative high-water mark, never a sum of cumulative log rows.

Also reads rollback metadata: a process can die after the curve replacement
but before usage.json/finally is published. In that case its observed time
must survive subsequent attempts. Unobserved time after a hard kill cannot be
reconstructed here and must be reconciled with the external allocation bill.
"""
    output = Path(output)
    values = [0.0]
    for filename in ("train.jsonl", "usage_history.jsonl"):
        path = output / filename
        if path.exists():
            for line in path.read_bytes().splitlines():
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    # A torn tail is retained by the recovery archive; it
                    # cannot supply a reliable time observation.
                    continue
                if isinstance(row, dict) and "wall_seconds" in row:
                    values.append(_seconds(row["wall_seconds"]))
    usage = output / "usage.json"
    if usage.exists():
        values.append(_seconds(json.loads(usage.read_text(encoding="utf-8"))["wall_seconds"]))
    for path in (output / "log_history").glob("recovery-*.json"):
        values.append(_seconds(json.loads(path.read_text(encoding="utf-8"))["observed_wall_seconds"]))
    return max(values)


def reconcile_training_log(output, checkpoint_step, invocation_id):
    """Keep exactly steps 1..checkpoint_step, archive first, then replace.

Archive names depend only on original bytes. Recovery intent is durable
before replacing train.jsonl. Every interruption leaves either the original
curve or its verified byte-for-byte archive, and a retry is idempotent.
"""
    output = Path(output)
    if type(checkpoint_step) is not int or checkpoint_step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    path = output / "train.jsonl"
    original = path.read_bytes() if path.exists() else b""
    lines = original.splitlines(keepends=True)
    complete = []
    for index, line in enumerate(lines):
        # Only an unterminated final append can be discarded as a torn tail.
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            break
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Malformed training log; original preserved for inspection") from error
        step = row.get("step") if isinstance(row, dict) else None
        if type(step) is not int or step != len(complete) + 1:
            raise ValueError("Training log must have unique consecutive steps from 1; original preserved")
        complete.append(line)
    if len(complete) < checkpoint_step:
        raise ValueError("Training log is missing committed steps; original preserved")
    retained = b"".join(complete[:checkpoint_step])
    if retained == original:
        return None
    digest = hashlib.sha256(original).hexdigest()
    history = output / "log_history"
    archive = history / f"train-{digest}.jsonl"
    if archive.exists():
        if archive.read_bytes() != original:
            raise ValueError("Existing training log archive differs; refusing to overwrite evidence")
    else:
        _atomic_bytes(archive, original)
    metadata = history / f"recovery-{digest}-step-{checkpoint_step:08d}.json"
    recovery = {
        "schema": "mdlm-training-log-recovery-v1", "invocation_id": invocation_id,
        "checkpoint_step": checkpoint_step, "original_sha256": digest,
        "archive": archive.relative_to(output).as_posix(), "original_bytes": len(original),
        "effective_sha256": hashlib.sha256(retained).hexdigest(),
        "retained_rows": checkpoint_step, "removed_complete_rows": len(complete) - checkpoint_step,
        "torn_tail_bytes": len(original) - sum(map(len, complete)),
        "observed_wall_seconds": recorded_wall_seconds(output),
        "meaning": "Durable recovery intent; archive is original evidence, train.jsonl is the effective curve",
    }
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        for key in ("schema", "checkpoint_step", "original_sha256", "archive", "effective_sha256"):
            if previous[key] != recovery[key]:
                raise ValueError("Existing recovery intent differs; refusing to overwrite evidence")
    else:
        _atomic_bytes(metadata, (json.dumps(recovery, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    # Re-sync even when reusing an intent published by a prior interrupted
    # attempt (which might have stopped after rename but before directory fsync).
    _sync_directory(history)
    _atomic_bytes(path, retained)
    return metadata.relative_to(output).as_posix()
