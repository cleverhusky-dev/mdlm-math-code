"""Verify frozen experiment inputs; optionally fetch missing upstream assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Input path escapes project: {relative}")
    return path


def load_assets(root: Path) -> list[dict]:
    spec = json.loads((root / "inputs.lock.json").read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or not spec.get("files"):
        raise ValueError("Invalid inputs.lock.json")
    seen = set()
    for entry in spec["files"]:
        asset_path(root, entry["path"])
        if entry["path"] in seen:
            raise ValueError(f"Duplicate input: {entry['path']}")
        seen.add(entry["path"])
        digest = entry["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid input digest: {entry['path']}")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0:
            raise ValueError(f"Invalid input size: {entry['path']}")
    return spec["files"]


def verify_file(path: Path, entry: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {entry['path']}")
    if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Input checksum mismatch: {entry['path']}; existing file was not changed")


def download_missing(root: Path, entry: dict) -> None:
    """Publish a download only after validation; never replace an existing input."""
    path = asset_path(root, entry["path"])
    if path.exists():
        verify_file(path, entry)
        return
    url = entry.get("url")
    if not url or not url.startswith("https://huggingface.co/"):
        raise FileNotFoundError(f"{entry['path']} must be built with scripts/rebuild_data.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        request = Request(url, headers={"User-Agent": "mdlm-sft-reproduction/1"})
        print(f"Downloading {entry['path']} ({entry['size_bytes']:,} bytes)", flush=True)
        with os.fdopen(fd, "wb") as target, urlopen(request, timeout=120) as source:
            written = 0
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                written += len(block)
                if written > entry["size_bytes"]:
                    raise ValueError(f"Download exceeds expected size: {entry['path']}")
                target.write(block)
        verify_file(temporary, entry)
        if path.exists():
            verify_file(path, entry)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing checkpoint/tokenizer files from pinned upstream revisions")
    parser.add_argument("--skip-checkpoint", action="store_true", help="Verify the small input bundle without the 2.69 GB checkpoint")
    parser.add_argument("--upstream-only", action="store_true", help="Verify or fetch upstream files before building the datasets")
    args = parser.parse_args()
    entries = [entry for entry in load_assets(ROOT)
               if (not args.upstream_only or entry.get("url")) and not (args.skip_checkpoint and entry["kind"] == "checkpoint")]
    # Detect incomplete prepared data before downloading a large model.
    for entry in entries:
        if not entry.get("url"):
            verify_file(asset_path(ROOT, entry["path"]), entry)
    for entry in entries:
        if args.download:
            download_missing(ROOT, entry)
        verify_file(asset_path(ROOT, entry["path"]), entry)
        print(f"OK  {entry['path']}", flush=True)
    print(json.dumps({"status": "passed", "verified_files": len(entries),
                      "checkpoint_verified": not args.skip_checkpoint}))


if __name__ == "__main__":
    main()
