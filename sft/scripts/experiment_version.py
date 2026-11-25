"""Record source identity for a clean Git checkout or a standalone source tree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed; install Git and use a committed checkout")
    return result.stdout.strip()


def capture_git_version(root: Path) -> dict:
    root = Path(root).resolve()
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise RuntimeError("The experiment project must be its own Git checkout")
    if git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise RuntimeError("Uncommitted changes found; commit the code and configuration before running")
    commit = git(root, "rev-parse", "HEAD")
    paths = git(root, "-c", "core.quotepath=false", "ls-files", "-z").split("\0")
    digests = {}
    for name in filter(None, paths):
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError(f"Expected an ordinary tracked project file: {name}")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if commit != git(root, "rev-parse", "HEAD") or git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise RuntimeError("Git checkout changed while recording the experiment")
    return {"commit": commit, "tags": git(root, "tag", "--points-at", "HEAD").splitlines(),
            "dirty": False, "tracked_files_sha256": digests}


def file_snapshot(root: Path) -> dict:
    """Hash executable source/configuration without requiring a Git installation.

    Dataset and weight identities are checked by their existing input locks.
    Runtime output, caches and README changes are deliberately excluded.
    """
    root = root.resolve()
    extensions = {".py", ".sh", ".json", ".txt", ".toml", ".yaml", ".yml", ".gz"}

    def collect():
        paths = list(root.glob("*.py")) + list(root.glob("*.sh"))
        paths += list(root.glob("requirements*.txt"))
        paths += [root / name for name in ("inputs.lock.json", "pyproject.toml")
                  if (root / name).exists()]
        for directory in ("mdlm_sft", "scripts", "experiments", "benchmarks", "tests", "datasets"):
            base = root / directory
            if base.is_symlink():
                raise RuntimeError(f"Source directory must not be a symlink: {directory}")
            paths += [path for path in base.rglob("*")
                      if path.suffix in extensions
                      and not any(part.startswith(".") or part == "__pycache__"
                                  for part in path.relative_to(base).parts)]
        digests = {}
        for path in sorted(set(paths)):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise RuntimeError(f"Expected an ordinary source file: {path}")
            if path.is_file():
                digests[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digests

    digests = collect()
    if not digests:
        raise RuntimeError("No project source/configuration files found")
    if collect() != digests:
        raise RuntimeError("Project source changed while recording the experiment")
    snapshot_sha256 = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return {"commit": None, "tags": [], "dirty": False, "source_mode": "file_snapshot",
            "snapshot_sha256": snapshot_sha256, "tracked_files_sha256": digests}


def capture_version(root: Path) -> dict:
    """Retain strict Git checks when present; otherwise use a portable SHA snapshot."""
    root = Path(root).resolve()
    if (root / ".git").exists():
        return capture_git_version(root)
    return file_snapshot(root)
