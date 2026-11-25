"""Create the shared Linux environment without changing the platform interpreter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
PYPI_INDEX = "https://pypi.org/simple"


def invoke(command: list[str], root: Path, *, capture: bool = False):
    return subprocess.run(command, cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None)


def validate_venv(root: Path) -> Path:
    environment = root / ".venv"
    config = environment / "pyvenv.cfg"
    python = environment / "bin/python"
    if environment.is_symlink() or not environment.is_dir() or not config.is_file() or not python.is_file():
        raise RuntimeError("Existing .venv is not a usable isolated environment; it was not replaced")
    values = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip().lower()
    if values.get("include-system-site-packages") != "false":
        raise RuntimeError(".venv must have include-system-site-packages = false")
    code = ("import json,sys; print(json.dumps({'version':list(sys.version_info[:2]),"
            "'prefix':sys.prefix,'base_prefix':sys.base_prefix}))")
    details = json.loads(invoke([str(python), "-I", "-c", code], root, capture=True).stdout)
    if details["version"] != [3, 12]:
        raise RuntimeError("Existing .venv must use Python 3.12; it was not replaced")
    if Path(details["prefix"]).resolve() != environment.resolve() or details["prefix"] == details["base_prefix"]:
        raise RuntimeError(".venv interpreter is not isolated from the platform Python")
    return python


def setup(root: Path, *, download: bool = False) -> None:
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Setup requires Linux and Python 3.12")
    root = root.resolve()
    # No user site or inherited Python search path may contaminate this environment.
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    environment = root / ".venv"
    if environment.exists() or environment.is_symlink():
        validate_venv(root)

    # Fail on missing/wrong upstream inputs before downloading large CUDA wheels.
    inputs = [sys.executable, "-I", str(root / "scripts/manage_inputs.py"), "--upstream-only"]
    if download:
        inputs.append("--download")
    invoke(inputs, root)
    if not environment.exists():
        invoke([sys.executable, "-I", "-m", "venv", str(environment)], root)
    python = validate_venv(root)
    pip = [str(python), "-I", "-m", "pip", "--isolated", "--disable-pip-version-check"]
    # Pin the local CUDA build as well as the public version: an existing CPU wheel
    # must not satisfy this step merely because its public version is 2.6.0.
    invoke([*pip, "install", "--no-cache-dir", "--index-url", TORCH_INDEX,
            "torch==2.6.0+cu124"], root)
    invoke([*pip, "install", "--no-cache-dir", "--index-url", PYPI_INDEX,
            "-r", str(root / "requirements.txt")], root)
    invoke([*pip, "check"], root)

    preflight = """
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
from mdlm_sft.runtime import validate_environment
from mdlm_sft.scoring import preflight_executor
validate_environment()
if torch.version.cuda != '12.4':
    raise RuntimeError('Expected the CUDA 12.4 PyTorch build, including on a no-GPU host')
print(json.dumps({'status': 'ready', 'torch': torch.__version__,
                  'cuda_runtime': torch.version.cuda, 'gpu_required_for_setup': False,
                  'executor': preflight_executor()}, indent=2))
"""
    invoke([str(python), "-I", "-c", preflight, str(root)], root)
    freeze = invoke([*pip, "freeze", "--all"], root, capture=True).stdout
    (environment / "installed-requirements.txt").write_text(freeze, encoding="utf-8", newline="\n")
    print("Environment ready. Run .venv/bin/python scripts/rebuild_data.py vars --download to prepare the SFT data.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing locked checkpoint/tokenizer inputs")
    args = parser.parse_args()
    setup(ROOT, download=args.download)


if __name__ == "__main__":
    main()
