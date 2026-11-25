"""Write a local RL configuration for a trusted SFT checkpoint and its dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdlm_rl.checkpoints import load_sft
from mdlm_rl.config import validate_config
from mdlm_rl.runtime import sha256


def project_path(path):
    """Path strings in an RL configuration are relative to the RL project."""
    try:
        return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()
    except ValueError:  # Windows paths on different drives
        return str(path.resolve())


def configure(checkpoint, sft_data, output):
    config = json.loads((ROOT / "experiments/experiment1/config.json").read_text(encoding="utf-8"))
    state = load_sft(checkpoint)
    if state["config"].get("smoke"):
        raise ValueError("Select a full SFT training checkpoint")
    source_digest = state["identity"]["data_manifest_sha256"]
    config["initial_global_step"] = state["progress"]["global_step"]
    del state
    config["initial_checkpoint"] = project_path(checkpoint)
    config["initial_checkpoint_sha256"] = sha256(checkpoint)
    bundled_manifest = ROOT / config["data_dir"] / "manifest.json"
    bundled_parent = json.loads(bundled_manifest.read_text(encoding="utf-8"))["parent_manifest_sha256"]
    if source_digest != bundled_parent:
        parent = sft_data / "manifest.json"
        if not parent.is_file() or sha256(parent) != source_digest:
            raise ValueError("Pass --sft-data pointing to the dataset used by this SFT checkpoint")
        config["data_dir"] = project_path(sft_data)
        config["expected_data_manifest_sha256"] = source_digest
    validate_config(config)
    content = json.dumps(config, indent=2) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") == content:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Existing configurations are never replaced by a different checkpoint.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted full SFT best.pt")
    parser.add_argument("--sft-data", type=Path, default=ROOT.parent / "sft/data/vars")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/config.local.json")
    args = parser.parse_args()
    configure(args.checkpoint, args.sft_data, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
