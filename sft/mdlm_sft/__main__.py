"""Command-line entry point; run from the project root with Python 3.12.

python -m mdlm_sft prepare --clean-data INPUT.jsonl --official-train TRAIN.parquet
python -m mdlm_sft train --init inputs/mdlm.ckpt
python -m mdlm_sft evaluate --kind pretrained --checkpoint inputs/mdlm.ckpt --output-dir runs/baseline

Each command accepts --help. Data and model weights are supplied separately.
Program execution/scoring requires Linux. Use torchrun for two-GPU training.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import sys

COMMANDS = {
    "prepare": "prepare_data",
    "train": "train",
    "evaluate": "evaluate",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"{sys.argv[0]} {args.command}", *sys.argv[2:]]
    import_module(f".{COMMANDS[args.command]}", __package__).main()


if __name__ == "__main__":
    main()
