#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash setup.sh [--download]

Prepare the shared environment and inputs before experiment0, then experiment1.
Requires Linux and Python 3.12; a GPU is not required for setup.

  --download  Fetch missing checkpoint/tokenizer files from locked upstream URLs.
              Existing files are verified before installation.
  --help      Show this help without installing anything.

PYTHON_BIN may select a Python 3.12 executable, for example:
  PYTHON_BIN=/path/to/python3.12 bash setup.sh

The isolated environment is created in .venv beside this script. Platform Python
and Conda are not modified. CUDA 12.4 PyTorch is installed even in no-GPU mode.
HELP
}

for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --download) ;;
        *) printf 'Unknown setup argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != Linux ]]; then
    printf 'Setup requires Linux. Run this script inside your server terminal.\n' >&2
    exit 1
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME

python_bin="${PYTHON_BIN:-}"
is_python312() {
    "$1" -I -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1
}
if [[ -n "$python_bin" ]]; then
    if ! is_python312 "$python_bin"; then
        printf 'PYTHON_BIN must point to a working Python 3.12 executable.\n' >&2
        exit 1
    fi
else
    for candidate in python3.12 python python3; do
        if command -v "$candidate" >/dev/null 2>&1 && is_python312 "$candidate"; then
            python_bin="$(command -v "$candidate")"
            break
        fi
    done
    if [[ -z "$python_bin" ]]; then
        printf 'Python 3.12 was not found. Activate your platform Python 3.12 environment or set PYTHON_BIN.\n' >&2
        exit 1
    fi
fi

exec "$python_bin" -I "$root/scripts/setup_environment.py" "$@"
