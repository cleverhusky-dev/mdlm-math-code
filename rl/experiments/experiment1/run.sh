#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
if [[ -n "${PYTHON_BIN:-}" ]]; then
  rl_python="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  rl_python="$project_root/.venv/bin/python"
elif [[ -x "$project_root/../sft/.venv/bin/python" ]]; then
  rl_python="$project_root/../sft/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  rl_python="$(command -v python3)"
else
  rl_python="$(command -v python)"
fi
exec "$rl_python" -B "$project_root/scripts/run_rl.py" \
  --config "$project_root/experiments/experiment1/config.json" "$@"
