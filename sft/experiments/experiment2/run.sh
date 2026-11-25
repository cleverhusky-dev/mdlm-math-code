#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
if [[ -n "${PYTHON_BIN:-}" ]]; then
  experiment_python="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  experiment_python="$project_root/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  experiment_python="$(command -v python3)"
else
  experiment_python="$(command -v python)"
fi
exec "$experiment_python" -B "$project_root/scripts/run_experiment.py" experiment2 "$@"
