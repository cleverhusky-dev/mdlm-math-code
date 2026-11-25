#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
if [[ -n "${PYTHON_BIN:-}" ]]; then
  benchmark_python="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  benchmark_python="$project_root/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  benchmark_python="$(command -v python3)"
else
  benchmark_python="$(command -v python)"
fi
exec "$benchmark_python" -B "$project_root/scripts/run_benchmark.py" "$@"
