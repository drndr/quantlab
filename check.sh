#!/usr/bin/env bash
set -euo pipefail

# Quantlab safety-check runner (agent-friendly)
# - runs fast unit tests
# - runs the golden series regression test
#
# Usage:
#   ./check.sh
#   ./check.sh --unit-only
#   ./check.sh --regression-only

MODE="all"
if [[ "${1-}" == "--unit-only" ]]; then
  MODE="unit"
elif [[ "${1-}" == "--regression-only" ]]; then
  MODE="regression"
elif [[ -n "${1-}" ]]; then
  echo "Unknown arg: ${1}" >&2
  exit 2
fi

# Run from the repo root; `import quantlab` needs the editable install (pip install -e .).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# The repo venv (Linux/macOS or Windows layout), else whatever python3 is on PATH.
PY="python3"
for candidate in "$ROOT/.venv/bin/python" "$ROOT/.venv/Scripts/python.exe"; do
  if [[ -x "$candidate" ]]; then
    PY="$candidate"
    break
  fi
done

banner() {
  echo ""
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

banner "Python"
"$PY" -V

banner "Import smoke test"
"$PY" -c "import quantlab, quantlab.engine.backtest, quantlab.strategies.trend_following; print('OK')"

if [[ "$MODE" == "all" || "$MODE" == "unit" ]]; then
  banner "Unit tests (fast)"
  # Skip regression tests here; they are run explicitly below.
  "$PY" -m pytest tests --ignore=tests/regression
fi

if [[ "$MODE" == "all" || "$MODE" == "regression" ]]; then
  banner "Regression: golden series (best_trend 2024–2025)"
  "$PY" -m pytest tests/regression/test_best_trend_2024_2025_series.py
fi

banner "DONE"
