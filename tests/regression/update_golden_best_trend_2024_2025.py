"""Regenerate golden series for best_trend 2024-2025.

Writes:
- tests/regression/golden/best_trend_2024_2025_series.csv.gz
- tests/regression/golden/best_trend_2024_2025_config.yaml

Run from the repo root with the venv active:
  python tests/regression/update_golden_best_trend_2024_2025.py

This is meant to be run *only* when intentional changes are made to execution
semantics (gates, costs, time filtering, trade extraction, etc.).
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from quantlab.paths import QUANTLAB
from quantlab.pipeline import run_config_periods


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
GOLDEN_DIR = REPO / "tests" / "regression" / "golden"
OUT_SERIES = GOLDEN_DIR / "best_trend_2024_2025_series.csv.gz"
OUT_CFG = GOLDEN_DIR / "best_trend_2024_2025_config.yaml"


def run_best_trend(cfg: dict) -> pd.DataFrame:
    period = ("2024-2025", dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    bt = run_config_periods(cfg, [period])[period[0]].copy()
    bt.index = pd.to_datetime(bt.index, utc=True)
    bt = bt.sort_index()
    return bt[["position", "returns_net", "equity"]]


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = QUANTLAB / "configs" / "trend_based" / "current.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    # Important: write an isolated snapshot of the config used for the golden.
    cfg_out = copy.deepcopy(cfg)
    OUT_CFG.write_text(yaml.safe_dump(cfg_out, sort_keys=False), encoding="utf-8")

    bt = run_best_trend(cfg_out)

    df = bt.copy()
    df.insert(0, "ts", df.index)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUT_SERIES, index=False, compression="gzip")

    print(f"Wrote config:  {OUT_CFG}")
    print(f"Wrote series:  {OUT_SERIES}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()
