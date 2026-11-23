"""Golden regression: does the code still trade exactly as before?

Two weeks of XAUUSD, one small case per rule (gate, time filter, engine feature) with simple
round-number settings. Independent of the live strategy config, so tuning it never touches this.
Each case's position must match golden.csv.gz exactly and its equity to 1e-10.

Update (ONLY when a behaviour change is intended), from the repo root:
    python tests/regression/test_golden.py --update
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.configs.schema import SCHEMA_GATE_NAMES
from quantlab.data.loaders import load_dukascopy_ohlc
from quantlab.paths import DATA_ROOTS
from quantlab.pipeline import backtest_config, run_config_periods

GOLDEN = Path(__file__).resolve().parent / "golden.csv.gz"

# An FOMC meeting in daylight-saving time (31 July), a month boundary and the early-August shocks.
PERIOD = ("golden", dt.date(2024, 7, 29), dt.date(2024, 8, 9))

NOT_COVERED = {"corr"}  # needs XAGUSD and EURUSD data


@dataclass(frozen=True)
class Case:
    cfg: dict
    differs_from: str | None = "trend_ema"  # the case must trade differently from this one
    leverage: float | None = None  # engine override (None = default)


def _gate(name: str, **params) -> dict:
    return {"gate": name, "params": params}


def _case(*gates: dict, differs_from: str | None = "trend_ema", leverage: float | None = None,
          ma_kind: str = "ema", **cfg) -> Case:
    return Case(
        {
            "symbol": "XAUUSD",
            "costs": {"fee_per_lot": 3.5, "spread_per_lot": 3.5},
            "trend": {"fast": 10, "slow": 30, "ma_kind": ma_kind},
            # A pipeline can't be empty; time_filter without a time_filter block does nothing.
            "pipeline": list(gates) or [_gate("time_filter")],
            **cfg,
        },
        differs_from,
        leverage,
    )


FOMC = {"FOMC": {"pre_hours": 2.0, "post_hours": 0.5}}
EMA_SEP = _gate("ema_sep", ema_fast=10, ema_slow=40, atr_n=10, sep_k=0.5)
SIZING = _gate("ema_strength_sizing", ema_fast=10, ema_slow=40, atr_n=10, strong_k=0.5, size_base=1.0, size_strong=2.0)
CHURN = _gate("churn", min_on_bars=3, cooldown_bars=8)

CASES: dict[str, Case] = {
    # base trend
    "trend_ema": _case(differs_from=None),
    "trend_sma": _case(ma_kind="sma"),
    # entry filters
    "htf_confirm": _case(_gate("htf_confirm", fast=10, slow=30)),
    "ema_sep": _case(EMA_SEP),
    "nochop": _case(_gate("nochop", ema=10, lookback=20, min_closes=12)),
    # time filter
    "time_filter_events": _case(_gate("time_filter"), time_filter={"events": FOMC}),
    "time_filter_months": _case(_gate("time_filter"), time_filter={"months": {"block": [8]}}),
    # sizing
    "ema_strength_sizing": _case(SIZING),
    "seasonality_cap": _case(SIZING, _gate("seasonality_cap", month_size_cap={8: 1.0}), differs_from="ema_strength_sizing"),
    # churn
    "churn": _case(CHURN),
    # exits
    "shock_exit_fixed": _case(_gate("shock_exit", shock_exit_abs_ret=0.004)),
    "shock_exit_sigma": _case(_gate("shock_exit", shock_exit_sigma_k=4.0, shock_exit_sigma_window=96, shock_cooldown_bars=6)),
    "mid_loss_limiter": _case(_gate("mid_loss_limiter", min_bars=6, max_bars=48, stop_ret=-0.002)),
    "no_recovery_exit": _case(_gate("no_recovery_exit", bar_n=12, min_ret=0.0)),
    "profit_milestone": _case(_gate("profit_milestone", bar_n=12, milestone_ret=0.002)),
    "rolling_max_exit": _case(_gate("rolling_max_exit", window_bars=12, min_bars=12, min_peak_ret=0.001)),
    # engine
    "margin_check": _case(SIZING, leverage=5.0, differs_from="ema_strength_sizing"),
    "warmup": _case(periods={"warmup_days": 5}),
    # gates together, in order
    "combined": _case(EMA_SEP, _gate("time_filter"), SIZING, CHURN, _gate("shock_exit", shock_exit_abs_ret=0.002),
                      time_filter={"events": FOMC}, differs_from="ema_sep"),
}


def run_case(case: Case) -> pd.DataFrame:
    bt_config = backtest_config(case.cfg)
    if case.leverage is not None:
        bt_config = replace(bt_config, leverage=case.leverage)
    return run_config_periods(case.cfg, [PERIOD], bt_config=bt_config)[PERIOD[0]]


def load_golden() -> pd.DataFrame:
    """Columns: `<case>.position`, `<case>.equity`; index: bar time (UTC)."""
    df = pd.read_csv(GOLDEN, compression="gzip", index_col="ts", float_precision="round_trip")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


@pytest.fixture(scope="module")
def golden() -> pd.DataFrame:
    return load_golden()


@pytest.fixture(scope="module")
def xauusd_data() -> None:
    """Datasets aren't tracked by git: skip if this clone has no XAUUSD data for the window."""
    for root in ("5m_ohlc", "15m_ohlc"):
        try:
            load_dukascopy_ohlc(symbol="XAUUSD", start=PERIOD[1], end=PERIOD[2], root=DATA_ROOTS[root])
        except FileNotFoundError as e:
            pytest.skip(str(e))


@pytest.mark.parametrize("name", CASES)
def test_trades_exactly_as_before(name: str, golden: pd.DataFrame, xauusd_data: None) -> None:
    actual = run_case(CASES[name])
    # A warmed-up run starts one bar earlier (the 00:00 bar is in the previous day's file), so the
    # other cases are empty there: compare each case on its own bars.
    golden = golden[[f"{name}.position", f"{name}.equity"]].dropna()
    assert actual.index.equals(golden.index), f"bars differ: {len(actual)} now, {len(golden)} in golden (price data changed?)"

    for col, atol in (("position", 0.0), ("equity", 1e-10)):
        a, g = actual[col].to_numpy(float), golden[f"{name}.{col}"].to_numpy(float)
        bad = np.flatnonzero(~np.isclose(a, g, rtol=0.0, atol=atol))
        assert not len(bad), (
            f"{name}: {col} differs on {len(bad)} bars, first at {actual.index[bad[0]]}: now {a[bad[0]]}, golden {g[bad[0]]}"
        )

    # returns_net isn't stored; it must be the bar-to-bar return of the (checked) equity.
    returns = actual["equity"].pct_change().fillna(0.0)
    np.testing.assert_allclose(actual["returns_net"].iloc[1:], returns.iloc[1:], rtol=0.0, atol=1e-12)


def test_golden_has_exactly_these_cases(golden: pd.DataFrame) -> None:
    assert list(dict.fromkeys(c.rsplit(".", 1)[0] for c in golden.columns)) == list(CASES)


@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c.differs_from])
def test_case_has_an_effect(name: str, golden: pd.DataFrame) -> None:
    # A case that trades like its reference would pass no matter what its rule does.
    ref = CASES[name].differs_from
    both = golden[[f"{name}.equity", f"{ref}.equity"]].dropna()
    assert not both.iloc[:, 0].equals(both.iloc[:, 1])
    assert (golden[f"{name}.position"] > 0).any()


def test_every_gate_has_a_case() -> None:
    used = {g["gate"] for c in CASES.values() for g in c.cfg["pipeline"]}
    assert used == SCHEMA_GATE_NAMES - NOT_COVERED


def update() -> None:
    runs = {name: run_case(case) for name, case in CASES.items()}
    df = pd.DataFrame({f"{n}.{c}": bt[c] for n, bt in runs.items() for c in ("position", "equity")})
    df.index.name = "ts"
    # Fixed mtime and line endings: an unchanged run writes an identical file (no git diff).
    df.to_csv(GOLDEN, lineterminator="\n", compression={"method": "gzip", "mtime": 1})
    print(f"Wrote {GOLDEN}: {len(CASES)} cases, {len(df):,} bars")


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit("Run with pytest, or with --update to rewrite golden.csv.gz")
    update()
