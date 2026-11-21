"""evaluate_from: indicators see the warm-up bars, trading and metrics start later."""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

import quantlab.pipeline as pipeline
from quantlab.configs.schema import validate_config_dict
from quantlab.strategies import BacktestConfig, TrendStrategy

CFG = {"trend": {"fast": 30, "slow": 110}, "pipeline": [{"gate": "time_filter", "params": {}}]}
PERIOD = ("P", dt.date(2020, 1, 1), dt.date(2020, 1, 31))


def _fake_data_from(first_day: dt.date):
    """Stand-in for load_period_data: hourly bars, but nothing before `first_day`."""

    def load(symbol, start, end, **_):
        lo = pd.Timestamp(max(start, first_day), tz="UTC")
        hi = pd.Timestamp(end + dt.timedelta(days=1), tz="UTC")
        idx = pd.date_range(lo, hi, freq="1h", inclusive="left")
        return {"prices": pd.Series(100.0, index=idx)}

    return load


def test_missing_warmup_data_warns_and_runs_with_what_exists(monkeypatch) -> None:
    # Dataset starts on the period start (like 2020 without Dec 2019).
    monkeypatch.setattr(pipeline, "load_period_data", _fake_data_from(dt.date(2020, 1, 1)))
    with pytest.warns(UserWarning, match="running with 0 days of warm-up"):
        (p,) = pipeline.prepare_periods(CFG, [PERIOD], warmup_days=21)
    assert p.load_start == p.start  # treated as a cold start


def test_full_warmup_data_does_not_warn(monkeypatch) -> None:
    # Warm-up window opens on a weekend: first bar 2 days late is not missing data.
    monkeypatch.setattr(pipeline, "load_period_data", _fake_data_from(dt.date(2019, 12, 13)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (p,) = pipeline.prepare_periods(CFG, [PERIOD], warmup_days=21)
    assert p.load_start == dt.date(2019, 12, 11)


def test_warmup_bars_feed_indicators_but_not_the_backtest() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)  # steady uptrend
    start = idx[10]
    cfg = BacktestConfig(fee_per_lot=0.0, spread_per_lot=0.0)
    strat = TrendStrategy(fast=2, slow=5)

    warm = strat.run_backtest(prices, config=cfg, evaluate_from=start).df
    cold = strat.run_backtest(prices[prices.index >= start], config=cfg).df

    assert warm.index[0] == start and len(warm) == len(cold) == 10
    # Warm: MAs already crossed before `start`, so it is long from the first tradable bar (lag=1).
    assert warm["position"].iloc[1] == 1.0
    # Cold: the slow MA needs 5 bars of its own before any signal.
    assert cold["position"].iloc[1] == 0.0


def test_warmup_days_defaults_to_cold_start_and_rejects_negatives() -> None:
    base = {"periods": {"mode": "yearly"}, "trend": {"fast": 30, "slow": 110}, "pipeline": []}
    assert validate_config_dict(base)["periods"]["warmup_days"] == 0
    bad = {**base, "periods": {"mode": "yearly", "warmup_days": -1}}
    try:
        validate_config_dict(bad)
    except Exception as e:  # pydantic.ValidationError
        assert "warmup_days" in str(e)
    else:
        raise AssertionError("negative warmup_days must be rejected")
