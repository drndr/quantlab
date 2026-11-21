"""The buy-and-hold baseline must be in the market from the first possible bar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies import BacktestConfig, BuyAndHoldStrategy
from quantlab.webui.runner import _buy_and_hold_config


def test_baseline_enters_immediately_in_a_rally() -> None:
    # Price rises every bar and never revisits the first close. With a 1x margin
    # check the baseline used to stay flat here and return 0%.
    idx = pd.date_range("2026-01-01", periods=50, freq="5min", tz="UTC")
    prices = pd.Series(np.linspace(2000.0, 2100.0, len(idx)), index=idx)

    cfg = _buy_and_hold_config(prices, BacktestConfig(fee_per_lot=3.5, spread_per_lot=3.5))
    df = BuyAndHoldStrategy().run_backtest(prices, config=cfg).df

    held = df["contract_units"] != 0
    assert not held.iloc[0] and held.iloc[1:].all()  # lag=1: in from bar 1, held to the end

    # ~1x notional: 0.5 oz at $2,000 on $1,000. Return = gold's move from the entry bar.
    oz = df["contract_units"].iloc[1]
    assert abs(oz - 0.5) < 1e-12
    expected = oz * (prices.iloc[-1] - prices.iloc[1]) / 1000.0
    assert abs((df["equity"].iloc[-1] / 1000.0 - 1.0) - expected) < 1e-4  # costs ~0.0035%
